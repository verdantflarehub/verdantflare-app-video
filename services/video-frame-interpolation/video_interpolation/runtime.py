"""Durable per-service queue; internal API with project-scoped Artifact inputs."""
from contextlib import asynccontextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sqlite3
import threading
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from .integrity import sha256
from .media import process, MediaError, TaskCancelledError
from .gpu_guard import (
    FencePersistenceError,
    GPUMemoryUnavailable,
    LeaseFenceGuard,
    StaleFenceError,
    read_runtime_identity,
    verify_internal_bearer,
)
from .schemas import InterpolationRequest

TASK_PATTERN = re.compile('video_task_[0-9a-f]{32}')


class RevokeLeaseRequest(BaseModel):
    lease_id: str = ''
    fencing_token: int = 0
    reason: str = ''


def source_path(req, root):
    if (root / 'artifacts').is_symlink():
        raise HTTPException(404, {'code': 'source_not_found'})
    directory = root / 'artifacts' / req.source_artifact_id
    metadata = directory / 'metadata.json'
    if directory.is_symlink() or metadata.is_symlink() or (not metadata.is_file()):
        raise HTTPException(404, {'code': 'source_not_found'})
    try:
        data = json.loads(metadata.read_text())
        if not isinstance(data, dict):
            raise ValueError('invalid metadata')
    except (ValueError, OSError):
        raise HTTPException(422, {'code': 'invalid_source'}) from None
    if data.get('project_id') != req.project_id or data.get('artifact_id') != req.source_artifact_id:
        raise HTTPException(404, {'code': 'source_not_found'})
    filename = data.get('filename', '')
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or (filename in {'.', '..'})
        or (not str(data.get('media_type', '')).startswith('video/'))
    ):
        raise HTTPException(422, {'code': 'invalid_source'})
    path = directory / filename
    if path.is_symlink() or not path.is_file() or path.resolve().parent != directory.resolve():
        raise HTTPException(404, {'code': 'source_not_found'})
    if path.stat().st_size != data.get('size') or data.get('sha256') != req.source_sha256 or sha256(path) != req.source_sha256:
        raise HTTPException(422, {'code': 'source_integrity_failed'})
    return path


class Queue:

    def __init__(self, root, artifact_root, engine, fence_guard: LeaseFenceGuard | None = None):
        (self.root, self.artifact_root, self.engine) = (Path(root).resolve(), Path(artifact_root).resolve(), engine)
        (self.request_type, self.service) = (InterpolationRequest, 'interpolate')
        self.root.mkdir(parents=True, exist_ok=True)
        self.fence_guard = fence_guard or LeaseFenceGuard(state_path=self.root / 'gpu_guard_state.json')
        self.identity = read_runtime_identity()
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self._active_task_id = None
        self._active_cancel_event = None
        self._active_settled_event = None
        self._active_proc = None
        self.db = sqlite3.connect(self.root / 'tasks.sqlite3', check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA busy_timeout=30000')
        self.db.execute(
            'CREATE TABLE IF NOT EXISTS tasks ('
            'id TEXT PRIMARY KEY, '
            'request TEXT NOT NULL, '
            'digest TEXT NOT NULL, '
            'state TEXT NOT NULL, '
            'result TEXT, '
            'error TEXT, '
            'created REAL NOT NULL, '
            'started REAL, '
            'finished REAL, '
            'worker_settled INTEGER NOT NULL DEFAULT 1'
            ')'
        )
        existing_cols = {row[1] for row in self.db.execute('PRAGMA table_info(tasks)').fetchall()}
        if 'started' not in existing_cols:
            self.db.execute('ALTER TABLE tasks ADD COLUMN started REAL')
        if 'finished' not in existing_cols:
            self.db.execute('ALTER TABLE tasks ADD COLUMN finished REAL')
        if 'worker_settled' not in existing_cols:
            self.db.execute('ALTER TABLE tasks ADD COLUMN worker_settled INTEGER NOT NULL DEFAULT 1')
        now = time.time()
        self.db.execute(
            "UPDATE tasks SET state='failed', error='interrupted', finished=COALESCE(finished, ?), worker_settled=1 "
            "WHERE state='running' OR worker_settled=0",
            (now,),
        )
        self.db.commit()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        with self.lock:
            if self._active_cancel_event is not None:
                self._active_cancel_event.set()
            if self._active_proc is not None and self._active_proc.poll() is None:
                self._active_proc.kill()
        if self.thread:
            self.thread.join()
        self.db.close()

    def active_task_ids(self):
        with self.lock:
            return [self._active_task_id] if self._active_task_id is not None else []

    def get(self, task_id):
        if not TASK_PATTERN.fullmatch(task_id):
            raise HTTPException(404, {'code': 'task_not_found'})
        with self.lock:
            row = self.db.execute(
                'SELECT state,result,error,created,started,finished,worker_settled FROM tasks WHERE id=?',
                (task_id,),
            ).fetchone()
            active_count = 1 if self._active_task_id is not None else 0
        if not row:
            raise HTTPException(404, {'code': 'task_not_found'})
        settled = bool(row[6])
        is_terminal = row[0] in ('succeeded', 'failed', 'cancelled')
        payload = {
            'video_task_id': task_id,
            'status': row[0],
            'result': json.loads(row[1]) if row[1] else None,
            'error': {'code': row[2], 'message': 'Video processing failed'} if row[2] else None,
            'created_at': row[3],
            'started_at': row[4],
            'finished_at': row[5],
            'worker_settled': settled,
        }
        try:
            proof = self.fence_guard.build_warm_reclaim_proof(
                task_id=task_id,
                active_tasks=active_count,
                settled=bool(settled and is_terminal),
                identity=self.identity,
            )
        except FencePersistenceError as exc:
            raise HTTPException(503, {'code': str(exc)}) from None
        if proof is not None:
            payload['reclaim_proof'] = proof
        return payload

    def submit(self, req, headers=None):
        canonical = req.model_dump_json()
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        stage_digest = hashlib.sha256(req.model_dump_json(exclude={'idempotency_key'}).encode()).hexdigest()
        try:
            _, replay_task_id = self.fence_guard.extract_and_validate(
                headers or {},
                task_id=req.idempotency_key,
                request_digest=stage_digest,
                identity=self.identity,
            )
        except FencePersistenceError as exc:
            raise HTTPException(503, {'code': str(exc)}) from None
        except PermissionError as exc:
            raise HTTPException(403, {'code': str(exc)}) from None
        except StaleFenceError as exc:
            raise HTTPException(409, {'code': str(exc)}) from None
        except ValueError as exc:
            raise HTTPException(409, {'code': str(exc)}) from None

        if replay_task_id is not None:
            with self.lock:
                replay_row = self.db.execute('SELECT id FROM tasks WHERE id=?', (replay_task_id,)).fetchone()
            if replay_row is not None and replay_task_id != req.idempotency_key:
                return self.get(replay_task_id)
            if replay_row is None and self.fence_guard.is_task_revoked_or_sealed(replay_task_id):
                raise HTTPException(409, {'code': 'stale_or_revoked_fencing_token'})

        with self.lock:
            row = self.db.execute('SELECT digest FROM tasks WHERE id=?', (req.idempotency_key,)).fetchone()
            if row:
                if row[0] != digest:
                    raise HTTPException(409, {'code': 'idempotency_conflict'})
                return self.get(req.idempotency_key)
            try:
                source_path(req, self.artifact_root)
                waiting = self.db.execute("SELECT count(*) FROM tasks WHERE state IN ('queued','running')").fetchone()[0]
                if waiting >= 8:
                    raise HTTPException(429, {'code': 'queue_full'})
                self.db.execute(
                    'INSERT INTO tasks (id, request, digest, state, result, error, created, started, finished, worker_settled) '
                    'VALUES (?,?,?,?,?,?,?,?,?,?)',
                    (req.idempotency_key, canonical, digest, 'queued', None, None, time.time(), None, None, 1),
                )
                self.db.commit()
            except BaseException:
                try:
                    self.fence_guard.seal_task(req.idempotency_key)
                except FencePersistenceError:
                    pass
                raise
            return self.get(req.idempotency_key)

    def revoke_lease(self, lease_id: str, fencing_token: int) -> dict[str, object]:
        try:
            affected = self.fence_guard.revoke_lease(lease_id, fencing_token)
        except FencePersistenceError as exc:
            raise HTTPException(503, {'code': str(exc)}) from None
        for tid in affected:
            try:
                self.cancel(tid)
            except HTTPException:
                pass
        try:
            with self.lock:
                active_count = len(self.active_task_ids())
                proof = self.fence_guard.reclaim_and_seal(
                    lease_id=lease_id,
                    fencing_token=fencing_token,
                    active_tasks=active_count,
                    settled=(active_count == 0),
                    identity=self.identity,
                ) or {}
        except FencePersistenceError as exc:
            raise HTTPException(503, {'code': str(exc)}) from None
        is_settled = bool(proof.get('reclaimed'))
        return {**proof, 'revoked': True, 'settled': is_settled}

    def reclaim_lease(self, lease_id: str, fencing_token: int) -> dict[str, object]:
        try:
            with self.lock:
                active_count = len(self.active_task_ids())
                proof = self.fence_guard.reclaim_and_seal(
                    lease_id=lease_id,
                    fencing_token=fencing_token,
                    active_tasks=active_count,
                    settled=True,
                    identity=self.identity,
                )
        except FencePersistenceError as exc:
            raise HTTPException(503, {'code': str(exc)}) from None
        if proof is None:
            raise HTTPException(400, {'code': 'missing_lease_or_token'})
        if not proof.get('reclaimed'):
            raise HTTPException(
                409,
                {
                    'code': 'worker_not_settled',
                    'settled': False,
                    'active_tasks': int(proof.get('active_tasks', 1)),
                    'reclaim_proof': proof,
                },
            )
        return {'ok': True, 'settled': True, 'active_tasks': 0, 'reclaim_proof': proof}

    def cancel(self, task_id):
        if not TASK_PATTERN.fullmatch(task_id):
            raise HTTPException(404, {'code': 'task_not_found'})
        settled_event = None
        with self.lock:
            row = self.db.execute('SELECT state FROM tasks WHERE id=?', (task_id,)).fetchone()
            if not row:
                raise HTTPException(404, {'code': 'task_not_found'})
            state = row[0]
            if state in ('succeeded', 'failed'):
                raise HTTPException(409, {'code': 'task_already_terminal'})
            if state == 'queued':
                now = time.time()
                self.db.execute(
                    "UPDATE tasks SET state='cancelled', result=NULL, error=NULL, finished=?, worker_settled=1 WHERE id=?",
                    (now, task_id),
                )
                self.db.commit()
                self.fence_guard.seal_task(task_id)
                return self.get(task_id)
            if state == 'cancelled':
                if self._active_task_id == task_id and self._active_settled_event is not None:
                    settled_event = self._active_settled_event
                else:
                    return self.get(task_id)
            elif state == 'running':
                if self._active_task_id == task_id and self._active_cancel_event is not None:
                    self._active_cancel_event.set()
                    if self._active_proc is not None and self._active_proc.poll() is None:
                        try:
                            self._active_proc.kill()
                        except OSError:
                            pass
                    settled_event = self._active_settled_event
                else:
                    now = time.time()
                    self.db.execute(
                        "UPDATE tasks SET state='cancelled', result=NULL, error=NULL, finished=?, worker_settled=1 WHERE id=?",
                        (now, task_id),
                    )
                    self.db.commit()
                    return self.get(task_id)
        if settled_event is not None:
            settled_event.wait(timeout=30.0)
        return self.get(task_id)

    def _register_proc(self, proc):
        with self.lock:
            self._active_proc = proc
            if proc is not None and self._active_cancel_event is not None and self._active_cancel_event.is_set():
                try:
                    proc.kill()
                except OSError:
                    pass

    def run_one(self):
        with self.lock:
            row = self.db.execute("SELECT id,request FROM tasks WHERE state='queued' ORDER BY created LIMIT 1").fetchone()
            if not row:
                return False
            (task_id, raw) = row
            try:
                self.fence_guard.verify_before_inference(task_id)
            except StaleFenceError:
                self.db.execute(
                    "UPDATE tasks SET state='cancelled', result=NULL, error=NULL, finished=?, worker_settled=1 WHERE id=?",
                    (time.time(), task_id),
                )
                self.db.commit()
                try:
                    self.fence_guard.seal_task(task_id)
                except FencePersistenceError:
                    pass
                return True
            except FencePersistenceError:
                self.db.execute(
                    "UPDATE tasks SET state='failed', result=NULL, error='fence_state_corrupted', finished=?, worker_settled=1 WHERE id=?",
                    (time.time(), task_id),
                )
                self.db.commit()
                return True
            started_at = time.time()
            cancel_event = threading.Event()
            settled_event = threading.Event()
            self._active_task_id = task_id
            self._active_cancel_event = cancel_event
            self._active_settled_event = settled_event
            self._active_proc = None
            self.db.execute(
                "UPDATE tasks SET state='running', started=?, worker_settled=0 WHERE id=?",
                (started_at, task_id),
            )
            self.db.commit()

        directory = None
        result = None
        failure_code = None
        began_engine = False
        try:
            req = self.request_type.model_validate_json(raw)
            if cancel_event.is_set():
                raise TaskCancelledError('cancelled')
            source = source_path(req, self.artifact_root)
            directory = self.root / 'projects' / req.project_id / task_id
            directory.mkdir(parents=True, exist_ok=True)
            memory = self.engine.begin_task()
            began_engine = True
            result = process(
                self.engine,
                source,
                directory,
                req,
                cancel_event=cancel_event,
                register_proc=self._register_proc,
            )
            result['gpu_memory'] = memory
        except (TaskCancelledError, StaleFenceError):
            cancel_event.set()
        except Exception as exc:
            if not cancel_event.is_set():
                code = 'invalid_media' if isinstance(exc, MediaError) else 'inference_failed'
                if isinstance(exc, GPUMemoryUnavailable):
                    code = 'gpu_memory_occupancy_changed'
                if isinstance(exc, HTTPException):
                    code = 'source_integrity_failed'
                failure_code = code
                logging.exception('%s_failed %s', self.service, task_id)
        finally:
            if began_engine:
                peak = self.engine.end_task()
                if result is not None and 'gpu_memory' in result:
                    result['gpu_memory']['peak_allocated_bytes'] = peak

            finished_at = time.time()
            with self.lock:
                if cancel_event.is_set():
                    if directory is not None and directory.exists():
                        shutil.rmtree(directory, ignore_errors=True)
                    self.db.execute(
                        "UPDATE tasks SET state='cancelled', result=NULL, error=NULL, finished=?, worker_settled=1 WHERE id=?",
                        (finished_at, task_id),
                    )
                    self.db.commit()
                    logging.info('%s_cancelled %s', self.service, task_id)
                elif failure_code is not None or result is None:
                    self.db.execute(
                        "UPDATE tasks SET state='failed', error=?, finished=?, worker_settled=1 WHERE id=?",
                        (failure_code or 'inference_failed', finished_at, task_id),
                    )
                    self.db.commit()
                else:
                    result.update(
                        model=self.engine.metadata,
                        input_sha256=req.source_sha256,
                        output_sha256=sha256(directory / 'output.mp4'),
                        preview_sha256=sha256(directory / 'preview.mp4'),
                    )
                    (directory / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
                    self.db.execute(
                        "UPDATE tasks SET state='succeeded', result=?, finished=?, worker_settled=1 WHERE id=?",
                        (json.dumps(result), finished_at, task_id),
                    )
                    self.db.commit()
                    logging.info('%s_completed %s', self.service, task_id)
                self._active_task_id = None
                self._active_cancel_event = None
                self._active_settled_event = None
                self._active_proc = None
                try:
                    self.fence_guard.seal_task(task_id)
                except FencePersistenceError:
                    logging.exception('%s_fence_seal_failed %s', self.service, task_id)
                settled_event.set()
        return True

    def run(self):
        while not self.stop.is_set():
            if not self.run_one():
                self.stop.wait(0.5)

    def content(self, task_id, filename):
        record = self.get(task_id)
        if record['status'] != 'succeeded':
            raise HTTPException(409, {'code': 'result_not_ready'})
        with self.lock:
            raw = self.db.execute('SELECT request FROM tasks WHERE id=?', (task_id,)).fetchone()[0]
        req = self.request_type.model_validate_json(raw)
        return self.root / 'projects' / req.project_id / task_id / filename


def create_app(engine_factory):
    service = 'interpolate'

    @asynccontextmanager
    async def lifespan(app):
        project_root = Path(os.environ.get('VIDEO_PROCESSING_ROOT', f'/data/{service}')).resolve()
        artifact_root = Path(os.environ.get('VIDEO_ARTIFACT_ROOT', '/source/video-mcp')).resolve()
        engine = engine_factory()
        queue = Queue(project_root, artifact_root, engine)
        app.state.queue = queue
        queue.start()
        try:
            yield
        finally:
            queue.close()

    app = FastAPI(title=f'Video {service} API', lifespan=lifespan)

    @app.get('/health')
    def health():
        queue = getattr(app.state, 'queue', None)
        if queue is None or queue.thread is None or (not queue.thread.is_alive()):
            raise HTTPException(503, {'code': 'not_ready'})
        active_ids = queue.active_task_ids()
        return {
            'status': 'ok',
            'service': service,
            'model': queue.engine.metadata,
            'gpu_worker_active': bool(active_ids),
            'active_task_ids': active_ids,
            'runtime_identity': queue.identity,
        }

    @app.post('/v1/interpolate', status_code=202)
    def generate_interpolation(req: InterpolationRequest, request: Request):
        return app.state.queue.submit(req, headers=request.headers)

    @app.post('/internal/v1/gpu-residency/revoke-lease')
    def revoke_lease(body: RevokeLeaseRequest, request: Request):
        if not verify_internal_bearer(request.headers.get('Authorization', '')):
            raise HTTPException(401, {'code': 'unauthorized'})
        queue = getattr(app.state, 'queue', None)
        if queue is None:
            raise HTTPException(503, {'code': 'not_ready'})
        return queue.revoke_lease(body.lease_id, body.fencing_token)

    @app.post('/internal/v1/gpu-guard/reclaim')
    def reclaim_lease(body: RevokeLeaseRequest, request: Request):
        if not verify_internal_bearer(request.headers.get('Authorization', '')):
            raise HTTPException(401, {'code': 'unauthorized'})
        queue = getattr(app.state, 'queue', None)
        if queue is None:
            raise HTTPException(503, {'code': 'not_ready'})
        return queue.reclaim_lease(body.lease_id, body.fencing_token)

    @app.get(f'/v1/{service}/{{task_id}}')
    def status(task_id: str):
        return app.state.queue.get(task_id)

    @app.delete(f'/v1/{service}/{{task_id}}')
    def cancel_interpolation(task_id: str):
        return app.state.queue.cancel(task_id)

    @app.get(f'/v1/{service}/{{task_id}}/content')
    def content(task_id: str):
        return FileResponse(app.state.queue.content(task_id, 'output.mp4'), media_type='video/mp4')

    @app.get(f'/v1/{service}/{{task_id}}/preview')
    def preview(task_id: str):
        return FileResponse(app.state.queue.content(task_id, 'preview.mp4'), media_type='video/mp4')

    return app
