"""Durable per-service queue; internal API with project-scoped Artifact inputs."""
from contextlib import asynccontextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from .integrity import sha256
from .media import process, MediaError
from .schemas import SRRequest
from .capacity import CapacityError
TASK_PATTERN = re.compile('video_task_[0-9a-f]{32}')

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
    if not isinstance(filename, str) or not filename or Path(filename).name != filename or (filename in {'.', '..'}) or (not str(data.get('media_type', '')).startswith('video/')):
        raise HTTPException(422, {'code': 'invalid_source'})
    path = directory / filename
    if path.is_symlink() or not path.is_file() or path.resolve().parent != directory.resolve():
        raise HTTPException(404, {'code': 'source_not_found'})
    if path.stat().st_size != data.get('size') or data.get('sha256') != req.source_sha256 or sha256(path) != req.source_sha256:
        raise HTTPException(422, {'code': 'source_integrity_failed'})
    return path

class Queue:

    def __init__(self, root, artifact_root, engine):
        (self.root, self.artifact_root, self.engine) = (Path(root).resolve(), Path(artifact_root).resolve(), engine)
        (self.request_type, self.service) = (SRRequest, 'sr')
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.db = sqlite3.connect(self.root / 'tasks.sqlite3', check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, request TEXT NOT NULL, digest TEXT NOT NULL, state TEXT NOT NULL, result TEXT, error TEXT, created REAL NOT NULL)')
        self.db.execute("UPDATE tasks SET state='failed', error='interrupted' WHERE state='running'")
        self.db.commit()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join()
        self.db.close()

    def get(self, task_id):
        if not TASK_PATTERN.fullmatch(task_id):
            raise HTTPException(404, {'code': 'task_not_found'})
        with self.lock:
            row = self.db.execute('SELECT state,result,error,created FROM tasks WHERE id=?', (task_id,)).fetchone()
        if not row:
            raise HTTPException(404, {'code': 'task_not_found'})
        return {'video_task_id': task_id, 'status': row[0], 'result': json.loads(row[1]) if row[1] else None, 'error': {'code': row[2], 'message': 'Video processing failed'} if row[2] else None, 'created_at': row[3]}

    def submit(self, req):
        canonical = req.model_dump_json()
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        with self.lock:
            row = self.db.execute('SELECT digest FROM tasks WHERE id=?', (req.idempotency_key,)).fetchone()
            if row:
                if row[0] != digest:
                    raise HTTPException(409, {'code': 'idempotency_conflict'})
                return self.get(req.idempotency_key)
            source_path(req, self.artifact_root)
            waiting = self.db.execute("SELECT count(*) FROM tasks WHERE state IN ('queued','running')").fetchone()[0]
            if waiting >= 8:
                raise HTTPException(429, {'code': 'queue_full'})
            self.db.execute('INSERT INTO tasks VALUES (?,?,?,?,?,?,?)', (req.idempotency_key, canonical, digest, 'queued', None, None, time.time()))
            self.db.commit()
            return self.get(req.idempotency_key)

    def run_one(self):
        with self.lock:
            row = self.db.execute("SELECT id,request FROM tasks WHERE state='queued' ORDER BY created LIMIT 1").fetchone()
            if not row:
                return False
            (task_id, raw) = row
            self.db.execute("UPDATE tasks SET state='running' WHERE id=?", (task_id,))
            self.db.commit()
        try:
            req = self.request_type.model_validate_json(raw)
            source = source_path(req, self.artifact_root)
            directory = self.root / 'projects' / req.project_id / task_id
            directory.mkdir(parents=True, exist_ok=False)
            result = process(self.engine, source, directory, req)
            result.update(model=self.engine.metadata, input_sha256=req.source_sha256, output_sha256=sha256(directory / 'output.mp4'), preview_sha256=sha256(directory / 'preview.mp4'))
            (directory / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
            with self.lock:
                self.db.execute("UPDATE tasks SET state='succeeded',result=? WHERE id=?", (json.dumps(result), task_id))
                self.db.commit()
            logging.info('%s_completed %s', self.service, task_id)
        except Exception as exc:
            code = 'invalid_media' if isinstance(exc, MediaError) else 'inference_failed'
            if isinstance(exc, CapacityError):
                code = str(exc) if str(exc) in {'capacity_profile_mismatch', 'gpu_memory_occupancy_changed'} else 'inference_failed'
            if isinstance(exc, HTTPException):
                code = 'source_integrity_failed'
            logging.exception('%s_failed %s', self.service, task_id)
            with self.lock:
                self.db.execute("UPDATE tasks SET state='failed',error=? WHERE id=?", (code, task_id))
                self.db.commit()
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
    service = 'sr'

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
        return {'status': 'ok', 'service': service, 'model': queue.engine.metadata}

    @app.post('/v1/sr', status_code=202)
    def generate_sr(req: SRRequest):
        return app.state.queue.submit(req)

    @app.get(f'/v1/{service}/{{task_id}}')
    def status(task_id: str):
        return app.state.queue.get(task_id)

    @app.get(f'/v1/{service}/{{task_id}}/content')
    def content(task_id: str):
        return FileResponse(app.state.queue.content(task_id, 'output.mp4'), media_type='video/mp4')

    @app.get(f'/v1/{service}/{{task_id}}/preview')
    def preview(task_id: str):
        return FileResponse(app.state.queue.content(task_id, 'preview.mp4'), media_type='video/mp4')
    return app
