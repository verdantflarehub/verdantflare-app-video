"""Task-number based same-node H3 post-processing adapter."""
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import httpx
from .channel_errors import MESSAGES
from .executor import ExecutionError, serialized, task_serialized
from .gpu_lease import LeaseBusy
from .tasks import TaskConflict
from .archive import ArchiveError, S3Archive
from .artifacts import ArtifactError


class H3LatentExecutor:
    service = 'h3-latent-upscale'

    def __init__(self, executor):
        self.executor = executor
        self.tasks, self.artifacts = executor.tasks, executor.artifacts
        self._client_override = None
        self._lock = executor._lock
        self.url = os.environ.get('H3_LATENT_RUNTIME_URL', 'http://video-h3-latent-upscaler:8000').rstrip('/')
        self.token = os.environ.get('H3_LATENT_RUNTIME_TOKEN', '')
        self.index = Path(os.environ.get('H3_LATENT_SOURCE_INDEX', '/data/h3-latent/source-index.sqlite'))
        self.node = os.environ.get('H3_LATENT_NODE_NAME', '')

    @property
    def client(self):
        return self._client_override if self._client_override is not None else self.executor.client

    @client.setter
    def client(self, value) -> None:
        self._client_override = value

    def archive(self):
        return S3Archive.from_environment()

    @property
    def headers(self):
        if not self.token:
            raise ExecutionError('H3 latent runtime is not configured')
        return {'Authorization': 'Bearer ' + self.token}

    def register_source(self, source):
        if source.service not in ('h3-vdn', 'h3-singularity') or source.status != 'succeeded':
            raise ExecutionError('source_not_ready: a successful supported H3 task is required')
        url, headers, _, _ = self.executor.runtime(self.executor._record_route(source))
        response = self.client.get(f'{url}/v1/videos/{source.runtime_task_id}', headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        if data.get('id') != source.runtime_task_id or data.get('status') != 'completed':
            raise ExecutionError('source_not_ready')
        bundle = (data.get('result') or {}).get('latent_bundle')
        if not isinstance(bundle, dict):
            raise ExecutionError('missing_latent_bundle: MP4-only tasks cannot be processed')
        if bundle.get('schema') != 'h3-latent-bundle/v1' or bundle.get('source_video_task_id') != source.video_task_id:
            raise ExecutionError('source_identity_mismatch')
        if not self.node or bundle.get('node') != self.node:
            raise ExecutionError('source_node_mismatch')
        relative = bundle.get('manifest_path')
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ExecutionError('invalid_source_descriptor')
        digest = bundle.get('manifest_sha256')
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ExecutionError('invalid_source_descriptor')
        self.index.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.index) as db:
            db.execute('PRAGMA journal_mode=DELETE')
            db.execute('PRAGMA synchronous=FULL')
            db.execute('''CREATE TABLE IF NOT EXISTS h3_sources (
                project_id TEXT NOT NULL, task_id TEXT NOT NULL, node TEXT NOT NULL,
                status TEXT NOT NULL, manifest_path TEXT, manifest_sha256 TEXT,
                PRIMARY KEY(project_id,task_id))''')
            values = (source.project_id, source.video_task_id, self.node, 'succeeded', relative, digest)
            old = db.execute('SELECT * FROM h3_sources WHERE project_id=? AND task_id=?', values[:2]).fetchone()
            if old is not None and old != values:
                raise ExecutionError('source_changed_after_registration')
            db.execute('INSERT OR IGNORE INTO h3_sources VALUES (?,?,?,?,?,?)', values)
        return digest

    @serialized
    def generate(self, source_video_task_id, *, project_id=None, idempotency_key=None,
                 profile_id=None, target_width=None, target_height=None, seed=None):
        # Existing MCP bearer authorization is workspace-wide. An explicitly
        # selected project further restricts this lookup; never guess file paths.
        source = self.tasks.get(source_video_task_id)
        if project_id is not None and source.project_id != project_id:
            raise ExecutionError('source_not_found')
        if source.service not in ('h3-vdn', 'h3-singularity'):
            raise ExecutionError('unsupported_source_route')
        source = self.executor.status(source_video_task_id)
        if source.status != 'succeeded':
            raise ExecutionError('source_not_ready')
        self.headers
        parameters = {}
        if (target_width is None) != (target_height is None):
            raise ValueError('provide both target dimensions')
        for key, value in [('target_width', target_width), ('target_height', target_height)]:
            if value is not None:
                if type(value) is not int or not 32 <= value <= 2048 or value % 32:
                    raise ValueError('invalid target dimensions')
                parameters[key] = value
        if seed is not None:
            if type(seed) is not int or not 0 <= seed <= 4294967295:
                raise ValueError('invalid seed')
            parameters['seed'] = seed
        if profile_id is not None:
            if not isinstance(profile_id, str) or not re.fullmatch('[a-z0-9][a-z0-9_.-]{0,95}', profile_id):
                raise ValueError('invalid profile')
            parameters['profile_id'] = profile_id
        request = {'service': self.service, 'source_video_task_id': source.video_task_id,
                   'project_id': source.project_id, **parameters}
        digest = 'sha256:' + hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        key = idempotency_key or 'h3-latent-' + digest[7:]
        if not isinstance(key, str) or not key.strip() or len(key) > 128:
            raise ValueError('invalid idempotency key')
        old = self.tasks.find_idempotency(source.project_id, key)
        if old:
            if old.input_digest != digest:
                raise TaskConflict('idempotency key has different input')
            return old
        self.register_source(source)
        try:
            self.archive().preflight()
        except ArchiveError as exc:
            raise ExecutionError('archive_unavailable: result archival must be ready before processing') from exc
        # Reject known unavailability before creating a queued task. Once a POST
        # is attempted, keep the existing ambiguous-submission recovery rules.
        try:
            response = self.client.get(self.url + '/health', headers=self.headers, timeout=5)
            response.raise_for_status()
            ready = response.json()
            if (not isinstance(ready, dict) or ready.get('ready') is not True
                    or type(ready.get('gpu_count')) is not int or ready['gpu_count'] != 1
                    or ready.get('cpu_offload') is not False
                    or not isinstance(ready.get('profile_id'), str) or not ready['profile_id']):
                raise ValueError('runtime_not_ready')
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise ExecutionError('runtime_not_ready: no processing task was submitted') from exc
        record = self.tasks.create(project_id=source.project_id, idempotency_key=key, input_digest=digest,
                                   request=request, runtime_task_id=None, status='queued',
                                   error={'code': 'submission_unconfirmed', 'message': 'Submission is pending confirmation'})
        record = self.tasks.update(record, service=self.service)
        return self.submit(record)

    def _fail_record(self, record, error):
        pstate = {
            k: v
            for k, v in (record.provider_state or {}).items()
            if k not in {'waiting_reason', 'gpu_dispatched'}
        }
        pstate['worker_settled'] = True
        updated = self.tasks.update(record, status='failed', error=error, provider_state=pstate)
        self.executor.stage_slot.release(updated.video_task_id, force=True)
        return self.executor.maybe_release_scheduler_lease(updated)

    def submit(self, record):
        payload = {key: value for key, value in record.request.items() if key != 'service'}
        payload['idempotency_key'] = record.video_task_id
        acquired = False
        grant_headers = {}
        try:
            local_gating = bool(self.executor.gpu.enabled or self.executor.stage_slot.enabled)
            if self.executor.scheduler.enabled:
                try:
                    sched_ok, record, grant_headers = self.executor.acquire_scheduler_grant(
                        record,
                        self.service,
                        runtime_url=self.url,
                        runtime_headers=self.headers,
                        bind_immediately=not local_gating,
                    )
                except LeaseBusy:
                    return self._fail_record(
                        record,
                        {'code': 'gpu_busy', 'message': MESSAGES['gpu_busy'], 'retryable': True},
                    )
                if not sched_ok:
                    return record

            if local_gating:
                try:
                    acquired = self.executor.acquire_local_gpu(record.video_task_id, self.service, record=record)
                except LeaseBusy:
                    return self._fail_record(
                        record,
                        {'code': 'gpu_busy', 'message': MESSAGES['gpu_busy'], 'retryable': True},
                    )
                if not acquired:
                    record = self.executor.rollback_unstarted_scheduler_grant(record)
                    return self.executor.defer_local_gpu_queue(record)

            if self.executor.scheduler.enabled:
                record, grant_headers = self.executor.ensure_scheduler_target_bound(
                    record,
                    runtime_url=self.url,
                    runtime_headers=self.headers,
                )

            if record.provider_state or self.executor.gpu.enabled or self.executor.stage_slot.enabled or self.executor.scheduler.enabled:
                pstate = {k: v for k, v in (record.provider_state or {}).items() if k != 'waiting_reason'}
                pstate['gpu_dispatched'] = True
                record = self.tasks.update(record, provider_state=pstate)

            response = self.client.post(
                self.url + '/v1/upscales',
                json=payload,
                headers={**self.headers, **grant_headers},
                timeout=30,
            )
            if response.status_code in {400, 404, 409, 422, 429, 503}:
                return self._fail_record(record, {
                    'code': 'upscale_request_rejected', 'message': 'Source, profile or runtime capacity is unavailable'})
            response.raise_for_status()
            return self.update(record, response.json())
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return self.tasks.update(record, error={'code': 'submission_unconfirmed', 'message': 'Query this task ID to recover'})
        finally:
            if acquired:
                self.executor.release_or_reconcile_local_gpu(record.video_task_id)

    def update(self, record, data):
        if data.get('video_task_id') != record.video_task_id or data.get('status') not in {'queued', 'running', 'succeeded', 'failed'}:
            raise ValueError('invalid runtime task response')
        pstate = dict(record.provider_state or {})
        for key in ('started_at', 'finished_at', 'worker_settled', 'gpu_epoch', 'reclaim_proof'):
            if key in data:
                pstate[key] = data[key]
        if data['status'] == 'succeeded':
            if pstate:
                pstate.setdefault('worker_settled', True)
                record = self.tasks.update(record, provider_state=pstate)
            return self.collect(record, data['result'])
        error = None if data['status'] != 'failed' else {'code': 'upscale_failed', 'message': 'H3 latent post-processing failed'}
        if data['status'] == 'failed' and (pstate or self.executor.stage_slot.enabled or self.executor.scheduler.enabled):
            pstate.setdefault('worker_settled', True)
        updated = self.tasks.update(record, status=data['status'], error=error, provider_state=pstate or None)
        if self.executor.stage_slot.is_task_settled(updated):
            self.executor.stage_slot.release(updated.video_task_id)
            updated = self.executor.maybe_release_scheduler_lease(updated)
        return updated

    def collect(self, record, result):
        if result.get('source_video_task_id') != record.request['source_video_task_id']:
            raise ValueError('source mismatch')
        for name in ('content', 'preview'):
            if not re.fullmatch('[0-9a-f]{64}', str(result.get(name + '_sha256', ''))):
                raise ValueError('invalid output digest')
        if not isinstance(result.get('media'), dict):
            raise ValueError('invalid output media')
        state = record.collection or {'result': result, 'artifacts': {}, 'archives': {}}
        if state['result'] != result:
            raise ValueError('runtime result changed after collection began')
        record = self.tasks.update(record, status='running', runtime_stage='archiving',
                                   collection=state, error=None)
        try:
            archive = self.archive()
            for name in ('content', 'preview'):
                digest = result[name + '_sha256']
                artifact_id = state['artifacts'].get(name)
                if artifact_id:
                    artifact = self.artifacts.get(artifact_id, record.project_id)
                    if artifact.sha256 != digest:
                        raise ValueError('collected output changed')
                else:
                    with self.client.stream('GET', f'{self.url}/v1/upscales/{record.video_task_id}/{name}', headers=self.headers, timeout=180) as response:
                        response.raise_for_status()
                        artifact = self.artifacts.create_from_chunks(project_id=record.project_id, operation=f'video.h3.latent.upscale.{name}',
                            filename=f'{record.video_task_id}-{name}.mp4', media_type='video/mp4',
                            chunks=response.iter_bytes(), expected_sha256=digest)
                    state['artifacts'][name] = artifact.artifact_id
                    record = self.tasks.update(record, collection=state)
                if name not in state['archives']:
                    state['archives'][name] = archive.store(artifact, self.artifacts.content_path(artifact), record.video_task_id, name)
                    record = self.tasks.update(record, collection=state)
        except (ArchiveError, ArtifactError, httpx.HTTPError, OSError):
            return self.tasks.update(record, error={'code': 'archive_pending',
                'message': 'Result delivery is pending; query this task to retry without repeating inference'})
        updated = self.tasks.update(record, status='succeeded', runtime_stage='completed',
            artifact_id=state['artifacts']['content'], error=None,
            media={**result['media'], 'preview_artifact_id': state['artifacts']['preview'], 'source_video_task_id': result['source_video_task_id']},
            runtime_metrics=result.get('runtime_metrics'))
        self.executor.stage_slot.release(updated.video_task_id)
        return self.executor.maybe_release_scheduler_lease(updated)

    @task_serialized
    def status(self, task_id):
        record = self.tasks.get(task_id)
        if record.service != self.service:
            raise ValueError('not an H3 latent task')
        if record.status == 'queued' and (record.provider_state or {}).get('gpu_dispatched') is False:
            return self.submit(record)
        if record.status in {'succeeded', 'failed'}:
            return record
        try:
            if record.collection:
                return self.collect(record, record.collection['result'])
            response = self.client.get(f'{self.url}/v1/upscales/{task_id}', headers=self.headers, timeout=15)
            if response.status_code == 404 and record.error and record.error.get('code') == 'submission_unconfirmed':
                return self.submit(record)
            response.raise_for_status()
            return self.update(record, response.json())
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise ExecutionError('Post-processing query or artifact registration unavailable; retain the task ID') from exc

    def result(self, task_id):
        record = self.status(task_id)
        if record.status != 'succeeded':
            raise ExecutionError('Post-processing has not succeeded')
        return record

    def public_status(self, record):
        value = {'video_task_id': record.video_task_id, 'source_video_task_id': record.request['source_video_task_id'],
                 'service': self.service, 'status': record.status, 'error': record.error,
                 'timing': record.timing, 'runtime_metrics': record.runtime_metrics}
        pstate = record.provider_state or {}
        if any(pstate.get(k) is not None for k in ('waiting_reason', 'queue_position', 'blocking_reason')):
            value['gpu_queue'] = {
                'waiting_reason': pstate.get('waiting_reason'),
                'queue_position': pstate.get('queue_position'),
                'blocking_reason': pstate.get('blocking_reason'),
                'blocking_holder_id': pstate.get('blocking_holder_id') or None,
            }
        return value

    def public_result(self, record, preview=False):
        artifact_id = record.media['preview_artifact_id'] if preview else record.artifact_id
        artifact = self.artifacts.get(artifact_id, record.project_id)
        return {**self.public_status(record), 'artifact_id': artifact_id, 'download_path': self.artifacts.download_path(artifact_id),
                'sha256': artifact.sha256, 'media': record.media,
                'download_url': record.collection['archives']['preview' if preview else 'content']['download_url']}
