"""Depth MCP adapter; public IDs and immutable Artifacts only."""
import hashlib
import json
import os
import httpx
from .artifacts import require_project_id
from .channel_errors import MESSAGES
from .executor import ExecutionError, serialized, task_serialized
from .gpu_lease import LeaseBusy
from .tasks import TaskConflict


class DepthExecutor:
    def __init__(self, executor):
        self.executor = executor
        self.artifacts, self.tasks = executor.artifacts, executor.tasks
        self._client_override = None
        self._lock = executor._lock
        self.url = os.environ.get('VIDEO_DEPTH_RUNTIME_URL', 'http://video-depth-anything:8000').rstrip('/')
        self.runtime_version = os.environ.get('VIDEO_DEPTH_RUNTIME_VERSION', 'video-depth-anything-v0.2.4')

    @property
    def client(self):
        return self._client_override if self._client_override is not None else self.executor.client

    @client.setter
    def client(self, value) -> None:
        self._client_override = value

    def _runtime_target(self, record=None):
        return self.url, {}, self.runtime_version

    @serialized
    def generate(self, *, project_id, idempotency_key, source_artifact_id,
                 model='video-depth-anything', output_format='mp4'):
        project_id = require_project_id(project_id)
        if not idempotency_key.strip() or len(idempotency_key) > 128:
            raise ValueError('idempotency_key is invalid')
        if model != 'video-depth-anything' or output_format != 'mp4':
            raise ValueError('unsupported depth model or output format')
        source = self.artifacts.get(source_artifact_id, project_id)
        if not source.media_type.startswith('video/'):
            raise ValueError('source artifact must be video')
        request = {'project_id': project_id, 'service': 'depth', 'model': model, 'output_format': output_format,
                   'source_artifact_id': source.artifact_id, 'source_sha256': source.sha256}
        digest = 'sha256:' + hashlib.sha256(json.dumps(request, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        existing = self.tasks.find_idempotency(project_id, idempotency_key)
        if existing:
            if existing.input_digest != digest:
                raise TaskConflict('idempotency key already exists with different input')
            return existing
        record = self.tasks.create(project_id=project_id, idempotency_key=idempotency_key, input_digest=digest,
                                   request=request, runtime_task_id='', status='queued')
        record = self.tasks.update(record, service='depth', runtime_version=self.runtime_version, runtime_task_id=record.video_task_id,
            error={'code': 'submission_unconfirmed', 'message': 'Submission is pending confirmation'})
        return self._submit(record)

    def _fail_record(self, record, error):
        pstate = {
            k: v
            for k, v in (record.provider_state or {}).items()
            if k not in {'waiting_reason', 'gpu_dispatched'}
        }
        pstate['worker_settled'] = True
        updated = self.tasks.update(record, status='failed', error=error, provider_state=pstate)
        self.executor.release_local_slots(updated.video_task_id, force=True)
        return self.executor.maybe_release_scheduler_lease(updated)

    def _submit(self, record):
        payload = {k: v for k, v in record.request.items() if k != 'service'}
        payload['idempotency_key'] = record.video_task_id
        acquired = False
        grant_headers = {}
        try:
            local_gating = bool((not self.executor.scheduler.enabled) and (self.executor.gpu.enabled or self.executor.stage_slot.enabled))
            if self.executor.engine_gate.enabled:
                slot_ident = self.executor.probe_runtime_identity(self.url)
                try:
                    acquired = self.executor.acquire_engine_slot(
                        record.video_task_id,
                        'depth',
                        runtime_identity=slot_ident,
                        record=record,
                    )
                except LeaseBusy:
                    return self._fail_record(
                        record,
                        {'code': 'gpu_busy', 'message': MESSAGES['gpu_busy'], 'retryable': True},
                    )
                if not acquired:
                    return self.executor.defer_local_gpu_queue(record, waiting_reason='engine_slot')
            elif local_gating:
                try:
                    acquired = self.executor.acquire_local_gpu(record.video_task_id, 'depth', record=record)
                except LeaseBusy:
                    return self._fail_record(
                        record,
                        {'code': 'gpu_busy', 'message': MESSAGES['gpu_busy'], 'retryable': True},
                    )
                if not acquired:
                    return self.executor.defer_local_gpu_queue(record)

            if self.executor.scheduler.enabled:
                try:
                    sched_ok, record, grant_headers = self.executor.acquire_scheduler_grant(
                        record,
                        'depth',
                        runtime_url=self.url,
                        bind_immediately=True,
                    )
                except LeaseBusy:
                    return self._fail_record(
                        record,
                        {'code': 'gpu_busy', 'message': MESSAGES['gpu_busy'], 'retryable': True},
                    )
                if not sched_ok:
                    return record

            if self.executor.scheduler.enabled:
                record, grant_headers = self.executor.ensure_scheduler_target_bound(
                    record,
                    runtime_url=self.url,
                )

            if record.provider_state or self.executor.gpu.enabled or self.executor.stage_slot.enabled or self.executor.scheduler.enabled:
                pstate = {k: v for k, v in (record.provider_state or {}).items() if k != 'waiting_reason'}
                pstate['gpu_dispatched'] = True
                record = self.tasks.update(record, provider_state=pstate)

            response = self.client.post(f'{self.url}/v1/depth', json=payload, headers=grant_headers or None, timeout=30)
            if response.status_code in {400, 404, 409, 422, 429}:
                return self._fail_record(record, {
                    'code': 'depth_request_rejected', 'message': 'Depth runtime rejected the source, parameters or queue capacity'})
            response.raise_for_status()
            return self._update(record, response.json())
        except (httpx.HTTPError, ValueError, KeyError):
            return self.tasks.update(record, error={'code': 'submission_unconfirmed',
                'message': 'Query this task ID to recover the persisted request'})
        finally:
            if acquired:
                self.executor.release_or_reconcile_local_gpu(record.video_task_id)

    def _update(self, record, data):
        status = data['status']
        if data.get('video_task_id') != record.video_task_id or status not in {'queued', 'running', 'succeeded', 'failed'}:
            raise ValueError('invalid runtime status')
        error = None
        if status == 'failed':
            code = (data.get('error') or {}).get('code')
            if code not in {'interrupted', 'invalid_media', 'source_integrity_failed', 'inference_failed'}:
                code = 'depth_failed'
            error = {'code': code, 'message': 'Depth conversion failed'}
        pstate = dict(record.provider_state or {})
        for key in ('started_at', 'finished_at', 'worker_settled', 'reclaim_proof'):
            if key in data:
                pstate[key] = data[key]
        if status in {'succeeded', 'failed'} and (pstate or self.executor.stage_slot.enabled or self.executor.scheduler.enabled):
            pstate.setdefault('worker_settled', True)
        updated = self.tasks.update(record, status=status, error=error, provider_state=pstate or None)
        if self.executor.stage_slot.is_task_settled(updated):
            self.executor.release_local_slots(updated.video_task_id)
            updated = self.executor.maybe_release_scheduler_lease(updated)
        return updated

    @task_serialized
    def status(self, video_task_id):
        record = self.tasks.get(video_task_id)
        if record.service != 'depth':
            raise ValueError('task is not a depth task')
        if record.status == 'queued' and (record.error or {}).get('code') == 'gpu_admission_reconciliation_required':
            return self.executor.reconcile_expired_scheduler_admission(record)
        if record.status == 'queued' and (record.provider_state or {}).get('gpu_dispatched') is False:
            record = self._submit(record)
            if (record.error or {}).get('code') == 'gpu_admission_reconciliation_required':
                return self.executor.reconcile_expired_scheduler_admission(record)
            return record
        if record.status in {'succeeded', 'failed'}:
            return record
        try:
            response = self.client.get(f'{self.url}/v1/depth/{record.video_task_id}', timeout=10)
            if response.status_code == 404:
                # The deterministic ID is also the runtime's durable idempotency key.
                # A retry after an ambiguous POST cannot enqueue duplicate inference.
                if not record.runtime_task_id or (record.error and record.error['code'] == 'submission_unconfirmed'):
                    return self._submit(record)
                return self.tasks.update(record, status='failed', error={
                    'code': 'runtime_task_lost', 'message': 'Depth runtime task is missing'})
            response.raise_for_status()
            return self._update(record, response.json())
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            raise ExecutionError('Depth status temporarily unavailable; keep the task ID') from exc

    def _artifact(self, record, endpoint, expected_sha256):
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ExecutionError('Depth result digest missing')
        try:
            with self.client.stream('GET', f'{self.url}/v1/depth/{record.video_task_id}/{endpoint}', timeout=120) as response:
                response.raise_for_status()
                return self.artifacts.create_from_chunks(project_id=record.project_id,
                    operation=f'video.depth.{endpoint}', filename=f'{record.video_task_id}-{endpoint}.mp4',
                    media_type='video/mp4', chunks=response.iter_bytes(), expected_sha256=expected_sha256)
        except httpx.HTTPError as exc:
            raise ExecutionError('Depth result download failed; retry this task ID') from exc

    @task_serialized
    def result(self, video_task_id):
        record = self.status(video_task_id)
        if record.status != 'succeeded':
            raise ExecutionError('Depth task has not succeeded')
        if record.artifact_id and record.media and record.media.get('preview_artifact_id'):
            return record
        try:
            response = self.client.get(f'{self.url}/v1/depth/{record.video_task_id}', timeout=10)
            response.raise_for_status()
            data = response.json()['result']
            if data['input_sha256'] != record.request['source_sha256']:
                raise ValueError('source mismatch')
            # Whitelist public metadata. Never forward arbitrary runtime payloads.
            keys = ('width', 'height', 'frames', 'frame_rate', 'duration_seconds', 'video_codec')
            media = {k: data['media'][k] for k in keys}
            for key in ('width', 'height', 'frames', 'frame_rate'):
                if media[key] != data['input_media'][key]:
                    raise ValueError('media mismatch')
            media['model'] = {k: data['model'][k] for k in ('model', 'encoder', 'code_revision', 'repository', 'revision', 'sha256')}
            media['parameters'] = {k: data['parameters'][k] for k in ('input_size', 'fp32', 'normalization', 'depth_min', 'depth_max', 'audio', 'crf')}
            media['input_sha256'] = data['input_sha256']
            media['output_sha256'] = data['output_sha256']
            media['preview_sha256'] = data['preview_sha256']
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise ExecutionError('Depth result metadata unavailable or invalid') from exc
        if not record.artifact_id:
            artifact = self._artifact(record, 'content', data['output_sha256'])
            record = self.tasks.update(record, artifact_id=artifact.artifact_id, media=media)
        preview = self._artifact(record, 'preview', data['preview_sha256'])
        media['preview_artifact_id'] = preview.artifact_id
        return self.tasks.update(record, media=media)

    def public_status(self, record):
        value = {'video_task_id': record.video_task_id, 'depth_task_id': record.video_task_id,
                 'status': record.status, 'created_at': record.created_at, 'updated_at': record.updated_at,
                 'error': record.error}
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
        return {**self.public_status(record), 'artifact_id': artifact_id,
                'download_path': self.artifacts.download_path(artifact_id), 'sha256': artifact.sha256,
                'source_artifact_id': record.request['source_artifact_id'],
                'media': {**record.media, 'width': record.media['width'] * 2} if preview else record.media}
