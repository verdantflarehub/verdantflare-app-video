"""Depth MCP adapter; public IDs and immutable Artifacts only."""
import hashlib
import json
import os
import httpx
from .artifacts import require_project_id
from .executor import ExecutionError, serialized
from .tasks import TaskConflict


class DepthExecutor:
    def __init__(self, executor):
        self.executor = executor
        self.artifacts, self.tasks, self.client = executor.artifacts, executor.tasks, executor.client
        self._lock = executor._lock
        self.url = os.environ.get('VIDEO_DEPTH_RUNTIME_URL', 'http://video-depth-anything-api:8000').rstrip('/')

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
        record = self.tasks.update(record, service='depth', runtime_version='video-depth-anything-api-v0.2.0', runtime_task_id=record.video_task_id,
            error={'code': 'submission_unconfirmed', 'message': 'Submission is pending confirmation'})
        return self._submit(record)

    def _submit(self, record):
        payload = {k: v for k, v in record.request.items() if k != 'service'}
        payload['idempotency_key'] = record.video_task_id
        try:
            response = self.client.post(f'{self.url}/v1/depth', json=payload, timeout=30)
            if response.status_code in {400, 404, 409, 422, 429}:
                return self.tasks.update(record, status='failed', error={
                    'code': 'depth_request_rejected', 'message': 'Depth runtime rejected the source, parameters or queue capacity'})
            response.raise_for_status()
            return self._update(record, response.json())
        except (httpx.HTTPError, ValueError, KeyError):
            return self.tasks.update(record, error={'code': 'submission_unconfirmed',
                'message': 'Query this task ID to recover the persisted request'})

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
        return self.tasks.update(record, status=status, error=error)

    @serialized
    def status(self, video_task_id):
        record = self.tasks.get(video_task_id)
        if record.service != 'depth':
            raise ValueError('task is not a depth task')
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

    @serialized
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
        return {'video_task_id': record.video_task_id, 'depth_task_id': record.video_task_id,
                'status': record.status, 'created_at': record.created_at, 'updated_at': record.updated_at,
                'error': record.error}

    def public_result(self, record, preview=False):
        artifact_id = record.media['preview_artifact_id'] if preview else record.artifact_id
        artifact = self.artifacts.get(artifact_id, record.project_id)
        return {**self.public_status(record), 'artifact_id': artifact_id,
                'download_path': self.artifacts.download_path(artifact_id), 'sha256': artifact.sha256,
                'source_artifact_id': record.request['source_artifact_id'],
                'media': {**record.media, 'width': record.media['width'] * 2} if preview else record.media}
