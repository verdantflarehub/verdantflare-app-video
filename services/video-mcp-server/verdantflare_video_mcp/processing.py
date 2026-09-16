"""MCP adapters for independent SR and interpolation runtimes."""
import hashlib
import json
import math
import os
import re
from fractions import Fraction
import httpx
from .artifacts import require_project_id
from .executor import ExecutionError, serialized
from .tasks import TaskConflict


class ProcessingExecutor:
    def __init__(self, executor, service):
        self.executor = executor
        self.artifacts, self.tasks, self.client = executor.artifacts, executor.tasks, executor.client
        self._lock = executor._lock
        if service not in {'sr', 'interpolate'}:
            raise ValueError('unsupported processing service')
        self.service = service
        self.backend = 'seedvr2' if service == 'sr' else 'rife'
        name = 'video-super-resolution-api' if service == 'sr' else 'video-frame-interpolation-api'
        prefix = 'VIDEO_SR' if service == 'sr' else 'VIDEO_INTERPOLATION'
        self.url = os.environ.get(f'{prefix}_RUNTIME_URL', f'http://{name}:8000').rstrip('/')
        self.runtime_version = os.environ.get(f'{prefix}_RUNTIME_VERSION', f'{name}-v0.1.0')

    def validate_parameters(self, supplied):
        defaults = ({'backend': 'seedvr2', 'quality_mode': 'standard', 'seed': 666} if self.service == 'sr'
                    else {'backend': 'rife', 'target_fps_num': 48, 'target_fps_den': 1})
        allowed = set(defaults) | ({'target_width', 'target_height'} if self.service == 'sr' else set())
        if set(supplied) - allowed:
            raise ValueError('unsupported processing parameters')
        values = {**defaults, **supplied}
        if values['backend'] != self.backend:
            raise ValueError('unsupported backend')
        if self.service == 'sr':
            for key in ('target_width', 'target_height'):
                value = values.get(key)
                if type(value) is not int or not 16 <= value <= 2048 or value % 16:
                    raise ValueError('target dimensions must be divisible by 16, within 16..2048')
            if values['quality_mode'] != 'standard' or type(values['seed']) is not int or not 0 <= values['seed'] <= 4294967295:
                raise ValueError('unsupported quality mode or seed')
        else:
            for key, limit in (('target_fps_num', 120000), ('target_fps_den', 1001)):
                if type(values[key]) is not int or not 1 <= values[key] <= limit:
                    raise ValueError('invalid target frame rate')
            fps = Fraction(values['target_fps_num'], values['target_fps_den'])
            if fps > 120:
                raise ValueError('target frame rate exceeds 120')
            values.update(target_fps_num=fps.numerator, target_fps_den=fps.denominator)
        return values

    @serialized
    def generate(self, *, project_id, idempotency_key, source_artifact_id,
                 **parameters):
        project_id = require_project_id(project_id)
        if not idempotency_key.strip() or len(idempotency_key) > 128:
            raise ValueError('idempotency_key is invalid')
        parameters = self.validate_parameters(parameters)
        source = self.artifacts.get(source_artifact_id, project_id)
        if not source.media_type.startswith('video/'):
            raise ValueError('source artifact must be video')
        request = {'project_id': project_id, 'service': self.service, **parameters,
                   'source_artifact_id': source.artifact_id, 'source_sha256': source.sha256}
        digest = 'sha256:' + hashlib.sha256(json.dumps(request, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        existing = self.tasks.find_idempotency(project_id, idempotency_key)
        if existing:
            if existing.input_digest != digest:
                raise TaskConflict('idempotency key already exists with different input')
            return existing
        record = self.tasks.create(project_id=project_id, idempotency_key=idempotency_key, input_digest=digest,
                                   request=request, runtime_task_id='', status='queued')
        record = self.tasks.update(record, service=self.service, runtime_version=self.runtime_version, runtime_task_id=record.video_task_id,
            error={'code': 'submission_unconfirmed', 'message': 'Submission is pending confirmation'})
        return self._submit(record)

    def _submit(self, record):
        payload = {k: v for k, v in record.request.items() if k != 'service'}
        payload['idempotency_key'] = record.video_task_id
        try:
            response = self.client.post(f'{self.url}/v1/{self.service}', json=payload, timeout=30)
            if response.status_code in {400, 404, 409, 422, 429}:
                return self.tasks.update(record, status='failed', error={
                    'code': f'{self.service}_request_rejected', 'message': 'Video processing runtime rejected the source, parameters or queue capacity'})
            response.raise_for_status()
            return self._update(record, response.json())
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
            return self.tasks.update(record, error={'code': 'submission_unconfirmed',
                'message': 'Query this task ID to recover the persisted request'})

    def _update(self, record, data):
        status = data['status']
        if data.get('video_task_id') != record.video_task_id or status not in {'queued', 'running', 'succeeded', 'failed'}:
            raise ValueError('invalid runtime status')
        error = None
        if status == 'failed':
            code = (data.get('error') or {}).get('code')
            if code not in {'interrupted', 'invalid_media', 'source_integrity_failed', 'inference_failed', 'capacity_profile_mismatch', 'gpu_memory_occupancy_changed'}:
                code = f'{self.service}_failed'
            error = {'code': code, 'message': 'Video processing conversion failed'}
        return self.tasks.update(record, status=status, error=error)

    @serialized
    def status(self, video_task_id):
        record = self.tasks.get(video_task_id)
        if record.service != self.service:
            raise ValueError('task is not a matching processing task')
        if record.status in {'succeeded', 'failed'}:
            return record
        try:
            response = self.client.get(f'{self.url}/v1/{self.service}/{record.video_task_id}', timeout=10)
            if response.status_code == 404:
                # The deterministic ID is also the runtime's durable idempotency key.
                # A retry after an ambiguous POST cannot enqueue duplicate inference.
                if not record.runtime_task_id or (record.error and record.error['code'] == 'submission_unconfirmed'):
                    return self._submit(record)
                return self.tasks.update(record, status='failed', error={
                    'code': 'runtime_task_lost', 'message': 'Video processing runtime task is missing'})
            response.raise_for_status()
            return self._update(record, response.json())
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ExecutionError('Video processing status temporarily unavailable; keep the task ID') from exc

    def _artifact(self, record, endpoint, expected_sha256):
        if not isinstance(expected_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', expected_sha256):
            raise ExecutionError('Video processing result digest missing')
        try:
            with self.client.stream('GET', f'{self.url}/v1/{self.service}/{record.video_task_id}/{endpoint}', timeout=120) as response:
                response.raise_for_status()
                return self.artifacts.create_from_chunks(project_id=record.project_id,
                    operation=f'video.{self.service}.{endpoint}', filename=f'{record.video_task_id}-{endpoint}.mp4',
                    media_type='video/mp4', chunks=response.iter_bytes(), expected_sha256=expected_sha256)
        except httpx.HTTPError as exc:
            raise ExecutionError('Video processing result download failed; retry this task ID') from exc

    @serialized
    def result(self, video_task_id):
        record = self.status(video_task_id)
        if record.status != 'succeeded':
            raise ExecutionError('Video processing task has not succeeded')
        if record.artifact_id and record.media and record.media.get('preview_artifact_id'):
            return record
        try:
            response = self.client.get(f'{self.url}/v1/{self.service}/{record.video_task_id}', timeout=10)
            response.raise_for_status()
            payload = response.json()
            if payload['video_task_id'] != record.video_task_id or payload['status'] != 'succeeded':
                raise ValueError('result task mismatch')
            data = payload['result']
            if data['input_sha256'] != record.request['source_sha256']:
                raise ValueError('source mismatch')
            if data['parameters']['backend'] != record.request['backend']:
                raise ValueError('backend mismatch')
            keys = ('width', 'height', 'frames', 'frame_rate', 'duration_seconds', 'video_codec', 'audio')
            media = {k: data['media'][k] for k in keys}
            source = {k: data['input_media'][k] for k in keys}
            preview = {k: data['preview_media'][k] for k in keys}
            for item in (source, media, preview):
                if any(type(item[k]) is not int or item[k] <= 0 for k in ('width', 'height', 'frames')):
                    raise ValueError('invalid dimensions')
                if Fraction(item['frame_rate']) <= 0 or type(item['audio']) is not bool:
                    raise ValueError('invalid media')
                if not math.isfinite(float(item['duration_seconds'])) or float(item['duration_seconds']) <= 0:
                    raise ValueError('invalid duration')
            if media['audio'] != source['audio'] or media['video_codec'] != 'h264':
                raise ValueError('media mismatch')
            if self.service == 'sr':
                if (media['width'], media['height']) != (record.request['target_width'], record.request['target_height']):
                    raise ValueError('resolution mismatch')
                if media['frames'] != source['frames'] or Fraction(media['frame_rate']) != Fraction(source['frame_rate']):
                    raise ValueError('timeline mismatch')
            else:
                fps = Fraction(record.request['target_fps_num'], record.request['target_fps_den'])
                if Fraction(media['frame_rate']) != fps or fps != 2 * Fraction(source['frame_rate']):
                    raise ValueError('frame rate mismatch')
                if media['frames'] != source['frames'] * 2 or any(media[k] != source[k] for k in ('width', 'height')):
                    raise ValueError('interpolation geometry mismatch')
            duration = float(Fraction(media['frames']) / Fraction(media['frame_rate']))
            if abs(duration - float(media['duration_seconds'])) > .001 or abs(duration - float(source['duration_seconds'])) > .001:
                raise ValueError('duration mismatch')
            if any(preview[k] != (media[k] * 2 if k == 'width' else media[k]) for k in keys):
                raise ValueError('preview mismatch')
            model = data['model']
            if model['backend'] != self.backend or not re.fullmatch(r'[0-9a-f]{40}', model['code_revision']):
                raise ValueError('model mismatch')
            if not re.fullmatch(r'(?:[0-9a-f]{40}|sha256:[0-9a-f]{64})', model['weights_revision']) or not re.fullmatch(r'[0-9a-f]{64}', model['manifest_sha256']):
                raise ValueError('model digest missing')
            filenames = ('seedvr2_ema_3b.pth', 'ema_vae.pth', 'pos_emb.pt', 'neg_emb.pt') if self.service == 'sr' else ('flownet.pkl',)
            weights = {name: model['weights_sha256'][name] for name in filenames}
            if not all(re.fullmatch(r'[0-9a-f]{64}', value) for value in weights.values()):
                raise ValueError('weight digest missing')
            media['model'] = {k: model[k] for k in ('backend', 'code_revision', 'weights_revision', 'manifest_sha256')}
            media['model']['weights_sha256'] = weights
            requested = {k: v for k, v in record.request.items() if k not in
                         {'project_id', 'service', 'source_artifact_id', 'source_sha256'}}
            if any(data['parameters'][k] != v for k, v in requested.items()):
                raise ValueError('parameter mismatch')
            # Fixed public encoding policy; arbitrary runtime keys never cross the MCP boundary.
            fixed = {'audio': 'aac_if_present', 'video_codec': 'h264', 'pixel_format': 'yuv420p', 'crf': 18}
            fixed.update({'scene_cut_threshold': .30, 'last_frame': 'hold'} if self.service == 'interpolate' else
                         {'sample_steps': 1, 'cfg_scale': 1.0, 'normalization': 'torch_rms_layer', 'color_fix': False})
            if any(data['parameters'][k] != v for k, v in fixed.items()):
                raise ValueError('processing policy mismatch')
            media['parameters'] = {**requested, **fixed}
            if self.service == 'sr':
                raw_window = data['parameters']['windowing']
                window_keys = ('window_frames', 'overlap_frames', 'capacity_profile_sha256', 'reserve_bytes',
                               'seam_method', 'window_seed', 'scene_cut_threshold', 'chunks',
                               'max_chunk_frames', 'output_frames')
                window = {k: raw_window[k] for k in window_keys}
                for key in ('window_frames', 'overlap_frames', 'reserve_bytes', 'chunks', 'max_chunk_frames', 'output_frames'):
                    if type(window[key]) is not int or window[key] < 0:
                        raise ValueError('invalid window metadata')
                if (window['window_frames'] < 9 or (window['window_frames'] - 1) % 4 or window['overlap_frames'] != 4
                    or not 0 < window['chunks'] <= media['frames']
                    or not 0 < window['max_chunk_frames'] <= min(window['window_frames'], media['frames'])
                    or window['output_frames'] != media['frames'] or window['seam_method'] != 'linear_overlap'
                    or window['window_seed'] != 'source_offset' or window['scene_cut_threshold'] != .30
                    or not re.fullmatch(r'[0-9a-f]{64}', window['capacity_profile_sha256'])):
                    raise ValueError('window contract mismatch')
                media['parameters']['windowing'] = window
            media['input_media'] = source
            media['preview_media'] = preview
            for key in ('input_sha256', 'output_sha256', 'preview_sha256'):
                if not re.fullmatch(r'[0-9a-f]{64}', data[key]):
                    raise ValueError('invalid digest')
                media[key] = data[key]
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError, ZeroDivisionError) as exc:
            raise ExecutionError('Video processing result metadata unavailable or invalid') from exc
        if record.artifact_id and record.media != media:
            raise ExecutionError('Video processing result changed after registration')
        if not record.artifact_id:
            artifact = self._artifact(record, 'content', data['output_sha256'])
            record = self.tasks.update(record, artifact_id=artifact.artifact_id, media=media)
        preview = self._artifact(record, 'preview', data['preview_sha256'])
        media['preview_artifact_id'] = preview.artifact_id
        return self.tasks.update(record, media=media)

    def public_status(self, record):
        return {'video_task_id': record.video_task_id, 'service': self.service,
                'status': record.status, 'created_at': record.created_at, 'updated_at': record.updated_at,
                'error': record.error}

    def public_result(self, record, preview=False):
        artifact_id = record.media['preview_artifact_id'] if preview else record.artifact_id
        artifact = self.artifacts.get(artifact_id, record.project_id)
        return {**self.public_status(record), 'artifact_id': artifact_id,
                'download_path': self.artifacts.download_path(artifact_id), 'sha256': artifact.sha256,
                'source_artifact_id': record.request['source_artifact_id'],
                'media': {**record.media, **record.media['preview_media']} if preview else record.media}
