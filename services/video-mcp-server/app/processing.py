"""MCP adapters for independent SR, interpolation, and 2K IC regeneration runtimes."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from datetime import UTC, datetime
from fractions import Fraction

import httpx

from .artifacts import require_project_id
from .channel_errors import MESSAGES
from .executor import (
    ExecutionError,
    _cancel_backoff_ready,
    _next_cancel_backoff_epoch,
    task_serialized,
)
from .gpu_lease import LeaseBusy
from .regenerate import (
    LOCKED_P2_CODE_REVISION,
    LOCKED_P2_WEIGHTS_SHA256,
    REGENERATE_2K_GRID_MODES,
    REGENERATE_2K_MODEL_ID,
    REGENERATE_2K_PROFILES,
    is_valid_h3_frame_bucket,
    preview_geometry,
    resolve_2k_target_geometry,
    temporal_windows,
)
from .tasks import TaskConflict, TaskRecord


SAFE_REJECTION_CODES = {
    'source_already_2k',
    'invalid_source_media',
    'unsupported_grid_mode',
    'unsupported_regenerate_profile',
    'source_not_found',
    'queue_full',
    'runtime_not_ready',
}

SAFE_RUNTIME_FAILURE_CODES = {
    'interrupted',
    'invalid_media',
    'invalid_source_media',
    'source_already_2k',
    'source_integrity_failed',
    'inference_failed',
    'capacity_profile_mismatch',
    'gpu_memory_occupancy_changed',
    'runtime_not_ready',
}


class ProcessingExecutor:
    def __init__(self, executor, service: str):
        self.executor = executor
        self.artifacts, self.tasks = executor.artifacts, executor.tasks
        self._client_override = None
        self._lock = executor._lock
        if service not in {'sr', 'interpolate', 'regenerate_2k'}:
            raise ValueError('unsupported processing service')
        self.service = service
        self.endpoint_path = 'regenerate-2k' if service == 'regenerate_2k' else service
        self.backend = (
            'seedvr2' if service == 'sr'
            else 'open_h3_2k_ic' if service == 'regenerate_2k'
            else 'rife'
        )
        if service == 'regenerate_2k':
            self.url = os.environ.get('SINGULARITY_RUNTIME_URL', executor.runtime_url).rstrip('/')
            self.runtime_version = os.environ.get('SINGULARITY_RUNTIME_VERSION', executor.runtime_version)
        else:
            name = 'video-super-resolution-api' if service == 'sr' else 'video-frame-interpolation-api'
            prefix = 'VIDEO_SR' if service == 'sr' else 'VIDEO_INTERPOLATION'
            self.url = os.environ.get(f'{prefix}_RUNTIME_URL', f'http://{name}:8000').rstrip('/')
            self.runtime_version = os.environ.get(f'{prefix}_RUNTIME_VERSION', f'{name}-v0.1.0')

    @property
    def client(self):
        return self._client_override if self._client_override is not None else self.executor.client

    @client.setter
    def client(self, value) -> None:
        self._client_override = value

    def _runtime_target(self, record: TaskRecord | None = None) -> tuple[str, dict[str, str], str]:
        if self.service != 'regenerate_2k':
            endpoint = (record.runtime_endpoint if record and record.runtime_endpoint else self.url).rstrip('/')
            version = (record.runtime_version if record and record.runtime_version else self.runtime_version)
            return endpoint, {}, version
        try:
            url, headers, version, _ = self.executor.runtime('h3-singularity')
        except ExecutionError:
            token = (
                os.environ.get('H3_SINGULARITY_RUNTIME_TOKEN', '').strip()
                or os.environ.get('SINGULARITY_RUNTIME_TOKEN', '').strip()
                or self.executor.singularity_token
            )
            url = (
                os.environ.get('SINGULARITY_RUNTIME_URL', '').strip()
                or os.environ.get('H3_RUNTIME_URL', '').strip()
                or self.url
            ).rstrip('/')
            version = os.environ.get('SINGULARITY_RUNTIME_VERSION', '').strip() or self.executor.runtime_version
            if not url or not token:
                raise ExecutionError('runtime_not_ready')
            headers = {'Authorization': f'Bearer {token}'}
        if record and record.runtime_endpoint:
            url = record.runtime_endpoint.rstrip('/')
        if record and record.runtime_version:
            version = record.runtime_version
        return url, headers, version

    def _resolve_regenerate_prompt(
        self,
        project_id: str,
        source_artifact_id: str,
        supplied_prompt: str | None,
    ) -> str:
        if supplied_prompt is not None:
            return supplied_prompt.strip()
        self.tasks.ensure_ready()
        candidates: list[TaskRecord] = []
        for path in self.tasks.root.glob('video_task_*.json'):
            try:
                candidate = TaskRecord.model_validate_json(path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            if candidate.project_id != project_id or candidate.service == 'regenerate_2k':
                continue
            source_match = (
                candidate.artifact_id == source_artifact_id
                or (candidate.delivery or {}).get('source_artifact_id') == source_artifact_id
            )
            if source_match:
                candidates.append(candidate)
        candidates.sort(key=lambda item: (item.created_at, item.video_task_id), reverse=True)
        for candidate in candidates:
            prompt = candidate.request.get('prompt')
            if isinstance(prompt, str) and prompt.strip():
                return prompt.strip()
        return ''

    def _inspect_regenerate_source_media(
        self,
        project_id: str,
        source,
        *,
        strict: bool,
    ) -> dict[str, object] | None:
        path = self.artifacts.content_path(source)
        head = b''
        try:
            with path.open('rb') as handle:
                head = handle.read(16)
        except OSError as exc:
            if strict:
                raise ValueError('invalid_source_media') from exc
            return None

        if strict or b'ftyp' in head:
            import subprocess
            try:
                proc = subprocess.run(
                    [
                        'ffprobe', '-v', 'error', '-count_frames',
                        '-show_streams', '-show_format', '-of', 'json', str(path),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                probe = json.loads(proc.stdout)
                videos = [
                    s for s in probe.get('streams', [])
                    if s.get('codec_type') == 'video'
                    and not (s.get('disposition') or {}).get('attached_pic')
                ]
                if len(videos) != 1:
                    raise ValueError('invalid_source_media')
                stream = videos[0]
                width = int(stream['width'])
                height = int(stream['height'])
                avg_rate = Fraction(stream.get('avg_frame_rate') or '0/1')
                r_rate = Fraction(stream.get('r_frame_rate') or '0/1')
                fps_frac = avg_rate if avg_rate > 0 else r_rate
                if fps_frac != Fraction(24, 1):
                    raise ValueError('invalid_source_media')
                raw_frames = stream.get('nb_read_frames') or stream.get('nb_frames')
                if raw_frames is not None:
                    frames = int(raw_frames)
                else:
                    duration = float(stream.get('duration') or probe.get('format', {}).get('duration') or 0)
                    frames = round(duration * 24)
                if not is_valid_h3_frame_bucket(frames):
                    raise ValueError('invalid_source_media')
                has_audio = any(s.get('codec_type') == 'audio' for s in probe.get('streams', []))
                return {
                    'width': width,
                    'height': height,
                    'fps': 24,
                    'frames': frames,
                    'has_audio': has_audio,
                }
            except ValueError:
                if b'ftyp' in head:
                    raise
            except (subprocess.SubprocessError, OSError, KeyError, TypeError, ZeroDivisionError) as exc:
                if b'ftyp' in head:
                    raise ValueError('invalid_source_media') from exc

        self.tasks.ensure_ready()
        for task_path in self.tasks.root.glob('video_task_*.json'):
            try:
                candidate = TaskRecord.model_validate_json(task_path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            if candidate.project_id != project_id:
                continue
            if candidate.artifact_id == source.artifact_id and isinstance(candidate.media, dict):
                m = candidate.media
                width = m.get('width')
                height = m.get('height')
                frames = m.get('frames')
                raw_fps = m.get('fps', m.get('frame_rate', 24))
                try:
                    fps_val = Fraction(str(raw_fps))
                except (ValueError, ZeroDivisionError):
                    fps_val = Fraction(0, 1)
                if isinstance(width, int) and isinstance(height, int) and isinstance(frames, int):
                    if fps_val != Fraction(24, 1) or not is_valid_h3_frame_bucket(frames):
                        raise ValueError('invalid_source_media')
                    return {
                        'width': width,
                        'height': height,
                        'fps': 24,
                        'frames': frames,
                        'has_audio': bool(m.get('audio', m.get('has_audio', False))),
                    }
        if strict:
            raise ValueError('invalid_source_media')
        return None

    def preflight(
        self,
        *,
        project_id: str,
        source_artifact_id: str,
        profile: str | None = None,
        grid_mode: str | None = None,
        prompt: str | None = None,
        seed: int | None = None,
        postprocess: dict | None = None,
    ) -> dict[str, object]:
        if self.service != 'regenerate_2k':
            raise ValueError('preflight is only supported for regenerate_2k')
        from .delivery import authorize
        project_id = require_project_id(project_id)
        authorize(self.client, project_id)
        validated = self.validate_parameters({
            'profile': profile,
            'grid_mode': grid_mode,
            'prompt': prompt,
            'seed': seed,
            'postprocess': postprocess,
        })
        source = self.artifacts.get(source_artifact_id, project_id)
        if not source.media_type.startswith('video/'):
            raise ValueError('invalid_source_media')
        source_media = self._inspect_regenerate_source_media(project_id, source, strict=True)
        geo = resolve_2k_target_geometry(
            int(source_media['width']),
            int(source_media['height']),
            validated['grid_mode'],
        )
        target_w, target_h, levels = geo['target_w'], geo['target_h'], geo['levels']
        regen_cap = self.executor.delivery.regenerate_capability()
        if not regen_cap['configured'] or regen_cap['readiness'] != 'ready':
            raise ValueError('runtime_not_ready')
        norm_post = validated.get('postprocess')
        if norm_post is not None:
            interp_cap = self.executor.delivery.capability()
            if not interp_cap['configured'] or interp_cap['readiness'] != 'ready':
                raise ExecutionError('interpolation backend is not ready')
        resolved_prompt = self._resolve_regenerate_prompt(
            project_id,
            source.artifact_id,
            validated.get('prompt'),
        )
        request = {
            'project_id': project_id,
            'service': self.service,
            'model': REGENERATE_2K_MODEL_ID,
            'profile': validated['profile'],
            'grid_mode': validated['grid_mode'],
            'prompt': resolved_prompt,
            'seed': validated['seed'],
            'source_artifact_id': source.artifact_id,
            'source_sha256': source.sha256,
        }
        if norm_post is not None:
            request['postprocess'] = norm_post
        digest = 'sha256:' + hashlib.sha256(
            json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
        ).hexdigest()
        resolved_output = {
            'width': target_w,
            'height': target_h,
            'fps': 24,
            'frames': int(source_media['frames']),
            'grid_mode': validated['grid_mode'],
            'levels': levels,
        }
        delivery_output = (
            {**resolved_output, 'fps': 48, 'frames': int(source_media['frames']) * 2}
            if norm_post is not None
            else None
        )
        return {
            'status': 'validated',
            'gpu_verified': False,
            'input_digest': digest,
            'source_artifact_id': source.artifact_id,
            'source_media': source_media,
            'resolved_output': resolved_output,
            'delivery_output': delivery_output,
            'postprocess': norm_post,
            'prompt': resolved_prompt,
            'profile': validated['profile'],
            'grid_mode': validated['grid_mode'],
            'seed': validated['seed'],
        }

    def validate_parameters(self, supplied: dict) -> dict:
        if self.service == 'regenerate_2k':
            from .delivery import normalize_regenerate_postprocess
            allowed = {'profile', 'grid_mode', 'prompt', 'seed', 'postprocess'}
            if set(supplied) - allowed:
                raise ValueError('unsupported processing parameters')
            profile = 'open_h3_2k_ic' if supplied.get('profile') is None else supplied['profile']
            if not isinstance(profile, str) or profile not in REGENERATE_2K_PROFILES:
                raise ValueError('unsupported_regenerate_profile')
            grid_mode = 'budget_1440p' if supplied.get('grid_mode') is None else supplied['grid_mode']
            if not isinstance(grid_mode, str) or grid_mode not in REGENERATE_2K_GRID_MODES:
                raise ValueError('unsupported_grid_mode')
            seed = 7 if supplied.get('seed') is None else supplied['seed']
            if type(seed) is not int or not 0 <= seed <= 4294967295:
                raise ValueError('invalid_request')
            prompt = supplied.get('prompt')
            if prompt is not None:
                if not isinstance(prompt, str) or len(prompt.strip()) > 16384:
                    raise ValueError('invalid_prompt')
            postprocess = normalize_regenerate_postprocess(supplied.get('postprocess'))
            values = {
                'profile': profile,
                'grid_mode': grid_mode,
                'prompt': prompt,
                'seed': seed,
            }
            if postprocess is not None:
                values['postprocess'] = postprocess
            return values

        defaults = (
            {'backend': 'seedvr2', 'quality_mode': 'standard', 'seed': 666}
            if self.service == 'sr'
            else {'backend': 'rife', 'target_fps_num': 48, 'target_fps_den': 1}
        )
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
            if (
                values['quality_mode'] != 'standard'
                or type(values['seed']) is not int
                or not 0 <= values['seed'] <= 4294967295
            ):
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

    def generate(
        self,
        *,
        project_id: str,
        idempotency_key: str,
        source_artifact_id: str,
        **parameters,
    ) -> TaskRecord:
        with self._lock:
            project_id = require_project_id(project_id)
            if not idempotency_key.strip() or len(idempotency_key) > 128:
                raise ValueError('idempotency_key is invalid')
            validated = self.validate_parameters(parameters)
            source = self.artifacts.get(source_artifact_id, project_id)
            if not source.media_type.startswith('video/'):
                raise ValueError('source artifact must be video')

            postprocess = validated.get('postprocess')
            authority = None
            inspected = None
            if self.service == 'regenerate_2k':
                inspected = self._inspect_regenerate_source_media(project_id, source, strict=False)
                if inspected is not None:
                    resolve_2k_target_geometry(
                        int(inspected['width']),
                        int(inspected['height']),
                        validated['grid_mode'],
                    )
                resolved_prompt = self._resolve_regenerate_prompt(
                    project_id,
                    source.artifact_id,
                    validated.get('prompt'),
                )
                request = {
                    'project_id': project_id,
                    'service': self.service,
                    'model': REGENERATE_2K_MODEL_ID,
                    'profile': validated['profile'],
                    'grid_mode': validated['grid_mode'],
                    'prompt': resolved_prompt,
                    'seed': validated['seed'],
                    'source_artifact_id': source.artifact_id,
                    'source_sha256': source.sha256,
                }
                if postprocess is not None:
                    from .delivery import authorize
                    authority = authorize(self.client, project_id)
                    request['postprocess'] = postprocess
            else:
                request = {
                    'project_id': project_id,
                    'service': self.service,
                    **validated,
                    'source_artifact_id': source.artifact_id,
                    'source_sha256': source.sha256,
                }

            digest = 'sha256:' + hashlib.sha256(
                json.dumps(request, ensure_ascii=False if self.service == 'regenerate_2k' else True, sort_keys=True, separators=(',', ':')).encode()
            ).hexdigest()

            self.executor._begin_reservation()
            try:
                with self.tasks.exclusive():
                    existing = self.tasks.find_idempotency(project_id, idempotency_key)
                    if existing:
                        if existing.input_digest != digest:
                            raise TaskConflict('idempotency key already exists with different input')
                        return existing
                    if postprocess is not None:
                        cap = self.executor.delivery.capability()
                        if not cap['configured'] or cap['readiness'] != 'ready':
                            raise ExecutionError('interpolation backend is not ready')
                    runtime_url, _, runtime_version = self._runtime_target()
                    record = self.tasks.create(
                        project_id=project_id,
                        idempotency_key=idempotency_key,
                        input_digest=digest,
                        request=request,
                        runtime_task_id='',
                        status='queued',
                        authority=authority,
                        runtime_endpoint=runtime_url,
                        runtime_version=runtime_version,
                        runtime_route='h3-singularity' if self.service == 'regenerate_2k' else None,
                    )
                    initial_pstate = {'source_frames': int(inspected['frames'])} if isinstance(inspected, dict) and type(inspected.get('frames')) is int else None
                    if initial_pstate is not None:
                        record = self.tasks.update(record, provider_state=initial_pstate)
                    initial_delivery = self.executor.delivery.initial(record) if postprocess is not None else None
                    record = self.tasks.update(
                        record,
                        service=self.service,
                        runtime_version=runtime_version,
                        runtime_task_id=record.video_task_id,
                        runtime_endpoint=runtime_url,
                        runtime_route='h3-singularity' if self.service == 'regenerate_2k' else None,
                        authority=authority,
                        delivery=initial_delivery,
                        error={'code': 'submission_unconfirmed', 'message': 'Submission is pending confirmation'},
                    )
                    with self.executor._state_guard:
                        self.executor._submitting_tasks.add(record.video_task_id)
            finally:
                self.executor._end_reservation()

        return self._submit(record, already_marked_submitting=True)

    @staticmethod
    def _extract_error_code(response: httpx.Response) -> str | None:
        try:
            body = response.json()
        except (ValueError, TypeError):
            return None
        if not isinstance(body, dict):
            return None
        for field in ('error', 'detail'):
            item = body.get(field)
            if isinstance(item, dict) and isinstance(item.get('code'), str):
                return item['code']
        return None

    def _fail_record(self, record: TaskRecord, error: dict[str, object]) -> TaskRecord:
        provider_state = {
            k: v
            for k, v in (record.provider_state or {}).items()
            if k not in {'waiting_reason', 'gpu_dispatched'}
        }
        provider_state['worker_settled'] = True
        updates: dict[str, object] = {
            'status': 'failed',
            'error': error,
            'provider_state': provider_state,
        }
        if record.delivery is not None:
            now = record.updated_at
            updates['delivery'] = {
                **record.delivery,
                'status': 'failed',
                'stage': 'regenerate_2k' if self.service == 'regenerate_2k' else record.delivery.get('stage'),
                'error': error,
                'completed_at': record.delivery.get('completed_at') or now,
            }
        updated = self.tasks.update(record, **updates)
        self.executor.stage_slot.release(updated.video_task_id, force=True)
        return updated

    def _submit(self, record: TaskRecord, *, already_marked_submitting: bool = False) -> TaskRecord:
        if not already_marked_submitting:
            with self.executor._state_guard:
                self.executor._submitting_tasks.add(record.video_task_id)

        url, headers, _ = self._runtime_target(record)
        payload = {
            k: v
            for k, v in record.request.items()
            if k not in {'service', 'model', 'postprocess'}
        }
        payload['idempotency_key'] = record.video_task_id
        acquired = False
        try:
            if self.service in {'interpolate', 'regenerate_2k'} and (
                self.executor.gpu.enabled or self.executor.stage_slot.enabled
            ):
                try:
                    acquired = self.executor.acquire_local_gpu(
                        record.video_task_id,
                        self.service,
                        record=record,
                    )
                except LeaseBusy:
                    return self._fail_record(
                        record,
                        {
                            'code': 'gpu_busy',
                            'message': MESSAGES['gpu_busy'],
                            'retryable': True,
                        },
                    )
                if not acquired:
                    return self.executor.defer_local_gpu_queue(record)

            dispatched_pstate = {
                k: v for k, v in (record.provider_state or {}).items() if k != 'waiting_reason'
            }
            dispatched_pstate['gpu_dispatched'] = True
            record = self.tasks.update(record, provider_state=dispatched_pstate)

            try:
                response = self.client.post(
                    f'{url}/v1/{self.endpoint_path}',
                    json=payload,
                    headers=headers,
                    timeout=30,
                )
                if response.status_code in {400, 404, 409, 422, 429} or (
                    self.service == 'regenerate_2k' and response.status_code == 503
                ):
                    extracted = self._extract_error_code(response)
                    code = extracted if extracted in SAFE_REJECTION_CODES else f'{self.service}_request_rejected'
                    message = MESSAGES.get(
                        code,
                        'Video processing runtime rejected the source, parameters or queue capacity',
                    )
                    return self._fail_record(
                        record,
                        {
                            'code': code,
                            'message': message,
                            'retryable': code in {'queue_full', 'runtime_not_ready'},
                        },
                    )
                response.raise_for_status()
                record = self.tasks.update(
                    record,
                    dispatched_at=record.dispatched_at or datetime.now(UTC).isoformat(),
                )
                return self._update(record, response.json())
            except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
                return self.tasks.update(
                    record,
                    error={
                        'code': 'submission_unconfirmed',
                        'message': 'Query this task ID to recover the persisted request',
                    },
                )
        finally:
            with self.executor._state_guard:
                self.executor._submitting_tasks.discard(record.video_task_id)
            if acquired:
                self.executor.release_or_reconcile_local_gpu(record.video_task_id)

    def _update(self, record: TaskRecord, data: dict) -> TaskRecord:
        status = data['status']
        if data.get('video_task_id') != record.video_task_id or status not in {
            'queued', 'running', 'succeeded', 'failed', 'cancelled'
        }:
            raise ValueError('invalid runtime status')
        error = None
        if status == 'failed':
            code = (data.get('error') or {}).get('code')
            if code not in SAFE_RUNTIME_FAILURE_CODES:
                code = f'{self.service}_failed'
            error = {
                'code': code,
                'message': MESSAGES.get(code, 'Video processing conversion failed'),
            }
        elif status == 'cancelled':
            if record.error and str(record.error.get('code', '')).startswith(('stage_timeout_', 'chain_active_timeout')):
                error = record.error
            else:
                error = {'code': 'cancelled', 'message': 'Video processing task was cancelled'}

        provider_state = dict(record.provider_state or {})
        for key in ('started_at', 'finished_at', 'worker_settled'):
            if key in data:
                provider_state[key] = data[key]
        if status in {'succeeded', 'failed', 'cancelled'} and 'worker_settled' not in data:
            provider_state.setdefault('worker_settled', True)

        pending_timeout = provider_state.get('_pending_timeout_error')
        if pending_timeout is not None:
            if provider_state.get('worker_settled') is True:
                provider_state.pop('_pending_timeout_error', None)
                status = 'failed'
                error = pending_timeout
            else:
                status = 'running'
                error = None

        stage = data.get('stage') if isinstance(data.get('stage'), str) else record.runtime_stage
        metrics = record.runtime_metrics
        result_block = data.get('result')
        if isinstance(result_block, dict) and isinstance(result_block.get('runtime_metrics'), dict):
            metrics = result_block['runtime_metrics']
        elif isinstance(data.get('runtime_metrics'), dict):
            metrics = data['runtime_metrics']

        updated = self.tasks.update(
            record,
            status=status,
            error=error,
            provider_state=provider_state or None,
            runtime_stage=stage,
            runtime_metrics=metrics,
        )
        if self.executor.stage_slot.is_task_settled(updated):
            self.executor.stage_slot.release(updated.video_task_id)
        return updated

    def cancel(self, video_task_id: str, *, timeout_error: dict[str, object] | None = None) -> TaskRecord:
        record = self.tasks.get(video_task_id)
        if record.service != self.service:
            raise ValueError('task is not a matching processing task')
        prev_state = dict(record.provider_state or {})
        now_epoch = time.time()
        attempts = int(prev_state.get('_cancel_attempts', 0)) + 1
        worker_settled = bool(prev_state.get('worker_settled', False))
        cancel_delivered = bool(prev_state.get('cancel_delivered', False))
        started_at = prev_state.get('started_at')
        finished_at = prev_state.get('finished_at')
        force_backoff = False

        if prev_state.get('gpu_dispatched') is False:
            worker_settled = True
            cancel_delivered = True
        else:
            url, headers, _ = self._runtime_target(record)
            try:
                response = self.client.delete(
                    f'{url}/v1/{self.endpoint_path}/{record.video_task_id}',
                    headers=headers,
                    timeout=15,
                )
                if response.status_code == 200:
                    body = response.json()
                    worker_settled = bool(body.get('worker_settled', False))
                    started_at = body.get('started_at', started_at)
                    finished_at = body.get('finished_at', finished_at)
                    if (
                        worker_settled
                        or body.get('cancel_requested') is True
                        or body.get('status') in {'cancelled', 'canceled', 'failed', 'succeeded', 'completed'}
                    ):
                        cancel_delivered = True
                    else:
                        force_backoff = True
                elif response.status_code == 409:
                    cancel_delivered = True
                    poll = self.client.get(
                        f'{url}/v1/{self.endpoint_path}/{record.video_task_id}',
                        headers=headers,
                        timeout=10,
                    )
                    if poll.status_code == 200:
                        body = poll.json()
                        if timeout_error is None:
                            return self._update(record, body)
                        worker_settled = bool(body.get('worker_settled', True))
                        started_at = body.get('started_at', started_at)
                        finished_at = body.get('finished_at', finished_at)
                elif response.status_code == 404:
                    worker_settled = True
                    cancel_delivered = True
                else:
                    worker_settled = False
            except (httpx.HTTPError, ValueError, TypeError):
                worker_settled = False

        next_retry_at = (
            0.0
            if cancel_delivered
            else _next_cancel_backoff_epoch(max(attempts, 2) if force_backoff else attempts, now_epoch)
        )
        provider_state = {
            **prev_state,
            'started_at': started_at,
            'finished_at': finished_at,
            'worker_settled': worker_settled,
            'cancel_delivered': cancel_delivered,
            '_cancel_requested': True,
            '_cancel_attempts': attempts,
            '_next_cancel_retry_at': next_retry_at,
        }
        provider_state.pop('waiting_reason', None)
        if prev_state.get('gpu_dispatched') is False:
            provider_state.pop('gpu_dispatched', None)
        if timeout_error is not None and not worker_settled:
            provider_state['_pending_timeout_error'] = timeout_error
            target_status = 'running'
            target_error = None
        else:
            provider_state.pop('_pending_timeout_error', None)
            target_status = 'failed' if timeout_error is not None else 'cancelled'
            target_error = timeout_error or {'code': 'cancelled', 'message': 'Video processing task was cancelled'}
        updated = self.tasks.update(
            record,
            status=target_status,
            error=target_error,
            provider_state=provider_state,
        )
        if worker_settled:
            self.executor.stage_slot.release(updated.video_task_id)
        return updated

    @task_serialized
    def status(self, video_task_id: str) -> TaskRecord:
        record = self.tasks.get(video_task_id)
        if record.service != self.service:
            raise ValueError('task is not a matching processing task')
        if record.status == 'queued' and (record.provider_state or {}).get('gpu_dispatched') is False:
            return self._submit(record)

        pstate_init = record.provider_state or {}
        if (
            pstate_init.get('worker_settled') is False
            and (pstate_init.get('_pending_timeout_error') is not None or pstate_init.get('_cancel_requested') is True)
            and pstate_init.get('cancel_delivered') is not True
            and _cancel_backoff_ready(pstate_init, time.time())
        ):
            record = self.cancel(video_task_id, timeout_error=pstate_init.get('_pending_timeout_error'))
            if record.status in {'succeeded', 'failed', 'cancelled'} and (record.provider_state or {}).get('worker_settled') is not False:
                return record
            if (record.provider_state or {}).get('cancel_delivered') is not True:
                return record

        if record.status in {'succeeded', 'failed', 'cancelled'}:
            if (record.provider_state or {}).get('worker_settled') is not False:
                return record
            url, headers, _ = self._runtime_target(record)
            try:
                poll = self.client.get(
                    f'{url}/v1/{self.endpoint_path}/{record.video_task_id}',
                    headers=headers,
                    timeout=10,
                )
                if poll.status_code == 200:
                    body = poll.json()
                    provider_state = {
                        **(record.provider_state or {}),
                        'worker_settled': bool(body.get('worker_settled', False)),
                    }
                    for key in ('started_at', 'finished_at'):
                        if key in body:
                            provider_state[key] = body[key]
                    record = self.tasks.update(record, provider_state=provider_state)
                    if provider_state.get('worker_settled') is True:
                        self.executor.stage_slot.release(record.video_task_id)
                elif poll.status_code == 404:
                    provider_state = {**(record.provider_state or {}), 'worker_settled': True}
                    record = self.tasks.update(record, provider_state=provider_state)
                    self.executor.stage_slot.release(record.video_task_id)
            except (httpx.HTTPError, ValueError, TypeError):
                pass
            return record

        url, headers, _ = self._runtime_target(record)
        try:
            response = self.client.get(
                f'{url}/v1/{self.endpoint_path}/{record.video_task_id}',
                headers=headers,
                timeout=10,
            )
            if response.status_code == 404:
                # The deterministic ID is also the runtime's durable idempotency key.
                # A retry after an ambiguous POST cannot enqueue duplicate inference.
                if not record.runtime_task_id or (
                    record.error and record.error.get('code') == 'submission_unconfirmed'
                ):
                    return self._submit(record)
                return self.tasks.update(
                    record,
                    status='failed',
                    error={'code': 'runtime_task_lost', 'message': 'Video processing runtime task is missing'},
                )
            response.raise_for_status()
            updated = self._update(record, response.json())
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ExecutionError('Video processing status temporarily unavailable; keep the task ID') from exc

        if (
            self.service in {'regenerate_2k', 'interpolate'}
            and updated.status == 'running'
            and not updated.request.get('postprocess')
            and not updated.idempotency_key.startswith('delivery:')
            and (updated.provider_state or {}).get('_pending_timeout_error') is None
        ):
            from .delivery import P2_WINDOW_TIMEOUT_SECONDS, P3_STAGE_TIMEOUT_SECONDS, _to_epoch, _window_count_from_record
            import time as _time
            started_epoch = _to_epoch((updated.provider_state or {}).get('started_at'))
            if started_epoch is not None:
                active_s = max(0.0, _time.time() - started_epoch)
                if self.service == 'regenerate_2k':
                    limit_s = P2_WINDOW_TIMEOUT_SECONDS * _window_count_from_record(updated)
                    code = 'stage_timeout_regenerate_2k'
                else:
                    limit_s = P3_STAGE_TIMEOUT_SECONDS
                    code = 'stage_timeout_interpolation'
                if active_s > limit_s:
                    return self.cancel(
                        updated.video_task_id,
                        timeout_error={'code': code, 'message': MESSAGES[code], 'retryable': False},
                    )
        return updated

    def _artifact(self, record: TaskRecord, endpoint: str, expected_sha256: str):
        if not isinstance(expected_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', expected_sha256):
            raise ExecutionError('Video processing result digest missing')
        url, headers, _ = self._runtime_target(record)
        try:
            with self.client.stream(
                'GET',
                f'{url}/v1/{self.endpoint_path}/{record.video_task_id}/{endpoint}',
                headers=headers,
                timeout=120,
            ) as response:
                response.raise_for_status()
                return self.artifacts.create_from_chunks(
                    project_id=record.project_id,
                    operation=f'video.{self.service}.{endpoint}',
                    filename=f'{record.video_task_id}-{endpoint}.mp4',
                    media_type='video/mp4',
                    chunks=response.iter_bytes(),
                    expected_sha256=expected_sha256,
                )
        except httpx.HTTPError as exc:
            raise ExecutionError('Video processing result download failed; retry this task ID') from exc

    @task_serialized
    def result(self, video_task_id: str) -> TaskRecord:
        record = self.status(video_task_id)
        if record.status != 'succeeded':
            raise ExecutionError('Video processing task has not succeeded')
        if record.artifact_id and record.media and record.media.get('preview_artifact_id'):
            return record
        url, headers, _ = self._runtime_target(record)
        try:
            response = self.client.get(
                f'{url}/v1/{self.endpoint_path}/{record.video_task_id}',
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            if payload['video_task_id'] != record.video_task_id or payload['status'] != 'succeeded':
                raise ValueError('result task mismatch')
            data = payload['result']
            if data['input_sha256'] != record.request['source_sha256']:
                raise ValueError('source mismatch')

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
                if data['parameters']['backend'] != record.request['backend']:
                    raise ValueError('backend mismatch')
                if (media['width'], media['height']) != (
                    record.request['target_width'],
                    record.request['target_height'],
                ):
                    raise ValueError('resolution mismatch')
                if media['frames'] != source['frames'] or Fraction(media['frame_rate']) != Fraction(source['frame_rate']):
                    raise ValueError('timeline mismatch')
            elif self.service == 'regenerate_2k':
                expected_geo = resolve_2k_target_geometry(
                    source['width'],
                    source['height'],
                    record.request['grid_mode'],
                )
                if (media['width'], media['height']) != (
                    expected_geo['target_w'],
                    expected_geo['target_h'],
                ):
                    raise ValueError('regenerate_2k geometry mismatch')
                if (
                    media['frames'] != source['frames']
                    or not is_valid_h3_frame_bucket(media['frames'])
                    or Fraction(media['frame_rate']) != Fraction(24, 1)
                    or Fraction(source['frame_rate']) != Fraction(24, 1)
                ):
                    raise ValueError('regenerate_2k timeline mismatch')
            else:
                if data['parameters']['backend'] != record.request['backend']:
                    raise ValueError('backend mismatch')
                fps = Fraction(record.request['target_fps_num'], record.request['target_fps_den'])
                if Fraction(media['frame_rate']) != fps or fps != 2 * Fraction(source['frame_rate']):
                    raise ValueError('frame rate mismatch')
                if media['frames'] != source['frames'] * 2 or any(
                    media[k] != source[k] for k in ('width', 'height')
                ):
                    raise ValueError('interpolation geometry mismatch')

            duration = float(Fraction(media['frames']) / Fraction(media['frame_rate']))
            if abs(duration - float(media['duration_seconds'])) > 0.001 or abs(
                duration - float(source['duration_seconds'])
            ) > 0.001:
                raise ValueError('duration mismatch')

            expected_pw, expected_ph = preview_geometry(media['width'], media['height'])
            if (preview['width'], preview['height']) != (expected_pw, expected_ph):
                raise ValueError('preview geometry mismatch')
            if any(preview[k] != media[k] for k in ('frames', 'frame_rate', 'duration_seconds', 'video_codec', 'audio')):
                raise ValueError('preview mismatch')

            model = data['model']
            if self.service == 'regenerate_2k':
                if (
                    model.get('backend') != 'open_h3_2k_ic'
                    or model.get('profile') != 'open_h3_2k_ic'
                    or model.get('code_revision') != LOCKED_P2_CODE_REVISION
                ):
                    raise ValueError('regenerate_2k model mismatch')
                if not re.fullmatch(
                    r'(?:[0-9a-f]{40}|sha256:[0-9a-f]{64})',
                    str(model.get('weights_revision', '')),
                ) or not re.fullmatch(r'[0-9a-f]{64}', str(model.get('manifest_sha256', ''))):
                    raise ValueError('model digest missing')
                weights = {name: model['weights_sha256'][name] for name in LOCKED_P2_WEIGHTS_SHA256}
                if weights != LOCKED_P2_WEIGHTS_SHA256:
                    raise ValueError('locked regenerate_2k weight digest mismatch')
                media['model'] = {
                    'backend': model['backend'],
                    'profile': model['profile'],
                    'code_revision': model['code_revision'],
                    'weights_revision': model['weights_revision'],
                    'manifest_sha256': model['manifest_sha256'],
                    'weights_sha256': weights,
                }

                geometry = data['geometry']
                if (
                    geometry.get('source_w'),
                    geometry.get('source_h'),
                    geometry.get('target_w'),
                    geometry.get('target_h'),
                    geometry.get('levels'),
                ) != (
                    expected_geo['source_w'],
                    expected_geo['source_h'],
                    expected_geo['target_w'],
                    expected_geo['target_h'],
                    expected_geo['levels'],
                ):
                    raise ValueError('regenerate_2k reported geometry mismatch')
                expected_windows = [[a, b] for a, b in temporal_windows(source['frames'])]
                if data.get('windows') != expected_windows:
                    raise ValueError('regenerate_2k windows mismatch')

                params = data['parameters']
                expected_prompt_sha = hashlib.sha256(
                    str(record.request['prompt']).encode('utf-8')
                ).hexdigest()
                expected_audio = 'copy' if source['audio'] else 'none'
                if (
                    params.get('profile') != record.request['profile']
                    or params.get('grid_mode') != record.request['grid_mode']
                    or params.get('seed') != record.request['seed']
                    or params.get('prompt_sha256') != expected_prompt_sha
                    or params.get('audio') != expected_audio
                    or params.get('video_codec') != 'h264'
                    or params.get('pixel_format') != 'yuv420p'
                    or params.get('crf') != 16
                    or params.get('tiles') != 2
                ):
                    raise ValueError('regenerate_2k parameter mismatch')
                if data['media'].get('audio_mode') != expected_audio:
                    raise ValueError('regenerate_2k audio_mode mismatch')
                media['audio_mode'] = expected_audio
                if source['audio']:
                    audio_sha = data['media'].get('audio_sha256')
                    if not isinstance(audio_sha, str) or not re.fullmatch(r'[0-9a-f]{64}', audio_sha):
                        raise ValueError('regenerate_2k audio_sha256 missing')
                    media['audio_sha256'] = audio_sha
                media['geometry'] = {
                    'source_w': expected_geo['source_w'],
                    'source_h': expected_geo['source_h'],
                    'target_w': expected_geo['target_w'],
                    'target_h': expected_geo['target_h'],
                    'levels': expected_geo['levels'],
                }
                media['windows'] = expected_windows
                media['parameters'] = {
                    'profile': params['profile'],
                    'grid_mode': params['grid_mode'],
                    'seed': params['seed'],
                    'prompt_sha256': expected_prompt_sha,
                    'audio': expected_audio,
                    'video_codec': 'h264',
                    'pixel_format': 'yuv420p',
                    'crf': 16,
                    'tiles': 2,
                }
            else:
                if model['backend'] != self.backend or not re.fullmatch(r'[0-9a-f]{40}', model['code_revision']):
                    raise ValueError('model mismatch')
                if not re.fullmatch(
                    r'(?:[0-9a-f]{40}|sha256:[0-9a-f]{64})',
                    model['weights_revision'],
                ) or not re.fullmatch(r'[0-9a-f]{64}', model['manifest_sha256']):
                    raise ValueError('model digest missing')
                filenames = (
                    ('seedvr2_ema_3b.pth', 'ema_vae.pth', 'pos_emb.pt', 'neg_emb.pt')
                    if self.service == 'sr'
                    else ('flownet.pkl',)
                )
                weights = {name: model['weights_sha256'][name] for name in filenames}
                if not all(re.fullmatch(r'[0-9a-f]{64}', value) for value in weights.values()):
                    raise ValueError('weight digest missing')
                media['model'] = {
                    k: model[k] for k in ('backend', 'code_revision', 'weights_revision', 'manifest_sha256')
                }
                media['model']['weights_sha256'] = weights

                requested = {
                    k: v
                    for k, v in record.request.items()
                    if k not in {'project_id', 'service', 'source_artifact_id', 'source_sha256'}
                }
                if any(data['parameters'][k] != v for k, v in requested.items()):
                    raise ValueError('parameter mismatch')
                # Fixed public encoding policy; arbitrary runtime keys never cross the MCP boundary.
                audio_policy = data['parameters'].get('audio')
                allowed_audio = (
                    {'aac_if_present', 'aac_copy_if_aligned'}
                    if self.service == 'interpolate'
                    else {'aac_if_present'}
                )
                if audio_policy not in allowed_audio:
                    raise ValueError('processing policy mismatch')
                fixed = {'audio': audio_policy, 'video_codec': 'h264', 'pixel_format': 'yuv420p', 'crf': 18}
                fixed.update(
                    {'scene_cut_threshold': 0.30, 'last_frame': 'hold'}
                    if self.service == 'interpolate'
                    else {'sample_steps': 1, 'cfg_scale': 1.0, 'normalization': 'torch_rms_layer', 'color_fix': False}
                )
                if any(data['parameters'][k] != v for k, v in fixed.items()):
                    raise ValueError('processing policy mismatch')
                media['parameters'] = {**requested, **fixed}
                if self.service == 'interpolate' and data['media'].get('audio_mode') in {'copy', 'transcode', 'none'}:
                    media['audio_mode'] = data['media']['audio_mode']
                if self.service == 'sr':
                    raw_window = data['parameters']['windowing']
                    window_keys = (
                        'window_frames', 'overlap_frames', 'capacity_profile_sha256', 'reserve_bytes',
                        'seam_method', 'window_seed', 'scene_cut_threshold', 'chunks',
                        'max_chunk_frames', 'output_frames',
                    )
                    window = {k: raw_window[k] for k in window_keys}
                    for key in (
                        'window_frames', 'overlap_frames', 'reserve_bytes', 'chunks',
                        'max_chunk_frames', 'output_frames',
                    ):
                        if type(window[key]) is not int or window[key] < 0:
                            raise ValueError('invalid window metadata')
                    if (
                        window['window_frames'] < 9
                        or (window['window_frames'] - 1) % 4
                        or window['overlap_frames'] != 4
                        or not 0 < window['chunks'] <= media['frames']
                        or not 0 < window['max_chunk_frames'] <= min(window['window_frames'], media['frames'])
                        or window['output_frames'] != media['frames']
                        or window['seam_method'] != 'linear_overlap'
                        or window['window_seed'] != 'source_offset'
                        or window['scene_cut_threshold'] != 0.30
                        or not re.fullmatch(r'[0-9a-f]{64}', window['capacity_profile_sha256'])
                    ):
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

        existing_comparable = (
            {k: v for k, v in record.media.items() if k != 'preview_artifact_id'}
            if record.media
            else None
        )
        if record.artifact_id and existing_comparable != media:
            raise ExecutionError('Video processing result changed after registration')
        if not record.artifact_id:
            artifact = self._artifact(record, 'content', data['output_sha256'])
            record = self.tasks.update(record, artifact_id=artifact.artifact_id, media=media)
        preview_art = self._artifact(record, 'preview', data['preview_sha256'])
        media_with_preview = {**media, 'preview_artifact_id': preview_art.artifact_id}
        return self.tasks.update(record, media=media_with_preview)

    def public_status(self, record: TaskRecord) -> dict:
        value = {
            'video_task_id': record.video_task_id,
            'service': self.service,
            'status': record.status,
            'created_at': record.created_at,
            'updated_at': record.updated_at,
            'error': record.error,
        }
        if self.service == 'regenerate_2k':
            value['delivery'] = record.delivery
            value['source_artifact_id'] = record.request.get('source_artifact_id')
            value['grid_mode'] = record.request.get('grid_mode')
            value['profile'] = record.request.get('profile')
        return value

    def public_result(self, record: TaskRecord, preview: bool = False) -> dict:
        artifact_id = record.media['preview_artifact_id'] if preview else record.artifact_id
        artifact = self.artifacts.get(artifact_id, record.project_id)
        base_media = {**record.media, **record.media['preview_media']} if preview else dict(record.media)
        enriched_media = {
            **base_media,
            'sha256': artifact.sha256,
            'size': int(artifact.size),
        }
        value = {
            **self.public_status(record),
            'project_id': record.project_id,
            'artifact_id': artifact_id,
            'download_path': self.artifacts.download_path(artifact_id),
            'sha256': artifact.sha256,
            'size': int(artifact.size),
            'source_artifact_id': record.request['source_artifact_id'],
            'media': enriched_media,
        }
        return value
