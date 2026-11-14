"""Durable, opt-in Singularity -> 2K IC -> RIFE delivery; no client polling dependency."""
from __future__ import annotations

from contextvars import ContextVar
from datetime import UTC, datetime
import hashlib
import json
import os
import secrets
import time
import uuid

import httpx

from .channel_errors import MESSAGES
from .executor import ExecutionError, _cancel_backoff_ready
from .regenerate import (
    REGENERATE_2K_GRID_MODES,
    REGENERATE_2K_PROFILES,
    is_valid_h3_frame_bucket,
    resolve_2k_target_geometry,
    temporal_windows,
)

CALLER = ContextVar('video_verified_caller', default=None)
OPTION = {'interpolate': {'profile': 'rife_2x'}}

P1_STAGE_TIMEOUT_SECONDS = 900.0
P2_WINDOW_TIMEOUT_SECONDS = 900.0
P3_STAGE_TIMEOUT_SECONDS = 300.0
LEGACY_P1_P3_CHAIN_TIMEOUT_SECONDS = 1200.0
P2_CHAIN_TIMEOUT_BY_WINDOWS = {1: 1800.0, 2: 2700.0, 3: 3600.0}


def request_id() -> str:
    # Project's service principal contract requires UUIDv7, including requests.
    value = ((time.time_ns() // 1_000_000) << 80) | (7 << 76)
    value |= secrets.randbits(12) << 64 | (2 << 62) | secrets.randbits(62)
    return str(uuid.UUID(int=value))


def _normalize_regenerate_block(block) -> dict[str, str]:
    if not isinstance(block, dict) or set(block) - {'profile', 'grid_mode'}:
        raise ValueError('postprocess.regenerate_2k requires profile=open_h3_2k_ic and optional grid_mode')
    profile = 'open_h3_2k_ic' if block.get('profile') is None else block.get('profile')
    if not isinstance(profile, str) or profile not in REGENERATE_2K_PROFILES:
        raise ValueError('unsupported_regenerate_profile')
    grid_mode = 'budget_1440p' if block.get('grid_mode') is None else block.get('grid_mode')
    if not isinstance(grid_mode, str) or grid_mode not in REGENERATE_2K_GRID_MODES:
        raise ValueError('unsupported_grid_mode')
    return {'profile': profile, 'grid_mode': grid_mode}


def normalize(value, route: str):
    if value is None:
        return None
    if route != 'h3-singularity' or not isinstance(value, dict) or not value:
        raise ValueError('postprocess requires h3-singularity and regenerate_2k/interpolate options')
    if set(value) - {'regenerate_2k', 'interpolate'}:
        raise ValueError('postprocess contains unsupported stage keys')
    normalized: dict[str, dict[str, str]] = {}
    if 'regenerate_2k' in value:
        normalized['regenerate_2k'] = _normalize_regenerate_block(value['regenerate_2k'])
    if 'interpolate' in value:
        if value['interpolate'] != {'profile': 'rife_2x'}:
            raise ValueError('postprocess requires h3-singularity and interpolate.profile=rife_2x')
        normalized['interpolate'] = {'profile': 'rife_2x'}
    return normalized


def normalize_regenerate_postprocess(value):
    if value is None:
        return None
    if not isinstance(value, dict) or value != OPTION:
        raise ValueError('regenerate_2k postprocess supports only interpolate.profile=rife_2x')
    return {'interpolate': {'profile': 'rife_2x'}}


def authorize(client, project_id, authority=None, *, require_write=True):
    actor = authority if authority is not None else CALLER.get()
    if not actor or not all(actor.get(k) for k in ('user_id', 'organization_id')):
        raise ValueError('verified Studio identity required')
    url = os.environ.get('VIDEO_PROJECT_SERVICE_URL', '').rstrip('/')
    token = os.environ.get('VIDEO_PROJECT_SERVICE_TOKEN', '')
    if not url or not token:
        raise ExecutionError('project authorization is not configured')
    try:
        response = client.post(
            url + '/internal/v1/project/open',
            headers={
                'Authorization': 'Bearer ' + token,
                'X-User-Id': actor['user_id'],
                'X-Organization-Id': actor['organization_id'],
                'X-Request-Id': request_id(),
            },
            json={'project_id': project_id, **({'require_write': True} if require_write else {})},
            timeout=15,
        )
        if response.status_code in (401, 403, 404):
            raise PermissionError('project access denied')
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ExecutionError('project authorization temporarily unavailable') from exc
    return dict(actor)


def _to_epoch(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return None
    return None


def _window_count_from_record(*records) -> int:
    for record in records:
        if record is None:
            continue
        frames = None
        resolved = record.request.get('resolved_output')
        if isinstance(resolved, dict) and type(resolved.get('frames')) is int:
            frames = resolved['frames']
        elif isinstance(record.provider_state, dict) and type(record.provider_state.get('source_frames')) is int:
            frames = record.provider_state['source_frames']
        elif isinstance(record.media, dict):
            if type(record.media.get('frames')) is int:
                frames = record.media['frames']
            elif isinstance(record.media.get('input_media'), dict) and type(record.media['input_media'].get('frames')) is int:
                frames = record.media['input_media']['frames']
        if frames is not None and is_valid_h3_frame_bucket(frames):
            return len(temporal_windows(frames))
    return 1


def _stage_active_and_queue_seconds(record, now_epoch: float) -> tuple[float, float]:
    if record is None:
        return 0.0, 0.0
    created_epoch = _to_epoch(record.created_at)
    dispatched_epoch = _to_epoch(record.dispatched_at)
    started_epoch = _to_epoch((record.provider_state or {}).get('started_at'))
    finished_epoch = _to_epoch((record.provider_state or {}).get('finished_at'))
    is_terminal = record.status in {'succeeded', 'failed', 'cancelled'}

    if started_epoch is None:
        if is_terminal:
            return 0.0, 0.0
        if dispatched_epoch is not None and created_epoch is not None:
            q_sec = max(0.0, dispatched_epoch - created_epoch) + max(0.0, now_epoch - dispatched_epoch)
        elif created_epoch is not None:
            q_sec = max(0.0, now_epoch - created_epoch)
        elif dispatched_epoch is not None:
            q_sec = max(0.0, now_epoch - dispatched_epoch)
        else:
            q_sec = 0.0
        return 0.0, round(q_sec, 3)

    if dispatched_epoch is not None and created_epoch is not None:
        q_sec = max(0.0, dispatched_epoch - created_epoch) + max(0.0, started_epoch - dispatched_epoch)
    elif created_epoch is not None:
        q_sec = max(0.0, started_epoch - created_epoch)
    elif dispatched_epoch is not None:
        q_sec = max(0.0, started_epoch - dispatched_epoch)
    else:
        q_sec = 0.0

    if is_terminal:
        end_epoch = finished_epoch if finished_epoch is not None else (_to_epoch(record.completed_at or record.updated_at) or started_epoch)
    else:
        end_epoch = now_epoch
    active_sec = max(0.0, end_epoch - started_epoch)
    return round(active_sec, 3), round(q_sec, 3)


class Delivery:
    def __init__(self, executor):
        self.executor = executor
        self.tasks = executor.tasks
        self._lock = executor._lock

    def capability(self):
        adapter = self.executor.processing['interpolate']
        result = {
            'profile': 'rife_2x',
            'backend': 'rife',
            'factor': 2,
            'generation_routes': ['h3-singularity'],
            'default_enabled': False,
            'configured': bool(
                os.environ.get('VIDEO_PROJECT_SERVICE_URL')
                and os.environ.get('VIDEO_PROJECT_SERVICE_TOKEN')
            ),
            'readiness': 'unreachable',
            'runtime_version': adapter.runtime_version,
        }
        if not result['configured']:
            result['readiness'] = 'not_configured'
            return result
        try:
            response = adapter.client.get(adapter.url + '/health', timeout=5)
            data = response.json()
            result['readiness'] = (
                'ready'
                if response.status_code == 200 and data.get('status') == 'ok' and data.get('service') == 'interpolate'
                else 'not_ready'
            )
        except (httpx.HTTPError, ValueError, AttributeError):
            pass
        return result

    def regenerate_capability(self):
        project_configured = bool(
            os.environ.get('VIDEO_PROJECT_SERVICE_URL')
            and os.environ.get('VIDEO_PROJECT_SERVICE_TOKEN')
        )
        runtime_url = ''
        runtime_headers: dict[str, str] = {}
        runtime_version = self.executor.processing['regenerate_2k'].runtime_version
        runtime_connected = False
        try:
            runtime_url, runtime_headers, runtime_version, _ = self.executor.runtime('h3-singularity')
            runtime_connected = True
        except ExecutionError:
            runtime_connected = False

        configured = project_configured and runtime_connected
        result = {
            'profile': 'open_h3_2k_ic',
            'backend': 'open_h3_2k_ic',
            'grid_modes': list(REGENERATE_2K_GRID_MODES),
            'default_grid_mode': 'budget_1440p',
            'generation_routes': ['h3-singularity'],
            'default_enabled': False,
            'configured': configured,
            'readiness': 'not_configured' if not configured else 'unreachable',
            'runtime_version': runtime_version,
        }
        if not configured:
            return result
        try:
            response = self.executor.client.get(f'{runtime_url}/health', headers=runtime_headers, timeout=5)
            data = response.json()
            if response.status_code == 200 and isinstance(data, dict) and data.get('ready') is True:
                caps = data.get('capabilities')
                if isinstance(caps, dict):
                    supported = caps.get('regenerate_2k_profiles')
                    ready = isinstance(supported, list) and 'open_h3_2k_ic' in supported
                else:
                    ready = True
                result['readiness'] = 'ready' if ready else 'not_ready'
                if isinstance(data.get('runtime_version'), str):
                    result['runtime_version'] = data['runtime_version']
            else:
                result['readiness'] = 'not_ready'
        except (httpx.HTTPError, ValueError, AttributeError):
            pass
        return result

    def prepare(self, request, digest, option, check=True):
        normalized = normalize(option, request['route'])
        if normalized is None:
            return request, digest
        if 'regenerate_2k' in normalized and isinstance(request.get('resolved_output'), dict):
            resolve_2k_target_geometry(
                request['resolved_output']['width'],
                request['resolved_output']['height'],
                normalized['regenerate_2k']['grid_mode'],
            )
        if check:
            if 'regenerate_2k' in normalized:
                rcap = self.regenerate_capability()
                if not rcap['configured'] or rcap['readiness'] != 'ready':
                    raise ExecutionError('runtime_not_ready')
            if 'interpolate' in normalized:
                cap = self.capability()
                if not cap['configured'] or cap['readiness'] != 'ready':
                    raise ExecutionError('interpolation backend is not ready')
        request = {**request, 'postprocess': normalized}
        canonical = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        return request, 'sha256:' + hashlib.sha256(canonical.encode()).hexdigest()

    @staticmethod
    def _profiles_for_record(record) -> tuple[str, list[str], bool, bool]:
        post = record.request.get('postprocess') or {}
        standalone_p2 = record.service == 'regenerate_2k'
        has_p2 = standalone_p2 or ('regenerate_2k' in post)
        has_p3 = 'interpolate' in post
        profiles: list[str] = []
        if 'regenerate_2k' in post:
            profiles.append('open_h3_2k_ic')
        if has_p3:
            profiles.append('rife_2x')
        profile = profiles[-1] if profiles else ('open_h3_2k_ic' if standalone_p2 else 'rife_2x')
        return profile, profiles, has_p2, has_p3

    @classmethod
    def initial(cls, record):
        profile, profiles, has_p2, _ = cls._profiles_for_record(record)
        standalone_p2 = record.service == 'regenerate_2k'
        stage = 'regenerate_2k' if standalone_p2 else 'generation'
        win_count = _window_count_from_record(record)
        stage_limit = (P2_WINDOW_TIMEOUT_SECONDS * win_count) if standalone_p2 else P1_STAGE_TIMEOUT_SECONDS
        chain_limit = P2_CHAIN_TIMEOUT_BY_WINDOWS[win_count] if has_p2 else LEGACY_P1_P3_CHAIN_TIMEOUT_SECONDS
        post = record.request.get('postprocess') or {}
        grid_mode = (post.get('regenerate_2k') or {}).get('grid_mode') or record.request.get('grid_mode')
        return {
            'status': 'queued',
            'stage': stage,
            'profile': profile,
            'profiles': profiles,
            'grid_mode': grid_mode,
            'source_artifact_id': record.request.get('source_artifact_id') if standalone_p2 else None,
            'regenerate_2k_artifact_id': None,
            'regenerate_2k_task_id': record.video_task_id if standalone_p2 else None,
            'final_artifact_id': None,
            'interpolation_task_id': None,
            'error': None,
            'submitted_at': record.created_at,
            'completed_at': None,
            'total_seconds': None,
            'timing': {
                'queue_seconds': 0.0,
                'generation_active_seconds': 0.0,
                'regenerate_2k_active_seconds': 0.0,
                'interpolation_active_seconds': 0.0,
                'chain_active_seconds': 0.0,
                'window_count': win_count,
                'active_limit_seconds': {
                    'stage': stage_limit,
                    'chain': chain_limit,
                },
            },
        }

    def save(self, record, **updates):
        value = {**(record.delivery or self.initial(record)), **updates}
        if value['status'] in ('succeeded', 'failed') and not value.get('completed_at'):
            now = datetime.now(UTC)
            value.update(
                completed_at=now.isoformat(),
                total_seconds=max(0, (now - datetime.fromisoformat(record.created_at)).total_seconds()),
            )
        return self.tasks.update(record, delivery=value)

    def _safe_task(self, task_id: str | None):
        if not task_id:
            return None
        try:
            return self.tasks.get(task_id)
        except Exception:
            return None

    def _compute_timing(
        self,
        parent,
        p2_record,
        p3_record,
        *,
        current_stage: str,
        now_epoch: float | None = None,
    ) -> tuple[dict, float, float]:
        now_epoch = time.time() if now_epoch is None else now_epoch
        _, _, has_p2, _ = self._profiles_for_record(parent)
        standalone_p2 = parent.service == 'regenerate_2k'

        win_count = _window_count_from_record(p2_record, parent)
        gen_active, gen_q = (0.0, 0.0) if standalone_p2 else _stage_active_and_queue_seconds(parent, now_epoch)
        regen_target = parent if standalone_p2 else p2_record
        p2_active, p2_q = _stage_active_and_queue_seconds(regen_target, now_epoch) if has_p2 else (0.0, 0.0)
        p3_active, p3_q = _stage_active_and_queue_seconds(p3_record, now_epoch)

        chain_active = round(gen_active + p2_active + p3_active, 3)
        queue_total = round(gen_q + p2_q + p3_q, 3)

        if current_stage == 'generation':
            stage_limit = P1_STAGE_TIMEOUT_SECONDS
            stage_active = gen_active
        elif current_stage == 'regenerate_2k':
            stage_limit = P2_WINDOW_TIMEOUT_SECONDS * win_count
            stage_active = p2_active
        elif current_stage == 'interpolation':
            stage_limit = P3_STAGE_TIMEOUT_SECONDS
            stage_active = p3_active
        else:
            stage_limit = P3_STAGE_TIMEOUT_SECONDS if p3_record else (P2_WINDOW_TIMEOUT_SECONDS * win_count if has_p2 else P1_STAGE_TIMEOUT_SECONDS)
            stage_active = 0.0

        chain_limit = P2_CHAIN_TIMEOUT_BY_WINDOWS[win_count] if has_p2 else LEGACY_P1_P3_CHAIN_TIMEOUT_SECONDS
        timing = {
            'queue_seconds': queue_total,
            'generation_active_seconds': gen_active,
            'regenerate_2k_active_seconds': p2_active,
            'interpolation_active_seconds': p3_active,
            'chain_active_seconds': chain_active,
            'window_count': win_count,
            'active_limit_seconds': {
                'stage': stage_limit,
                'chain': chain_limit,
            },
        }
        return timing, stage_active, chain_active

    def _check_timeout_or_settle(
        self,
        parent,
        stage_record,
        *,
        stage_name: str,
        p2_record=None,
        p3_record=None,
    ):
        timing, stage_active, chain_active = self._compute_timing(
            parent,
            p2_record,
            p3_record,
            current_stage=stage_name,
        )
        stage_limit = timing['active_limit_seconds']['stage']
        chain_limit = timing['active_limit_seconds']['chain']
        delivery_doc = parent.delivery or {}
        pending = delivery_doc.get('_pending_timeout_error')
        pending_task_id = delivery_doc.get('_pending_timeout_task_id')
        pending_stage = delivery_doc.get('_pending_timeout_stage')

        def _dispatch_stage_cancel(rec, err):
            if stage_name == 'generation':
                return self.executor.cancel_generation(rec, timeout_error=err)
            if stage_name == 'regenerate_2k':
                return self.executor.processing['regenerate_2k'].cancel(
                    rec.video_task_id,
                    timeout_error=err,
                )
            return self.executor.processing['interpolate'].cancel(
                rec.video_task_id,
                timeout_error=err,
            )

        if pending is not None and (
            pending_task_id == stage_record.video_task_id
            or (pending_task_id is None and (pending_stage is None or pending_stage == stage_name))
        ):
            pstate = stage_record.provider_state or {}
            if (
                pstate.get('worker_settled') is False
                and pstate.get('cancel_delivered') is not True
                and _cancel_backoff_ready(pstate, time.time())
            ):
                stage_record = _dispatch_stage_cancel(stage_record, pending)
                if stage_record.video_task_id == parent.video_task_id:
                    parent = stage_record
                timing, _, _ = self._compute_timing(
                    parent,
                    stage_record if stage_name == 'regenerate_2k' and parent.service != 'regenerate_2k' else p2_record,
                    stage_record if stage_name == 'interpolation' else p3_record,
                    current_stage=stage_name,
                )
            settled = (stage_record.provider_state or {}).get('worker_settled') is not False
            if not settled:
                return self.save(parent, status='running', stage=stage_name, timing=timing), True
            cleaned = {
                k: v
                for k, v in (parent.delivery or {}).items()
                if k not in {'_pending_timeout_error', '_pending_timeout_stage', '_pending_timeout_task_id'}
            }
            parent = self.tasks.update(parent, delivery=cleaned)
            return self.save(
                parent,
                status='failed',
                stage=stage_name,
                final_artifact_id=None,
                error=pending,
                timing=timing,
            ), True

        is_active = stage_record.status not in {'succeeded', 'failed', 'cancelled'} and (
            (stage_record.provider_state or {}).get('started_at') is not None
        )
        if not is_active:
            return timing, False

        stage_over = stage_active > stage_limit
        chain_over = chain_active > chain_limit
        if not stage_over and not chain_over:
            return timing, False

        if stage_over and (not chain_over or (stage_active - stage_limit) >= (chain_active - chain_limit)):
            code = f'stage_timeout_{stage_name}'
        else:
            code = 'chain_active_timeout'
        timeout_error = {
            'code': code,
            'message': MESSAGES[code],
            'retryable': False,
        }

        cancelled = _dispatch_stage_cancel(stage_record, timeout_error)
        if stage_record.video_task_id == parent.video_task_id:
            parent = cancelled

        settled = (cancelled.provider_state or {}).get('worker_settled') is not False
        timing, _, _ = self._compute_timing(
            parent,
            cancelled if stage_name == 'regenerate_2k' and parent.service != 'regenerate_2k' else p2_record,
            cancelled if stage_name == 'interpolation' else p3_record,
            current_stage=stage_name,
        )
        if not settled:
            return self.save(
                parent,
                status='running',
                stage=stage_name,
                timing=timing,
                _pending_timeout_error=timeout_error,
                _pending_timeout_stage=stage_name,
                _pending_timeout_task_id=cancelled.video_task_id,
            ), True
        return self.save(
            parent,
            status='failed',
            stage=stage_name,
            final_artifact_id=None,
            error=timeout_error,
            timing=timing,
        ), True

    def tick(self, task_id: str):
        # Keep workflow coordination separate from result locks: submitting a
        # child acquires the identity lock, so never hold a parent result lock.
        with self.tasks.task_lock(task_id, purpose='delivery', blocking=False) as acquired:
            if acquired:
                return self._tick(task_id)
            return self.tasks.get(task_id)

    def _tick(self, task_id: str):
        parent = self.tasks.get(task_id)
        post = parent.request.get('postprocess')
        if not post or (parent.delivery or {}).get('status') in ('succeeded', 'failed'):
            return parent
        try:
            authorize(self.executor.client, parent.project_id, parent.authority)
        except PermissionError:
            return self.save(
                parent,
                status='failed',
                error={'code': 'project_access_denied', 'message': 'Project access was revoked'},
            )

        standalone_p2 = parent.service == 'regenerate_2k'
        _, _, has_p2, has_p3 = self._profiles_for_record(parent)

        # --- Stage 1: Native Generation (P1) or Standalone Regenerate-2K (P2) ---
        parent = self.executor.status(task_id)
        first_stage = 'regenerate_2k' if standalone_p2 else 'generation'
        outcome, halted = self._check_timeout_or_settle(
            parent,
            parent,
            stage_name=first_stage,
        )
        if halted:
            return outcome
        timing = outcome

        if parent.status in ('failed', 'cancelled'):
            if (parent.provider_state or {}).get('worker_settled') is False:
                return self.save(parent, status='running', stage=first_stage, timing=timing)
            return self.save(
                parent,
                status='failed',
                stage=first_stage,
                final_artifact_id=None,
                error=parent.error,
                timing=timing,
            )
        if parent.status != 'succeeded':
            return self.save(parent, status=parent.status, stage=first_stage, timing=timing)

        # Result retrieval is idempotent and preserves the original Artifact.
        parent = self.executor.result(task_id)
        if standalone_p2:
            m1_id = parent.request.get('source_artifact_id')
            m2_id = parent.artifact_id
            p2_record = parent
        else:
            m1_id = parent.artifact_id
            m2_id = (parent.delivery or {}).get('regenerate_2k_artifact_id')
            p2_record = None

        # --- Stage 2: P2 Open-H3-Regenerate-2K-IC (when parent is P1 and postprocess has regenerate_2k) ---
        if has_p2 and not standalone_p2:
            regen_adapter = self.executor.processing['regenerate_2k']
            grid_mode = post['regenerate_2k']['grid_mode']
            parent = self.save(
                parent,
                status='running',
                stage='regenerate_2k',
                source_artifact_id=m1_id,
                grid_mode=grid_mode,
            )
            p2_id = parent.delivery.get('regenerate_2k_task_id')
            if not p2_id:
                p2_child = regen_adapter.generate(
                    project_id=parent.project_id,
                    idempotency_key=f'delivery:{task_id}:open_h3_2k_ic:{grid_mode}',
                    source_artifact_id=m1_id,
                    profile='open_h3_2k_ic',
                    grid_mode=grid_mode,
                    prompt=parent.request.get('prompt', ''),
                    seed=7 if parent.request.get('seed') is None else int(parent.request['seed']),
                )
                p2_id = p2_child.video_task_id
                parent = self.save(parent, regenerate_2k_task_id=p2_id)
            p2_record = regen_adapter.status(p2_id)
            outcome, halted = self._check_timeout_or_settle(
                parent,
                p2_record,
                stage_name='regenerate_2k',
                p2_record=p2_record,
            )
            if halted:
                return outcome
            timing = outcome

            if p2_record.status in ('failed', 'cancelled'):
                if (p2_record.provider_state or {}).get('worker_settled') is False:
                    return self.save(parent, status='running', stage='regenerate_2k', timing=timing)
                return self.save(
                    parent,
                    status='failed',
                    stage='regenerate_2k',
                    final_artifact_id=None,
                    error=p2_record.error,
                    timing=timing,
                )
            if p2_record.status != 'succeeded':
                return self.save(parent, status='running', stage='regenerate_2k', timing=timing)

            p2_record = regen_adapter.result(p2_id)
            m2_id = p2_record.artifact_id
            timing, _, _ = self._compute_timing(parent, p2_record, None, current_stage='completed' if not has_p3 else 'interpolation')
            if not has_p3:
                p2_public = regen_adapter.public_result(p2_record)
                return self.save(
                    parent,
                    status='succeeded',
                    stage='completed',
                    source_artifact_id=m1_id,
                    regenerate_2k_artifact_id=m2_id,
                    final_artifact_id=m2_id,
                    preview_artifact_id=p2_record.media['preview_artifact_id'],
                    result=p2_public,
                    error=None,
                    timing=timing,
                )
            parent = self.save(
                parent,
                status='running',
                stage='interpolation',
                source_artifact_id=m1_id,
                regenerate_2k_artifact_id=m2_id,
                timing=timing,
            )

        # --- Stage 3: P3 RIFE 2x Interpolation ---
        if has_p3:
            interp_source_id = m2_id if has_p2 else m1_id
            interp_key = f'delivery:{task_id}:rife_2x:2k' if has_p2 else f'delivery:{task_id}:rife_2x'
            parent = self.save(
                parent,
                status='running',
                stage='interpolation',
                source_artifact_id=m1_id,
                **({'regenerate_2k_artifact_id': m2_id} if has_p2 else {}),
            )
            adapter = self.executor.processing['interpolate']
            child_id = parent.delivery.get('interpolation_task_id')
            if not child_id:
                # Atomic process lock + stable runtime idempotency recover the gap
                # between a child POST response and persisting the parent link.
                child = adapter.generate(
                    project_id=parent.project_id,
                    idempotency_key=interp_key,
                    source_artifact_id=interp_source_id,
                    backend='rife',
                    target_fps_num=48,
                    target_fps_den=1,
                )
                child_id = child.video_task_id
                parent = self.save(parent, interpolation_task_id=child_id)
            child = adapter.status(child_id)
            outcome, halted = self._check_timeout_or_settle(
                parent,
                child,
                stage_name='interpolation',
                p2_record=p2_record,
                p3_record=child,
            )
            if halted:
                return outcome
            timing = outcome

            if child.status in ('failed', 'cancelled'):
                if (child.provider_state or {}).get('worker_settled') is False:
                    return self.save(parent, status='running', stage='interpolation', timing=timing)
                return self.save(
                    parent,
                    status='failed',
                    stage='interpolation',
                    final_artifact_id=None,
                    error=child.error,
                    timing=timing,
                )
            if child.status == 'succeeded':
                child = adapter.result(child_id)
                result = adapter.public_result(child)
                timing, _, _ = self._compute_timing(parent, p2_record, child, current_stage='completed')
                return self.save(
                    parent,
                    status='succeeded',
                    stage='completed',
                    final_artifact_id=child.artifact_id,
                    preview_artifact_id=child.media['preview_artifact_id'],
                    result=result,
                    error=None,
                    timing=timing,
                )
        return parent

    def sync(self):
        for path in self.tasks.root.glob('video_task_*.json'):
            try:
                record = self.tasks.get(path.stem)
                if record.request.get('postprocess') and (record.delivery or {}).get('status') not in ('succeeded', 'failed'):
                    self.tick(record.video_task_id)
            except (ExecutionError, OSError, ValueError):
                # Preserve durable intent on unavailable authorization/runtime or
                # Artifact registration; never create a replacement generation.
                continue
        self.executor.dispatch_queued_local_tasks()
