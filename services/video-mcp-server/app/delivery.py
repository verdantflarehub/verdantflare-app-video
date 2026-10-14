"""Durable, opt-in Singularity -> RIFE delivery; no client polling dependency."""
from contextvars import ContextVar
from datetime import UTC, datetime
import hashlib
import json
import os
import uuid

import httpx

from .executor import ExecutionError, serialized

CALLER = ContextVar('video_verified_caller', default=None)
OPTION = {'interpolate': {'profile': 'rife_2x'}}


def normalize(value, route):
    if value is None:
        return None
    if value != OPTION or route != 'h3-singularity':
        raise ValueError('postprocess requires h3-singularity and interpolate.profile=rife_2x')
    return {'interpolate': {'profile': 'rife_2x'}}


def authorize(client, project_id, authority=None):
    actor = authority if authority is not None else CALLER.get()
    if not actor or not all(actor.get(k) for k in ('user_id', 'organization_id')):
        raise ValueError('verified Studio identity required')
    url = os.environ.get('VIDEO_PROJECT_SERVICE_URL', '').rstrip('/')
    token = os.environ.get('VIDEO_PROJECT_SERVICE_TOKEN', '')
    if not url or not token:
        raise ExecutionError('project authorization is not configured')
    try:
        response = client.post(url + '/internal/v1/project/open',
            headers={'Authorization': 'Bearer ' + token, 'X-User-Id': actor['user_id'],
                     'X-Organization-Id': actor['organization_id'], 'X-Request-Id': str(uuid.uuid4())},
            json={'project_id': project_id, 'require_write': True}, timeout=15)
        if response.status_code in (401, 403, 404):
            raise PermissionError('project access denied')
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ExecutionError('project authorization temporarily unavailable') from exc
    return dict(actor)


class Delivery:
    def __init__(self, executor):
        self.executor = executor
        self.tasks = executor.tasks
        self._lock = executor._lock

    def capability(self):
        adapter = self.executor.processing['interpolate']
        result = {'profile': 'rife_2x', 'backend': 'rife', 'factor': 2,
                  'generation_routes': ['h3-singularity'], 'default_enabled': False,
                  'configured': bool(os.environ.get('VIDEO_PROJECT_SERVICE_URL') and
                                     os.environ.get('VIDEO_PROJECT_SERVICE_TOKEN')),
                  'readiness': 'unreachable', 'runtime_version': adapter.runtime_version}
        if not result['configured']:
            result['readiness'] = 'not_configured'
            return result
        try:
            response = adapter.client.get(adapter.url + '/health', timeout=5)
            data = response.json()
            result['readiness'] = 'ready' if response.status_code == 200 and data.get('status') == 'ok' and data.get('service') == 'interpolate' else 'not_ready'
        except (httpx.HTTPError, ValueError, AttributeError):
            pass
        return result

    def prepare(self, request, digest, option, check=True):
        normalized = normalize(option, request['route'])
        if normalized is None:
            return request, digest
        if check:
            cap = self.capability()
            if not cap['configured'] or cap['readiness'] != 'ready':
                raise ExecutionError('interpolation backend is not ready')
        request = {**request, 'postprocess': normalized}
        canonical = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        return request, 'sha256:' + hashlib.sha256(canonical.encode()).hexdigest()

    @staticmethod
    def initial(record):
        return {'status': 'queued', 'stage': 'generation', 'profile': 'rife_2x',
                'source_artifact_id': None, 'final_artifact_id': None,
                'interpolation_task_id': None, 'error': None,
                'submitted_at': record.created_at, 'completed_at': None, 'total_seconds': None}

    def save(self, record, **updates):
        value = {**(record.delivery or self.initial(record)), **updates}
        if value['status'] in ('succeeded', 'failed') and not value.get('completed_at'):
            now = datetime.now(UTC)
            value.update(completed_at=now.isoformat(),
                         total_seconds=max(0, (now-datetime.fromisoformat(record.created_at)).total_seconds()))
        return self.tasks.update(record, delivery=value)

    @serialized
    def tick(self, task_id):
        parent = self.tasks.get(task_id)
        if not parent.request.get('postprocess') or (parent.delivery or {}).get('status') in ('succeeded', 'failed'):
            return parent
        try:
            authorize(self.executor.client, parent.project_id, parent.authority)
        except PermissionError:
            return self.save(parent, status='failed', error={'code': 'project_access_denied', 'message': 'Project access was revoked'})
        parent = self.executor.status(task_id)
        if parent.status in ('failed', 'cancelled'):
            return self.save(parent, status='failed', stage='generation', error=parent.error)
        if parent.status != 'succeeded':
            return self.save(parent, status=parent.status, stage='generation')
        # Result retrieval is idempotent and preserves the original Artifact.
        parent = self.executor.result(task_id)
        parent = self.save(parent, status='running', stage='interpolation', source_artifact_id=parent.artifact_id)
        adapter = self.executor.processing['interpolate']
        child_id = parent.delivery.get('interpolation_task_id')
        if not child_id:
            # Atomic process lock + stable runtime idempotency recover the gap
            # between a child POST response and persisting the parent link.
            child = adapter.generate(project_id=parent.project_id,
                idempotency_key='delivery:' + task_id + ':rife_2x',
                source_artifact_id=parent.artifact_id, backend='rife', target_fps_num=48, target_fps_den=1)
            child_id = child.video_task_id
            parent = self.save(parent, interpolation_task_id=child_id)
        child = adapter.status(child_id)
        if child.status == 'failed':
            return self.save(parent, status='failed', error=child.error)
        if child.status == 'succeeded':
            child = adapter.result(child_id)
            result = adapter.public_result(child)
            return self.save(parent, status='succeeded', stage='completed', final_artifact_id=child.artifact_id,
                preview_artifact_id=child.media['preview_artifact_id'], result=result, error=None)
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
