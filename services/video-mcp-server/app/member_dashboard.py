"""Higgsfield member pages: project-authorized operations, no management proxy."""
from datetime import UTC, datetime
from contextlib import contextmanager
import hashlib
import json
import os
import sqlite3
import threading

import httpx
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

from . import titles
from .artifacts import ArtifactError, ArtifactNotFound
from .dashboard import Submission, public_task
from .tasks import TaskConflict, TaskNotFound


class MemberError(Exception):
    def __init__(self, code, status=409):
        self.code, self.status = code, status


class MemberDashboard:
    def __init__(self, dashboard):
        self.dashboard = dashboard
        self.executor = dashboard.executor
        self.adapter = self.executor.higgsfield
        self.lock = threading.RLock()

    @contextmanager
    def quotes(self):
        root = self.executor.tasks.root
        root.mkdir(parents=True, exist_ok=True)
        path = root / 'higgsfield-web-quotes.sqlite3'
        db = sqlite3.connect(path, timeout=30)
        os.chmod(path, 0o600)
        try:
            db.execute('CREATE TABLE IF NOT EXISTS quotes (identity TEXT PRIMARY KEY, value TEXT NOT NULL)')
            with db:
                yield db
        finally:
            db.close()

    def identity(self, project, attempt):
        actor = self.adapter.authorize(project)
        if not isinstance(attempt, str) or not 1 <= len(attempt) <= 128:
            raise ValueError('invalid_attempt')
        return hashlib.sha256(json.dumps([actor, project, attempt], sort_keys=True).encode()).hexdigest()

    def record(self, task):
        record = self.executor.tasks.get(task)
        if record.service != 'higgsfield':
            raise TaskNotFound('not_found')
        self.adapter.authorize_record(record)
        return record

    def normalize(self, body):
        inputs = Submission.model_validate(body)
        if inputs.route != 'higgsfield' or not inputs.model:
            raise ValueError('higgsfield_model_required')
        # This rejected supplier combination remains unavailable in the webpage.
        if inputs.model == 'grok_video_v15_lite' and inputs.references.images:
            raise MemberError('image_to_video_not_verified')
        values = inputs.model_dump(exclude={'title', 'idempotency_key', 'route'})
        request, digest = self.adapter.normalize(**values)
        return inputs, request, digest

    def preflight(self, body):
        # Authorize before resolving any referenced content.
        identity = self.identity(body.get('project_id'), body.get('idempotency_key'))
        inputs, request, digest = self.normalize(body)
        if not self.adapter.enabled or not self.adapter.configured():
            raise MemberError('higgsfield_not_configured')
        quote = self.adapter.quote(request)
        if type(quote.get('can_submit')) is not bool or not quote.get('expires_at'):
            raise MemberError('quote_admission_unavailable', 503)
        value = {'input_digest': digest, 'quoted_credits': quote['credits'],
                 **{k:quote[k] for k in ('can_submit','blocked_reason','observed_at','expires_at')}}
        with self.lock, self.quotes() as db:
            db.execute('INSERT OR REPLACE INTO quotes VALUES (?,?)',
                       (identity, json.dumps({'quote': value})))
        return value

    def create(self, body):
        identity = self.identity(body.get('project_id'), body.get('idempotency_key'))
        inputs, request, digest = self.normalize(body)
        with self.lock, self.executor.tasks.exclusive():
            old = self.executor.tasks.find_idempotency(inputs.project_id, inputs.idempotency_key)
            if old:
                self.record(old.video_task_id)
                if old.input_digest != digest:
                    raise TaskConflict('idempotency_conflict')
                return public_task(old, self.executor.tasks)
            with self.quotes() as db:
                row = db.execute('SELECT value FROM quotes WHERE identity=?', (identity,)).fetchone()
            if not row:
                raise MemberError('quote_required')
            saved = json.loads(row[0])['quote']
            if saved['input_digest'] != digest:
                raise MemberError('quote_changed')
            if datetime.fromisoformat(saved['expires_at']) <= datetime.now(UTC):
                raise MemberError('quote_expired')
            if not saved['can_submit']:
                raise MemberError(saved['blocked_reason'] or 'quote_blocked')
            current = self.adapter.quote(request)
            if not current.get('can_submit'):
                raise MemberError(current.get('blocked_reason') or 'quote_blocked')
            if current['credits'] != saved['quoted_credits']:
                raise MemberError('quote_changed')
            title = titles.validate_title(inputs.title) if inputs.title else None
            marker = titles.PROVIDED_TITLE.set(title)
            try:
                record = self.adapter.generate(idempotency_key=inputs.idempotency_key,
                    price_ceiling=current['credits'], **inputs.model_dump(exclude={'title','idempotency_key','route'}))
            finally:
                titles.PROVIDED_TITLE.reset(marker)
            return public_task(record, self.executor.tasks)

    def dispatch(self, action, request, body):
        self.adapter.authorize_member()
        query = request.query_params
        if action == 'capabilities':
            values = self.adapter.capabilities()
            for value in values:
                if value['model'] == 'grok_video_v15_lite':
                    value['modes'] = ['text_to_video']
                    value['unavailable_modes'] = {'image_to_video':'media_contract_mismatch'}
            return {'routes':values}
        if action == 'projects':
            value = self.adapter.project_request('list', {'require_write':True, 'limit':50,
                                                          'cursor':query.get('cursor','')})
            return {'items':[{k:p[k] for k in ('project_id','name')} for p in value['items']],
                    'next_cursor':value.get('next_cursor','')}
        if action == 'assets':
            project = query.get('project_id')
            self.adapter.authorize(project)
            items = []
            for path in self.executor.artifacts.artifacts_root.glob('art_*/metadata.json'):
                try:
                    item = self.executor.artifacts.get(path.parent.name, project)
                    if item.media_type.startswith(('image/', 'video/')):
                        items.append(item.model_dump())
                except (ArtifactError, ValueError):
                    continue
            return {'items': sorted(items, key=lambda a:a['created_at'], reverse=True)}
        if action.startswith('assets/') and action.endswith('/content'):
            project = query.get('project_id')
            self.adapter.authorize(project)
            item = self.executor.artifacts.get(action.split('/')[1], project)
            if not item.media_type.startswith(('image/', 'video/')):
                raise MemberError('not_found', 404)
            return FileResponse(self.executor.artifacts.content_path(item), media_type=item.media_type,
                                headers={'Cache-Control':'private, no-store'})
        if action.startswith('imports/'):
            self.adapter.authorize(body.get('project_id'))
            method = action.split('/')[1]
            allowed = {'prepare','chunk','status','commit'}
            if method not in allowed:
                raise ValueError('invalid_import_operation')
            if method == 'prepare' and os.path.splitext(body.get('filename',''))[1].lower() not in {'.jpg','.jpeg','.png','.webp','.mp4'}:
                raise ValueError('unsupported_reference_media')
            if method == 'prepare' and (type(body.get('size')) is not int or not 0 < body['size'] <= 128*1024*1024):
                raise ValueError('unsupported_reference_size')
            return getattr(self.executor.imports, method)(**body)
        if action == 'preflight':
            return self.preflight(body)
        if action == 'attempt':
            self.identity(body.get('project_id'), body.get('idempotency_key'))
            record = self.executor.tasks.find_idempotency(body['project_id'], body['idempotency_key'])
            if not record:
                return {'state':'not_found'}
            self.record(record.video_task_id)
            return {'state':'found', 'task':public_task(record, self.executor.tasks)}
        if action == 'tasks' and request.method == 'POST':
            return self.create(body)
        if action == 'tasks':
            items = []
            for record in self.dashboard.records():
                if record.service != 'higgsfield':
                    continue
                try:
                    self.adapter.authorize_record(record)
                except PermissionError:
                    continue
                items.append(public_task(record, self.executor.tasks))
            return {'tasks':items}
        parts = action.split('/')
        if len(parts) in {2,3} and parts[0] == 'tasks':
            record = self.record(parts[1])
            if len(parts) == 2:
                value = self.dashboard.detail(record.video_task_id)
                value['billing'] = {'quoted_credits':(record.provider_state or {}).get('quoted_credits'),
                                    'provider_charge':None, 'reconciliation_status':'unreconciled'}
                return value
            if parts[2] == 'content':
                return FileResponse(self.adapter.content_path(record), media_type='video/mp4',
                    filename=record.video_task_id+'.mp4', headers={'Cache-Control':'private, no-store'})
            if parts[2] == 'thumbnail':
                return self.dashboard.thumbnail(record.video_task_id)
            if parts[2] == 'result':
                # A pending result is a normal state, not an HTTP failure.
                self.adapter.status(record)
                return self.dashboard.detail(record.video_task_id)
        raise MemberError('not_found', 404)

    async def endpoint(self, request):
        try:
            raw = bytearray()
            if request.method == 'POST':
                async for chunk in request.stream():
                    raw.extend(chunk)
                    if len(raw) > 1024*1024:
                        raise MemberError('request_too_large', 413)
            body = json.loads(raw) if raw else {}
            if not isinstance(body, dict):
                raise ValueError('invalid_request')
            value = await run_in_threadpool(self.dispatch, request.url.path.split('/api/higgsfield/')[1], request, body)
            return value if isinstance(value, Response) else JSONResponse(value, headers={'Cache-Control':'no-store'})
        except MemberError as error:
            code, status = error.code, error.status
        except PermissionError:
            code, status = 'permission_denied', 403
        except (TaskNotFound, ArtifactNotFound):
            code, status = 'not_found', 404
        except TaskConflict:
            code, status = 'idempotency_conflict', 409
        except (ValidationError, ValueError, TypeError, KeyError):
            code, status = 'invalid_request', 400
        except httpx.HTTPStatusError as error:
            code, status = 'channel_unavailable', 503
            try:
                candidate = error.response.json().get('error')
                if candidate in {'authorization_required','account_budget_exceeded','account_concurrency_exceeded',
                                 'account_reconciliation_required','insufficient_credits','cost_not_allowed','quote_changed'}:
                    code = candidate
            except (ValueError, AttributeError):
                pass
        except (httpx.HTTPError, OSError, ArtifactError):
            code, status = 'channel_unavailable', 503
        return JSONResponse({'error':code}, status_code=status, headers={'Cache-Control':'no-store'})

    def routes(self):
        get = ('projects','capabilities','assets','assets/{asset_id:str}/content','tasks','tasks/{task_id:str}',
               'tasks/{task_id:str}/content','tasks/{task_id:str}/thumbnail')
        post = ('preflight','tasks','attempt','imports/prepare','imports/chunk','imports/status','imports/commit',
                'tasks/{task_id:str}/result')
        return [Route('/api/higgsfield/'+path, self.endpoint, methods=[method])
                for method, paths in [('GET',get),('POST',post)] for path in paths]
