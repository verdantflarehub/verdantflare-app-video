"""Single-owner Huobao v2 adapter. All business operations go through Studio."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from urllib.parse import urlsplit
import uuid

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

MAX_BODY = 48 * 1024 * 1024
MAX_MEDIA = 32 * 1024 * 1024
MAX_TOTAL = 96 * 1024 * 1024
FORMATS = {'image/png': '.png', 'image/jpeg': '.jpg', 'image/webp': '.webp',
           'video/mp4': '.mp4', 'audio/wav': '.wav', 'audio/mpeg': '.mp3'}
KINDS = {'image_url': 'images', 'video_url': 'videos', 'audio_url': 'audios'}
ID = re.compile(r'hb_[0-9a-f]{32}\Z')
ARTIFACT = re.compile(r'art_[0-9a-f]{32}\Z')


@dataclass(frozen=True)
class Settings:
    studio_origin: str
    studio_token: str
    api_key: str
    signing_key: str
    project_id: str
    owner_user_id: str
    owner_org_id: str
    data_dir: Path
    public_origin: str = 'http://huobao-h3-bridge:8000'
    media_origin: str = 'https://huobao-trial.verdantflarehub.com'
    media_internal_origin: str = 'http://huobao-drama:5679'

    def validate(self):
        for value in (self.studio_origin, self.public_origin, self.media_origin, self.media_internal_origin):
            p = urlsplit(value)
            if p.scheme not in {'https', 'http'} or not p.hostname or p.username or p.password or p.path or p.query or p.fragment:
                raise ValueError('origins must have no path, credentials or query')
        if urlsplit(self.studio_origin).scheme != 'https':
            raise ValueError('Studio requires HTTPS')
        for value in (self.studio_token, self.api_key, self.signing_key):
            if len(value) < 32:
                raise ValueError('dedicated credentials of at least 32 characters required')
        for value in (self.project_id, self.owner_user_id, self.owner_org_id):
            if uuid.UUID(value).version != 7:
                raise ValueError('real UUIDv7 identity and project required')


class UpstreamError(Exception):
    def __init__(self, code='studio_unavailable', retry=True):
        self.code, self.retry = code, retry


class Studio:
    def __init__(self, settings, client):
        self.settings, self.client = settings, client
        self.headers = {'Authorization': 'Bearer ' + settings.studio_token,
                        'Accept': 'application/json, text/event-stream'}

    async def call(self, name, args):
        try:
            r = await self.client.post(self.settings.studio_origin + '/mcp', headers=self.headers,
                json={'jsonrpc': '2.0', 'id': uuid.uuid4().hex, 'method': 'tools/call',
                      'params': {'name': name, 'arguments': args}})
            if r.status_code != 200:
                raise UpstreamError('studio_access_denied' if r.status_code in {401, 403} else 'studio_unavailable',
                                    r.status_code not in {401, 403})
            if r.headers.get('content-type', '').startswith('text/event-stream'):
                values = [json.loads(line[5:].strip()) for line in r.text.splitlines() if line.startswith('data:')]
                result = next(v for v in reversed(values) if 'result' in v or 'error' in v)
            else:
                result = r.json()
            if result.get('error'):
                raise UpstreamError('studio_tool_rejected', False)
            tool = result['result']
            if tool.get('isError'):
                # Do not expose raw upstream errors: they can contain source URLs or identities.
                raise UpstreamError('studio_tool_rejected', False)
            data = tool.get('structuredContent')
            if data is None:
                data = json.loads(next(c['text'] for c in tool.get('content', []) if c.get('type') == 'text'))
            if not isinstance(data, dict) or data.get('error'):
                raise UpstreamError('studio_tool_rejected', False)
            return data
        except (httpx.RequestError, ValueError, KeyError, StopIteration, TypeError):
            raise UpstreamError() from None

    async def authorize(self):
        return await self.call('project.open', {'project_id': self.settings.project_id, 'require_write': True})


def normalize(body):
    if not isinstance(body, dict) or set(body) - {'model', 'content', 'duration', 'resolution', 'ratio'}:
        raise HTTPException(400, 'unsupported_fields')
    if body.get('model') != 'MiniMax-H3':
        raise HTTPException(400, 'model_must_be_MiniMax-H3')
    duration = body.get('duration', 5)
    if type(duration) is not int or not 4 <= duration <= 15:
        raise HTTPException(400, 'duration_must_be_4_to_15')
    if body.get('resolution', '768P') != '768P':
        raise HTTPException(400, 'only_768P_supported')
    ratio = body.get('ratio', 'adaptive')
    if ratio not in {'21:9', '16:9', '4:3', '1:1', '3:4', '9:16', 'adaptive'}:
        raise HTTPException(400, 'unsupported_ratio')
    items = body.get('content')
    if not isinstance(items, list) or not 2 <= len(items) <= 13:
        raise HTTPException(400, 'prompt_and_1_to_12_references_required')
    prompt, refs = [], []
    for item in items:
        if not isinstance(item, dict):
            raise HTTPException(400, 'invalid_content')
        kind = item.get('type')
        if kind == 'text' and set(item) == {'type', 'text'} and isinstance(item['text'], str):
            prompt.append(item['text'])
        elif kind in KINDS and set(item) <= {'type', kind, 'role'}:
            expected = {'image_url': 'reference_image', 'video_url': 'reference_video', 'audio_url': 'reference_audio'}[kind]
            if item.get('role', expected) != expected:
                raise HTTPException(400, 'first_last_frame_not_supported_use_reference_image')
            nested = item.get(kind)
            if not isinstance(nested, dict) or set(nested) != {'url'} or not isinstance(nested['url'], str):
                raise HTTPException(400, 'invalid_reference')
            refs.append({'kind': KINDS[kind], 'source': nested['url'], 'purpose': expected})
        else:
            raise HTTPException(400, 'unsupported_content')
    text = '\n'.join(prompt).strip()
    if not text or len(text) > 7000 or not refs:
        raise HTTPException(400, 'prompt_and_reference_required')
    for kind, limit in [('images', 9), ('videos', 3), ('audios', 3)]:
        if sum(r['kind'] == kind for r in refs) > limit:
            raise HTTPException(400, 'reference_count_exceeded')
    if len(refs) > 12 or ratio == 'adaptive' and all(r['kind'] == 'audios' for r in refs):
        raise HTTPException(400, 'audio_only_requires_explicit_ratio')
    return {'prompt': text, 'duration_seconds': duration, 'aspect_ratio': ratio, 'media': refs}


def source_info(source, kind, settings):
    """Return inline bytes or an exact internal URL; reject redirects and encoded paths."""
    if source.startswith('data:'):
        header, sep, encoded = source.partition(',')
        mime = header[5:].removesuffix(';base64')
        if not sep or header != 'data:' + mime + ';base64' or mime not in FORMATS:
            raise HTTPException(400, 'unsupported_data_url')
        if not mime.startswith({'images': 'image/', 'videos': 'video/', 'audios': 'audio/'}[kind]):
            raise HTTPException(400, 'reference_type_mismatch')
        try:
            content = base64.b64decode(encoded, validate=True)
        except ValueError:
            raise HTTPException(400, 'invalid_base64') from None
        if not 0 < len(content) <= MAX_MEDIA:
            raise HTTPException(413, 'media_too_large_or_empty')
        return FORMATS[mime], content, None
    parsed = urlsplit(source)
    origin = f'{parsed.scheme}://{parsed.netloc}'
    if origin not in {settings.media_origin, settings.media_internal_origin} or parsed.query or parsed.fragment:
        raise HTTPException(400, 'only_huobao_static_media_allowed')
    if not re.fullmatch(r'/static/(?:uploads|images|videos|audio|audios)/[A-Za-z0-9_-]+\.(?:png|jpg|jpeg|webp|mp4|wav|mp3)', parsed.path):
        raise HTTPException(400, 'unsafe_media_path')
    suffix = Path(parsed.path).suffix
    allowed = {'images': {'.png', '.jpg', '.jpeg', '.webp'}, 'videos': {'.mp4'}, 'audios': {'.wav', '.mp3'}}
    if suffix not in allowed[kind]:
        raise HTTPException(400, 'reference_type_mismatch')
    return suffix, None, settings.media_internal_origin + parsed.path


class Ledger:
    def __init__(self, settings):
        self.root = settings.data_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / 'jobs.sqlite', check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS owner (binding TEXT NOT NULL)')
        binding = json.dumps([settings.owner_user_id, settings.owner_org_id, settings.project_id])
        old = self.db.execute('SELECT binding FROM owner').fetchone()
        if old and old[0] != binding:
            raise ValueError('ledger_identity_mismatch')
        if not old:
            self.db.execute('INSERT INTO owner VALUES (?)', (binding,))
        self.db.execute('''CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY, idem TEXT UNIQUE, digest TEXT, created REAL,
            status TEXT, payload TEXT, refs TEXT, native TEXT, error TEXT)''')
        self.db.commit()

    def get(self, task):
        if not ID.fullmatch(task):
            raise HTTPException(404, 'task_not_found')
        row = self.db.execute('SELECT * FROM jobs WHERE id=?', (task,)).fetchone()
        if row is None:
            raise HTTPException(404, 'task_not_found')
        return dict(row)

    def update(self, task, **values):
        if not set(values) <= {'status', 'payload', 'refs', 'native', 'error'}:
            raise ValueError('invalid ledger field')
        self.db.execute('UPDATE jobs SET ' + ','.join(k + '=?' for k in values) + ' WHERE id=?', (*values.values(), task))
        self.db.commit()

    def create(self, payload, key):
        encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        if key:
            if not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', key):
                raise HTTPException(400, 'invalid_idempotency_key')
            old = self.db.execute('SELECT * FROM jobs WHERE idem=?', (key,)).fetchone()
        else:
            old = self.db.execute('SELECT * FROM jobs WHERE digest=? AND created>? ORDER BY created DESC LIMIT 1',
                                  (digest, time.time() - 86400)).fetchone()
        if old:
            if old['digest'] != digest:
                raise HTTPException(409, 'idempotency_conflict')
            return old['id']
        if self.db.execute("SELECT count(*) FROM jobs WHERE status NOT IN ('succeeded','failed')").fetchone()[0] >= 4:
            raise HTTPException(429, 'queue_full')
        if shutil.disk_usage(self.root).free < 256 * 1024 * 1024:
            raise HTTPException(507, 'insufficient_storage')
        task = 'hb_' + uuid.uuid4().hex
        self.db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)',
                        (task, key, digest, time.time(), 'queued', encoded, None, None, None))
        self.db.commit()
        return task


def create_app(settings: Settings, client=None):
    settings.validate()
    backend = client or httpx.AsyncClient(timeout=httpx.Timeout(120, connect=10), follow_redirects=False)
    studio = Studio(settings, backend)
    ledger = Ledger(settings)
    workers = {}
    gate = asyncio.Semaphore(1)

    async def import_media(task, payload):
        references = {'images': [], 'videos': [], 'audios': []}
        total = 0
        for index, media in enumerate(payload['media']):
            suffix, inline, url = source_info(media['source'], media['kind'], settings)
            path = ledger.root / (task + '-' + str(index) + suffix)
            if not path.exists():
                if inline is None:
                    content = bytearray()
                    async with backend.stream('GET', url) as response:
                        if response.status_code != 200:
                            raise HTTPException(400, 'huobao_media_unavailable')
                        async for chunk in response.aiter_bytes():
                            content.extend(chunk)
                            if len(content) > MAX_MEDIA:
                                raise HTTPException(413, 'media_too_large')
                    inline = bytes(content)
                if not inline:
                    raise HTTPException(400, 'empty_media')
                temp = path.with_suffix('.part')
                with temp.open('wb') as output:
                    output.write(inline)
                    output.flush()
                    os.fsync(output.fileno())
                temp.replace(path)
            size = path.stat().st_size
            total += size
            if total > MAX_TOTAL:
                raise HTTPException(413, 'total_media_too_large')
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            state = await studio.call('video.import_prepare', {
                'project_id': settings.project_id, 'idempotency_key': f'{task}:input:{index}',
                'filename': f'reference-{index}{suffix}', 'size': size, 'sha256': sha, 'purpose': media['purpose']})
            args = {'project_id': settings.project_id, 'import_id': state['import_id']}
            with path.open('rb') as source:
                offset = state['offset']
                source.seek(offset)
                while offset < size:
                    chunk = source.read(min(512 * 1024, state.get('chunk_max_bytes', 512 * 1024)))
                    result = await studio.call('video.import_chunk', {**args, 'offset': offset,
                        'content_base64': base64.b64encode(chunk).decode(), 'sha256': hashlib.sha256(chunk).hexdigest()})
                    if result.get('offset') != offset + len(chunk):
                        raise UpstreamError('invalid_import_offset', False)
                    offset += len(chunk)
            result = await studio.call('video.import_commit', args)
            references[media['kind']].append({'artifact_id': result['artifact']['artifact_id'], 'purpose': media['purpose']})
        return references

    async def run(task):
        try:
            async with gate:
                for attempt in range(8):
                    try:
                        row = ledger.get(task)
                        if row['native'] or row['status'] in {'succeeded', 'failed'}:
                            return
                        await studio.authorize()
                        payload = json.loads(row['payload'])
                        refs = json.loads(row['refs']) if row['refs'] else await import_media(task, payload)
                        payload.pop('media', None)
                        ledger.update(task, refs=json.dumps(refs), payload=json.dumps(payload), status='running', error=None)
                        for path in ledger.root.glob(task + '-*'):
                            path.unlink()
                        args = {**payload, 'references': refs, 'project_id': settings.project_id,
                                'model': 'minimax-h3-ref2va', 'route': 'h3-singularity', 'quality_profile': 'du-0'}
                        await studio.call('video.preflight', args)
                        native = await studio.call('video.create', {**args, 'idempotency_key': task})
                        ledger.update(task, native=native['task_id'], status='running', error=None)
                        return
                    except HTTPException as exc:
                        ledger.update(task, status='failed', error=str(exc.detail))
                        return
                    except UpstreamError as exc:
                        # A lost create response is recovered with the SAME key, never a new generation.
                        ledger.update(task, error=exc.code, status='recovering')
                        if not exc.retry:
                            ledger.update(task, status='failed')
                            return
                    except (httpx.RequestError, OSError, KeyError, ValueError):
                        ledger.update(task, error='bridge_recovery_required', status='recovering')
                    await asyncio.sleep(min(2 ** attempt, 60))
        finally:
            workers.pop(task, None)

    def schedule(task):
        if task not in workers:
            workers[task] = asyncio.create_task(run(task))

    @asynccontextmanager
    async def lifespan(_app):
        for row in ledger.db.execute("SELECT id FROM jobs WHERE native IS NULL AND status NOT IN ('succeeded','failed')"):
            schedule(row['id'])
        yield
        tasks = list(workers.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        ledger.db.close()
        if client is None:
            await backend.aclose()

    app = FastAPI(title='Huobao H3 bridge', version='0.5.0', lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.ledger = ledger

    @app.middleware('http')
    async def auth(request, call_next):
        if request.url.path not in {'/live', '/health'} and not request.url.path.startswith('/v2/content/'):
            if not hmac.compare_digest(request.headers.get('authorization', '').encode(), ('Bearer ' + settings.api_key).encode()):
                return JSONResponse({'error': {'message': 'unauthorized'}}, status_code=401)
        result = await call_next(request)
        result.headers['Cache-Control'] = 'no-store'
        return result

    @app.exception_handler(UpstreamError)
    async def upstream_error(_request, exc):
        return JSONResponse({'error': {'message': exc.code}}, status_code=503 if exc.retry else 403)

    @app.get('/live')
    async def live():
        return {'live': True}

    @app.get('/health')
    async def health():
        await studio.authorize()
        return {'ready': True, 'route': 'h3-singularity'}

    @app.post('/v2/video_generation')
    async def submit(request: Request):
        await studio.authorize()
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > MAX_BODY:
                raise HTTPException(413, 'request_too_large')
        try:
            body = json.loads(data)
        except ValueError:
            raise HTTPException(400, 'invalid_json') from None
        payload = normalize(body)
        for media in payload['media']:
            source_info(media['source'], media['kind'], settings)
        task = ledger.create(payload, request.headers.get('Idempotency-Key'))
        row = ledger.get(task)
        if not row['native'] and row['status'] not in {'succeeded', 'failed'}:
            schedule(task)
        return {'task_id': task}

    def signature(task, expires):
        return hmac.new(settings.signing_key.encode(), f'{task}:{expires}'.encode(), hashlib.sha256).hexdigest()

    async def native_result(row):
        result = await studio.call('video.result', {'task_id': row['native']})
        if not ARTIFACT.fullmatch(str(result.get('artifact_id', ''))):
            raise UpstreamError('invalid_result_identity', False)
        return result

    @app.get('/v2/query/video_generation/{task}')
    async def status(task: str):
        await studio.authorize()
        row = ledger.get(task)
        if not row['native']:
            if row['status'] not in {'failed', 'succeeded'}:
                schedule(task)
            return {'task_id': task, 'status': 'failed' if row['status'] == 'failed' else 'processing',
                    'error': {'message': row['error']} if row['error'] else None}
        status = await studio.call('video.status', {'task_id': row['native']})
        state = status.get('status')
        if state in {'failed', 'cancelled'}:
            ledger.update(task, status='failed', error='h3_generation_failed')
            return {'task_id': task, 'status': 'failed', 'error': {'message': 'h3_generation_failed'}}
        if state not in {'succeeded', 'completed'}:
            return {'task_id': task, 'status': 'processing'}
        result = await native_result(row)
        ledger.update(task, status='succeeded')
        expires = int(time.time()) + 900
        return {'task_id': task, 'status': 'succeeded',
                'content': {'url': f'{settings.public_origin}/v2/content/{task}.mp4?expires={expires}&signature={signature(task, expires)}'},
                'vf': {'project_id': settings.project_id, 'task_id': row['native'],
                       'artifact_id': result['artifact_id'], 'media': result.get('media'),
                       'route': result.get('runtime_route'), 'runtime_version': result.get('runtime_version')}}

    @app.get('/v2/content/{filename}')
    async def content(filename: str, expires: int = 0, signature: str = ''):
        task = filename.removesuffix('.mp4')
        if filename != task + '.mp4' or not ID.fullmatch(task):
            raise HTTPException(404, 'task_not_found')
        now = int(time.time())
        expected = hmac.new(settings.signing_key.encode(), f'{task}:{expires}'.encode(), hashlib.sha256).hexdigest()
        if not now < expires <= now + 900 or not hmac.compare_digest(signature, expected):
            raise HTTPException(403, 'invalid_download_signature')
        await studio.authorize()
        row = ledger.get(task)
        if not row['native']:
            raise HTTPException(409, 'result_not_ready')
        result = await native_result(row)
        url = settings.studio_origin + '/artifacts/' + result['artifact_id'] + '/content'
        response = await backend.send(backend.build_request('GET', url, headers=studio.headers), stream=True)
        if response.status_code != 200:
            await response.aclose()
            raise UpstreamError('result_download_unavailable')
        headers = {'Content-Disposition': f'attachment; filename="{task}.mp4"'}
        if 'content-length' in response.headers:
            headers['Content-Length'] = response.headers['content-length']
        return StreamingResponse(response.aiter_bytes(), media_type='video/mp4', headers=headers,
                                 background=BackgroundTask(response.aclose))

    return app


def main():
    import uvicorn
    settings = Settings(studio_origin=os.environ['STUDIO_ORIGIN'], studio_token=os.environ['STUDIO_BEARER_TOKEN'],
        api_key=os.environ['HUOBAO_API_KEY'], signing_key=os.environ['HUOBAO_SIGNING_KEY'],
        project_id=os.environ['HUOBAO_PROJECT_ID'], owner_user_id=os.environ['HUOBAO_OWNER_USER_ID'],
        owner_org_id=os.environ['HUOBAO_OWNER_ORG_ID'], data_dir=Path(os.environ.get('HUOBAO_BRIDGE_DATA', '/data')),
        public_origin=os.environ.get('HUOBAO_BRIDGE_ORIGIN', 'http://huobao-h3-bridge:8000'),
        media_origin=os.environ.get('HUOBAO_MEDIA_ORIGIN', 'https://huobao-trial.verdantflarehub.com'),
        media_internal_origin=os.environ.get('HUOBAO_MEDIA_INTERNAL_ORIGIN', 'http://huobao-drama:5679'))
    uvicorn.run(create_app(settings), host='0.0.0.0', port=8000, workers=1, access_log=False)


if __name__ == '__main__':
    main()
