import base64
from dataclasses import replace
import json
import time

from fastapi.testclient import TestClient
import httpx
import pytest
from h3_api.huobao import Settings, create_app, normalize, source_info, Ledger

UUID = '0199c0a0-0000-7000-8000-000000000021'
ART = 'art_' + 'a' * 32


def settings(tmp_path):
    return Settings('https://studio.example', 's'*40, 'k'*40, 'h'*40, UUID, UUID, UUID, tmp_path)


def body():
    return {'model': 'MiniMax-H3', 'duration': 4, 'resolution': '768P', 'ratio': '16:9',
            'content': [{'type': 'text', 'text': 'reference test'},
                        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,YWJj'}, 'role': 'reference_image'}]}


class Backend:
    def __init__(self):
        self.calls = []
        self.denied = False
        self.lost_create = False
        self.fail_once = False
        self.native_ids = set()

    def handle(self, request):
        if request.url.path.endswith('/content'):
            return httpx.Response(200, content=b'video-bytes', headers={'Content-Type': 'video/mp4'})
        data = json.loads(request.content)['params']
        name, args = data['name'], data['arguments']
        self.calls.append((name, args))
        if self.denied:
            return httpx.Response(401)
        result = {}
        if name == 'video.import_prepare':
            result = {'import_id': 'imp_test', 'offset': 0, 'chunk_max_bytes': 524288}
        elif name == 'video.import_chunk':
            result = {'offset': args['offset'] + len(base64.b64decode(args['content_base64']))}
        elif name == 'video.import_commit':
            result = {'artifact': {'artifact_id': ART}}
        elif name == 'video.create':
            self.native_ids.add(args['idempotency_key'])
            if self.fail_once and not self.lost_create:
                self.lost_create = True
                raise httpx.ReadTimeout('lost response', request=request)
            result = {'task_id': 'video_task_'+'b'*32}
        elif name == 'video.status':
            result = {'status': 'succeeded'}
        elif name == 'video.result':
            result = {'artifact_id': ART, 'runtime_route': 'h3-singularity', 'media': {'fps': 24}}
        return httpx.Response(200, json={'jsonrpc': '2.0', 'id': 'test', 'result': {'structuredContent': result}})


def test_roundtrip_idempotency_and_signed_download(tmp_path):
    s = settings(tmp_path)
    back = Backend()
    with TestClient(create_app(s, httpx.AsyncClient(transport=httpx.MockTransport(back.handle)))) as c:
        headers = {'Authorization': 'Bearer '+s.api_key, 'Idempotency-Key': 'huobao-task-1'}
        assert c.post('/v2/video_generation', json=body()).status_code == 401
        assert c.post('/v2/video_generation', headers=headers, json={}).status_code == 400
        r = c.post('/v2/video_generation', headers=headers, json=body())
        assert r.status_code == 200
        task = r.json()['task_id']
        assert c.post('/v2/video_generation', headers=headers, json=body()).json()['task_id'] == task
        modified = body(); modified['duration'] = 5
        assert c.post('/v2/video_generation', headers=headers, json=modified).status_code == 409
        for _ in range(50):
            r = c.get('/v2/query/video_generation/'+task, headers=headers).json()
            if r['status'] == 'succeeded':
                break
            time.sleep(.01)
        assert r['status'] == 'succeeded'
        url = r['content']['url'].replace(s.public_origin, '')
        assert c.get(url).content == b'video-bytes'
        assert c.get(url[:-1] + ('a' if url[-1] != 'a' else 'b')).status_code == 403
        assert c.get('/v2/content/'+task+'.mp4').status_code == 403
        back.denied = True
        assert c.get(url).status_code == 403
        assert c.get('/v2/query/video_generation/'+task, headers=headers).status_code == 403
        assert c.post('/v2/video_generation', headers=headers, json=body()).status_code == 403
    create = next(args for name, args in back.calls if name == 'video.create')
    assert create['project_id'] == s.project_id and create['route'] == 'h3-singularity'
    assert create['references']['images'][0]['artifact_id'] == ART
    assert not list(tmp_path.glob('hb_*-*'))


@pytest.mark.parametrize('url', [
    'http://169.254.169.254/latest/meta-data',
    'https://huobao-trial.verdantflarehub.com.evil/static/uploads/a.png',
    'https://huobao-trial.verdantflarehub.com/static/uploads/%2e%2e/a.png',
    'https://huobao-trial.verdantflarehub.com/static/uploads/a.png?token=x',
    'https://user@huobao-trial.verdantflarehub.com/static/uploads/a.png',
    'https://huobao-trial.verdantflarehub.com/static/uploads/../a.png',
    'file:///tmp/a.png', 'data:image/png;base64,%%%'
])
def test_unsafe_sources(tmp_path, url):
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        source_info(url, 'images', settings(tmp_path))


def test_modes_and_spoofing_rejected():
    from fastapi import HTTPException
    for change in ({'resolution': '2K'}, {'project_id': UUID}, {'model': 'MiniMax-H3-fast'}, {'duration': True}):
        with pytest.raises(HTTPException):
            normalize({**body(), **change})
    b = body(); b['content'][1]['role'] = 'first_frame'
    with pytest.raises(HTTPException):
        normalize(b)


def test_ledger_restart_and_binding(tmp_path):
    s = settings(tmp_path)
    ledger = Ledger(s)
    task = ledger.create(normalize(body()), 'stable')
    ledger.db.close()
    ledger = Ledger(s)
    assert ledger.create(normalize(body()), 'stable') == task
    assert ledger.create(normalize(body()), None) == task
    ledger.db.close()
    with pytest.raises(ValueError, match='identity_mismatch'):
        Ledger(replace(s, project_id='0199c0a0-0000-7000-8000-000000000022'))


def test_unknown_task_and_redirect(tmp_path):
    s = settings(tmp_path)
    back = Backend()
    def handle(request):
        if request.method == 'GET':
            return httpx.Response(302, headers={'location': 'http://169.254.169.254/'})
        return back.handle(request)
    with TestClient(create_app(s, httpx.AsyncClient(transport=httpx.MockTransport(handle)))) as c:
        headers = {'Authorization': 'Bearer '+s.api_key}
        assert c.get('/v2/query/video_generation/hb_'+'f'*32, headers=headers).status_code == 404
        b = body(); b['content'][1]['image_url']['url'] = s.media_origin+'/static/uploads/test.png'
        task = c.post('/v2/video_generation', headers=headers, json=b).json()['task_id']
        for _ in range(50):
            result = c.get('/v2/query/video_generation/'+task, headers=headers).json()
            if result['status'] == 'failed': break
            time.sleep(.01)
        assert result['status'] == 'failed'
        assert not any(name == 'video.create' for name, _ in back.calls)


def test_create_response_loss_recovers_same_key(tmp_path):
    s = settings(tmp_path)
    back = Backend(); back.fail_once = True
    with TestClient(create_app(s, httpx.AsyncClient(transport=httpx.MockTransport(back.handle)))) as c:
        headers = {'Authorization': 'Bearer '+s.api_key}
        task = c.post('/v2/video_generation', headers=headers, json=body()).json()['task_id']
        for _ in range(100):
            result = c.get('/v2/query/video_generation/'+task, headers=headers).json()
            if result['status'] == 'succeeded': break
            time.sleep(.02)
        assert result['status'] == 'succeeded'
        assert len(back.native_ids) == 1
        assert sum(name == 'video.import_commit' for name, _ in back.calls) == 1
