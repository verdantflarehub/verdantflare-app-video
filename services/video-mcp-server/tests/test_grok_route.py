import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
from PIL import Image

from app.artifacts import ArtifactStore
from app.executor import VideoExecutor
from app.delivery import CALLER
from app.tasks import TaskStore, TaskConflict


class GrokRouteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = patch.dict('os.environ', {'XAI_BASE_URL':'https://grok.example/v1',
            'XAI_API_KEY':'private-fixture','XAI_VIDEO_ENABLED':'true'})
        self.env.start()
        self.calls = []

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def executor(self, handler):
        def respond(r):
            self.calls.append(r)
            return handler(r)
        return VideoExecutor(ArtifactStore(self.root), TaskStore(self.root),
                             httpx.Client(transport=httpx.MockTransport(respond)))

    def inputs(self, **updates):
        p = dict(project_id='demo', idempotency_key='one', route='grok',
                 model='grok-imagine-video-1.5', prompt='Exact prompt',
                 duration_seconds=5, aspect_ratio='16:9', references={})
        p.update(updates)
        return p

    def poll(self, ex, task):
        rec = ex.tasks.get(task.video_task_id)
        state = dict(rec.provider_state or {})
        state['next_poll_at'] = 0
        ex.tasks.update(rec, provider_state=state)
        return ex.status(task.video_task_id)

    def test_submission_is_deferred_and_idempotent(self):
        ex = self.executor(lambda r: httpx.Response(200,json={'request_id':'remote-1'}))
        task = ex.generate(**self.inputs())
        self.assertEqual(len(self.calls), 0)
        self.assertEqual(task.service, 'grok')
        same = ex.generate(**self.inputs())
        self.assertEqual(task.video_task_id, same.video_task_id)
        with self.assertRaises(TaskConflict):
            ex.generate(**self.inputs(prompt='different'))
        task = ex.status(task.video_task_id)
        self.assertEqual(task.runtime_task_id,'remote-1')
        self.assertEqual(json.loads(self.calls[0].content)['duration'], 5)

    def test_dashboard_grok_defaults_and_private_state(self):
        from app.dashboard import Dashboard
        from starlette.applications import Starlette
        from starlette.testclient import TestClient
        ex = self.executor(lambda r: httpx.Response(200, json={'request_id':'remote-1'}))
        payload = self.inputs(duration_seconds=1)
        payload.pop('model')
        payload.pop('aspect_ratio')
        with TestClient(Starlette(routes=Dashboard(ex).routes())) as client:
            result = client.post('/api/tasks', json=payload)
            self.assertEqual(result.status_code, 200)
            data = result.json()
            self.assertEqual(data['model'], 'grok-imagine-video-1.5')
            self.assertEqual(data['aspect_ratio'], '16:9')
            self.assertEqual(data['service'], 'grok')
            self.assertNotIn('provider_state', data)
            self.assertEqual(self.calls, [])
            self.assertEqual(client.post('/api/tasks', json={**payload,'prompt':'changed'}).status_code,409)
    def test_timeout_never_reposts_even_after_restart(self):
        def handler(r):
            raise httpx.ReadTimeout('private-fixture',request=r)
        ex = self.executor(handler)
        task = ex.generate(**self.inputs())
        task = ex.status(task.video_task_id)
        self.assertEqual(task.runtime_stage, 'submission_unknown')
        self.assertNotIn('private-fixture', str(task.error))
        restarted = self.executor(lambda r: self.fail('Must not repost'))
        self.poll(restarted, task)
        self.assertEqual(len(self.calls), 1)

    def test_http_200_failed_is_failure(self):
        ex = self.executor(lambda r: httpx.Response(200,json={'request_id':'remote-1'} if r.method=='POST' else {'status':'failed','error':{'message':'secret'}}))
        task = ex.generate(**self.inputs())
        ex.status(task.video_task_id)
        task = self.poll(ex, task)
        self.assertEqual(task.status,'failed')
        self.assertNotIn('secret', str(task.error))

    def test_empty_202_is_pending(self):
        ex = self.executor(lambda r: httpx.Response(200,json={'request_id':'remote-1'}) if r.method=='POST' else httpx.Response(202))
        task = ex.generate(**self.inputs())
        ex.status(task.video_task_id)
        task = self.poll(ex,task)
        self.assertEqual(task.runtime_stage, 'remote_pending')
        self.assertIsNone(task.timing['total_seconds'])

    def test_invalid_combinations_never_submit(self):
        ex = self.executor(lambda r: self.fail('Must not submit'))
        for update in ({'model':'minimax-h3-ref2va'}, {'duration_seconds':0}, {'duration_seconds':16},
                       {'quality_profile':'du-0'}, {'resolution':'1080p'}, {'aspect_ratio':'21:9'},
                       {'mode':'image_to_video'}, {'references':{'audios':[{}]}}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                ex.generate(**self.inputs(**update))

    def test_first_frame_inherits_input_and_rejects_conflicts(self):
        ex = self.executor(lambda r: httpx.Response(200,json={'request_id':'remote-1'}))
        stream=io.BytesIO()
        Image.new('RGB',(18,32),'green').save(stream,'PNG')
        image=ex.artifacts.create_from_chunks(project_id='demo', operation='input', filename='frame.png', media_type='image/png',chunks=[stream.getvalue()])
        refs={'images':[{'artifact_id':image.artifact_id,'purpose':'first_frame'}]}
        with self.assertRaisesRegex(ValueError,'aspect_ratio_conflict'):
            ex.generate(**self.inputs(references=refs))
        task=ex.generate(**self.inputs(references=refs,aspect_ratio=None))
        self.assertEqual(task.request['aspect_ratio'],'9:16')
        ex.status(task.video_task_id)
        payload=json.loads(self.calls[-1].content)
        self.assertNotIn('aspect_ratio',payload)
        self.assertTrue(payload['image']['url'].startswith('data:image/png;base64,'))

    def test_remote_done_collects_once_and_hides_url(self):
        def handler(r):
            return httpx.Response(200,json={'request_id':'remote-1'} if r.method=='POST' else
                {'status':'done','video':{'url':'https://vidgen.x.ai/result.mp4?secret=fixture'}})
        ex=self.executor(handler)
        ex.grok.download_client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200,content=b'video-fixture',headers={'content-type':'video/mp4'})))
        ex.grok.probe=lambda p: {'duration_ms':5000,'frame_rate':30,'width':1280,'height':720,'video_codec':'h264','audio_codec':None}
        task=ex.generate(**self.inputs())
        ex.status(task.video_task_id)
        with patch('socket.getaddrinfo',return_value=[(2,1,6,'',('8.8.8.8',443))]):
            task=self.poll(ex,task)
        self.assertEqual(task.status,'succeeded')
        self.assertEqual(task.media['frame_rate'],30)
        self.assertNotIn('result_url',task.provider_state)
        result=ex.result(task.video_task_id)
        self.assertEqual(task.artifact_id,result.artifact_id)
        self.assertEqual(sum(r.method=='POST' for r in self.calls),1)

    def test_download_denies_private_and_unapproved_targets(self):
        ex=self.executor(lambda r: None)
        with self.assertRaises(ValueError):
            ex.grok.allowed_url('https://untrusted.example/video')
        with patch('socket.getaddrinfo',return_value=[(2,1,6,'',('127.0.0.1',443))]), self.assertRaises(ValueError):
            ex.grok.allowed_url('https://vidgen.x.ai/video')

    def test_gateway_content_auth_is_bound_to_original_task(self):
        ex = self.executor(lambda r: httpx.Response(200, json={'request_id': 'remote-1'} if r.method == 'POST' else
            {'status': 'done', 'video': {'url': '/v1/videos/remote-1/content'}}))
        downloads = []
        def download(r):
            downloads.append(r)
            self.assertEqual(str(r.url), 'https://grok.example/v1/videos/remote-1/content')
            self.assertEqual(r.headers.get('Authorization'), 'Bearer private-fixture')
            return httpx.Response(200, content=b'video-fixture', headers={'content-type': 'video/mp4'})
        ex.grok.download_client = httpx.Client(transport=httpx.MockTransport(download))
        ex.grok.probe = lambda p: {'duration_ms': 5000, 'frame_rate': 30, 'width': 1280, 'height': 720}
        task = ex.generate(**self.inputs())
        ex.status(task.video_task_id)
        with patch('socket.getaddrinfo', return_value=[(2,1,6,'',('8.8.8.8',443))]):
            task = self.poll(ex, task)
            self.assertEqual(task.status, 'succeeded')
            for url in ['https://vidgen.x.ai/video.mp4', 'https://grok.example/v1/videos/other/content',
                        'https://grok.example/v1/videos/remote-1/content?next=other']:
                _, headers = ex.grok.result_request(url, 'remote-1')
                self.assertNotIn('Authorization', headers)
            for url in ['/v1/videos/other/content', '//evil.example/v1/videos/remote-1/content']:
                with self.assertRaises(ValueError):
                    ex.grok.result_request(url, 'remote-1')
        self.assertEqual(len(downloads), 1)
        self.assertEqual(sum(r.method == 'POST' for r in self.calls), 1)


class GrokChannelRouteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.token = 'g' * 32
        self.env = patch.dict('os.environ', {
            'VIDEO_GROK_CHANNEL_URL': 'http://grok-channel:8001',
            'VIDEO_GROK_CHANNEL_TOKEN': self.token,
            'XAI_API_KEY': '',
            'VIDEO_PROJECT_SERVICE_URL': 'http://project',
            'VIDEO_PROJECT_SERVICE_TOKEN': 'p' * 40,
        })
        self.env.start()
        self.caller_token = CALLER.set({'organization_id': 'org', 'user_id': 'user'})
        self.calls = []

    def tearDown(self):
        CALLER.reset(self.caller_token)
        self.env.stop()
        self.tmp.cleanup()

    def test_grok_channel_submit_status_and_content(self):
        call_id = '11111111-2222-3333-4444-555555555555'
        def handler(r: httpx.Request):
            self.calls.append(r)
            if r.url.host == 'project':
                return httpx.Response(200, json={})
            self.assertEqual(r.headers.get('Authorization'), 'Bearer ' + self.token)
            if r.method == 'POST' and r.url.path == '/v1/calls':
                return httpx.Response(200, json={
                    'call_id': call_id,
                    'accepted': True,
                    'state': 'in_progress',
                })
            if r.method == 'GET' and r.url.path == f'/v1/calls/{call_id}':
                return httpx.Response(200, json={
                    'call_id': call_id,
                    'accepted': True,
                    'state': 'completed',
                })
            if r.method == 'GET' and r.url.path == f'/v1/calls/{call_id}/result':
                return httpx.Response(200, content=b'grok-channel-mp4', headers={'content-type': 'video/mp4'})
            return httpx.Response(404)

        ex = VideoExecutor(ArtifactStore(self.root), TaskStore(self.root),
                           httpx.Client(transport=httpx.MockTransport(handler)))
        ex.require_channel = Mock()
        ex.grok.probe = lambda p: {'duration_ms': 5000, 'frame_rate': 30, 'width': 1280, 'height': 720}
        task = ex.generate(project_id='demo', idempotency_key='grok-ch-1', route='grok',
                           model='grok-imagine-video', prompt='Neon city', duration_seconds=5,
                           aspect_ratio='16:9', references={})
        self.assertEqual(task.runtime_task_id, call_id)
        ex.require_channel.assert_called_once_with('grok', 'grok-imagine-video')
        state = dict(task.provider_state or {})
        state['next_poll_at'] = 0
        ex.tasks.update(task, provider_state=state)
        task = ex.status(task.video_task_id)
        self.assertEqual(task.status, 'succeeded')
        self.assertTrue(task.artifact_id)

    def test_read_errors_preserve_acceptance_without_assuming_unknown_calls_accepted(self):
        for failure_at, accepted in [('status', True), ('result', True), ('status', None)]:
            with self.subTest(failure_at=failure_at, accepted=accepted):
                call_id = '11111111-2222-3333-4444-555555555555'
                def handler(r):
                    if r.url.host == 'project':
                        return httpx.Response(200, json={})
                    if r.method == 'POST':
                        return httpx.Response(200, json={'call_id': call_id, 'accepted': accepted,
                            'state': 'in_progress' if accepted else 'acceptance_unknown'})
                    if failure_at == 'result' and not r.url.path.endswith('/result'):
                        return httpx.Response(200, json={'call_id': call_id, 'accepted': True, 'state': 'completed'})
                    return httpx.Response(503, json={'accepted': False, 'error': {
                        'source': 'grok', 'code': 'provider_unavailable',
                        'message': 'Read operation is temporarily unavailable.'}})
                ex = VideoExecutor(ArtifactStore(self.root), TaskStore(self.root),
                    httpx.Client(transport=httpx.MockTransport(handler)))
                ex.require_channel = Mock()
                task = ex.generate(project_id='demo', idempotency_key=f'{failure_at}-{accepted}', route='grok',
                    model='grok-imagine-video', prompt='A blue ball', duration_seconds=5,
                    aspect_ratio='16:9', references={})
                ex.tasks.update(task, provider_state={**task.provider_state, 'next_poll_at': 0})
                observed = ex.status(task.video_task_id)
                self.assertEqual(observed.error['code'], 'provider_unavailable')
                self.assertIs(observed.error['accepted'], True if accepted else False)
                self.assertEqual(observed.runtime_task_id, call_id)


class GoogleChannelRouteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.token = 'v' * 32
        self.env = patch.dict('os.environ', {
            'VIDEO_GOOGLE_CHANNEL_URL': 'http://google-channel:8001',
            'VIDEO_GOOGLE_CHANNEL_TOKEN': self.token,
            'GOOGLE_VIDEO_MODEL': 'gemini-omni-1.1-flash',
            'VIDEO_PROJECT_SERVICE_URL': 'http://project',
            'VIDEO_PROJECT_SERVICE_TOKEN': 'p' * 40,
        })
        self.env.start()
        self.caller_token = CALLER.set({'organization_id': 'org', 'user_id': 'user'})
        self.calls = []

    def tearDown(self):
        CALLER.reset(self.caller_token)
        self.env.stop()
        self.tmp.cleanup()

    def test_google_channel_submit_status_and_content(self):
        call_id = '22222222-3333-4444-5555-666666666666'
        def handler(r: httpx.Request):
            self.calls.append(r)
            if r.url.host == 'project':
                return httpx.Response(200, json={})
            self.assertEqual(r.headers.get('Authorization'), 'Bearer ' + self.token)
            if r.method == 'POST' and r.url.path == '/v1/calls':
                return httpx.Response(200, json={
                    'call_id': call_id,
                    'accepted': True,
                    'state': 'in_progress',
                })
            if r.method == 'GET' and r.url.path == f'/v1/calls/{call_id}':
                return httpx.Response(200, json={
                    'call_id': call_id,
                    'accepted': True,
                    'state': 'completed',
                })
            if r.method == 'GET' and r.url.path == f'/v1/calls/{call_id}/result':
                return httpx.Response(200, content=b'google-omni-mp4', headers={'content-type': 'video/mp4'})
            return httpx.Response(404)

        ex = VideoExecutor(ArtifactStore(self.root), TaskStore(self.root),
                           httpx.Client(transport=httpx.MockTransport(handler)))
        ex.require_channel = Mock()
        ex.grok.probe = lambda p: {'duration_ms': 4000, 'frame_rate': 24, 'width': 1280, 'height': 720}
        task = ex.generate(project_id='demo', idempotency_key='google-ch-1', route='google',
                           model='gemini-omni-1.1-flash', prompt='Ocean waves', duration_seconds=4,
                           aspect_ratio='16:9', references={})
        self.assertEqual(task.runtime_task_id, call_id)
        ex.require_channel.assert_called_once_with('google', 'gemini-omni-1.1-flash')
        state = dict(task.provider_state or {})
        state['next_poll_at'] = 0
        ex.tasks.update(task, provider_state=state)
        task = ex.status(task.video_task_id)
        self.assertEqual(task.status, 'succeeded')
        self.assertTrue(task.artifact_id)

