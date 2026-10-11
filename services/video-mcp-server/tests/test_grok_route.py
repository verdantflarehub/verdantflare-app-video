import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from PIL import Image

from app.artifacts import ArtifactStore
from app.executor import VideoExecutor
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
