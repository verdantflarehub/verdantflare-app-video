import json
import httpx
import os
from unittest.mock import patch
import unittest
import test_sol_route
from app.executor import ExecutionError, VideoExecutor
from app.resources import Resources


class VdnRouteTest(unittest.TestCase):
    tearDown = test_sol_route.SolRouteTest.tearDown
    def setUp(self):
        test_sol_route.SolRouteTest.setUp(self)
        self.route_env = patch.dict(os.environ, {
            'H3_RUNTIME_ROUTES': json.dumps({'h3-vdn': {
                'url':'http://vdn.example:8000', 'version':'video-minimax-h3-vdn-v0.3.0',
                'requires_token':True}}), 'H3_VDN_RUNTIME_TOKEN':'test-vdn-token'})
        self.route_env.start()
        self.addCleanup(self.route_env.stop)
        self.executor = VideoExecutor(self.assets, self.tasks, self.http)

    def test_vdn_ref2va_contract_and_identity(self):
        row = self.executor.generate(**self.kw, route='h3-vdn')
        call = self.calls[-1]
        self.assertEqual(call.url.host, 'vdn.example')
        self.assertEqual(call.headers['Authorization'], 'Bearer test-vdn-token')
        body = json.loads(call.content)
        self.assertEqual((body['task'], body['num_inference_steps']), ('ref2va', 4))
        self.assertEqual(body['idempotency_key'], row.video_task_id)
        self.assertEqual(body['conditions'][0]['size'], 7)
        self.assertEqual(len(body['conditions'][0]['sha256']), 64)
        self.assertEqual((row.service, row.runtime_route, row.request['model']),
                         ('h3-vdn', 'h3-vdn', 'minimax-h3-ref2va'))
        self.executor.status(row.video_task_id)
        self.assertEqual(self.calls[-1].url.host, 'vdn.example')
        self.assertEqual(self.executor.generate(**self.kw, route='h3-vdn').video_task_id, row.video_task_id)
        self.assertEqual(len([r for r in self.calls if r.method == 'POST']), 1)

    def test_missing_vdn_token_never_uses_sol_token_or_h3(self):
        with patch.dict(os.environ, {'H3_VDN_RUNTIME_TOKEN':''}):
            executor = VideoExecutor(self.assets, self.tasks, self.http)
            with self.assertRaises(ExecutionError):
                executor.generate(**self.kw, route='h3-vdn')
            self.assertFalse(Resources.route_connected('h3-vdn'))
        with patch.dict(os.environ, {'H3_RUNTIME_ROUTES':'{}'}):
            executor = VideoExecutor(self.assets, self.tasks, self.http)
            with self.assertRaises(ExecutionError):
                executor.generate(**self.kw, route='h3-vdn')
        self.assertFalse(self.calls)

    def test_vdn_rejects_unsupported_duration_model_and_total_count(self):
        for update in ({'duration_seconds':6}, {'model':'minimax-h3-t2va'},
                       {'references':{'images':self.kw['references']['images']*9,
                                      'videos':self.kw['references']['images']*3,
                                      'audios':self.kw['references']['images']}}):
            with self.assertRaises(ValueError):
                self.executor.generate(**{**self.kw, **update}, route='h3-vdn')
        self.assertFalse(self.calls)

    def test_explicit_audio_precedes_video_soundtrack_numbering(self):
        video = self.assets.create_from_chunks(project_id='demo',operation='test',filename='clip.mp4',media_type='video/mp4',chunks=[b'video'])
        audio = self.assets.create_from_chunks(project_id='demo',operation='test',filename='voice.wav',media_type='audio/wav',chunks=[b'audio'])
        refs = dict(self.kw['references'], videos=[{'artifact_id':video.artifact_id,'purpose':'motion'}],
                    audios=[{'artifact_id':audio.artifact_id,'purpose':'voice'}])
        self.executor.generate(**{**self.kw,'references':refs},route='h3-vdn')
        body = json.loads(self.calls[-1].content)
        self.assertEqual([r['type'] for r in body['conditions']],['image','audio','video'])
        self.assertIn(audio.artifact_id, body['conditions'][1]['uri'])
        self.assertIn('<Audio 1> is the approved voice reference',body['prompt'])

    def test_removed_and_unknown_routes_never_fall_back(self):
        for route in ('h3-sol-4090','minimax-h3-sol-ref2va-4090','h3-vdn-typo','missing-channel'):
            with self.assertRaises(ExecutionError):
                self.executor.generate(**self.kw,route=route)
        self.assertFalse(self.calls)

    def test_readiness_failures_never_submit_and_remain_idempotent(self):
        scenarios = [
            (httpx.ConnectError("private-host secret-token"), "runtime_unavailable"),
            (httpx.ConnectTimeout("private-host"), "runtime_unavailable"),
            (httpx.ReadTimeout("private-host"), "runtime_health_check_failed"),
            (httpx.Response(503, json={"ready": False}), "runtime_not_ready"),
            (httpx.Response(200, json={"ready": False}), "runtime_not_ready"),
            (httpx.Response(401, text="secret-token"), "runtime_health_check_failed"),
            (httpx.Response(200, json={"ready": "true"}), "runtime_health_check_failed"),
            (httpx.Response(200, json=[]), "runtime_health_check_failed"),
            (httpx.Response(200, text="bad-json"), "runtime_health_check_failed"),
        ]
        for index, (outcome, code) in enumerate(scenarios):
            with self.subTest(code=code, index=index):
                calls = []
                def handle(request):
                    calls.append(request)
                    self.assertEqual((request.method, request.url.path), ("GET", "/health"))
                    self.assertEqual(request.headers["Authorization"], "Bearer test-vdn-token")
                    if isinstance(outcome, Exception):
                        raise outcome
                    return outcome
                with httpx.Client(transport=httpx.MockTransport(handle)) as client:
                    executor = VideoExecutor(self.assets, self.tasks, client)
                    kw = dict(self.kw, idempotency_key=f"health-{index}", route="h3-vdn")
                    row = executor.generate(**kw)
                    self.assertEqual((row.status, row.error["code"]), ("failed", code))
                    self.assertFalse(row.runtime_task_id)
                    self.assertNotIn("secret-token", str(row.error))
                    self.assertEqual(executor.generate(**kw).video_task_id, row.video_task_id)
                    self.assertEqual(executor.status(row.video_task_id).error, row.error)
                    self.assertEqual(len(calls), 1)

    def test_post_failure_classification_and_no_automatic_replay(self):
        scenarios = [
            (httpx.ConnectError("private-host"), "runtime_unavailable"),
            (httpx.ConnectTimeout("private-host"), "runtime_unavailable"),
            (httpx.ReadTimeout("private-host"), "submission_unconfirmed"),
            (httpx.WriteError("private-host"), "submission_unconfirmed"),
            (httpx.Response(500, text="secret-token"), "submission_unconfirmed"),
            (httpx.Response(200, json={}), "submission_unconfirmed"),
            (httpx.Response(200, json={"id": ""}), "submission_unconfirmed"),
            (httpx.Response(200, json={"id": None}), "submission_unconfirmed"),
            (httpx.Response(200, json=[]), "submission_unconfirmed"),
        ]
        for index, (outcome, code) in enumerate(scenarios):
            with self.subTest(code=code, index=index):
                calls = []
                def handle(request):
                    calls.append(request)
                    if request.url.path == "/health":
                        return httpx.Response(200, json={"ready": True})
                    if isinstance(outcome, Exception):
                        raise outcome
                    return outcome
                with httpx.Client(transport=httpx.MockTransport(handle)) as client:
                    executor = VideoExecutor(self.assets, self.tasks, client)
                    kw = dict(self.kw, idempotency_key=f"post-{index}", route="h3-vdn")
                    row = executor.generate(**kw)
                    self.assertEqual((row.status, row.error["code"]), ("failed", code))
                    self.assertFalse(row.runtime_task_id)
                    self.assertNotIn("secret-token", str(row.error))
                    self.assertEqual(executor.generate(**kw).video_task_id, row.video_task_id)
                    executor.status(row.video_task_id)
                    self.assertEqual([r.method for r in calls], ["GET", "POST"])
