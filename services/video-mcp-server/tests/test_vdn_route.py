import json
import os
from unittest.mock import patch
import unittest
import test_sol_route
from verdantflare_video_mcp.executor import ExecutionError, VideoExecutor
from verdantflare_video_mcp.resources import Resources


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
        self.assertEqual((body['task'], body['num_inference_steps']), ('ref2va', 8))
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
