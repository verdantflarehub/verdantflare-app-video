import copy
import http.client
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from resident_api import State, serve, validate
from resident_worker import execute
from task_store import TaskStore

SOURCE = 'http://video-mcp-server:8000'


def request():
    return dict(model='MiniMaxAI/MiniMax-H3',task='ref2va',prompt='Test motion',seed=7,
                seconds=5,num_inference_steps=8,num_outputs_per_prompt=1,flow_shift=12.0,audio_flow_shift=3.0,
                target={'short_edge':768,'aspect_ratio':'9:16','duration_seconds':5.0},idempotency_key='attempt-1',
                conditions=[dict(type='image',role='reference',uri=SOURCE+'/runtime-artifacts/art_'+'a'*32+'/content',sha256='a'*64,size=12)])


class ResidentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = TaskStore(self.temp.name, 'test-instance')
        self.addCleanup(self.store.close)
        self.state = State('test-instance')
        self.server = serve(self.state, self.store, 'test-token', SOURCE, host='127.0.0.1', port=0)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def call(self, method, path, payload=None, authorized=True):
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        headers = {'Authorization':'Bearer test-token'} if authorized else {}
        conn.request(method, path, body=json.dumps(payload) if payload is not None else None, headers=headers)
        response = conn.getresponse()
        result = (response.status, json.loads(response.read()))
        conn.close()
        return result

    def test_http_auth_readiness_idempotency_conflict_and_cancel(self):
        self.assertEqual(self.call('POST', '/v1/videos', request(), False)[0], 401)
        self.assertEqual(self.call('GET', '/live')[0], 200)
        self.assertEqual(self.call('GET', '/health')[0], 503)
        self.assertEqual(self.call('POST', '/v1/videos', request())[0], 503)
        self.state.update(ready=True, stage='ready')
        status, task = self.call('POST', '/v1/videos', request())
        self.assertEqual(status, 200)
        self.assertEqual(self.call('POST', '/v1/videos', request())[1]['id'], task['id'])
        self.assertEqual(self.call('POST', '/v1/videos', dict(request(), prompt='different'))[0], 409)
        self.assertEqual(self.call('GET', '/v1/videos/'+task['id']+'/content')[0], 404)
        self.assertEqual(self.call('DELETE', '/v1/videos/'+task['id'])[1]['status'], 'cancelled')

    def test_rejects_unsupported_modes_and_unsafe_artifacts(self):
        base = request()
        validate(base, SOURCE)
        cases = [dict(request(), task='t2va'), dict(request(), num_inference_steps=50), dict(request(), seconds=True),
                 dict(request(), task=[]), dict(request(), conditions=[[]]), dict(request(), conditions=[])]
        for uri in [SOURCE+'.attacker/runtime-artifacts/art_'+'a'*32+'/content',
                    SOURCE+'/runtime-artifacts/../secret', base['conditions'][0]['uri']+'?redirect=http://attacker']:
            bad=copy.deepcopy(base);bad['conditions'][0]['uri']=uri;cases.append(bad)
        for payload in cases:
            self.assertEqual(self.call('POST', '/v1/videos', payload)[0], 400)

    def test_four_step_sampling_and_eight_step_compatibility(self):
        self.state.update(ready=True, stage='ready')
        for steps in (4, 8):
            payload = dict(request(), num_inference_steps=steps, idempotency_key=f'nfe-{steps}')
            status, task = self.call('POST', '/v1/videos', payload)
            self.assertEqual(status, 200)
            accepted = self.store.take()
            self.assertEqual(accepted['id'], task['id'])
            self.assertEqual(accepted['request']['num_inference_steps'], steps)
            self.assertEqual(self.call('POST', '/v1/videos', payload)[1]['id'], task['id'])
            changed = dict(payload, num_inference_steps=8 if steps == 4 else 4)
            self.assertEqual(self.call('POST', '/v1/videos', changed)[0], 409)
            self.store.finish(task['id'], error='test_finished')
        for steps in (True, 4.0, '4', 0, 5, 50):
            self.assertEqual(self.call('POST', '/v1/videos', dict(request(), num_inference_steps=steps))[0], 400)

    def test_second_process_owner_cannot_invalidate_active_tasks(self):
        task=self.store.submit('key', {})
        with self.assertRaisesRegex(RuntimeError, 'another runtime'):
            TaskStore(self.temp.name, 'second')
        self.assertEqual(self.store.get(task['id'])['status'], 'queued')

    def test_download_failure_never_calls_gpu_and_is_queryable(self):
        task=self.store.submit('key', request())
        task=self.store.take()
        with patch('resident_worker.download', side_effect=ValueError('bad hash')), patch('resident_worker.Engine') as engine:
            execute(self.store, task, engine, {})
            engine.generate.assert_not_called()
        self.assertEqual(self.store.get(task['id'])['error'], 'artifact_download_failed')

    def test_invalid_media_is_never_published(self):
        task=self.store.submit('key', request());task=self.store.take()
        with patch('resident_worker.download', return_value={}), patch('resident_worker.Engine') as engine, patch('resident_worker.inspect_media', side_effect=RuntimeError('bad media')):
            with self.assertRaises(RuntimeError):execute(self.store, task, engine, {})
        self.assertFalse((self.store.root/task['id']/'video.mp4').exists())


if __name__ == '__main__':
    unittest.main()
