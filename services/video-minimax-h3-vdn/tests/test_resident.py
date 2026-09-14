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
    return dict(schema_version=1, task='t2va', prompt='A river at dawn', seed=7,
                frames=124, steps=8, input_artifacts=[], idempotency_key='attempt-1')


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
        base.update(task='i2va', input_artifacts=[dict(role='first', uri=SOURCE+'/runtime-artifacts/art_'+'a'*32+'/content', sha256='a'*64, size=12)])
        validate(base, SOURCE)
        cases = [dict(request(), task='ref2va'), dict(request(), steps=50), dict(request(), frames=True),
                 dict(request(), task=[]), dict(request(), input_artifacts=[[]])]
        for uri in [SOURCE+'.attacker/runtime-artifacts/art_'+'a'*32+'/content',
                    SOURCE+'/runtime-artifacts/../secret', base['input_artifacts'][0]['uri']+'?redirect=http://attacker']:
            bad=copy.deepcopy(base);bad['input_artifacts'][0]['uri']=uri;cases.append(bad)
        for payload in cases:
            self.assertEqual(self.call('POST', '/v1/videos', payload)[0], 400)

    def test_second_process_owner_cannot_invalidate_active_tasks(self):
        task=self.store.submit('key', {})
        with self.assertRaisesRegex(RuntimeError, 'another runtime'):
            TaskStore(self.temp.name, 'second')
        self.assertEqual(self.store.get(task['id'])['status'], 'queued')

    def test_download_failure_never_calls_gpu_and_is_queryable(self):
        task=self.store.submit('key', dict(request(), input_artifacts=[]))
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
