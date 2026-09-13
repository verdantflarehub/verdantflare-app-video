import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException
from fastapi.testclient import TestClient
from main import app, DepthRequest, Queue
from media import convert, MediaError, probe
from model import verify


class FakeEngine:
    def infer(self, frames, fps):
        # Deterministic fixture exercises actual RGB decode and MP4 export, not GPU inference.
        return frames[:, :, :, 0].astype(np.float32)


class DepthApiTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.art = 'art_' + 'a' * 32
        directory = self.root / 'source/artifacts' / self.art
        directory.mkdir(parents=True)
        source = directory / 'source.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=64x48:rate=30000/1001',
                        '-frames:v', '6', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)], check=True)
        self.digest = hashlib.sha256(source.read_bytes()).hexdigest()
        self.metadata = {'artifact_id': self.art, 'project_id': 'p', 'media_type': 'video/mp4',
                         'filename': 'source.mp4', 'size': source.stat().st_size, 'sha256': self.digest}
        (directory / 'metadata.json').write_text(json.dumps(self.metadata))
        self.queue = Queue(self.root / 'tasks', self.root / 'source', FakeEngine())
        app.state.queue = self.queue
        self.client = TestClient(app)
        self.payload = dict(project_id='p', idempotency_key='video_task_' + 'b' * 32,
                            source_artifact_id=self.art, source_sha256=self.digest)

    def tearDown(self):
        self.queue.close()
        self.temp.cleanup()

    def test_schema_rejects_path_url_unknown_model_and_extra_fields(self):
        for change in ({'source_artifact_id': '/tmp/source.mp4'}, {'source_artifact_id': 'https://example.com/a.mp4'},
                       {'model': 'midas'}, {'output_format': 'png'}, {'project_id': '../p'}, {'path': '/etc/passwd'}):
            self.assertEqual(self.client.post('/v1/depth', json={**self.payload, **change}).status_code, 422)

    def test_health_not_ready_without_worker(self):
        self.assertEqual(self.client.get('/health').status_code, 503)

    def test_cross_project_and_tampering(self):
        self.assertEqual(self.client.post('/v1/depth', json={**self.payload, 'project_id': 'other'}).status_code, 404)
        self.assertEqual(self.client.post('/v1/depth', json={**self.payload, 'source_sha256': '0'*64}).status_code, 422)

    def test_async_idempotency_actual_media_and_restart(self):
        first = self.client.post('/v1/depth', json=self.payload)
        self.assertEqual(first.status_code, 202)
        self.assertEqual(first.json()['status'], 'queued')
        self.assertEqual(self.client.post('/v1/depth', json=self.payload).json(), first.json())
        self.assertEqual(self.client.post('/v1/depth', json={**self.payload, 'source_sha256': '0'*64}).status_code, 409)
        task_id = self.payload['idempotency_key']
        self.assertEqual(self.client.get(f'/v1/depth/{task_id}/content').status_code, 409)
        self.assertTrue(self.queue.run_one())
        self.assertFalse(self.queue.run_one())
        data = self.client.get(f'/v1/depth/{task_id}').json()
        self.assertEqual(data['status'], 'succeeded')
        self.assertEqual(data['result']['media']['frames'], 6)
        self.assertEqual(data['result']['media']['frame_rate'], '30000/1001')
        self.assertEqual(data['result']['preview_media']['width'], 128)
        content = self.client.get(f'/v1/depth/{task_id}/content').content
        self.assertEqual(hashlib.sha256(content).hexdigest(), data['result']['output_sha256'])
        self.queue.close()
        self.queue = Queue(self.root / 'tasks', self.root / 'source', FakeEngine())
        self.assertEqual(self.queue.get(task_id)['status'], 'succeeded')
        self.assertEqual(self.queue.submit(DepthRequest(**self.payload))['status'], 'succeeded')

    def test_running_task_interrupted_on_restart(self):
        self.queue.submit(DepthRequest(**self.payload))
        self.queue.db.execute("UPDATE tasks SET state='running'")
        self.queue.db.commit()
        self.queue.close()
        self.queue = Queue(self.root / 'tasks', self.root / 'source', FakeEngine())
        self.assertEqual(self.queue.get(self.payload['idempotency_key'])['error']['code'], 'interrupted')
        self.assertFalse(self.queue.run_one())

    def test_invalid_model_fails_integrity_check(self):
        with patch.dict('os.environ', {'DEPTH_MODEL_ROOT': str(self.root)}):
            with self.assertRaisesRegex(RuntimeError, 'model_integrity_failed'):
                verify()

    def test_failed_inference_never_exposes_content(self):
        self.queue.submit(DepthRequest(**self.payload))
        with patch.object(self.queue.engine, 'infer', side_effect=RuntimeError('private internal path')):
            self.queue.run_one()
        data = self.queue.get(self.payload['idempotency_key'])
        self.assertEqual(data['status'], 'failed')
        self.assertNotIn('private', json.dumps(data))
        with self.assertRaises(HTTPException):
            self.queue.content(self.payload['idempotency_key'], 'depth.mp4')


if __name__ == '__main__':
    unittest.main()
