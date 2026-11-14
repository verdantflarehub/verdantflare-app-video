import json
from pathlib import Path
import tempfile
import time
import unittest
from fastapi.testclient import TestClient
from h3_latent_upscaler.api import create_app
from h3_latent_upscaler.queue import Queue, QueueError
from h3_latent_upscaler.resources import ResourceError


class QueueTests(unittest.TestCase):
    def test_restart_and_idempotency(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = Queue(tmp)
            first = {'idempotency_key': 'video_task_' + 'a' * 32, 'source_video_task_id': 'video_task_' + 'c' * 32, 'project_id': 'p'}
            queued = {**first, 'idempotency_key': 'video_task_' + 'b' * 32}
            self.assertEqual(queue.submit(first), queue.submit(first))
            queue.submit(queued)
            running = queue.take()
            queue.close()
            resumed = Queue(tmp)
            self.assertEqual(resumed.get(running['video_task_id'])['error']['code'], 'interrupted')
            self.assertEqual(resumed.get(queued['idempotency_key'])['status'], 'queued')
            with self.assertRaisesRegex(QueueError, 'idempotency_conflict'):
                resumed.submit({**first, 'project_id': 'another'})
            resumed.close()

    def test_authenticated_task_submission_and_no_mp4_fields(self):
        class Sources:
            def resolve(self, project, source):
                if project != 'p': raise ResourceError('source_not_found')
                return {'manifest': {}}
        class Engine:
            profile = {'id': 'fixture'}
            def validate_request(self, manifest, request): pass
            def generate(self, resources, request, directory):
                (directory / 'output.mp4').write_bytes(b'fixture')
                return {'source_video_task_id': request['source_video_task_id']}
        with tempfile.TemporaryDirectory() as tmp:
            queue = Queue(tmp)
            app = create_app(queue, Sources(), Engine, 'test-token')
            with TestClient(app) as client:
                for _ in range(100):
                    if client.get('/health').status_code == 200: break
                    time.sleep(.01)
                data = {'idempotency_key': 'video_task_' + 'a' * 32, 'project_id': 'p', 'source_video_task_id': 'video_task_' + 'b' * 32}
                self.assertEqual(client.post('/v1/upscales', json=data).status_code, 401)
                headers = {'Authorization': 'Bearer test-token'}
                self.assertEqual(client.post('/v1/upscales', json={**data, 'mp4': 'file.mp4'}, headers=headers).status_code, 422)
                for length in ('bad', '-1'):
                    self.assertEqual(client.post('/v1/upscales', content=b'{}', headers={**headers, 'content-length': length}).status_code, 400)
                self.assertEqual(client.post('/v1/upscales', content=b'x' * 16385, headers={**headers, 'content-length': '0'}).status_code, 413)
                response = client.post('/v1/upscales', json=data, headers=headers)
                self.assertEqual(response.status_code, 200)
                task = response.json()['video_task_id']
                for _ in range(100):
                    status = client.get('/v1/upscales/' + task, headers=headers).json()
                    if status['status'] == 'succeeded': break
                    time.sleep(.01)
                self.assertEqual(status['status'], 'succeeded')
                self.assertEqual(client.post('/v1/upscales', json=data, headers=headers).json()['video_task_id'], task)
            queue.close()
