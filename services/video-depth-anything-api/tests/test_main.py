import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import main


class DepthApiTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def test_health(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'ok')

    def test_rejects_unknown_model(self):
        response = self.client.post('/v1/depth', json={'project_id': 'p', 'idempotency_key': 'k', 'source_artifact_id': 'x', 'model': 'midas'})
        self.assertEqual(response.status_code, 400)

    def test_missing_source(self):
        response = self.client.post('/v1/depth', json={'project_id': 'p', 'idempotency_key': 'k', 'source_artifact_id': '/missing.mp4'})
        self.assertEqual(response.status_code, 404)

    @patch.object(main, '_pipe')
    @patch.object(main.cv2, 'VideoCapture')
    @patch.object(main.cv2, 'VideoWriter')
    def test_processes_frames(self, writer_cls, capture_cls, pipe):
        with tempfile.NamedTemporaryFile(suffix='.mp4') as source:
            capture = capture_cls.return_value
            capture.get.side_effect = [30.0, 2, 2]
            capture.read.side_effect = [(True, object()), (False, None)]
            writer = writer_cls.return_value
            image = type('Depth', (), {'resize': lambda self, size: [[1, 2], [3, 4]]})()
            pipe.return_value.return_value = {'depth': image}
            with patch.object(main.cv2, 'cvtColor', return_value=object()), patch.object(main.np, 'asarray', return_value=main.np.array([[1, 2], [3, 4]])):
                response = self.client.post('/v1/depth', json={'project_id': 'p', 'idempotency_key': 'k', 'source_artifact_id': source.name})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['frames'], 1)
            writer.write.assert_called_once()


if __name__ == '__main__':
    unittest.main()
