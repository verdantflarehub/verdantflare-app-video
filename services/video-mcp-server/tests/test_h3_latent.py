import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import httpx
from verdantflare_video_mcp.artifacts import ArtifactStore
from verdantflare_video_mcp.tasks import TaskStore
from verdantflare_video_mcp.executor import VideoExecutor, ExecutionError


class H3LatentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {'H3_LATENT_RUNTIME_TOKEN': 'test-only',
            'H3_LATENT_NODE_NAME': 'node-a', 'H3_LATENT_SOURCE_INDEX': str(self.root / 'index.sqlite'),
            'H3_VDN_RUNTIME_TOKEN': 'vdn-test', 'H3_RUNTIME_ROUTES': json.dumps({'h3-vdn': {
                'url': 'http://vdn', 'service': 'h3-vdn', 'version': 'test', 'task': 'ref2va'}})})
        self.env.start(); self.addCleanup(self.env.stop)
        self.tasks = TaskStore(self.root)
        self.source = self.tasks.create(project_id='project-a', idempotency_key='source', input_digest='source',
            request={'route': 'h3-vdn'}, runtime_task_id='vdn_' + 'b' * 32, status='succeeded')
        self.source = self.tasks.update(self.source, service='h3-vdn', runtime_route='h3-vdn')
        self.posts = []
        self.bundle = {'schema': 'h3-latent-bundle/v1', 'source_video_task_id': self.source.video_task_id,
                       'node': 'node-a', 'manifest_path': 'h3-vdn/tasks/source/manifest.json', 'manifest_sha256': 'a' * 64}
        def handler(request):
            if request.method == 'GET' and request.url.host == 'vdn':
                return httpx.Response(200, json={'id': self.source.runtime_task_id, 'status': 'completed',
                                                'result': {'latent_bundle': self.bundle}})
            if request.method == 'POST':
                payload = json.loads(request.content)
                self.posts.append(payload)
                return httpx.Response(200, json={'video_task_id': payload['idempotency_key'], 'status': 'queued'})
            return httpx.Response(404)
        self.executor = VideoExecutor(ArtifactStore(self.root), self.tasks, httpx.Client(transport=httpx.MockTransport(handler)))
        self.adapter = self.executor.processing['h3-latent-upscale']

    def test_task_id_only_resolves_source_and_deduplicates(self):
        first = self.adapter.generate(self.source.video_task_id)
        second = self.adapter.generate(self.source.video_task_id)
        self.assertEqual(first.video_task_id, second.video_task_id)
        self.assertEqual(len(self.posts), 1)
        self.assertEqual(self.posts[0]['source_video_task_id'], self.source.video_task_id)
        self.assertNotIn('source_artifact_id', self.posts[0])
        self.assertEqual(first.service, 'h3-latent-upscale')

    def test_restart_after_first_persisted_write_recovers_same_task_and_route(self):
        # Simulate a process dying after the durable create and before HTTP POST.
        with patch.object(self.adapter, 'submit', side_effect=RuntimeError('simulated process exit')):
            with self.assertRaisesRegex(RuntimeError, 'simulated process exit'):
                self.adapter.generate(self.source.video_task_id)
        saved = [json.loads(path.read_text()) for path in self.tasks.root.glob('video_task_*.json')
                 if path.stem != self.source.video_task_id]
        self.assertEqual(len(saved), 1)
        record = saved[0]
        self.assertEqual(record['service'], 'h3-latent-upscale')
        self.assertEqual(record['runtime_task_id'], record['video_task_id'])
        self.assertEqual(record['error']['code'], 'submission_unconfirmed')
        recovered = VideoExecutor(ArtifactStore(self.root), TaskStore(self.root), self.executor.client)
        result = recovered.status(record['video_task_id'])
        self.assertEqual(result.video_task_id, record['video_task_id'])
        self.assertEqual(result.status, 'queued')
        self.assertEqual(len(self.posts), 1)
        self.assertEqual(self.posts[0]['idempotency_key'], record['video_task_id'])
        self.assertEqual(self.posts[0]['source_video_task_id'], self.source.video_task_id)

    def test_selected_project_cannot_read_other_source(self):
        with self.assertRaisesRegex(ExecutionError, 'source_not_found'):
            self.adapter.generate(self.source.video_task_id, project_id='project-b')
        self.assertFalse(self.posts)

    def test_mp4_only_rejected_before_queue(self):
        self.bundle = None
        with self.assertRaisesRegex(ExecutionError, 'missing_latent_bundle'):
            self.adapter.generate(self.source.video_task_id)
        self.assertFalse(self.posts)

    def test_other_node_rejected(self):
        self.bundle['node'] = 'node-b'
        with self.assertRaisesRegex(ExecutionError, 'source_node_mismatch'):
            self.adapter.generate(self.source.video_task_id)
        self.assertFalse(self.posts)
