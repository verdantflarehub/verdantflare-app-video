import json
import unittest
from unittest.mock import patch
import httpx
import test_timeline_audio
from app.singularity import runtime_supports_first_frame
from app.tasks import TaskConflict


class ImageGuidanceTests(unittest.TestCase):
    def setUp(self):
        self.f = test_timeline_audio.TimelineRouteTest()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.e = self.f.e
        self.supported = True
        self.posts = []
        def respond(req):
            if req.url.path == '/health':
                return httpx.Response(200, json={'ready': True, 'capabilities': {
                    'audio_guidance_modes': ['reference', 'timeline'],
                    'image_guidance_modes': ['reference', 'first_frame'] if self.supported else ['reference']}})
            if '/by-idempotency/' in req.url.path:
                return httpx.Response(404)
            if req.method == 'POST':
                self.posts.append(json.loads(req.content))
                return httpx.Response(202, json={'id': 'singularity_' + 'c' * 32})
            return httpx.Response(200, json={'status': 'queued'})
        self.client = httpx.Client(transport=httpx.MockTransport(respond))
        self.addCleanup(self.client.close)
        self.e.client = self.client
        self.args = {**self.f.args, 'image_guidance': 'first_frame'}

    def preflight(self, args=None):
        from app import server
        with patch.object(server, 'executor', self.e):
            return server.video_preflight(**{k:v for k,v in (args or self.args).items() if k != 'idempotency_key'})

    def test_preflight_dispatch_proof_and_idempotency(self):
        from app.execution_proof import proof
        preview = self.preflight()
        self.assertFalse(preview.is_error)
        self.assertEqual(preview.structured_content['image_guidance'], 'first_frame')
        row = self.e.generate(**self.args)
        self.assertEqual(row.input_digest, preview.structured_content['input_digest'])
        self.assertEqual(self.posts[0]['image_guidance'], 'first_frame')
        self.assertEqual(self.posts[0]['audio_guidance'], 'timeline')
        self.assertEqual(self.posts[0]['conditions'][0]['sha256'], self.f.f.assets.get(self.args['references']['images'][0]['artifact_id'], 'demo').sha256)
        self.assertEqual(self.e.generate(**self.args).video_task_id, row.video_task_id)
        with self.assertRaises(TaskConflict):
            self.e.generate(**{**self.args, 'image_guidance': None})
        row = self.e.tasks.update(row, status='completed', artifact_id=self.f.asset.artifact_id)
        self.assertEqual(proof(row, self.f.f.assets)['parameters']['image_guidance'], 'first_frame')
        self.assertEqual(len(self.posts), 1)

    def test_public_create_and_generate_forward_first_frame(self):
        from app import server
        with patch.object(server, 'executor', self.e):
            for index, entry in enumerate((server.video_create, server.video_generate)):
                args = {**self.args, 'idempotency_key': 'public-first-frame-' + str(index), 'title': 'First-frame check'}
                result = entry(**args)
                self.assertFalse(result.is_error)
                self.assertEqual(self.posts[-1]['image_guidance'], 'first_frame')
                self.assertEqual(self.posts[-1]['audio_guidance'], 'timeline')
        self.assertEqual(len(self.posts), 2)

    def test_default_unchanged(self):
        kw = {k:v for k,v in self.f.args.items() if k != 'idempotency_key'}
        first = self.e._normalize(**kw)
        self.assertEqual(first, self.e._normalize(**kw, image_guidance=None))
        self.assertNotIn('image_guidance', first[0])

    def test_missing_old_runtime_blocks_without_submit(self):
        self.supported = False
        self.assertTrue(self.preflight().is_error)
        row = self.e.generate(**self.args)
        self.assertEqual(row.error['code'], 'first_frame_not_supported')
        self.assertFalse(self.posts)
        for health in (None, [], {}, {'capabilities': None}, {'capabilities': {'image_guidance_modes': 'first_frame'}}):
            self.assertFalse(runtime_supports_first_frame(health))

    def test_wrong_canvas_and_missing_image_rejected(self):
        args = {**self.args, 'aspect_ratio': '16:9'}
        self.assertTrue(self.preflight(args).is_error)
        with self.assertRaisesRegex(ValueError, 'canvas_mismatch'):
            self.e.generate(**args)
        args = {**self.args, 'references': {'audios': self.args['references']['audios']}}
        with self.assertRaisesRegex(ValueError, 'requires_image'):
            self.e.generate(**args)
        self.assertFalse(self.posts)

    def test_other_routes_rejected(self):
        for route in ('fal', 'grok', 'google', 'higgsfield', 'h3', 'h3-sol'):
            args = {**self.args, 'route': route, 'audio_guidance': None}
            with self.assertRaisesRegex(ValueError, 'image_guidance_requires_h3_singularity'):
                self.e.generate(**args)
            self.assertTrue(self.preflight(args).is_error)
        self.assertFalse(self.posts)

    def test_recovery_keeps_image_and_audio(self):
        with patch('app.delivery.authorize', return_value={'user_id':'user','organization_id':'org'}), patch.object(self.e.delivery, 'capability', return_value={'configured':True,'readiness':'ready'}):
            row = self.e.generate(**self.args, postprocess={'interpolate': {'profile': 'rife_2x'}})
            row = self.e.tasks.update(row, runtime_task_id='', status='queued')
            recovered = self.e.status(row.video_task_id)
        self.assertEqual(recovered.runtime_task_id, 'singularity_'+'c'*32)
        self.assertEqual(self.posts[0], self.posts[1])

    def test_recovery_on_downgraded_runtime_does_not_dispatch(self):
        from app.executor import ExecutionError
        with patch('app.delivery.authorize', return_value={'user_id':'user','organization_id':'org'}), patch.object(self.e.delivery, 'capability', return_value={'configured':True,'readiness':'ready'}):
            row = self.e.generate(**self.args, postprocess={'interpolate': {'profile': 'rife_2x'}})
            row = self.e.tasks.update(row, runtime_task_id='', status='queued')
            self.supported = False
            with self.assertRaises(ExecutionError):
                self.e.status(row.video_task_id)
        self.assertEqual(len(self.posts), 1)

    def test_capabilities_and_registered_schemas(self):
        from app import server
        for supported in (False, True):
            self.supported = supported
            with patch.object(self.e, 'catalog_view', return_value={}):
                cap = next(r for r in self.e.capabilities()['routes'] if r['route']=='h3-singularity')['image_guidance']
            self.assertEqual('first_frame' in cap['modes'], supported)
            self.assertFalse(cap['gpu_verified'])
        for item in server.VIDEO_TOOLS_SCHEMA:
            if item['name'] in ('video.create', 'video.preflight'):
                self.assertIn('image_guidance', item['inputSchema']['properties'])
