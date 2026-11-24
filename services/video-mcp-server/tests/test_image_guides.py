import copy
import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import httpx
import jsonschema
from app.artifacts import ArtifactStore
from app.delivery import CALLER
from app.executor import VideoExecutor
from app.image_guides import runtime_supports
from app.tasks import TaskConflict, TaskStore


class ImageGuidesTests(unittest.TestCase):
    def setUp(self):
        self.project, self.user, self.org = (
            '00000000-0000-7000-8000-'+f'{n:012x}' for n in range(1, 4))
        self.actor = {'user_id': self.user, 'organization_id': self.org}
        token = CALLER.set(dict(self.actor))
        self.addCleanup(CALLER.reset, token)
        self.enterContext(patch.dict(os.environ, {
            'VIDEO_PROJECT_SERVICE_URL': 'https://project.example.test',
            'VIDEO_PROJECT_SERVICE_TOKEN': 'test-project-token'*2,
            'H3_RUNTIME_URL': 'http://fasth3.example:8000',
            'H3_RUNTIME_ROUTE': 'h3-fasth3',
            'H3_FASTH3_RUNTIME_TOKEN': 'test-fasth3-token',
            'VIDEO_INTERPOLATION_RUNTIME_URL': 'http://interpolate.example:8000',
        }))
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        assets, tasks = ArtifactStore(root), TaskStore(root)
        self.f = SimpleNamespace(assets=assets, tasks=tasks, kw={
            'project_id': self.project, 'idempotency_key': 'guide-attempt',
            'model': 'minimax-h3-ref2va', 'prompt': 'Approved motion',
            'duration_seconds': 5, 'aspect_ratio': '9:16',
        })
        self.supported = True
        self.posts = []
        self.project_write = True
        self.revoke_on_health = False
        self.health_reads = 0
        self.by_key = None
        def respond(req):
            if req.url.host == 'project.example.test':
                self.assertEqual(req.headers['authorization'], 'Bearer '+'test-project-token'*2)
                self.assertEqual(req.headers['x-user-id'], self.user)
                self.assertEqual(req.headers['x-organization-id'], self.org)
                if req.url.path == '/internal/v1/project/open':
                    self.assertEqual(json.loads(req.content)['project_id'], self.project)
                    return httpx.Response(200 if self.project_write else 403, json={})
                self.assertEqual(req.url.path, '/internal/v1/project/read_scope')
                reply = {
                    'items': [{'project_id': self.project, 'can_write': self.project_write}],
                    'next_cursor': '', 'authorization_revision': 'a'*64,
                }
                return httpx.Response(200, headers={'content-type': 'application/json'},
                    stream=httpx.ByteStream(json.dumps(reply).encode()))
            if req.url.host == 'interpolate.example':
                self.assertEqual(req.url.path, '/health')
                return httpx.Response(200, json={'status': 'ok', 'service': 'interpolate'})
            if req.url.path == '/health':
                self.health_reads += 1
                if self.revoke_on_health:
                    self.project_write = False
                caps = {'image_guidance_modes': ['reference', 'first_frame']}
                if self.supported:
                    caps['image_guides'] = {'schema': 'single-frame-v1', 'maximum': 8, 'index_base': 0}
                return httpx.Response(200, json={'ready': True, 'capabilities': caps})
            if '/by-idempotency/' in req.url.path:
                return httpx.Response(200, json=self.by_key) if self.by_key else httpx.Response(404)
            if req.method == 'POST':
                self.posts.append(json.loads(req.content))
                return httpx.Response(202, json={'id': 'fasth3_'+'c'*32})
            return httpx.Response(200, json={'status': 'queued'})
        self.client = httpx.Client(transport=httpx.MockTransport(respond))
        self.addCleanup(self.client.close)
        self.e = VideoExecutor(assets, tasks, self.client)
        project = self.f.kw['project_id']
        def asset(name, mime, data):
            return self.f.assets.create_from_chunks(project_id=project, operation='test', filename=name,
                media_type=mime, chunks=[data]).artifact_id
        source = asset('source.png', 'image/png', b'full-photo')
        first = asset('first.png', 'image/png', b'first-photo')
        ending = asset('ending.png', 'image/png', b'ending-photo')
        motion = asset('motion.mp4', 'video/mp4', b'motion')
        self.guides = [{'image_index': 1, 'frame_index': 0}, {'image_index': 2, 'frame_index': 110},
                       {'image_index': 2, 'frame_index': 123}]
        self.args = {**self.f.kw, 'route': 'h3-fasth3', 'quality_profile': 'fasth3-nvfp4',
                     'image_guidance': 'reference', 'image_guides': self.guides,
                     'references': {'images': [{'artifact_id': x, 'purpose': 'approved photo'} for x in (source, first, ending)],
                                    'videos': [{'artifact_id': motion, 'purpose': 'stick camera motion only'}]}}
        def media(path, mime):
            if mime.startswith('video/'):
                return {'type': 'video', 'width': 768, 'height': 1344, 'duration_seconds': 124/24,
                        'resampled_frames': 124, 'fps': 24, 'has_audio': False}
            width, height = (1672, 941) if path.read_bytes() == b'full-photo' else (768, 1344)
            return {'type': 'image', 'width': width, 'height': height, 'has_audio': False}
        self.enterContext(patch('app.media_validation.validate_reference_media', side_effect=media))

    def preflight(self, args=None):
        from app import server
        with patch.object(server, 'executor', self.e):
            return server.video_preflight(**{k: v for k, v in (args or self.args).items() if k != 'idempotency_key'})

    def uncertain_postprocess_intent(self):
        # Real local create/preparation against HTTP peers, then simulate a lost receipt.
        row = self.e.generate(**self.args, postprocess={'interpolate': {'profile': 'rife_2x'}})
        self.assertEqual(row.authority, self.actor)
        self.assertEqual(row.request['postprocess'], {'interpolate': {'profile': 'rife_2x'}})
        return self.e.tasks.update(row, runtime_task_id='', status='queued')

    def test_preflight_dispatch_rebuild_proof_and_changed_key_conflict(self):
        from app.execution_proof import proof
        preview = self.preflight()
        self.assertFalse(preview.is_error, preview.content)
        self.assertEqual(preview.structured_content['image_guides'], self.guides)
        row = self.e.generate(**self.args)
        self.assertEqual(row.input_digest, preview.structured_content['input_digest'])
        self.assertEqual(self.posts[0]['image_guides'], self.guides)
        self.assertEqual([x['type'] for x in self.posts[0]['conditions']], ['image']*3+['video'])
        self.assertEqual(self.e._build_local_generation_payload(row), self.posts[0])
        self.assertEqual(self.e.generate(**self.args).video_task_id, row.video_task_id)
        for guides in (None, [{'image_index': 2, 'frame_index': 109}, self.guides[-1]]):
            with self.assertRaises(TaskConflict):
                self.e.generate(**{**self.args, 'image_guides': guides})
        row = self.e.tasks.update(row, status='completed', artifact_id=self.args['references']['videos'][0]['artifact_id'])
        self.assertEqual(proof(row, self.f.assets)['parameters']['image_guides'], self.guides)
        self.assertEqual(len(self.posts), 1)

    def test_public_create_generate_forward_without_coercion(self):
        from app import server
        with patch.object(server, 'executor', self.e):
            for i, entry in enumerate((server.video_create, server.video_generate)):
                result = entry(**{**self.args, 'idempotency_key': 'public-guide-'+str(i), 'title': 'Camera guide'})
                self.assertFalse(result.is_error, result.content)
                self.assertEqual(self.posts[-1]['image_guides'], self.guides)
            for value in (True, 110.0, '110'):
                guides = [self.guides[0], {'image_index': 2, 'frame_index': value}]
                result = server.video_create(**{**self.args, 'idempotency_key': 'bad-'+str(value), 'image_guides': guides})
                self.assertTrue(result.is_error)
        self.assertEqual(len(self.posts), 2)

    def test_null_is_absent_for_old_routes_and_old_digest(self):
        args = {k: v for k, v in self.args.items() if k not in ('idempotency_key', 'image_guides')}
        for route, profile in (('h3-fasth3', 'fasth3-nvfp4'), ('h3-singularity', 'du-0'), ('h3', 'du-0')):
            kw = {**args, 'route': route, 'quality_profile': profile, 'image_guidance': None}
            old = self.e._normalize(**kw)
            self.assertEqual(old, self.e._normalize(**kw, image_guides=None))
            self.assertNotIn('image_guides', old[0])

    def test_missing_capability_blocks_dispatch_but_not_original_task_read(self):
        row = self.e.generate(**self.args)
        self.supported = False
        self.assertEqual(self.e.status(row.video_task_id).video_task_id, row.video_task_id)
        self.assertTrue(self.preflight().is_error)
        other = self.e.generate(**{**self.args, 'idempotency_key': 'unsupported'})
        self.assertEqual(other.error['code'], 'image_guides_not_supported')
        self.assertEqual(len(self.posts), 1)
        for health in (None, {}, {'capabilities': None}, {'capabilities': {'image_guides': True}},
                       {'capabilities': {'image_guides': {'schema': 'single-frame-v1', 'maximum': True, 'index_base': 0}}}):
            self.assertFalse(runtime_supports(health))

    def test_range_order_canvas_and_conflicting_mode_block_pre_submit(self):
        invalid = [[], self.guides[::-1], self.guides+[self.guides[-1]],
                   [{'image_index': 0, 'frame_index': 0}], [{'image_index': 3, 'frame_index': 0}],
                   [{'image_index': 2, 'frame_index': 124}], [{'image_index': 2, 'frame_index': 0, 'frames': 14}]]
        for guides in invalid:
            with self.subTest(guides=guides), self.assertRaises(ValueError):
                self.e.generate(**{**self.args, 'image_guides': guides})
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            self.e.generate(**{**self.args, 'image_guidance': 'first_frame'})
        self.assertFalse(self.posts)

    def test_other_routes_and_processing_preflight_reject_nonnull(self):
        from app import server
        for route in ('h3-singularity', 'h3-sol', 'fal', 'grok', 'google', 'higgsfield', 'comfyui'):
            with self.subTest(route=route), self.assertRaises(ValueError):
                self.e.generate(**{**self.args, 'route': route, 'image_guidance': None})
        with patch.object(server, 'executor', self.e):
            for extra in ({'source_artifact_id': 'art_'+'a'*32}, {'route': 'regenerate_2k'},
                          {'mode': 'character_swap'}, {'character_image_artifact_id': 'art_'+'b'*32},
                          {'route': 'lipsync'}, {'mode': 'lipsync'}, {'model': 'latentsync-1.6'},
                          {'audio_artifact_id': 'art_'+'d'*32}):
                result = server.video_preflight(**{**{k: v for k, v in self.args.items() if k != 'idempotency_key'}, **extra})
                self.assertTrue(result.is_error)
        self.assertFalse(self.posts)

    def test_registered_schema_rejects_extra_keys_and_noninteger_types(self):
        from app import server
        for entry in server.VIDEO_TOOLS_SCHEMA:
            if entry['name'] not in ('video.create', 'video.preflight', 'video.generate'):
                continue
            schema = entry['inputSchema']['properties']['image_guides']
            jsonschema.validate(self.guides, schema)
            jsonschema.validate(None, schema)
            for value in (True, '110', 110.5):
                with self.assertRaises(jsonschema.ValidationError):
                    jsonschema.validate([{'image_index': 2, 'frame_index': value}], schema)
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate([{'image_index': 2, 'frame_index': 110, 'frames': 14}], schema)

    def test_recovery_preserves_original_payload_and_capability_is_rechecked(self):
        # Existing recovery rules permit redispatch only for the original postprocess-enabled request.
        row = self.uncertain_postprocess_intent()
        request, digest = copy.deepcopy(row.request), row.input_digest
        with patch.object(self.e, 'acquire_local_gpu', return_value=True):
            recovered = self.e.status(row.video_task_id)
        self.assertEqual(recovered.runtime_task_id, 'fasth3_'+'c'*32)
        self.assertEqual(self.posts[0], self.posts[1])
        self.assertEqual(recovered.request, request)
        self.assertEqual(recovered.input_digest, digest)
        self.assertEqual(recovered.authority, self.actor)
        self.supported = False
        row = self.e.tasks.update(recovered, runtime_task_id='', status='queued')
        from app.executor import ExecutionError
        with self.assertRaisesRegex(ExecutionError, 'remains unconfirmed') as caught:
            self.e.status(row.video_task_id)
        self.assertEqual(str(caught.exception.__cause__), 'image_guides_not_supported_or_not_ready')
        self.assertEqual(len(self.posts), 2)

    def test_nonpostprocess_unknown_original_never_redispatches(self):
        row = self.e.generate(**self.args)
        self.e.tasks.update(row, runtime_task_id='', status='queued')
        recovered = self.e.status(row.video_task_id)
        self.assertEqual(recovered.error['code'], 'submission_unconfirmed')
        self.assertFalse(recovered.runtime_task_id)
        self.assertEqual(len(self.posts), 1)

    def test_found_original_does_not_require_new_guide_capability(self):
        row = self.e.generate(**self.args)
        self.e.tasks.update(row, runtime_task_id='', status='queued')
        self.by_key = {'id': 'fasth3_'+'c'*32}
        self.supported = False
        reads = self.health_reads
        recovered = self.e.status(row.video_task_id)
        self.assertEqual(recovered.runtime_task_id, self.by_key['id'])
        self.assertEqual(recovered.request['image_guides'], self.guides)
        self.assertEqual(self.health_reads, reads)
        self.assertEqual(len(self.posts), 1)

    def test_permission_revoked_during_new_health_check_blocks_original_replay(self):
        row = self.uncertain_postprocess_intent()
        self.revoke_on_health = True
        with self.assertRaises(PermissionError):
            self.e.status(row.video_task_id)
        self.assertFalse(self.project_write)
        self.assertEqual(len(self.posts), 1)

    def test_background_recovery_uses_saved_actor_after_restart(self):
        row = self.uncertain_postprocess_intent()
        request, digest = copy.deepcopy(row.request), row.input_digest
        restarted = VideoExecutor(self.f.assets, TaskStore(self.f.tasks.root.parent), self.client)
        token = CALLER.set(None)
        try:
            with patch.object(restarted, 'acquire_local_gpu', return_value=True):
                recovered = restarted.status(row.video_task_id)
        finally:
            CALLER.reset(token)
        self.assertEqual(recovered.authority, self.actor)
        self.assertEqual(recovered.request, request)
        self.assertEqual(recovered.input_digest, digest)
        self.assertEqual(self.posts[0], self.posts[1])
        self.assertEqual(len(self.posts), 2)

    def test_background_recovery_rechecks_revoked_saved_actor(self):
        row = self.uncertain_postprocess_intent()
        self.revoke_on_health = True
        token = CALLER.set(None)
        try:
            with self.assertRaises(PermissionError):
                self.e.status(row.video_task_id)
        finally:
            CALLER.reset(token)
        self.assertFalse(self.project_write)
        self.assertEqual(len(self.posts), 1)

    def test_background_recovery_without_saved_actor_cannot_replay(self):
        row = self.uncertain_postprocess_intent()
        self.e.tasks.update(row, authority=None)
        from app.executor import ExecutionError
        token = CALLER.set(None)
        try:
            with self.assertRaises(ExecutionError) as caught:
                self.e.status(row.video_task_id)
        finally:
            CALLER.reset(token)
        self.assertIsInstance(caught.exception.__cause__, ValueError)
        self.assertEqual(str(caught.exception.__cause__), 'verified Studio identity required')
        self.assertEqual(len(self.posts), 1)

    def test_actual_tool_runner_does_not_cast_bool_float_or_string(self):
        from app import server
        with patch.object(server, 'executor', self.e):
            for value in (True, 110.0, '110'):
                args = {**self.args, 'idempotency_key': 'typed-'+str(value),
                        'image_guides': [{'image_index': 2, 'frame_index': value}]}
                result = asyncio.run(server.mcp._tool_manager.call_tool('video.create', args, None))
                self.assertTrue(result.is_error)
        self.assertFalse(self.posts)

    def test_capability_requires_real_complete_runtime_declaration(self):
        for supported in (False, True):
            self.supported = supported
            with patch.object(self.e, 'catalog_view', return_value={}):
                route = next(r for r in self.e.capabilities()['routes'] if r['route'] == 'h3-fasth3')
            self.assertEqual('image_guides' in route, supported)
            if supported:
                self.assertFalse(route['image_guides']['gpu_verified'])
                self.assertFalse(route['image_guides']['hard_geometry_lock'])
