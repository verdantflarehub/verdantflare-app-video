import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
import wave
from unittest.mock import patch

import httpx
import test_singularity_route
from app.singularity import validate_timeline_wav, runtime_supports_timeline
from app.tasks import TaskConflict


def wav_bytes(samples=384000, width=2):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(width)
        stream.setframerate(48000)
        stream.writeframes(b'\0' * samples * width)
    return buffer.getvalue()


class TimelineWavTest(unittest.TestCase):
    def test_exact_pcm_length_and_corrupt_or_wrong_formats(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'a.wav'
            path.write_bytes(wav_bytes())
            self.assertEqual(validate_timeline_wav(path, 192)['sample_count'], 384000)
            for width in (2, 3, 4):
                path.write_bytes(wav_bytes(width=width))
                self.assertEqual(validate_timeline_wav(path, 192)['sample_count'], 384000)
            for data, message in [(wav_bytes(383998), 'duration_mismatch'),
                                  (wav_bytes(384002), 'duration_mismatch'),
                                  (wav_bytes(width=1), 'pcm_wav'),
                                  (b'not a wav', 'pcm_wav'),
                                  (wav_bytes()[:-10], 'truncated')]:
                path.write_bytes(data)
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    validate_timeline_wav(path, 192)

    def test_malformed_runtime_declarations_fail_closed(self):
        for health in (None, [], {}, {'capabilities': None}, {'capabilities': {'audio_guidance_modes': None}},
                       {'capabilities': {'audio_guidance_modes': 'timeline'}}):
            self.assertFalse(runtime_supports_timeline(health))


class TimelineRouteTest(unittest.TestCase):
    def setUp(self):
        self.f = test_singularity_route.SingularityRouteTest()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.addCleanup(self.f.tearDown)
        self.e = self.f.executor
        self.supported = True
        self.posts = []
        def respond(request):
            if request.url.path == '/health':
                return httpx.Response(200, json={'ready': True, 'capabilities': {
                    'audio_guidance_modes': ['reference', 'timeline'] if self.supported else ['reference']}})
            if request.method == 'POST':
                self.posts.append(json.loads(request.content))
                return httpx.Response(202, json={'id': 'singularity_'+'c'*32})
            return httpx.Response(200, json={'status': 'queued'})
        self.client = httpx.Client(transport=httpx.MockTransport(respond))
        self.addCleanup(self.client.close)
        self.e.client = self.client
        self.asset = self.f.assets.create_from_chunks(project_id='demo', operation='test', filename='dialogue.wav',
            media_type='audio/wav', chunks=[wav_bytes()])
        self.args = {**self.f.kw, 'route': 'h3-singularity', 'duration_seconds': 8, 'audio_guidance': 'timeline',
                     'references': {**self.f.kw['references'], 'audios': [{'artifact_id': self.asset.artifact_id, 'purpose': 'approved timed dialogue'}]}}
        media = patch('app.media_validation.validate_reference_media', side_effect=lambda p,m:
            {'type': 'image', 'width': 768, 'height': 1344, 'has_audio': False} if m.startswith('image/') else
            {'type': 'audio', 'duration_seconds': 8, 'has_audio': True, 'sample_rate': 48000})
        media.start(); self.addCleanup(media.stop)

    def test_preflight_create_idempotency_and_proof_keep_timeline(self):
        from app import server
        from app.execution_proof import proof
        with patch.object(server, 'executor', self.e):
            preflight = server.video_preflight(**{k:v for k,v in self.args.items() if k != 'idempotency_key'})
        self.assertFalse(preflight.is_error)
        self.assertEqual(preflight.structured_content['audio_guidance'], 'timeline')
        self.assertFalse(self.posts)
        row = self.e.generate(**self.args)
        self.assertEqual(row.input_digest, preflight.structured_content['input_digest'])
        self.assertEqual(row.request['audio_guidance'], 'timeline')
        self.assertEqual(self.posts[0]['audio_guidance'], 'timeline')
        self.assertEqual(len(self.posts[0]['conditions']), 2)
        self.assertEqual(self.e.generate(**self.args).video_task_id, row.video_task_id)
        self.assertEqual(len(self.posts), 1)
        with self.assertRaises(TaskConflict):
            self.e.generate(**{**self.args, 'audio_guidance': None})
        row = self.e.tasks.update(row, status='completed', artifact_id=self.asset.artifact_id)
        self.assertEqual(proof(row, self.f.assets)['parameters']['audio_guidance'], 'timeline')

    def test_old_runtime_blocks_without_post(self):
        from app import server
        self.supported = False
        with patch.object(server, 'executor', self.e):
            preview = server.video_preflight(**{k:v for k,v in self.args.items() if k != 'idempotency_key'})
        self.assertTrue(preview.is_error)
        row = self.e.generate(**self.args)
        self.assertEqual(row.error['code'], 'timeline_audio_not_supported')
        self.assertFalse(self.posts)

    def test_other_routes_and_bad_track_count_rejected_before_dispatch(self):
        from app import server
        for route in ['fal', 'grok', 'google', 'higgsfield', 'h3', 'h3-sol']:
            args = {**self.args, 'route': route}
            with self.subTest(route=route), self.assertRaisesRegex(ValueError, 'requires_h3_singularity'):
                self.e.generate(**args)
            with patch.object(server, 'executor', self.e):
                result = server.video_preflight(**{k:v for k,v in args.items() if k != 'idempotency_key'})
            self.assertTrue(result.is_error)
        for refs in [self.f.kw['references'], {**self.args['references'], 'audios': self.args['references']['audios'] * 2}]:
            with self.assertRaisesRegex(ValueError, 'one_audio_and_no_video'):
                self.e.generate(**{**self.args, 'references': refs})
        self.assertFalse(self.posts)

    def test_recovery_dispatch_retains_timeline_and_runtime_key(self):
        option = {'interpolate': {'profile': 'rife_2x'}}
        with patch('app.delivery.authorize', return_value={'user_id':'user','organization_id':'org'}), \
                patch.object(self.e.delivery, 'capability', return_value={'configured':True,'readiness':'ready'}):
            row = self.e.generate(**self.args, postprocess=option)
            row = self.e.tasks.update(row, runtime_task_id='', status='queued')
            recovered_posts = []
            def respond(request):
                if request.method == 'POST':
                    recovered_posts.append(json.loads(request.content))
                    return httpx.Response(202,json={'id':'singularity_'+'d'*32})
                if '/by-idempotency/' in request.url.path:
                    return httpx.Response(404)
                return httpx.Response(200,json={'status':'queued'})
            with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                self.e.client = client
                recovered = self.e.status(row.video_task_id)
            self.assertEqual(recovered.runtime_task_id, 'singularity_'+'d'*32)
            self.assertEqual(recovered_posts[0]['audio_guidance'], 'timeline')
            self.assertEqual(recovered_posts[0]['idempotency_key'], row.video_task_id)
            self.assertEqual(recovered_posts[0]['conditions'], self.posts[0]['conditions'])

    def test_capability_does_not_advertise_unsupported_runtime_and_schema_has_field(self):
        from app import server
        for supported in (False, True):
            self.supported = supported
            with patch.object(self.e, 'catalog_view', return_value={}):
                cap = next(r for r in self.e.capabilities()['routes'] if r['route']=='h3-singularity')
            self.assertEqual('timeline' in cap['audio_guidance']['modes'], supported)
            self.assertFalse(cap['audio_guidance']['lip_sync_verified'])
        schemas = {t['name']: t['inputSchema'] for t in server.VIDEO_TOOLS_SCHEMA}
        for name in ['video.create', 'video.preflight']:
            self.assertIn('audio_guidance', schemas[name]['properties'])
