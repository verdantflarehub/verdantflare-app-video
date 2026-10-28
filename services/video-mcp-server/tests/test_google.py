import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
from PIL import Image

from app.artifacts import ArtifactStore
from app.channel_errors import ChannelError
from app.delivery import CALLER
from app.providers.google import DEFAULT_MODEL, GoogleVideoAdapter
from app.tasks import TaskConflict, TaskStore


class GoogleVideoTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.env = patch.dict('os.environ', {'VIDEO_GOOGLE_CHANNEL_URL': 'http://google-channel:8001',
            'VIDEO_GOOGLE_CHANNEL_TOKEN': 'g' * 40, 'GOOGLE_VIDEO_MODEL': DEFAULT_MODEL,
            'VIDEO_PROJECT_SERVICE_URL': 'http://project', 'VIDEO_PROJECT_SERVICE_TOKEN': 'p' * 40})
        self.env.start()
        self.addCleanup(self.env.stop)
        token = CALLER.set({'organization_id': 'org', 'user_id': 'user'})
        self.addCleanup(CALLER.reset, token)
        self.requests = []
        self.mode = 'success'
        self.channel_call = '11111111-2222-4333-8444-555555555555'
        def upstream(request):
            self.requests.append(request)
            if request.url.host == 'project':
                return httpx.Response(200, json={})
            self.assertEqual(request.headers['X-Organization-Id'], 'org')
            if request.url.path == '/v1/video-preflight':
                if self.mode == 'error':
                    return httpx.Response(503, json={'error': {'code': 'price_table_expired', 'source': 'google',
                        'message': 'The Google video price table has expired.', 'retryable': False}})
                return httpx.Response(200, json={'schema_version': 1, 'channel_id': 'google',
                    'model_id': DEFAULT_MODEL, 'can_submit': True, 'blocked_reason': None,
                    'quote': {'amount': None, 'known_amount': '0.5068', 'unit': 'USD', 'source': 'price_table'}})
            if request.method == 'POST':
                if self.mode == 'timeout':
                    raise httpx.ReadTimeout('synthetic timeout', request=request)
                if self.mode == 'error':
                    return httpx.Response(409, json={'error': 'budget_exceeded', 'accepted': False})
                return httpx.Response(200, json={'call_id': self.channel_call, 'accepted': True,
                    'state': 'in_progress', 'pricing': {'amount': None, 'known_amount': '0.5068'}})
            if request.url.path.endswith('/result'):
                return httpx.Response(200, content=b'google-video-test', headers={'content-type': 'video/mp4'})
            if self.mode == 'failed':
                return httpx.Response(200, json={'call_id': self.channel_call, 'accepted': True, 'state': 'failed',
                    'error': {'code': 'google_invalid_argument', 'source': 'google',
                        'message': 'duration must use a valid video length.', 'retryable': False}})
            if self.mode == 'unknown':
                return httpx.Response(200, json={'call_id': self.channel_call, 'accepted': None,
                    'state': 'acceptance_unknown', 'error': {'code': 'google_transport_error', 'source': 'google',
                        'message': 'Query the original interaction.', 'retryable': True}})
            return httpx.Response(200, json={'call_id': self.channel_call, 'accepted': True, 'state': 'completed',
                'usage': {'total_input_tokens': 100}, 'settlement': {'amount': None, 'unit': 'USD', 'source': 'unknown'}})
        client = httpx.Client(transport=httpx.MockTransport(upstream))
        self.addCleanup(client.close)
        self.executor = SimpleNamespace(client=client, tasks=TaskStore(root), artifacts=ArtifactStore(root),
            require_channel=Mock(), grok=SimpleNamespace(probe=lambda path: {'duration_ms': 5000,
                'width': 1280, 'height': 720, 'frame_rate': 24, 'video_codec': 'h264'}))
        self.adapter = GoogleVideoAdapter(self.executor)

    def inputs(self, **changes):
        return {'project_id': 'demo', 'model': DEFAULT_MODEL, 'prompt': 'test', 'duration_seconds': 5,
                'aspect_ratio': '16:9', 'references': {}, **changes}

    def generate(self, **changes):
        return self.adapter.generate(idempotency_key='one', **self.inputs(**changes))

    def poll(self, task):
        state = dict(task.provider_state or {}, next_poll_at=0)
        return self.adapter.status(self.executor.tasks.update(task, provider_state=state))

    def test_exact_omni_parameters_and_original_result_are_preserved(self):
        task = self.generate()
        self.executor.require_channel.assert_called_once_with('google', DEFAULT_MODEL)
        posted = json.loads([r for r in self.requests if r.url.path == '/v1/calls'][0].content)
        self.assertEqual(posted['model_id'], DEFAULT_MODEL)
        self.assertEqual(posted['payload']['duration_seconds'], 5)
        self.assertEqual(posted['payload']['resolution'], '720p')
        self.assertEqual(posted['budget_limit'], '2')
        self.assertIsNone(task.provider_state['pricing']['amount'])
        complete = self.poll(task)
        self.assertEqual(complete.status, 'succeeded')
        self.assertIsNone(complete.provider_state['settlement']['amount'])
        self.assertEqual(complete.provider_state['usage']['total_input_tokens'], 100)
        self.assertEqual(self.generate().video_task_id, task.video_task_id)
        self.assertEqual(sum(r.url.path == '/v1/calls' for r in self.requests), 1)

    def test_preflight_uses_same_payload_and_does_not_create_task(self):
        request, _ = self.adapter.normalize(**self.inputs())
        result = self.adapter.preflight(request, budget_limit='0')
        self.assertIsNone(result['quote']['amount'])
        self.assertEqual(result['quote']['known_amount'], '0.5068')
        posted = json.loads(self.requests[-1].content)
        self.assertEqual(posted['budget_limit'], '0')
        self.assertEqual(posted['payload']['duration_seconds'], 5)
        self.assertIsNone(self.executor.tasks.find_idempotency('demo', 'one'))
        self.assertFalse(any(r.url.path == '/v1/calls' for r in self.requests))
        self.mode = 'error'
        with self.assertRaises(ChannelError) as caught:
            self.adapter.preflight(request)
        self.assertEqual(caught.exception.code, 'price_table_expired')
        self.assertIn('Google video', caught.exception.public()['message'])

    def test_invalid_inputs_never_call_channel_or_project(self):
        for changes in ({'duration_seconds': 2}, {'duration_seconds': 11}, {'duration_seconds': True},
                        {'resolution': '480p'}, {'resolution': '1080p'}, {'aspect_ratio': '1:1'},
                        {'references': {'videos': [{}]}}, {'references': {'audios': [{}]}},
                        {'references': {'images': [{}, {}]}}, {'mode': 'image_to_video'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.generate(**changes)
        self.assertEqual(self.requests, [])

    def test_first_frame_becomes_one_checked_image(self):
        data = io.BytesIO()
        Image.new('RGB', (16, 9), 'green').save(data, 'PNG')
        asset = self.executor.artifacts.create_from_chunks(project_id='demo', operation='input',
            filename='frame.png', media_type='image/png', chunks=[data.getvalue()])
        task = self.generate(references={'images': [{'artifact_id': asset.artifact_id, 'purpose': 'first_frame'}]})
        posted = json.loads(self.requests[-1].content)['payload']
        self.assertEqual(posted['mode'], 'image_to_video')
        self.assertEqual(posted['image']['mime_type'], 'image/png')
        self.assertNotIn('images', posted)
        self.assertEqual(task.request['duration_seconds'], 5)

    def test_dynamic_admission_precedes_external_calls_and_existing_task_bypasses_it(self):
        task = self.generate()
        self.executor.require_channel.side_effect = ChannelError('channel_unavailable', source='google')
        count = len(self.requests)
        self.assertEqual(self.generate().video_task_id, task.video_task_id)
        self.assertEqual(len(self.requests), count)
        with self.assertRaises(ChannelError):
            self.adapter.generate(idempotency_key='two', **self.inputs())
        self.assertEqual(len(self.requests), count)
        with self.assertRaises(TaskConflict):
            self.adapter.generate(idempotency_key='one', budget_limit='3', **self.inputs())

    def test_missing_verified_actor_never_fabricates_identity(self):
        caller = CALLER.set(None)
        try:
            with self.assertRaises(ChannelError) as caught:
                self.generate()
            self.assertEqual(caught.exception.code, 'unauthorized')
        finally:
            CALLER.reset(caller)
        self.assertEqual(self.requests, [])

    def test_retired_models_do_not_submit_and_original_legacy_task_is_recoverable(self):
        old_inputs = self.inputs(model='veo-3.1-lite-generate-preview', duration_seconds=5)
        with self.assertRaises(ChannelError) as caught:
            self.adapter.generate(idempotency_key='old', **old_inputs)
        self.assertEqual(caught.exception.code, 'model_retired')
        request, digest = self.adapter.normalize(**old_inputs)
        self.assertEqual(request['duration_seconds'], 4)
        old = self.executor.tasks.create(project_id='demo', idempotency_key='old', input_digest=digest,
            request=request, runtime_task_id=self.channel_call, status='running', authority={'organization_id': 'org', 'user_id': 'user'})
        old = self.executor.tasks.update(old, service='google', runtime_route='google', runtime_stage='remote_pending',
            provider_state={'organization_id': 'org', 'budget_limit': '2'})
        self.assertEqual(self.adapter.generate(idempotency_key='old', **old_inputs).video_task_id, old.video_task_id)
        self.assertEqual(self.poll(old).status, 'succeeded')
        self.assertFalse(any(r.method == 'POST' for r in self.requests))

    def test_safe_provider_error_and_unknown_status_survive_projection(self):
        task = self.generate()
        self.mode = 'unknown'
        unknown = self.poll(task)
        self.assertEqual(unknown.status, 'running')
        self.assertEqual(unknown.runtime_stage, 'submission_unknown')
        self.assertEqual(unknown.error['code'], 'google_transport_error')
        self.assertTrue(unknown.error['retryable'])
        self.mode = 'failed'
        failed = self.poll(unknown)
        self.assertEqual(failed.status, 'failed')
        self.assertEqual(failed.runtime_task_id, self.channel_call)
        self.assertEqual(failed.error['code'], 'google_invalid_argument')
        self.assertEqual(failed.error['message'], 'duration must use a valid video length.')
        self.assertTrue(failed.error['accepted'])

    def test_timeout_does_not_repeat_create_and_http_reject_preserves_reason(self):
        self.mode = 'timeout'
        unknown = self.generate()
        self.assertEqual(unknown.runtime_stage, 'submission_unknown')
        self.assertEqual(self.generate().video_task_id, unknown.video_task_id)
        self.assertEqual(sum(r.url.path == '/v1/calls' for r in self.requests), 1)
        self.mode = 'error'
        rejected = self.adapter.generate(idempotency_key='two', **self.inputs())
        self.assertEqual(rejected.status, 'failed')
        self.assertEqual(rejected.error['code'], 'budget_exceeded')

    def test_wrong_actual_duration_does_not_become_success(self):
        task = self.generate()
        self.executor.grok.probe = lambda path: {'duration_ms': 4000, 'width': 1280, 'height': 720}
        failed = self.poll(task)
        self.assertEqual(failed.status, 'failed')
        self.assertEqual(failed.error['code'], 'media_contract_mismatch')
        self.assertEqual(failed.media['duration_ms'], 4000)
        self.assertIsNone(failed.artifact_id)
