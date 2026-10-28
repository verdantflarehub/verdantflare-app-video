"""fal admission and execution use identical payloads and keep safe diagnostics."""
import hashlib
import json
import unittest
from unittest import mock

import httpx

from app.executor import VideoExecutor
from app.channel_errors import ChannelError
import test_fal_route


class FalPreflightTest(unittest.TestCase):
    def setUp(self):
        self.fixture = test_fal_route.FalRouteTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)

    def executor(self, handler):
        client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        return VideoExecutor(self.fixture.artifacts, self.fixture.tasks, client)

    def preflight_inputs(self, **updates):
        value = self.fixture.inputs(**updates)
        value.pop('idempotency_key')
        value.pop('route')
        return value

    @staticmethod
    def quote(request, *, allowed=True, reason=None):
        body = json.loads(request.content)
        digest = hashlib.sha256(json.dumps([body['model_id'], body['payload']], sort_keys=True,
            separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
        return httpx.Response(200, json={'schema_version': 1, 'channel_id': 'fal', 'model_id': body['model_id'],
            'can_submit': allowed, 'blocked_reason': reason, 'pricing': {'model_id': body['model_id'],
            'input_digest': digest, 'estimate': {'amount': '0.25', 'unit': 'USD', 'source': 'official_price_table'}}})

    def test_preflight_payload_equals_create_without_task_or_reservation(self):
        seen = []
        def handler(request):
            seen.append(request)
            if request.url.path == '/v1/video-preflight':
                return self.quote(request)
            return httpx.Response(200, json={'call_id': 'fal-original', 'accepted': True, 'state': 'queued'})
        executor = self.executor(handler)
        before = list(self.fixture.tasks.root.glob('*.json'))
        quoted = executor.fal.preflight(**self.preflight_inputs())
        self.assertTrue(quoted['can_submit'])
        self.assertEqual(list(self.fixture.tasks.root.glob('*.json')), before)
        executor.generate(**self.fixture.inputs())
        self.assertEqual([r.url.path for r in seen], ['/v1/video-preflight', '/v1/calls'])
        first, second = [json.loads(r.content) for r in seen]
        self.assertEqual(first['payload'], second['payload'])
        self.assertEqual(first['model_id'], second['model_id'])
        self.assertEqual(first['budget_limit'], second['budget_limit'])

    def test_historical_lock_returns_specific_error_without_create(self):
        executor = self.executor(lambda request: self.quote(request, allowed=False, reason='historical_cost_unknown'))
        result = executor.fal.preflight(**self.preflight_inputs())
        self.assertFalse(result['can_submit'])
        self.assertEqual(result['error']['code'], 'historical_cost_unknown')
        self.assertEqual(result['error']['source'], 'fal')
        self.assertFalse(result['error']['accepted'])
        self.assertFalse(list(self.fixture.tasks.root.glob('*.json')))

    def test_registry_gate_blocks_new_calls_but_not_existing_idempotency(self):
        calls = []
        executor = self.executor(lambda request: calls.append(request) or httpx.Response(200,
            json={'call_id': 'fal-original', 'accepted': True, 'state': 'queued'}))
        first = executor.generate(**self.fixture.inputs())
        with mock.patch.object(executor, 'require_channel', side_effect=ChannelError('channel_unavailable', source='fal', accepted=False)):
            self.assertEqual(executor.generate(**self.fixture.inputs()).video_task_id, first.video_task_id)
            with self.assertRaises(ChannelError):
                executor.generate(**self.fixture.inputs(idempotency_key='new'))
        self.assertEqual(len(calls), 1)

    def test_accepted_false_retains_provider_quota_error(self):
        executor = self.executor(lambda request: httpx.Response(200, json={'call_id': 'fal-rejected',
            'accepted': False, 'state': 'rejected', 'reason_code': 'provider_quota_exhausted', 'error': {
                'code': 'provider_quota_exhausted', 'message': 'fal account has insufficient credits or quota',
                'source': 'fal', 'retryable': False, 'accepted': False}}))
        result = executor.generate(**self.fixture.inputs())
        self.assertEqual(result.status, 'failed')
        self.assertEqual(result.error['code'], 'provider_quota_exhausted')
        self.assertIn('credits', result.error['message'])

    def test_http_diagnostic_survives_compatibility_error_string(self):
        executor = self.executor(lambda request: httpx.Response(429, json={'error': 'provider_rate_limited', 'diagnostic': {
            'code': 'provider_rate_limited', 'message': 'fal rate limit was reached; retain the original attempt',
            'source': 'fal', 'retryable': True, 'accepted': False}}))
        result = executor.generate(**self.fixture.inputs())
        self.assertEqual(result.error['code'], 'provider_rate_limited')
        self.assertTrue(result.error['retryable'])
        self.assertIn('original attempt', result.error['message'])

    def test_terminal_failure_keeps_content_error(self):
        def handler(request):
            if request.method == 'POST':
                return httpx.Response(200, json={'call_id': 'fal-original', 'state': 'queued', 'accepted': True})
            return httpx.Response(200, json={'call_id': 'fal-original', 'state': 'completed', 'accepted': True,
                'error': {'code': 'provider_content_rejected', 'message': 'fal rejected the request under its content rules',
                          'source': 'fal', 'retryable': False, 'accepted': True}})
        executor = self.executor(handler)
        created = executor.generate(**self.fixture.inputs())
        result = executor.status(created.video_task_id)
        self.assertEqual(result.status, 'failed')
        self.assertEqual(result.error['code'], 'provider_content_rejected')
        self.assertTrue(result.error['accepted'])
