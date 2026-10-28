import json
import unittest

import httpx

from app.channel_errors import ChannelError, channel_error, check_response, tool_errors


class ChannelErrorsTest(unittest.TestCase):
    def test_http_failure_survives_mcp_text_and_structured_content(self):
        @tool_errors
        def invoke(route='higgsfield'):
            check_response(httpx.Response(503, json={'error': 'provider_not_found'}), route, accepted=False)
        result = invoke()
        wire = result.model_dump(by_alias=True)
        self.assertTrue(wire['isError'])
        self.assertEqual(json.loads(result.content[0].text), wire['structuredContent'])
        self.assertEqual(wire['structuredContent']['error'], {
            'code': 'provider_not_found', 'source': 'higgsfield', 'http_status': 503,
            'retryable': False, 'accepted': False,
            'message': 'The provider returned not found for this operation.'})

    def test_safe_channel_envelope_preserves_real_reason_and_removes_secrets(self):
        error = channel_error({'accepted': True, 'error': {
            'source': 'google', 'code': 'google_invalid_argument', 'retryable': False,
            'message': 'duration must use Ns; https://private.example/result?token=secret Bearer secret'}}, source='google').public()
        self.assertEqual(error['code'], 'google_invalid_argument')
        self.assertTrue(error['accepted'])
        self.assertIn('duration must use Ns', error['message'])
        self.assertNotIn('secret', error['message'])
        self.assertNotIn('private.example', error['message'])

    def test_arbitrary_body_and_foreign_source_are_not_trusted(self):
        for body in ({'error': 'private password'}, {'error': {'source':'fal', 'code':'model_retired', 'message':'raw private body'}}):
            error = channel_error(body, source='google').public()
            self.assertNotIn('private', json.dumps(error))

    def test_unknown_shape_is_safe_and_signature_is_preserved(self):
        @tool_errors
        def invoke(project_id: str):
            check_response(httpx.Response(502, json=['credential']), 'fal')
        self.assertEqual(invoke.__annotations__, {'project_id': str})
        self.assertEqual(invoke('demo').model_dump(by_alias=True)['structuredContent']['error']['code'], 'channel_http_error')

    def test_terminal_envelope_preserves_admission_fact(self):
        error = channel_error({'accepted':False,'error':{'code':'provider_quota_exceeded',
            'source':'fal','message':'Daily video allowance is exhausted.','retryable':False}},source='fal')
        self.assertIsInstance(error, ChannelError)
        self.assertEqual(error.public()['message'], 'Daily video allowance is exhausted.')
        self.assertFalse(error.public()['accepted'])

    def test_private_json_paths_and_case_insensitive_urls_are_redacted(self):
        error = channel_error({'error': {'source':'google','code':'google_invalid_argument',
            'accepted':False, 'message':'{"api_key":"fixture-secret"} /var/run/private/config HTTPS://host.invalid/?secret=fixture'}},
            source='google', accepted=True).public()
        self.assertTrue(error['accepted'])
        for private in ('fixture-secret', '/var/run', 'host.invalid'):
            self.assertNotIn(private, error['message'])

    def test_string_error_prefers_safe_diagnostic(self):
        error = channel_error({'error':'provider_failed', 'diagnostic':{'code':'provider_quota_exhausted',
            'source':'fal','message':'Monthly quota exhausted','accepted':False}}, source='fal').public()
        self.assertEqual(error['code'],'provider_quota_exhausted')
        self.assertFalse(error['accepted'])
