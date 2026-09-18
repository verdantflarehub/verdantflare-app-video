import asyncio
import json
import unittest
from unittest.mock import patch

import httpx
from app import server
from app.executor import ExecutionError
from app.tasks import TaskNotFound


class LatentToolErrorsTests(unittest.TestCase):
    def test_public_error_codes_and_redaction(self):
        adapter = server.executor.processing['h3-latent-upscale']
        cases = [
            (TaskNotFound('/private/tasks'), 'task_not_found', False),
            (ValueError('https://private.invalid?token=secret'), 'invalid_request', False),
            (ExecutionError('missing_latent_bundle: /private/source'), 'missing_latent_bundle', False),
            (ExecutionError('source_node_mismatch'), 'source_node_mismatch', False),
            (httpx.ConnectError('secret-internal-host'), 'postprocessing_unavailable', True),
        ]
        for error, code, retryable in cases:
            with self.subTest(code=code), patch.object(adapter, 'generate', side_effect=error):
                result = server.video_h3_latent_generate('video_task_' + '0' * 32)
                self.assertTrue(result.is_error)
                data = result.structured_content
                self.assertEqual(data['error']['code'], code)
                self.assertEqual(data['error']['retryable'], retryable)
                self.assertEqual(json.loads(result.content[0].text), data)
                self.assertNotIn('private', json.dumps(data))
                self.assertNotIn('secret', json.dumps(data))

    def test_query_tools_preserve_safe_errors(self):
        adapter = server.executor.processing['h3-latent-upscale']
        for name, method in [('status', 'status'), ('result', 'result'), ('preview', 'result')]:
            with self.subTest(name=name), patch.object(adapter, method, side_effect=TaskNotFound('internal')):
                response = getattr(server, 'video_h3_latent_' + name)('video_task_' + '0' * 32)
                self.assertEqual(response.structured_content['error']['code'], 'task_not_found')
                self.assertTrue(response.is_error)

    def test_decorator_preserves_task_id_only_schema(self):
        tools = asyncio.run(server.mcp.list_tools())
        tool = next(t for t in tools if t.name == 'video.h3.latent.upscale.generate')
        self.assertEqual(tool.input_schema['required'], ['source_video_task_id'])
        self.assertNotIn('args', tool.input_schema['properties'])
