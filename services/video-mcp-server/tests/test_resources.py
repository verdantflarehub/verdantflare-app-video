"""MCP status survives retirement; legacy APIs cannot serve inventory."""
from __future__ import annotations
import os
import unittest
from unittest import mock
from starlette.applications import Starlette
from starlette.testclient import TestClient
from app.resources import Resources, stamp
from app.server import BearerAuthMiddleware

class ResourcesTest(unittest.TestCase):
    def test_retired_apis_require_auth_and_return_gone(self):
        resources = Resources()
        app = Starlette(routes=resources.routes())
        app.add_middleware(BearerAuthMiddleware)
        paths = ['/api/models', '/api/models/h3/instances',
                 '/api/models/h3/instances/old-instance',
                 '/api/models/h3/instances/old-instance/gpus/old-gpu?window=15m']
        with mock.patch.dict(os.environ, {'VIDEO_MCP_BEARER_TOKEN': 'test-only'}), TestClient(app) as client:
            for path in paths:
                self.assertEqual(client.get(path).status_code, 401)
                response = client.get(path, headers={'Authorization': 'Bearer test-only'})
                self.assertEqual(response.status_code, 410)
                self.assertEqual(response.json(), {'error': 'runtime_resources_retired'})
                self.assertEqual(response.headers['cache-control'], 'no-store')
            response = client.get('/api/mcp/status', headers={'Authorization': 'Bearer test-only'})
            self.assertEqual(response.status_code, 200)
            self.assertIn('protocol', response.json())
            self.assertIn('requests', response.json())

    def test_mcp_request_counts_and_protocol_staleness(self):
        now = [1000.0]
        resources = Resources(clock=lambda: now[0])
        resources.record_request(False)
        resources.record_request(True)
        resources.protocol = {'status': 'ready', 'state': 'fresh', 'sampled_at': stamp(now[0])}
        self.assertEqual(resources.mcp_status()['requests']['count'], 2)
        self.assertEqual(resources.mcp_status()['requests']['errors'], 1)
        now[0] += 31
        self.assertEqual(resources.mcp_status()['protocol']['state'], 'stale')
        self.assertEqual(resources.protocol['state'], 'fresh')
