from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import httpx
from starlette.testclient import TestClient

from verdantflare_video_mcp.artifacts import ArtifactStore
from verdantflare_video_mcp.dashboard import Dashboard
from verdantflare_video_mcp.executor import ExecutionError, VideoExecutor
from verdantflare_video_mcp.tasks import TaskStore


class DashboardTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.artifacts = ArtifactStore(root)
        self.tasks = TaskStore(root)
        self.calls = []

        def handler(request):
            self.calls.append(request)
            if request.url.path == '/health':
                return httpx.Response(200, json={'status': 'ok'})
            if request.method == 'POST':
                return httpx.Response(200, json={'id': 'private-runtime-id'})
            return httpx.Response(200, json={'status': 'completed'})

        self.executor = VideoExecutor(self.artifacts, self.tasks, httpx.Client(transport=httpx.MockTransport(handler)))
        self.dashboard = Dashboard(self.executor)
        self.env = mock.patch.dict(os.environ, {'VIDEO_MCP_BEARER_TOKEN':'test-only-token'})
        self.env.start()
        from verdantflare_video_mcp.server import BearerAuthMiddleware
        from starlette.applications import Starlette
        app = Starlette(routes=self.dashboard.routes())
        app.add_middleware(BearerAuthMiddleware)
        self.client = TestClient(app)
        self.headers = {'Authorization':'Bearer test-only-token'}
        self.image = self.artifacts.create_from_chunks(project_id='demo', operation='test', filename='ref.png',
                                                       media_type='image/png', chunks=[b'reference'])
        self.payload = {'project_id':'demo', 'idempotency_key':'shot/attempt-1', 'route':'h3',
                        'prompt':'Continuous camera motion', 'duration_seconds':5, 'aspect_ratio':'9:16',
                        'references':{'images':[{'artifact_id':self.image.artifact_id, 'purpose':'identity'}]}}

    def tearDown(self):
        self.client.close()
        self.executor.client.close()
        self.env.stop()
        self.temp.cleanup()

    def test_shell_is_public_but_api_requires_token(self):
        page = self.client.get('/dashboard')
        self.assertEqual(page.status_code, 200)
        self.assertIn('H3-Sol', page.text)
        self.assertIn("script-src 'self'", page.headers['content-security-policy'])
        for path in ('/api/dashboard', '/api/tasks/video_task_'+'a'*32):
            self.assertEqual(self.client.get(path).status_code, 401)
        self.assertEqual(self.client.post('/api/tasks', json=self.payload).status_code, 401)
        self.assertEqual(self.client.get('/dashboard/static/dashboard.js').status_code, 200)
        self.assertEqual(self.client.get('/dashboard/static/server.py').status_code, 404)
        self.assertEqual(self.client.get('/dashboard/').status_code, 200)
        with mock.patch.dict(os.environ, {'VIDEO_MCP_BEARER_TOKEN':''}):
            self.assertEqual(self.client.get('/api/dashboard').status_code, 503)

    def test_submit_detail_and_background_status(self):
        response = self.client.post('/api/tasks', json=self.payload, headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text)
        task_id = response.json()['video_task_id']
        self.assertNotIn('private-runtime-id', response.text)
        self.dashboard.sync()
        response = self.client.get('/api/tasks/'+task_id, headers=self.headers)
        self.assertEqual(response.json()['status'], 'succeeded')
        self.assertEqual(response.json()['references'][0]['artifact_id'], self.image.artifact_id)
        self.assertNotIn('runtime_task_id', response.text)
        self.assertNotIn(self.executor.runtime_url, response.text)
        snapshot = self.client.get('/api/dashboard', headers=self.headers).json()
        self.assertEqual(snapshot['counts']['succeeded'], 1)
        self.assertEqual(snapshot['services']['h3-sol'], 'not_connected')

    def test_result_updates_do_not_change_completion_time(self):
        record = self.tasks.create(project_id='demo', idempotency_key='timing', request={}, input_digest='x', runtime_task_id='r', status='queued')
        complete = self.tasks.update(record, status='succeeded')
        updated = self.tasks.update(complete, artifact_id=self.image.artifact_id)
        self.assertEqual(complete.completed_at, updated.completed_at)
        self.assertIsNotNone(updated.completed_at)

    def test_filter_pagination_and_safe_untrusted_prompt(self):
        for i in range(27):
            self.tasks.create(project_id='demo' if i < 26 else 'other', idempotency_key=str(i),
                              request={'prompt':'<script>alert(1)</script>', 'service':'h3'}, input_digest=str(i),
                              runtime_task_id=str(i), status='queued')
        data = self.client.get('/api/dashboard?project_id=demo&page=2&page_size=24&q=script', headers=self.headers).json()
        self.assertEqual(data['total'], 26)
        self.assertEqual(len(data['tasks']), 2)
        self.assertEqual(data['counts']['queued'], 26)
        self.assertEqual(self.client.get('/api/dashboard?service=h3-sol', headers=self.headers).json()['total'], 0)
        self.assertEqual(self.client.get('/api/dashboard?page=bad', headers=self.headers).status_code, 400)

    def test_sol_rejected_without_fallback_and_inputs_are_strict(self):
        self.assertEqual(self.client.post('/api/tasks', json={**self.payload,'route':'h3-sol'}, headers=self.headers).status_code, 409)
        self.assertEqual(len(self.calls), 0)
        for change in ({'duration_seconds':True}, {'duration_seconds':16}, {'seed':42}, {'aspect_ratio':'16:9'}, {'references':{'images':[]}}):
            self.assertEqual(self.client.post('/api/tasks', json={**self.payload,**change}, headers=self.headers).status_code, 400)
        self.assertEqual(self.client.post('/api/tasks', content=b'x'*65537, headers=self.headers).status_code, 413)

    def test_import_rejects_untrusted_source_and_does_not_leak_it(self):
        response = self.client.post('/api/artifacts/import', json={'project_id':'demo','source_url':'https://untrusted.example/private?secret=value',
                                    'filename':'ref.png', 'expected_sha256':'a'*64}, headers=self.headers)
        self.assertEqual(response.status_code, 400)
        self.assertNotIn('secret', response.text)

    def test_duplicate_concurrent_submissions_only_call_runtime_once(self):
        kwargs = dict(self.payload)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: self.executor.generate(model='minimax-h3-ref2va', **kwargs), range(2)))
        self.assertEqual(results[0].video_task_id, results[1].video_task_id)
        self.assertEqual(len(self.calls), 1)

    def test_uncertain_submission_persists_and_is_not_retried(self):
        def fail(request):
            raise httpx.ReadTimeout('private request payload')
        self.executor.client = httpx.Client(transport=httpx.MockTransport(fail))
        first = self.client.post('/api/tasks', json=self.payload, headers=self.headers)
        self.assertEqual(first.status_code, 502)
        self.assertNotIn('private', first.text)
        second = self.client.post('/api/tasks', json=self.payload, headers=self.headers)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()['error']['code'], 'submission_unconfirmed')

    def test_recover_interrupted_submission_and_reject_result_before_success(self):
        record = self.tasks.create(project_id='demo', idempotency_key='interrupted', request={}, input_digest='x', runtime_task_id='', status='queued')
        self.dashboard.recover_incomplete_submissions()
        self.assertEqual(self.tasks.get(record.video_task_id).status, 'failed')
        self.assertEqual(self.client.post('/api/tasks/'+record.video_task_id+'/result', headers=self.headers).status_code, 502)

    def test_lifespan_runs_without_affecting_mcp_route(self):
        from verdantflare_video_mcp import server
        async def poll_once():
            await asyncio.sleep(3600)
        with mock.patch.object(server, 'dashboard', self.dashboard), mock.patch.object(self.dashboard, 'poll', poll_once), \
             mock.patch.object(server, 'artifacts', self.artifacts), mock.patch.object(server, 'tasks', self.tasks):
            with TestClient(server.app) as client:
                self.assertEqual(client.get('/health').status_code, 200)
                response = client.post('/mcp', headers={**self.headers, 'Host':'localhost:8000', 'Accept':'application/json, text/event-stream'},
                                       json={'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'2025-03-26','capabilities':{},'clientInfo':{'name':'test','version':'1'}}})
                self.assertEqual(response.status_code, 200, response.text)
