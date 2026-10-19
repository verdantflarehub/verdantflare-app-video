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

from app.artifacts import ArtifactStore
from app.dashboard import Dashboard
from app.executor import ExecutionError, VideoExecutor
from app.tasks import TaskStore


class DashboardTest(unittest.TestCase):
    def test_project_title_and_category_search_preserve_id_filter(self):
        self.tasks.create(project_id='demo', idempotency_key='project-label',
                          request={'route': 'h3', 'prompt': 'plain'}, input_digest='label',
                          runtime_task_id='label', status='failed')
        options = [{'id': 'demo', 'title': '最后一块冰', 'category': 'creative_video',
                    'category_label': '创意视频制作', 'label': '创意视频制作 · 最后一块冰'}]
        with mock.patch('app.dashboard.project_options', return_value=options):
            for term in ('最后一块冰', '创意视频制作'):
                result = self.client.get('/api/dashboard', params={'q': term, 'project_id': 'demo'}, headers=self.headers).json()
                self.assertEqual(result['total'], 1)
                self.assertEqual(result['projects'], ['demo'])
                self.assertEqual(result['project_options'], options)
            result = self.client.get('/api/dashboard', params={'q': '最后一块冰', 'project_id': 'other'}, headers=self.headers).json()
            self.assertEqual(result['total'], 0)

    def test_catalog_endpoint_matches_capabilities_and_remains_read_only(self):
        response = self.client.get('/api/model-catalog',headers=self.headers)
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json(),self.executor.capabilities()['catalog'])
        self.assertEqual(self.client.get('/api/model-catalog').status_code,401)
        self.assertEqual(self.client.post('/api/model-catalog',headers=self.headers).status_code,405)
        self.assertEqual(self.client.get('/api/dashboard',headers=self.headers).status_code,200)

    def test_fal_catalog_execution_projection_keeps_snapshot_and_fail_closed_states(self):
        catalog = {'state':'ready', 'snapshot_id':'approved-digest', 'channels':[
            {'id':'fal', 'models':[
                {'id':'h3-ref', 'upstream_model_id':'minimax/h3/reference-to-video'},
                {'id':'h3-text', 'upstream_model_id':'minimax/h3/text-to-video'}]}],
            'groups':[], 'conflicts':[]}
        with mock.patch.object(self.executor.catalog, 'snapshot', return_value=catalog):
            with mock.patch.object(self.executor.fal, 'connected', return_value=True):
                response = self.client.get('/api/model-catalog',headers=self.headers)
                self.assertEqual(response.status_code,200)
                value = response.json()
                self.assertEqual(value['snapshot_id'],'approved-digest')
                self.assertEqual(value['execution']['fal']['h3-ref']['state'],'adapter_connected_unverified')
                self.assertEqual(value['execution']['fal']['h3-text']['state'],'catalog_only')
                self.assertNotIn('execution',catalog)
            with mock.patch.object(self.executor.fal, 'connected', return_value=False):
                self.assertEqual(self.executor.catalog_view()['execution']['fal']['h3-ref']['state'],'adapter_disconnected')

    def test_mcp_capabilities_exposes_channel_directory_without_enabling_generation(self):
        catalog = {'state':'ready', 'snapshot_id':'approved-digest', 'channels':[
            {'id':'fal', 'models':[{'id':'h3max-ref',
                'upstream_model_id':'minimax/h3-max/reference-to-video'}]}],
            'groups':[], 'conflicts':[]}
        directory = {'schema_version':1, 'channel_id':'fal', 'version':'declared-digest',
            'capabilities':[{'capability_id':'video.h3max.camera_controls', 'type':'video',
                'model_id':'minimax/h3-max', 'endpoint_id':'minimax/h3-max/camera-controls',
                'display_name':'Camera Controls', 'execution_state':'catalog_only'}]}
        with (mock.patch.object(self.executor.catalog, 'snapshot', return_value=catalog),
              mock.patch.object(self.executor.fal, 'channel_capabilities', return_value=directory)):
            response = self.client.get('/api/model-catalog', headers=self.headers)
            self.assertEqual(response.status_code, 200)
            value = response.json()
            self.assertEqual(value['channel_capabilities']['fal'], directory)
            self.assertEqual(value['execution']['fal']['h3max-ref']['state'], 'catalog_only')
            mcp_capabilities = self.executor.capabilities()
            self.assertEqual(mcp_capabilities['catalog']['channel_capabilities']['fal'], directory)
            self.assertEqual(mcp_capabilities['channel_capabilities']['fal'], directory)
            self.assertNotIn('channel_capabilities', catalog)

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
        from app.server import BearerAuthMiddleware
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
        self.assertIn('Video MCP', page.text)
        self.assertIn("script-src 'self'", page.headers['content-security-policy'])
        for path in ('/api/dashboard', '/api/tasks/video_task_'+'a'*32):
            self.assertEqual(self.client.get(path).status_code, 401)
        self.assertEqual(self.client.post('/api/tasks', json=self.payload).status_code, 401)
        self.assertEqual(self.client.get('/dashboard/frontend/index.html').status_code, 200)
        self.assertEqual(self.client.get('/dashboard/tasks/video_task_example').status_code, 200)
        self.assertEqual(self.client.get('/dashboard/frontend/dashboard.js').status_code, 200)
        self.assertEqual(self.client.get('/dashboard/frontend/studio-embed.js').status_code, 200)
        self.assertEqual(self.client.get('/dashboard/static/studio-embed.js').status_code, 200)
        embed_page = self.client.get('/dashboard?embed=1')
        self.assertEqual(embed_page.status_code, 200)
        self.assertIn('studio-embed.js', embed_page.text)
        self.assertEqual(self.client.get('/dashboard/frontend/server.py').status_code, 404)
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

    def test_channel_project_search_counts_and_pagination(self):
        for i, route in enumerate(('h3', 'h3-sol', 'h3-vdn', 'h3-singularity', 'fal', 'higgsfield', 'grok')):
            self.tasks.create(project_id='demo', idempotency_key=f'channel-{i}',
                              request={'route': route, 'prompt': 'needle'}, input_digest=str(i),
                              runtime_task_id=str(i), status='queued' if i == 0 else 'failed')
        self.tasks.create(project_id='other', idempotency_key='other', request={'route': 'h3', 'prompt': 'needle'},
                          input_digest='other', runtime_task_id='other', status='queued')
        with mock.patch.object(self.executor.fal, 'authorize_record'), \
             mock.patch.object(self.executor.higgsfield, 'authorize_record'):
            data = self.client.get('/api/dashboard?channel=local&project_id=demo&q=needle&status=failed&page=2&page_size=2', headers=self.headers).json()
            self.assertEqual(data['total'], 3)
            self.assertEqual(len(data['tasks']), 1)
            self.assertEqual(data['counts']['queued'], 1)
            self.assertEqual(data['counts']['failed'], 3)
            self.assertEqual({c['id'] for c in data['channels']}, {'local', 'fal', 'higgsfield', 'grok'})
            self.assertEqual(data['projects'], ['demo', 'other'])
            for channel in ('fal', 'higgsfield', 'grok'):
                result = self.client.get(f'/api/dashboard?channel={channel}&project_id=demo', headers=self.headers).json()
                self.assertEqual(result['total'], 1)
                self.assertEqual(result['tasks'][0]['presentation']['channel_id'], channel)
            result = self.client.get('/api/dashboard?channel=local&service=fal', headers=self.headers).json()
            self.assertEqual(result['total'], 0)
            result = self.client.get('/api/dashboard?channel=unknown', headers=self.headers).json()
            self.assertEqual(result['total'], 0)
        with mock.patch.object(self.executor.fal, 'authorize_record', side_effect=PermissionError), \
             mock.patch.object(self.executor.higgsfield, 'authorize_record', side_effect=PermissionError):
            result = self.client.get('/api/dashboard?channel=fal', headers=self.headers).json()
            self.assertEqual(result['total'], 0)
            self.assertNotIn('fal', {c['id'] for c in result['channels']})
            self.assertNotIn('higgsfield', {c['id'] for c in result['channels']})

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
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["status"], "failed")
        self.assertEqual(first.json()["error"]["code"], "submission_unconfirmed")
        self.assertNotIn('private', first.text)
        second = self.client.post('/api/tasks', json=self.payload, headers=self.headers)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()['error']['code'], 'submission_unconfirmed')
        self.assertEqual(first.json()['video_task_id'], second.json()['video_task_id'])

    def test_recover_interrupted_submission_and_reject_result_before_success(self):
        record = self.tasks.create(project_id='demo', idempotency_key='interrupted', request={}, input_digest='x', runtime_task_id='', status='queued')
        self.dashboard.recover_incomplete_submissions()
        self.assertEqual(self.tasks.get(record.video_task_id).status, 'failed')
        self.assertEqual(self.client.post('/api/tasks/'+record.video_task_id+'/result', headers=self.headers).status_code, 502)

    def test_lifespan_runs_without_affecting_mcp_route(self):
        from app import server
        async def poll_once():
            await asyncio.sleep(3600)
        with mock.patch.object(server, 'dashboard', self.dashboard), mock.patch.object(self.dashboard, 'poll', poll_once), \
             mock.patch.object(server, 'artifacts', self.artifacts), mock.patch.object(server, 'tasks', self.tasks), \
             mock.patch.object(server, 'executor', self.executor), \
             mock.patch.object(self.executor.delivery, 'sync', return_value=None), \
             mock.patch.object(server.registry, 'start', new_callable=mock.AsyncMock), \
             mock.patch.object(server.registry, 'stop', new_callable=mock.AsyncMock):
            with TestClient(server.app) as client:
                self.assertEqual(client.get('/health').status_code, 200)
                response = client.post('/mcp', headers={**self.headers, 'Host':'localhost:8000', 'Accept':'application/json, text/event-stream'},
                                       json={'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'2025-03-26','capabilities':{},'clientInfo':{'name':'test','version':'1'}}})
                self.assertEqual(response.status_code, 200, response.text)

    def test_mcp_submission_failure_returns_task_id_and_safe_error(self):
        from app import server
        def fail(request):
            raise httpx.ConnectError("private-secret")
        self.executor.client = httpx.Client(transport=httpx.MockTransport(fail))
        with mock.patch.object(server, "executor", self.executor):
            result = server.video_generate(model="minimax-h3-ref2va", **self.payload)
        data = result.model_dump(by_alias=True)["structuredContent"]
        self.assertEqual(data["status"], "failed")
        self.assertEqual(data["error"]["code"], "runtime_unavailable")
        self.assertEqual(self.tasks.get(data["video_task_id"]).error, data["error"])
        self.assertNotIn("private-secret", str(data))

    def test_thumbnail_extracts_only_from_succeeded_result_videos_with_cache_and_safe_errors(self):
        import subprocess
        video_art = self.artifacts.create_from_chunks(
            project_id="demo", operation="test", filename="result.mp4",
            media_type="video/mp4", chunks=[b"fake-mp4-content"],
        )
        local_task = self.tasks.create(
            project_id="demo", idempotency_key="thumb-local",
            request={"route": "h3-singularity", "prompt": "local video",
                     "references": {"images": [{"artifact_id": self.image.artifact_id, "purpose": "identity"}]}},
            input_digest="thumb-local", runtime_task_id="rt-local", status="succeeded",
        )
        self.tasks.update(local_task, artifact_id=video_art.artifact_id)

        fal_task = self.tasks.create(
            project_id="demo", idempotency_key="thumb-fal",
            request={"route": "fal", "prompt": "fal video"},
            input_digest="thumb-fal", runtime_task_id="rt-fal", status="succeeded",
        )
        self.tasks.update(fal_task, delivery={"final_artifact_id": video_art.artifact_id})

        hf_file = Path(self.temp.name) / "hf_result.mp4"
        hf_file.write_bytes(b"fake-hf-mp4")
        hf_task = self.tasks.create(
            project_id="demo", idempotency_key="thumb-hf",
            request={"route": "higgsfield", "prompt": "hf video"},
            input_digest="thumb-hf", runtime_task_id="rt-hf", status="succeeded",
        )
        self.tasks.update(hf_task, media={"sha256": "b" * 64, "width": 720, "height": 1280, "frame_rate": 24})

        # Queued/failed tasks with input reference images must NEVER impersonate result thumbnails.
        queued_with_ref = self.tasks.create(
            project_id="demo", idempotency_key="thumb-queued-ref",
            request={"route": "h3", "prompt": "queued",
                     "references": {"images": [{"artifact_id": self.image.artifact_id, "purpose": "identity"}]}},
            input_digest="thumb-q", runtime_task_id="rt-q", status="queued",
        )
        failed_with_ref = self.tasks.create(
            project_id="demo", idempotency_key="thumb-failed-ref",
            request={"route": "h3", "prompt": "failed",
                     "references": {"images": [{"artifact_id": self.image.artifact_id, "purpose": "identity"}]}},
            input_digest="thumb-f", runtime_task_id="rt-f", status="failed",
        )
        non_video_result = self.tasks.create(
            project_id="demo", idempotency_key="thumb-image-art",
            request={"route": "h3", "prompt": "wrong media"},
            input_digest="thumb-img", runtime_task_id="rt-img", status="succeeded",
        )
        self.tasks.update(non_video_result, artifact_id=self.image.artifact_id)

        ffmpeg_calls = []
        def fake_ffmpeg(cmd, **kwargs):
            ffmpeg_calls.append(cmd)
            Path(cmd[-1]).write_bytes(b"\xff\xd8\xff\xe0JFIF-test-frame")
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        with mock.patch("app.dashboard.subprocess.run", side_effect=fake_ffmpeg), \
             mock.patch.object(self.executor.fal, "authorize_record"), \
             mock.patch.object(self.executor.higgsfield, "authorize_record"), \
             mock.patch.object(self.executor.higgsfield, "content_path", return_value=hf_file):
            for tid in (queued_with_ref.video_task_id, failed_with_ref.video_task_id, non_video_result.video_task_id):
                res = self.client.get(f"/api/tasks/{tid}/thumbnail", headers=self.headers)
                self.assertEqual(res.status_code, 404)
                self.assertEqual(res.json(), {"error": "preview_unavailable"})
            self.assertEqual(len(ffmpeg_calls), 0)

            res_local = self.client.get(f"/api/tasks/{local_task.video_task_id}/thumbnail", headers=self.headers)
            self.assertEqual(res_local.status_code, 200)
            self.assertEqual(res_local.headers["content-type"], "image/jpeg")
            self.assertTrue(res_local.content.startswith(b"\xff\xd8"))
            self.assertEqual(len(ffmpeg_calls), 1)

            # Second request and fal task sharing same video digest hit disk cache without re-running ffmpeg.
            res_local_cached = self.client.get(f"/api/tasks/{local_task.video_task_id}/thumbnail", headers=self.headers)
            res_fal = self.client.get(f"/api/tasks/{fal_task.video_task_id}/thumbnail", headers=self.headers)
            self.assertEqual(res_local_cached.status_code, 200)
            self.assertEqual(res_fal.status_code, 200)
            self.assertEqual(len(ffmpeg_calls), 1)

            res_hf = self.client.get(f"/api/tasks/{hf_task.video_task_id}/thumbnail", headers=self.headers)
            self.assertEqual(res_hf.status_code, 200)
            self.assertTrue(res_hf.content.startswith(b"\xff\xd8"))
            self.assertEqual(len(ffmpeg_calls), 2)

        # Permission denied must return 403 and never leak file paths or existence.
        with mock.patch.object(self.executor.fal, "authorize_record", side_effect=PermissionError("org mismatch")):
            denied = self.client.get(f"/api/tasks/{fal_task.video_task_id}/thumbnail", headers=self.headers)
            self.assertEqual(denied.status_code, 403)
            self.assertEqual(denied.json(), {"error": "permission_denied"})

        # Extraction failure on corrupt file returns 422 without leaking paths or stderr.
        corrupt_art = self.artifacts.create_from_chunks(
            project_id="demo", operation="test", filename="corrupt.mp4",
            media_type="video/mp4", chunks=[b"corrupt-mp4-bytes"],
        )
        corrupt_task = self.tasks.create(
            project_id="demo", idempotency_key="thumb-corrupt",
            request={"route": "h3-singularity", "prompt": "corrupt"},
            input_digest="thumb-c", runtime_task_id="rt-c", status="succeeded",
        )
        self.tasks.update(corrupt_task, artifact_id=corrupt_art.artifact_id)
        with mock.patch("app.dashboard.subprocess.run", side_effect=subprocess.CalledProcessError(1, ["ffmpeg"], stderr=b"/secret/path/corrupt.mp4")):
            failed_extract = self.client.get(f"/api/tasks/{corrupt_task.video_task_id}/thumbnail", headers=self.headers)
            self.assertEqual(failed_extract.status_code, 422)
            self.assertEqual(failed_extract.json(), {"error": "preview_extraction_failed"})
            self.assertNotIn("secret", failed_extract.text)

