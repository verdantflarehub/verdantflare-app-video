import json
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import httpx

import test_processing
import test_singularity_route
from app.delivery import CALLER, OPTION
from app.executor import ExecutionError, VideoExecutor
from app.tasks import TaskConflict, TaskStore


class DeliveryTest(unittest.TestCase):
    def setUp(self):
        self.fixture = test_processing.InterpolationMCPTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.e = self.fixture.executor
        self.auth = patch('app.delivery.authorize', return_value={'user_id': 'user', 'organization_id': 'org'})
        self.auth.start()
        self.addCleanup(self.auth.stop)
        with self.e.tasks.exclusive():
            self.parent = self.e.tasks.create(project_id='p', idempotency_key='parent', input_digest='generation',
                request={'postprocess': OPTION, 'route': 'h3-singularity'}, runtime_task_id='native', status='succeeded')
            self.parent = self.e.tasks.update(self.parent, artifact_id=self.fixture.source.artifact_id)
        # Parent is already a validated native video; real processing adapter is
        # exercised through MockTransport, including metadata and result hashes.
        self.native_result = patch.object(self.e, 'result', side_effect=lambda tid: self.e.tasks.get(tid))
        self.native_result.start()
        self.addCleanup(self.native_result.stop)

    def test_background_completion_and_repeat_ticks_preserve_source(self):
        self.e.delivery.sync()
        parent = self.e.tasks.get(self.parent.video_task_id)
        self.assertEqual(parent.delivery['status'], 'running')
        self.assertEqual(parent.status, 'succeeded')
        self.fixture.state = 'succeeded'
        self.e.delivery.sync()
        self.e.delivery.sync()
        parent = self.e.tasks.get(parent.video_task_id)
        self.assertEqual(parent.artifact_id, self.fixture.source.artifact_id)
        self.assertEqual(parent.delivery['status'], 'succeeded')
        self.assertNotEqual(parent.delivery['final_artifact_id'], parent.artifact_id)
        self.assertEqual(len([r for r in self.fixture.requests if r.method == 'POST']), 1)

    def test_crash_after_child_submit_recovers_same_child(self):
        save = self.e.delivery.save
        def crash(parent, **kw):
            if 'interpolation_task_id' in kw:
                raise OSError('crash before parent link persisted')
            return save(parent, **kw)
        with patch.object(self.e.delivery, 'save', side_effect=crash):
            self.e.delivery.sync()
        self.e.delivery.sync()
        parent = self.e.tasks.get(self.parent.video_task_id)
        self.assertTrue(parent.delivery['interpolation_task_id'])
        self.assertEqual(len(list(self.e.tasks.root.glob('video_task_*.json'))), 2)
        self.assertEqual(len([r for r in self.fixture.requests if r.method == 'POST']), 1)

    def test_failed_interpolation_keeps_successful_generation(self):
        self.e.delivery.sync()
        self.fixture.state = 'failed'
        self.e.delivery.sync()
        row = self.e.tasks.get(self.parent.video_task_id)
        self.assertEqual(row.status, 'succeeded')
        self.assertEqual(row.delivery['status'], 'failed')
        self.assertIsNone(row.delivery['final_artifact_id'])
        self.assertEqual(row.artifact_id, self.fixture.source.artifact_id)

    def test_revoked_access_does_not_dispatch(self):
        with patch('app.delivery.authorize', side_effect=PermissionError()):
            self.e.delivery.sync()
        self.assertEqual(self.e.tasks.get(self.parent.video_task_id).delivery['error']['code'], 'project_access_denied')
        self.assertFalse(self.fixture.requests)

    def test_concurrent_executors_create_one_child(self):
        from app.artifacts import ArtifactStore
        from pathlib import Path
        other = VideoExecutor(ArtifactStore(Path(self.fixture.temp.name)), TaskStore(Path(self.fixture.temp.name)), self.fixture.client)
        with patch.object(other, 'result', side_effect=lambda tid: other.tasks.get(tid)):
            with ThreadPoolExecutor(2) as pool:
                futures = [pool.submit(e.delivery.tick, self.parent.video_task_id) for e in (self.e, other)]
                for result in futures:
                    result.result()
        self.assertEqual(len(list(self.e.tasks.root.glob('video_task_*.json'))), 2)
        self.assertEqual(len([r for r in self.fixture.requests if r.method == 'POST']), 1)


class GenerationOptionTest(unittest.TestCase):
    def setUp(self):
        self.f = test_singularity_route.SingularityRouteTest()
        self.f.setUp()
        self.addCleanup(self.f.tearDown)
        self.e = self.f.executor
        auth = patch('app.delivery.authorize', return_value={'user_id': 'user', 'organization_id': 'org'})
        auth.start()
        self.addCleanup(auth.stop)
        cap = patch.object(self.e.delivery, 'capability', return_value={'configured': True, 'readiness': 'ready'})
        cap.start()
        self.addCleanup(cap.stop)

    def test_enabled_option_persists_intent_and_changes_idempotency(self):
        row = self.e.generate(**self.f.kw, route='h3-singularity', postprocess=OPTION)
        self.assertEqual(row.request['postprocess'], OPTION)
        self.assertEqual(row.authority['user_id'], 'user')
        self.assertEqual(row.delivery['stage'], 'generation')
        self.assertNotIn('postprocess', json.loads(self.f.calls[-1].content))
        again = self.e.generate(**self.f.kw, route='h3-singularity', postprocess=OPTION)
        self.assertEqual(row.video_task_id, again.video_task_id)
        with self.assertRaises(TaskConflict):
            self.e.generate(**self.f.kw, route='h3-singularity')

    def test_disabled_option_preserves_existing_digest(self):
        first = self.e.generate(**self.f.kw, route='h3-singularity')
        second = self.e.generate(**self.f.kw, route='h3-singularity', postprocess=None)
        self.assertEqual(first.input_digest, second.input_digest)
        self.assertIsNone(first.delivery)

    def test_invalid_option_and_backend_failure_precede_generation(self):
        for value, route in [({}, 'h3-singularity'), (OPTION, 'h3-vdn'), ({'interpolate': {'profile': 'film'}}, 'h3-singularity')]:
            with self.assertRaises(ValueError):
                self.e.generate(**self.f.kw, route=route, postprocess=value)
        with patch.object(self.e.delivery, 'capability', return_value={'configured': True, 'readiness': 'not_ready'}), self.assertRaises(ExecutionError):
            self.e.generate(**self.f.kw, route='h3-singularity', postprocess=OPTION)
        self.assertFalse(self.f.calls)
        self.assertIsNone(self.e.tasks.find_idempotency(self.f.kw['project_id'], self.f.kw['idempotency_key']))

    def test_preflight_and_public_schema_expose_opt_in_without_dispatch(self):
        from app import server
        args = {**self.f.kw, 'route': 'h3-singularity', 'postprocess': OPTION}
        args.pop('idempotency_key')
        with patch.object(server, 'executor', self.e), patch.object(server, 'authorize', return_value={'user_id': 'user', 'organization_id': 'org'}):
            preview = server.video_preflight(**args).structured_content
        self.assertEqual(preview['delivery_output']['fps'], 48)
        self.assertEqual(preview['delivery_output']['frames'], preview['resolved_output']['frames'] * 2)
        schemas = {t['name']: t['inputSchema'] for t in server.VIDEO_TOOLS_SCHEMA}
        self.assertIn('postprocess', schemas['video.create']['properties'])
        self.assertIn('postprocess', schemas['video.preflight']['properties'])
        self.assertIn('idempotency_key', schemas['video.interpolate']['required'])
        self.assertFalse(self.f.calls)

    def test_interrupted_generation_submission_reuses_runtime_key(self):
        row = self.e.generate(**self.f.kw, route='h3-singularity', postprocess=OPTION)
        row = self.e.tasks.update(row, runtime_task_id='', status='queued')
        posts = []
        def reply(request):
            if request.method == 'POST':
                posts.append(json.loads(request.content))
                return httpx.Response(202, json={'id': 'singularity_'+'a'*32})
            if '/by-idempotency/' in request.url.path:
                return httpx.Response(404)
            return httpx.Response(200, json={'status': 'queued'})
        with httpx.Client(transport=httpx.MockTransport(reply)) as client:
            self.e.client = client
            recovered = self.e.status(row.video_task_id)
            self.assertEqual(recovered.runtime_task_id, 'singularity_'+'a'*32)
            self.assertEqual(posts[0]['idempotency_key'], row.video_task_id)


class AuthorityTest(unittest.TestCase):
    def test_authorization_uses_project_uuid7_request_contract(self):
        import uuid
        from app.delivery import authorize
        def respond(request):
            identity = uuid.UUID(request.headers['X-Request-Id'])
            self.assertEqual(identity.version, 7)
            self.assertEqual(json.loads(request.content), {'project_id': 'p', 'require_write': True})
            return httpx.Response(200, json={})
        with httpx.Client(transport=httpx.MockTransport(respond)) as client, patch.dict(os.environ, {'VIDEO_PROJECT_SERVICE_URL': 'https://project', 'VIDEO_PROJECT_SERVICE_TOKEN': 'internal'}):
            self.assertEqual(authorize(client, 'p', {'user_id': 'u', 'organization_id': 'o'})['user_id'], 'u')

    def test_authority_revalidation_denial_and_missing_identity(self):
        from app.delivery import authorize
        with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(403))) as client:
            with self.assertRaises(ValueError):
                authorize(client, 'p')
            with patch.dict(os.environ, {'VIDEO_PROJECT_SERVICE_URL': 'https://project', 'VIDEO_PROJECT_SERVICE_TOKEN': 'internal'}):
                with self.assertRaises(PermissionError):
                    authorize(client, 'p', {'user_id': 'u', 'organization_id': 'o'})


if __name__ == '__main__':
    unittest.main()
