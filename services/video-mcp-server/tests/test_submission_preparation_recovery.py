import unittest
from unittest.mock import patch

import httpx
import test_singularity_route
from app.executor import ExecutionError, VideoExecutor


class SubmissionPreparationRecoveryTest(unittest.TestCase):
    def setUp(self):
        test_singularity_route.SingularityRouteTest.setUp(self)

    def tearDown(self):
        test_singularity_route.SingularityRouteTest.tearDown(self)

    def test_preparation_error_is_terminal_and_does_not_submit(self):
        with patch.object(self.executor, 'acquire_local_gpu', side_effect=ExecutionError('private endpoint detail')):
            self.executor.gpu.enabled = True
            row = self.executor.generate(**self.kw, route='h3-fasth3')
        self.assertEqual(row.status, 'failed')
        self.assertEqual(row.error['code'], 'submission_preparation_failed')
        self.assertNotIn('private', str(row.error))
        self.assertIn('local_gpu_admission', row.error['message'])
        self.assertTrue(row.provider_state['worker_settled'])
        self.assertFalse(row.runtime_task_id)
        self.assertFalse([r for r in self.calls if r.method == 'POST'])
        again = self.executor.generate(**self.kw, route='h3-fasth3')
        self.assertEqual(again.video_task_id, row.video_task_id)
        self.assertFalse([r for r in self.calls if r.method == 'POST'])

    def test_process_interruption_preserves_pre_submit_intent(self):
        with patch.object(self.executor, '_dispatch_local_generation', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.executor.generate(**self.kw, route='h3-fasth3')
        row = self.tasks.find_idempotency('demo', self.kw['idempotency_key'])
        self.assertIs(row.provider_state['gpu_dispatched'], False)
        restarted = VideoExecutor(self.assets, self.tasks, self.http)
        with patch.object(restarted, '_dispatch_local_generation', return_value=row) as dispatch:
            restarted.status(row.video_task_id)
        dispatch.assert_called_once()

    def test_scheduler_binding_failure_before_post_is_recorded(self):
        self.executor.scheduler.enabled = True
        def admitted(record, *args, **kwargs):
            return True, record, {}
        with patch.object(self.executor, 'acquire_scheduler_grant', side_effect=admitted), \
             patch.object(self.executor, 'ensure_scheduler_target_bound', side_effect=ExecutionError('binding failed')):
            row = self.executor.generate(**self.kw, route='h3-fasth3')
        self.assertEqual(row.status, 'failed')
        self.assertEqual(row.error['code'], 'submission_preparation_failed')
        self.assertIn('target_binding', row.error['message'])
        self.assertTrue(row.provider_state['worker_settled'])
        self.assertFalse([r for r in self.calls if r.method == 'POST'])

    def unknown_record(self):
        row = self.executor.generate(**self.kw, route='h3-fasth3')
        return self.tasks.update(row, runtime_task_id='', provider_state={}, error=None)

    def test_legacy_missing_marker_is_explicitly_unknown_not_redispatched(self):
        row = self.unknown_record()
        self.executor.client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(404)))
        self.addCleanup(self.executor.client.close)
        with patch.object(self.executor, '_dispatch_local_generation') as dispatch:
            for _ in range(3):
                result = self.executor.status(row.video_task_id)
        dispatch.assert_not_called()
        self.assertEqual(result.status, 'queued')
        self.assertEqual(result.error['code'], 'submission_unconfirmed')
        self.assertFalse(result.runtime_task_id)
        self.assertNotIn('worker_settled', result.provider_state)

    def test_post_timeout_retains_dispatch_marker_and_unknown_state(self):
        def handler(req):
            if req.url.path == '/health':
                return httpx.Response(200, json={'ready': True})
            if req.method == 'POST':
                raise httpx.ReadTimeout('lost reply')
            return httpx.Response(404)
        self.executor.client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(self.executor.client.close)
        row = self.executor.generate(**self.kw, route='h3-fasth3')
        self.assertEqual(row.error['code'], 'submission_unconfirmed')
        self.assertIs(row.provider_state['gpu_dispatched'], True)
        self.assertNotEqual(row.status, 'failed')
        with patch.object(self.executor, '_dispatch_local_generation') as dispatch:
            self.executor.status(row.video_task_id)
        dispatch.assert_not_called()
