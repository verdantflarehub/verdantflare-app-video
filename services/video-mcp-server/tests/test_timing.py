import tempfile
import unittest
from pathlib import Path
from app.tasks import TaskStore
from app.dashboard import public_task

class TimingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = TaskStore(Path(self.tmp.name))
        self.record = self.store.create(project_id='test', idempotency_key='time', input_digest='abc', request={}, runtime_task_id='', status='queued')

    def test_completed_and_dashboard(self):
        record = self.record.model_copy(update=dict(created_at='2026-09-16T00:00:00+00:00', dispatched_at='2026-09-16T00:00:12+00:00', completed_at='2026-09-16T00:08:26+00:00', status='succeeded'))
        timing = public_task(record)['timing']
        self.assertEqual((timing['queue_seconds'], timing['processing_seconds'], timing['total_seconds']), (12, 494, 506))
        self.assertIsNone(timing['processing_elapsed_seconds'])

    def test_running_does_not_invent_final_duration(self):
        record = self.store.update(self.record, dispatched_at=self.record.created_at, status='running')
        self.assertIsNone(record.timing['processing_seconds'])
        self.assertIsNone(record.timing['total_seconds'])
        self.assertGreaterEqual(record.timing['processing_elapsed_seconds'], 0)
        updated = self.store.update(record, dispatched_at='2099-01-01T00:00:00+00:00')
        self.assertEqual(updated.dispatched_at, record.dispatched_at)

    def test_legacy_record_and_reload(self):
        record = self.store.update(self.record, status='succeeded')
        loaded = TaskStore(Path(self.tmp.name)).get(record.video_task_id)
        self.assertIsNone(loaded.timing['queue_seconds'])
        self.assertIsNone(loaded.timing['processing_seconds'])
        self.assertIsNotNone(loaded.timing['total_seconds'])

    def test_queue_live_elapsed(self):
        timing = public_task(self.record)['timing']
        self.assertIsNone(timing['queue_seconds'])
        self.assertGreaterEqual(timing['queue_elapsed_seconds'], 0)

    def test_higgsfield_generation_stops_before_delayed_delivery(self):
        record = self.record.model_copy(update=dict(service='higgsfield',
            created_at='2026-10-14T16:11:40+00:00', dispatched_at='2026-10-14T16:11:48+00:00',
            completed_at='2026-10-14T16:28:13+00:00', status='succeeded',
            provider_state={'generation_completed_at':'2026-10-14T16:13:46+00:00',
                            'generation_time_source':'gateway_result_collected'}))
        timing = public_task(record)['timing']
        self.assertEqual(timing['processing_seconds'], 118)
        self.assertEqual(timing['total_seconds'], 993)
        self.assertIsNone(timing['processing_elapsed_seconds'])
        self.assertEqual(timing['generation_time_source'], 'gateway_result_collected')
        collecting = record.model_copy(update={'completed_at':None, 'status':'running', 'runtime_stage':'collecting'})
        timing = public_task(collecting)['timing']
        self.assertEqual(timing['processing_seconds'], 118)
        self.assertIsNone(timing['processing_elapsed_seconds'])

    def test_higgsfield_missing_or_invalid_evidence_does_not_invent_runtime(self):
        record = self.record.model_copy(update=dict(service='higgsfield',
            dispatched_at=self.record.created_at, completed_at=self.record.created_at, status='succeeded'))
        for stamp in [None, 'bad', '2099-01-01T00:00:00+00:00', '2000-01-01T00:00:00+00:00', '2026-10-14T16:13:46']:
            with self.subTest(stamp=stamp):
                timing = public_task(record.model_copy(update={'provider_state':{'generation_completed_at':stamp}}))['timing']
                self.assertIsNone(timing['processing_seconds'])
                self.assertIsNone(timing['processing_elapsed_seconds'])
                self.assertIsNone(timing['generation_completed_at'])
