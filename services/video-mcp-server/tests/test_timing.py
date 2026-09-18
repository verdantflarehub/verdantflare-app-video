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
