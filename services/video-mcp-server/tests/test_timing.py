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

    def test_invalid_dispatch_does_not_block_persisted_task_or_dashboard(self):
        for stamp in ('bad', '2026-11-03T00:00:01', '2026-99-03T00:00:01Z'):
            with self.subTest(stamp=stamp):
                record = self.record.model_copy(update=dict(
                    created_at='2026-11-03T00:00:00Z', dispatched_at=stamp,
                    completed_at='2026-11-03T00:00:03Z', status='succeeded',
                    artifact_id='retained-output'))
                self.store._write(record)
                loaded = TaskStore(Path(self.tmp.name)).get(record.video_task_id)
                view = public_task(loaded)
                self.assertEqual(loaded.dispatched_at, stamp)
                self.assertEqual(view['artifact_id'], 'retained-output')
                self.assertEqual(view['status'], 'succeeded')
                self.assertIsNone(view['timing']['queue_seconds'])
                self.assertIsNone(view['timing']['processing_seconds'])
                self.assertEqual(view['timing']['total_seconds'], 3)

    def test_reversed_intervals_remain_unknown_not_zero(self):
        record = self.record.model_copy(update=dict(
            created_at='2026-11-03T00:00:00Z',
            dispatched_at='2026-11-03T00:00:01Z',
            completed_at='2026-11-02T23:59:59Z', status='succeeded'))
        timing = public_task(record)['timing']
        self.assertEqual(timing['queue_seconds'], 1)
        self.assertIsNone(timing['processing_seconds'])
        self.assertIsNone(timing['total_seconds'])

    def test_timezone_offsets_and_real_zero_duration_remain_valid(self):
        record = self.record.model_copy(update=dict(
            created_at='2026-11-03T08:00:00+08:00',
            dispatched_at='2026-11-03T00:00:00Z',
            completed_at='2026-11-03T01:00:03+01:00', status='succeeded'))
        timing = public_task(record)['timing']
        self.assertEqual((timing['queue_seconds'], timing['processing_seconds'],
                          timing['total_seconds']), (0, 3, 3))

    def test_running_invalid_time_does_not_invent_elapsed_duration(self):
        for stamp in ('bad', '2026-11-03T00:00:01', '9999-12-31T00:00:00Z'):
            with self.subTest(stamp=stamp):
                queued = self.record.model_copy(update={'created_at': stamp})
                self.assertIsNone(public_task(queued)['timing']['queue_elapsed_seconds'])
                running = self.record.model_copy(update={
                    'dispatched_at': stamp, 'status': 'running'})
                timing = public_task(running)['timing']
                self.assertIsNone(timing['processing_elapsed_seconds'])
                self.assertIsNone(timing['processing_seconds'])
                self.assertIsNone(timing['total_seconds'])

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
