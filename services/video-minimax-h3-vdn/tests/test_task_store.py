from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from task_store import TaskStore, Conflict, QueueFull


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = TaskStore(self.temp.name, 'instance-a', capacity=2)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_idempotency_before_capacity_and_conflicts(self):
        task = self.store.submit('a', {'prompt':'one'})
        self.store.submit('b', {'prompt':'two'})
        self.assertEqual(self.store.submit('a', {'prompt':'one'})['id'], task['id'])
        with self.assertRaises(Conflict): self.store.submit('a', {'prompt':'changed'})
        with self.assertRaises(QueueFull): self.store.submit('c', {})

    def test_single_worker_and_cancel_boundary(self):
        first = self.store.submit('a', {})
        second = self.store.submit('b', {})
        self.assertEqual(self.store.take()['id'], first['id'])
        self.assertIsNone(self.store.take())
        with self.assertRaises(Conflict): self.store.cancel(first['id'])
        self.assertEqual(self.store.cancel(second['id'])['status'], 'cancelled')
        self.store.finish(first['id'], error='inference_failed')
        self.assertIsNone(self.store.take())

    def test_restart_never_replays_work(self):
        first = self.store.submit('a', {'prompt':'original'})
        second = self.store.submit('b', {})
        self.store.take()
        self.store.close()
        self.store = TaskStore(self.temp.name, 'instance-b')
        for task in (first, second):
            row = self.store.get(task['id'])
            self.assertEqual((row['status'], row['error']), ('failed','runtime_restarted'))
        self.assertIsNone(self.store.take())
        self.assertEqual(self.store.submit('a', {'prompt':'original'})['status'], 'failed')

    def test_output_only_for_completed_tasks(self):
        task = self.store.submit('a', {})
        path = Path(self.temp.name) / task['id'] / 'video.mp4'
        path.parent.mkdir();path.write_bytes(b'fixture')
        with self.assertRaises(KeyError): self.store.output(task['id'])
        self.store.take(); self.store.finish(task['id'], result={'sha256':'fixture'})
        self.assertEqual(self.store.output(task['id']), path)
        with self.assertRaises(Conflict): self.store.finish(task['id'], error='late_failure')
        with self.assertRaises(KeyError): self.store.get('../file')


if __name__ == '__main__': unittest.main()
