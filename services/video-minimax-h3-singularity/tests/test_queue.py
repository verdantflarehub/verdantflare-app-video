import tempfile
import unittest
from pathlib import Path

from h3_singularity.queue import Queue, QueueError


class QueueTest(unittest.TestCase):
    def request(self, suffix="a"):
        return {
            "idempotency_key": "video_task_" + suffix * 32,
            "model": "MiniMaxAI/MiniMax-H3",
            "task": "ref2va",
            "prompt": "test",
            "seconds": 15,
            "conditions": [],
            "target": {"short_edge": 768, "aspect_ratio": "9:16"},
        }

    def test_idempotency_and_lifecycle(self):
        with tempfile.TemporaryDirectory() as root:
            queue = Queue(Path(root))
            first = queue.submit(self.request())
            same = queue.submit(self.request())
            self.assertEqual(first["id"], same["id"])
            task = queue.take()
            self.assertEqual(task["status"], "in_progress")
            queue.update(task["id"], stage="generating")
            queue.update(task["id"], status="completed", stage="completed", result={"content_sha256": "a"})
            self.assertEqual(queue.get(task["id"])["status"], "completed")
            queue.close()

    def test_conflicting_idempotency_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            queue = Queue(Path(root))
            queue.submit(self.request())
            changed = self.request()
            changed["prompt"] = "different"
            with self.assertRaisesRegex(QueueError, "idempotency_conflict"):
                queue.submit(changed)
            queue.close()


if __name__ == "__main__":
    unittest.main()
