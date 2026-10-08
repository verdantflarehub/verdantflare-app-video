"""Exercise recovery through the real HTTP application and durable queue."""

import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from h3_singularity.api import MAX_REQUEST_BYTES, create_app
from h3_singularity.errors import RuntimeErrorCode
from h3_singularity.queue import Queue


def request(key="a"):
    return {
        "idempotency_key": "video_task_" + key * 32,
        "model": "MiniMaxAI/MiniMax-H3", "task": "ref2va",
        "prompt": "A landscape", "seconds": 5,
        "conditions": [{"type": "image", "role": "reference", "uri": "http://video-mcp-server:8000/runtime-artifacts/art_" + "b" * 32 + "/content",
                        "size": 3, "sha256": "a" * 64}],
        "target": {"aspect_ratio": "16:9"},
    }


class FakeEngine:
    def health(self):
        return {"ready": True}

    def generate(self, *args):
        return {"execution_instance_id": "test-engine", "media": {}}


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.queue = Queue(self.temp.name)

    def tearDown(self):
        self.queue.close()
        self.temp.cleanup()

    def wait_for(self, predicate):
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        self.fail("worker did not reach expected state")

    def test_loading_is_live_but_not_ready(self):
        initialized = threading.Event()

        def factory():
            initialized.wait(3)
            return FakeEngine()

        with TestClient(create_app(self.queue, factory, "test-token")) as client:
            try:
                self.assertEqual(client.get("/live").status_code, 200)
                self.assertEqual(client.get("/health").status_code, 503)
            finally:
                initialized.set()
            self.wait_for(lambda: client.get("/health").status_code == 200)

    def test_initialization_failure_is_not_live(self):
        def factory():
            raise RuntimeErrorCode("model_missing")

        with TestClient(create_app(self.queue, factory, "test-token")) as client:
            self.wait_for(lambda: client.get("/live").status_code == 503)
            self.assertEqual(client.get("/live").json()["error"], "model_missing")

    def test_oom_fails_liveness_and_retains_task_without_replay(self):
        engine = FakeEngine()
        task = self.queue.submit(request())
        with patch.object(engine, "generate", side_effect=RuntimeErrorCode("dit_out_of_memory")) as generate:
            with patch("h3_singularity.api._download_references", return_value={}):
                with TestClient(create_app(self.queue, lambda: engine, "test-token")) as client:
                    self.wait_for(lambda: client.get("/live").status_code == 503)
                    self.assertEqual(client.get("/live").json()["error"], "dit_out_of_memory")
                    failed = self.queue.get(task["id"])
                    self.assertEqual(failed["status"], "failed")
                    self.assertEqual(failed["error"]["code"], "dit_out_of_memory")
                    generate.assert_called_once()
        self.queue.close()
        self.queue = Queue(self.temp.name)
        self.assertEqual(self.queue.submit(request())["id"], task["id"])
        self.assertIsNone(self.queue.take())

    def test_directory_failure_fails_only_that_task(self):
        first = self.queue.submit(request())
        (self.queue.root / first["id"]).mkdir()
        second = self.queue.submit(request("b"))
        with patch("h3_singularity.api._download_references", return_value={}):
            with TestClient(create_app(self.queue, FakeEngine, "test-token")) as client:
                self.wait_for(lambda: self.queue.get(second["id"])["status"] == "completed")
                self.assertEqual(self.queue.get(first["id"])["error"]["code"], "FileExistsError")
                self.assertEqual(client.get("/live").status_code, 200)

    def test_queue_failure_is_not_live(self):
        with patch.object(self.queue, "take", side_effect=RuntimeError("storage unavailable")):
            with TestClient(create_app(self.queue, FakeEngine, "test-token")) as client:
                self.wait_for(lambda: client.get("/live").status_code == 503)
                self.assertEqual(client.get("/health").status_code, 503)

    def test_chunked_body_limit_and_authentication(self):
        with TestClient(create_app(self.queue, FakeEngine, "test-token")) as client:
            self.assertEqual(client.post("/v1/videos", json=request()).status_code, 401)
            response = client.post(
                "/v1/videos", content=iter([b" " * MAX_REQUEST_BYTES, b"x"]),
                headers={"Authorization": "Bearer test-token"},
            )
            self.assertEqual(response.status_code, 413)

    def test_idempotency_recovery_works_when_runtime_is_not_ready(self):
        task = self.queue.submit(request())

        def unavailable():
            raise RuntimeErrorCode("model_unavailable")

        with TestClient(create_app(self.queue, unavailable, "test-token")) as client:
            headers = {"Authorization": "Bearer test-token"}
            self.wait_for(lambda: client.get("/live").status_code == 503)
            found = client.get("/v1/videos/by-idempotency/" + request()["idempotency_key"], headers=headers)
            self.assertEqual(found.json()["id"], task["id"])
            replay = client.post("/v1/videos", json=request(), headers=headers)
            self.assertEqual(replay.status_code, 200)
            self.assertEqual(replay.json()["id"], task["id"])
            conflict = client.post("/v1/videos", json={**request(), "prompt": "changed"}, headers=headers)
            self.assertEqual(conflict.status_code, 409)


if __name__ == "__main__":
    unittest.main()
