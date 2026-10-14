"""HTTP contract checks without a running model or container."""

import json
import unittest

import httpx
from fastapi.testclient import TestClient

from h3_api.app import MAX_REQUEST_BYTES, create_app


TOKEN = "unit-test-token"
TASK = "singularity_" + "a" * 32
KEY = "video_task_" + "b" * 32


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.seen = []

        def upstream(request):
            self.seen.append((request.method, request.url.path, request.headers.get("Authorization"), request.content))
            if request.url.path == "/health":
                return httpx.Response(200, json={"ready": True})
            if request.url.path == "/v1/videos":
                return httpx.Response(200, json={"id": TASK, "status": "queued"})
            if request.url.path.endswith("/content"):
                return httpx.Response(200, content=b"sample-mp4-bytes", headers={"content-type": "video/mp4"})
            return httpx.Response(200, json={"id": TASK, "status": "completed"})

        self.backend = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        self.app = create_app("http://video-minimax-h3-singularity:8000", TOKEN, "runtime-test-token", self.backend)
        self.client = TestClient(self.app, headers={"Authorization": f"Bearer {TOKEN}"})

    def tearDown(self):
        self.client.close()

    def test_auth_and_health_are_separate(self):
        self.assertEqual(TestClient(self.app).get("/v1/videos/" + TASK).status_code, 401)
        self.assertEqual(self.client.get("/health").json(), {"ready": True, "runtime": "h3-singularity"})
        self.assertEqual(self.client.get("/live").status_code, 200)
        self.assertEqual(self.client.get("/capabilities").json()["task"], "ref2va")

    def test_submission_and_recovery_paths_keep_runtime_contract(self):
        payload = {"idempotency_key": KEY, "model": "MiniMaxAI/MiniMax-H3", "task": "ref2va"}
        submitted = self.client.post("/v1/videos", json=payload)
        self.assertEqual(submitted.status_code, 200)
        self.assertEqual(submitted.json()["id"], TASK)
        self.assertEqual(json.loads(self.seen[-1][3]), payload)
        self.assertEqual(self.client.get("/v1/videos/by-idempotency/" + KEY).json()["id"], TASK)
        self.assertEqual(self.client.get("/v1/videos/" + TASK).json()["status"], "completed")
        self.assertTrue(all(item[2] == "Bearer runtime-test-token" for item in self.seen))

    def test_content_is_relayed_and_paths_are_bounded(self):
        response = self.client.get("/v1/videos/" + TASK + "/content")
        self.assertEqual(response.content, b"sample-mp4-bytes")
        self.assertEqual(response.headers["content-type"], "video/mp4")
        count = len(self.seen)
        self.assertEqual(self.client.get("/v1/videos/other-service-task").status_code, 422)
        self.assertEqual(self.client.get("/v1/videos/bad-id/content").status_code, 422)
        self.assertEqual(self.client.get("/v1/videos/by-idempotency/bad-key").status_code, 422)
        self.assertEqual(len(self.seen), count)

    def test_large_or_invalid_request_never_reaches_runtime(self):
        self.assertEqual(self.client.post("/v1/videos", content=b"x" * (MAX_REQUEST_BYTES + 1)).status_code, 413)
        self.assertEqual(self.client.post("/v1/videos", content=b"not json").status_code, 422)
        self.assertEqual(self.seen, [])

    def test_upstream_failure_fails_closed(self):
        async def fail(_request):
            raise httpx.ConnectError("unavailable")

        backend = httpx.AsyncClient(transport=httpx.MockTransport(fail))
        app = create_app("http://runtime:8000", TOKEN, "runtime-test-token", backend)
        with TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"}) as client:
            self.assertEqual(client.get("/health").status_code, 503)
            self.assertEqual(client.post("/v1/videos", json={}).status_code, 503)
            self.assertEqual(client.get("/v1/videos/" + TASK + "/content").status_code, 503)


if __name__ == "__main__":
    unittest.main()
