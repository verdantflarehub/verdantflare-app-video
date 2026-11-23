"""Regression and performance tests for Video MCP first-load dashboard optimization."""
from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import httpx
from starlette.applications import Starlette
from starlette.testclient import TestClient

from app.artifacts import ArtifactStore
from app.dashboard import Dashboard
from app.executor import VideoExecutor
from app.server import BearerAuthMiddleware
from app.tasks import TaskStore


class DashboardFirstLoadPerfTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.tasks = TaskStore(root)
        self.artifacts = ArtifactStore(root)

        def handler(request):
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok"})
            return httpx.Response(200, json={"status": "completed"})

        self.executor = VideoExecutor(
            self.artifacts,
            self.tasks,
            httpx.Client(transport=httpx.MockTransport(handler)),
        )
        self.dashboard = Dashboard(self.executor)
        self.env = mock.patch.dict(os.environ, {"VIDEO_MCP_BEARER_TOKEN": "perf-test-token"}, clear=False)
        self.env.start()
        app = Starlette(routes=self.dashboard.routes())
        app.add_middleware(BearerAuthMiddleware)
        self.client = TestClient(app)
        self.headers = {"Authorization": "Bearer perf-test-token"}

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_snapshot_deduplicates_project_authorization_calls(self):
        """30 higgsfield tasks in the same project must trigger authorize_record only once."""
        for i in range(30):
            rec = self.tasks.create(
                project_id="proj-same",
                idempotency_key=f"hf-{i}",
                request={"route": "higgsfield", "model": "kling-3.0", "prompt": f"shot {i}"},
                input_digest=f"d-{i}",
                runtime_task_id=f"rt-hf-{i}",
                status="succeeded",
            )
            self.tasks.update(rec, service="higgsfield", authority={"organization_id": "org-1"})

        call_count = 0

        def fake_auth(record):
            nonlocal call_count
            call_count += 1
            return {"organization_id": "org-1", "user_id": "u-1"}

        with mock.patch.object(self.executor.higgsfield, "authorize_record", side_effect=fake_auth), \
             mock.patch("app.dashboard.project_options", return_value=[{"id": "proj-same", "label": "短剧 · 测试项目"}]):
            resp = self.client.get("/api/dashboard?page=1&page_size=24", headers=self.headers)

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["total"], 30)
        self.assertEqual(len(body["tasks"]), 24)
        self.assertEqual(call_count, 1)

    def test_snapshot_cold_start_does_not_block_on_slow_catalog_view(self):
        """Slow upstream catalog_view (2s+) must not block initial /api/dashboard response."""
        self.tasks.create(
            project_id="proj-fast",
            idempotency_key="local-1",
            request={"route": "h3", "model": "minimax-h3-ref2va", "prompt": "fast first load"},
            input_digest="dfast",
            runtime_task_id="rt-fast",
            status="succeeded",
        )

        bg_started = threading.Event()
        release_bg = threading.Event()

        def slow_catalog_view():
            bg_started.set()
            release_bg.wait(timeout=3.0)
            return {"channels": [], "execution": {}}

        try:
            with mock.patch.object(self.executor, "catalog_view", side_effect=slow_catalog_view), \
                 mock.patch("app.dashboard.project_options", return_value=[]):
                t0 = time.monotonic()
                resp = self.client.get("/api/dashboard?page=1&page_size=24", headers=self.headers)
                elapsed = time.monotonic() - t0
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["total"], 1)
            self.assertLess(elapsed, 0.8, f"Cold start blocked for {elapsed:.2f}s on slow catalog_view")
            self.assertTrue(bg_started.wait(timeout=1.0), "Background catalog refresh should be triggered")
        finally:
            release_bg.set()

    def test_snapshot_only_serializes_current_page_and_preserves_counts(self):
        """With 40 records and page_size=10, counts reflect all 40 while only 10 are returned."""
        for i in range(40):
            self.tasks.create(
                project_id="proj-page",
                idempotency_key=f"p-{i}",
                request={"route": "h3", "model": "minimax-h3-ref2va", "prompt": f"scene {i}"},
                input_digest=f"dp-{i}",
                runtime_task_id=f"rt-p-{i}",
                status="succeeded" if i % 2 == 0 else "failed",
            )

        with mock.patch("app.dashboard.project_options", return_value=[]):
            resp = self.client.get("/api/dashboard?page=1&page_size=10", headers=self.headers)

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["total"], 40)
        self.assertEqual(len(body["tasks"]), 10)
        self.assertEqual(body["counts"]["succeeded"], 20)
        self.assertEqual(body["counts"]["failed"], 20)


if __name__ == "__main__":
    unittest.main()
