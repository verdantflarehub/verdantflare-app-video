"""HTTP regression for undispatched grants without running a model."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from h3_fasth3.api import create_app
from h3_fasth3.queue import Queue


class ReclaimApiTests(unittest.TestCase):
    def test_unknown_lease_requires_bound_container_identity(self):
        ident = {"pod_uid": "pod-a", "container_id": "container-a", "boot_id": "boot-a",
                 "visible_gpu_uuids": ["GPU-A"], "supervisor_pid": os.getpid(), "supervisor_starttime_ticks": 1}
        class Engine:
            def health(self): return {"ready": True}
        with tempfile.TemporaryDirectory() as tmp, patch("h3_fasth3.api.read_runtime_identity", return_value=ident):
            queue = Queue(Path(tmp) / "tasks")
            self.addCleanup(queue.close)
            app = create_app(queue, lambda: Engine(), "test-token")
            with TestClient(app) as client:
                headers = {"Authorization": "Bearer test-token"}
                body = {"lease_id": "undispatched", "fencing_token": 90}
                endpoint = "/internal/v1/gpu-guard/reclaim"
                for extra in ({}, {"pod_uid": "pod-a"}, {"pod_uid": "pod-a", "container_id": "old-container", "boot_id": "boot-a"}):
                    response = client.post(endpoint, json={**body, **extra}, headers=headers)
                    self.assertEqual(response.status_code, 409)
                    self.assertEqual(response.json()["detail"]["code"], "fence_scope_unresolved")
                response = client.post(endpoint, json={**body, **{k: ident[k] for k in ("pod_uid", "container_id", "boot_id")}}, headers=headers)
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.json()["reclaim_proof"]["reclaimed"])
                # Known lease is resolved durably; the old caller remains valid.
                self.assertEqual(client.post(endpoint, json=body, headers=headers).status_code, 200)
                bad = client.post("/internal/v1/gpu-residency/revoke-lease", json={**body, "fencing_token": 91}, headers=headers)
                self.assertEqual(bad.status_code, 409)
                self.assertEqual(bad.json()["detail"]["code"], "lease_fence_conflict")
