import threading
import unittest
from unittest.mock import patch

import httpx

import test_singularity_route
from app.executor import VideoExecutor
from app.gpu_lease import GpuCoordinator, K8sApi, MemoryApi


class GPURecoveryTest(unittest.TestCase):
    def setUp(self):
        test_singularity_route.SingularityRouteTest.setUp(self)
        self.api = MemoryApi()
        self.executor.gpu = GpuCoordinator(self.api, "h3", "h3/test", enabled=True)

    def tearDown(self):
        test_singularity_route.SingularityRouteTest.tearDown(self)

    def holder(self):
        return ((self.api.get_lease("gpu-task-h3") or {}).get("spec") or {}).get("holderIdentity", "")

    def unknown(self):
        def handler(request):
            if request.url.path == "/health":
                return httpx.Response(200, json={"ready": True})
            raise httpx.ReadTimeout("lost response", request=request)
        self.executor.client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(self.executor.client.close)
        return self.executor.generate(**self.kw, route="h3-singularity")

    def test_unknown_submission_remains_protected_without_runtime_id(self):
        row = self.unknown()
        self.assertEqual(row.error["code"], "submission_unconfirmed")
        self.assertFalse(row.runtime_task_id)
        self.assertEqual(self.holder(), "h3/test")
        self.executor.reconcile_gpu()
        music = GpuCoordinator(self.api, "music3", "music3/test", enabled=True)
        self.assertFalse(music.try_acquire())

    def test_restart_protects_persisted_unknown_submission(self):
        row = self.unknown()
        restarted = VideoExecutor(self.assets, self.tasks, self.http)
        restarted.gpu = GpuCoordinator(self.api, "h3", "h3/test", enabled=True)
        self.api.leases.clear()
        restarted.reconcile_gpu()
        self.assertEqual(self.holder(), "h3/test")
        self.assertEqual(self.tasks.get(row.video_task_id).idempotency_key, self.kw["idempotency_key"])

    def test_restart_cannot_overwrite_a_conflicting_active_role(self):
        self.unknown()
        self.api.leases.clear()
        music = GpuCoordinator(self.api, "music3", "music3/test", enabled=True)
        self.assertTrue(music.try_acquire())
        self.executor.reconcile_gpu()
        self.assertEqual(self.holder(), "")
        self.assertEqual(self.api.get_lease(music.lease_name)["spec"]["holderIdentity"], music.holder)

    def test_terminal_legacy_unknown_does_not_mean_backend_stopped(self):
        row = self.unknown()
        self.tasks.update(row, status="failed")
        self.api.leases.clear()
        self.executor.reconcile_gpu()
        self.assertEqual(self.holder(), "h3/test")

    def test_record_corruption_is_not_an_idle_signal(self):
        row = self.executor.generate(**self.kw, route="h3-singularity")
        (self.tasks.root / (row.video_task_id + ".json")).write_text("{")
        with self.assertRaises(Exception):
            self.executor.reconcile_gpu()
        self.assertEqual(self.holder(), "h3/test")

    def test_unreadable_directory_is_not_an_idle_signal(self):
        self.executor.generate(**self.kw, route="h3-singularity")
        with patch("app.executor.os.scandir", side_effect=PermissionError("unavailable")):
            with self.assertRaises(PermissionError):
                self.executor.reconcile_gpu()
        self.assertEqual(self.holder(), "h3/test")

    def test_confirmed_success_takes_precedence_over_legacy_error(self):
        row = self.unknown()
        self.tasks.update(row, status="succeeded")
        self.executor.reconcile_gpu()
        self.assertEqual(self.holder(), "")

    def test_runtime_404_does_not_release_an_unconfirmed_execution(self):
        row = self.executor.generate(**self.kw, route="h3-singularity")
        self.executor.client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(404)))
        self.addCleanup(self.executor.client.close)
        updated = self.executor.status(row.video_task_id)
        self.executor.reconcile_gpu()
        self.assertNotEqual(updated.status, "failed")
        self.assertEqual(updated.error["code"], "runtime_task_lost")
        self.assertEqual(self.holder(), "h3/test")

    def test_confirmed_terminal_releases_after_last_active_task(self):
        first = self.executor.generate(**self.kw, route="h3-singularity")
        second = self.executor.generate(**{**self.kw, "idempotency_key": "second"}, route="h3-singularity")
        self.tasks.update(first, status="succeeded")
        self.executor.reconcile_gpu()
        self.assertEqual(self.holder(), "h3/test")
        self.tasks.update(second, status="succeeded")
        self.executor.reconcile_gpu()
        self.assertEqual(self.holder(), "")

    def test_background_reconcile_during_post_does_not_release(self):
        observed = []
        def handler(request):
            if request.url.path == "/health":
                return httpx.Response(200, json={"ready": True})
            thread = threading.Thread(target=self.executor.reconcile_gpu)
            thread.start()
            thread.join(2)
            self.assertFalse(thread.is_alive())
            observed.append(self.holder())
            return httpx.Response(200, json={"id": "singularity_" + "a" * 32})
        self.executor.client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(self.executor.client.close)
        self.executor.generate(**self.kw, route="h3-singularity")
        self.assertEqual(observed, ["h3/test"])

    def test_busy_submission_does_not_claim_role_while_waiting_for_admission(self):
        with self.executor._lock:
            thread = threading.Thread(target=self.executor.reconcile_gpu)
            thread.start()
            thread.join(2)
            self.assertFalse(thread.is_alive())
        self.assertEqual(self.holder(), "")

    def test_known_rejection_releases_only_its_own_idle_role(self):
        def handler(request):
            return httpx.Response(200, json={"ready": True}) if request.url.path == "/health" else httpx.Response(422)
        self.executor.client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(self.executor.client.close)
        row = self.executor.generate(**self.kw, route="h3-singularity")
        self.assertEqual(row.error["code"], "runtime_rejected")
        self.assertEqual(self.holder(), "")

    def test_control_plane_failure_does_not_discard_successful_receipt(self):
        with patch.object(self.executor, "reconcile_gpu", side_effect=OSError("test control plane outage")):
            row = self.executor.generate(**self.kw, route="h3-singularity")
        self.assertTrue(row.runtime_task_id)
        self.assertEqual(self.tasks.get(row.video_task_id).runtime_task_id, row.runtime_task_id)
        self.assertEqual(self.holder(), "h3/test")

    def test_live_owner_is_not_revoked_by_transient_peer_claim(self):
        self.executor.generate(**self.kw, route="h3-singularity")
        gpu = self.executor.gpu
        self.api.replace_lease("gpu-task-music3", gpu._spec("music3/contender"), None)
        self.executor.reconcile_gpu()
        self.assertEqual(self.holder(), "h3/test")

    def test_new_holder_between_reads_cannot_be_overwritten(self):
        gpu = self.executor.gpu
        original = self.api.get_lease
        reads = 0
        def read(name):
            nonlocal reads
            if name == gpu.lease_name:
                reads += 1
                if reads == 2:
                    self.api.replace_lease(name, gpu._spec("h3/another"), None)
            return original(name)
        self.api.get_lease = read
        self.assertFalse(gpu.try_acquire())
        self.assertEqual(self.holder(), "h3/another")

    def test_create_conflict_does_not_overwrite_winning_holder(self):
        api, calls = K8sApi("test"), []
        def request(method, path, **kwargs):
            calls.append(method)
            return httpx.Response(409 if method == "POST" else 200,
                json={"metadata": {"resourceVersion": "1"}, "spec": {"holderIdentity": "winner"}},
                request=httpx.Request(method, "http://test" + path))
        api._request = request
        self.assertEqual(api.replace_lease("gpu-task-h3", {"holderIdentity": "loser"}, None), 409)
        self.assertEqual(calls, ["POST"])


if __name__ == "__main__":
    unittest.main()
