"""Expired admission must not become another queue/acquire or runtime POST."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import httpx

from app.artifacts import ArtifactStore
from app.dashboard import public_task
from app.executor import VideoExecutor
from app.tasks import TaskStore


class ExpiredAdmissionTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.tasks = TaskStore(root / "tasks")
        self.http_calls = []

        def reject_network(request):
            self.http_calls.append((request.method, request.url.path))
            raise AssertionError("No runtime request expected during expired admission")

        client = httpx.Client(transport=httpx.MockTransport(reject_network))
        self.addCleanup(client.close)
        self.executor = VideoExecutor(ArtifactStore(root / "artifacts"), self.tasks, client)
        self.executor.scheduler = Mock(enabled=True)
        self.executor.probe_runtime_identity = Mock(return_value={})
        self.executor.runtime = Mock(return_value=("http://runtime:8000", {}, "h3", "test"))
        self.executor.gpu.enabled = False
        self.executor.stage_slot.enabled = False
        self.executor.stage_slot.queue_seconds = 1200
        self.record = self.tasks.create(
            project_id="proj_expired", idempotency_key="expired-test", input_digest="digest",
            request={"route": "h3-fasth3"}, runtime_task_id="", status="queued",
        )
        self.record = self.tasks.update(self.record, provider_state={"gpu_dispatched": False})
        self.lease = {"lease_id": "lease-test", "state": "expired",
                      "blocking_reason": "queue_ttl_or_pod_expired", "attempt_id": "att-test"}

    def acquire(self, record=None, **kwargs):
        return self.executor.acquire_scheduler_grant(
            record or self.record, "h3-fasth3", runtime_url="http://runtime:8000", **kwargs,
        )

    def cache(self, lease=None, **state):
        self.record = self.tasks.update(self.record, provider_state={
            **self.record.provider_state, "_gpu_lease": lease or self.lease,
            "_gpu_attempt_id": "att-test", **state,
        })

    def test_queue_expiry_fails_public_status_and_does_not_resubmit(self):
        self.cache({**self.lease, "state": "queued"})
        self.executor.scheduler.get_lease.return_value = self.lease
        failed = self.executor.status(self.record.video_task_id)
        self.assertEqual(failed.status, "failed")
        self.assertEqual(public_task(failed)["error"]["code"], "gpu_admission_expired")
        self.assertTrue(failed.error["retryable"])
        self.assertEqual(failed.error["reason"], "queue_ttl_or_pod_expired")
        for _ in range(3):
            self.assertEqual(self.executor.status(failed.video_task_id).status, "failed")
        self.executor.scheduler.get_lease.assert_called_once_with("lease-test")
        self.executor.scheduler.acquire.assert_not_called()
        self.executor.scheduler.release.assert_not_called()
        self.assertEqual(self.http_calls, [])

    def test_cached_expiry_does_not_acquire_or_probe(self):
        self.cache()
        ok, record, headers = self.acquire()
        self.assertFalse(ok)
        self.assertEqual(headers, {})
        self.assertEqual(record.status, "failed")
        self.executor.scheduler.acquire.assert_not_called()
        self.executor.probe_runtime_identity.assert_not_called()

    def test_acquire_itself_can_return_expired(self):
        self.executor.scheduler.acquire.return_value = self.lease
        ok, record, _ = self.acquire()
        self.assertFalse(ok)
        self.assertEqual(record.status, "failed")
        self.assertEqual(record.provider_state["_gpu_lease"], self.lease)

    def test_expiry_fails_delivery_without_losing_chain_metadata(self):
        self.record = self.tasks.update(self.record, delivery={
            "status": "running", "stage": "generation", "chain_id": "original-chain",
        })
        self.cache()
        _, record, _ = self.acquire()
        self.assertEqual(record.delivery["status"], "failed")
        self.assertEqual(record.delivery["error"], record.error)
        self.assertEqual(record.delivery["chain_id"], "original-chain")
        self.assertTrue(record.delivery["completed_at"])

    def test_normal_queue_and_draining_keep_waiting(self):
        for state in ("queued", "draining"):
            with self.subTest(state=state):
                self.executor.scheduler.acquire.return_value = {**self.lease, "state": state}
                ok, record, _ = self.acquire()
                self.assertFalse(ok)
                self.assertEqual(record.status, "queued")
                self.assertIsNone(record.error)
                self.assertEqual(record.provider_state["lease_state"], state)

    def test_uncertain_dispatch_preserves_identity_and_never_retries(self):
        cases = [
            {"provider_state": {"gpu_dispatched": True}},
            {"provider_state": {}},
            {"runtime_task_id": "runtime-original"},
            {"dispatched_at": "2026-11-16T00:00:00+00:00"},
            {"provider_state": {"gpu_dispatched": False, "worker_settled": False}},
            {"error": {"code": "submission_unconfirmed"}},
        ]
        for index, changes in enumerate(cases):
            with self.subTest(changes=changes):
                record = self.tasks.create(
                    project_id="proj_expired", idempotency_key=f"uncertain-{index}", input_digest="digest",
                    request={"route": "h3-fasth3"}, runtime_task_id="", status="queued",
                )
                record = self.tasks.update(record, **{**changes, "provider_state": {
                    **changes.get("provider_state", {"gpu_dispatched": False}), "_gpu_lease": self.lease,
                }})
                for _ in range(2):
                    ok, record, _ = self.acquire(record)
                    self.assertFalse(ok)
                    self.assertEqual(record.status, "queued")
                    self.assertEqual(record.error["code"], "gpu_admission_reconciliation_required")
                    self.assertFalse(record.error["retryable"])
                    self.assertEqual(record.runtime_task_id, changes.get("runtime_task_id", ""))
        self.executor.scheduler.acquire.assert_not_called()
        self.executor.scheduler.release.assert_not_called()
        self.assertEqual(self.http_calls, [])

    def test_even_partial_binding_needs_reconciliation(self):
        for field in ("pod_uid", "container_id", "boot_id"):
            with self.subTest(field=field):
                self.cache({**self.lease, field: "bound-identity"})
                _, record, _ = self.acquire()
                self.assertEqual(record.error["code"], "gpu_admission_reconciliation_required")
                self.assertNotIn("worker_settled", record.provider_state)

    def test_never_granted_candidate_identity_converges_after_restart(self):
        self.cache({**self.lease, "pod_uid": "candidate", "container_id": "candidate", "boot_id": "candidate"})
        self.record = self.tasks.update(self.record, error={"code": "gpu_admission_reconciliation_required"})
        self.executor.scheduler.get_lease.return_value = {**self.record.provider_state["_gpu_lease"], "never_granted": True}
        self.executor.reconcile_expired_admissions()
        record = self.tasks.get(self.record.video_task_id)
        self.assertEqual(record.status, "failed")
        self.assertEqual(record.error["code"], "gpu_admission_expired")
        self.executor.scheduler.acquire.assert_not_called()
        self.assertEqual(self.http_calls, [])

    def test_never_granted_does_not_override_dispatch_uncertainty(self):
        for changes in ({"runtime_task_id": "accepted"}, {"error": {"code": "submission_unconfirmed"}}):
            record = self.tasks.update(self.record, **changes)
            result = self.executor._expired_scheduler_admission(record, {**self.lease, "never_granted": True})
            self.assertEqual(result.status, "queued")

    def test_background_scan_retains_granted_and_missing_history(self):
        self.cache({**self.lease, "pod_uid": "candidate"})
        for extra in ({}, {"never_granted": False}, {"state": "queued", "never_granted": True}):
            self.executor.scheduler.get_lease.return_value = {**self.record.provider_state["_gpu_lease"], **extra}
            self.executor.reconcile_expired_admissions()
            self.assertEqual(self.tasks.get(self.record.video_task_id).status, "queued")
        self.executor.scheduler.acquire.assert_not_called()

    def test_inherited_expiry_never_claims_parent_worker_settled(self):
        ok, record, _ = self.acquire(inherited_lease=self.lease)
        self.assertFalse(ok)
        self.assertFalse(record.error["retryable"])
        self.assertNotIn("worker_settled", record.provider_state)
        self.executor.scheduler.release.assert_not_called()

    def test_expired_unstarted_task_can_be_cancelled_without_dispatch(self):
        self.cache()
        self.executor.scheduler.release.return_value = {**self.lease, "state": "released"}
        cancelled = self.executor.cancel_generation(self.record)
        self.assertEqual(cancelled.status, "cancelled")
        self.assertEqual(self.executor.status(cancelled.video_task_id).status, "cancelled")
        self.executor.scheduler.acquire.assert_not_called()
        self.assertEqual(self.http_calls, [])
