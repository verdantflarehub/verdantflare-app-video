import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

import httpx

from app.artifacts import ArtifactStore
from app.executor import VideoExecutor
from app.tasks import TaskStore


class EngineRetirementTest(unittest.TestCase):
    service = "character_swap"

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(503)))
        self.addCleanup(client.close)
        self.executor = VideoExecutor(ArtifactStore(root), TaskStore(root), client)
        gpu = self.executor.gpu = Mock(enabled=True, idle_seconds=600)
        scheduler = self.executor.scheduler = Mock(enabled=True)
        scheduler._journal_lock = threading.Lock()
        scheduler._pending_releases = {}
        scheduler.list_leases.return_value = []
        gpu.engine_replicas.return_value = 0
        self.executor.engine_journal.wake(gpu, self.service, time.time())
        gpu.engine_replicas.return_value = 1
        gpu.scale_engine.reset_mock()
        self.executor._resolve_service_runtime_target = lambda _: ("http://engine", {})
        self.failed_task("failed-wake")

    def failed_task(self, key):
        return self.executor.tasks.create(project_id="test", idempotency_key=key, input_digest=key,
            request={"service": self.service}, runtime_task_id=None, status="failed")

    def test_concurrent_submit_blocks_rollback(self):
        other = self.failed_task("still-returning")
        self.executor._submitting_tasks.add(other.video_task_id)
        self.executor.retire_idle_engine(self.service, rollback=True)
        self.executor.gpu.scale_engine.assert_not_called()

    def test_unbound_failed_wake_rolls_back(self):
        self.executor.retire_idle_engine(self.service, rollback=True)
        self.executor.gpu.scale_engine.assert_called_once_with(self.service, 0)

    def test_collector_blocks_rollback(self):
        other = self.failed_task("collecting")
        self.executor.collector.active[other.video_task_id] = threading.current_thread()
        self.executor.retire_idle_engine(self.service, rollback=True)
        self.executor.gpu.scale_engine.assert_not_called()

    def test_station_lease_blocks_rollback(self):
        self.executor.scheduler.list_leases.return_value = [
            {"stage_id": self.service, "state": "granted"}]
        self.executor.retire_idle_engine(self.service, rollback=True)
        self.executor.gpu.scale_engine.assert_not_called()

    def test_different_ready_pod_blocks_rollback(self):
        self.executor.engine_journal.observed(self.service, "woken_pod_uid", "original")
        self.executor.client = Mock()
        self.executor.client.get.return_value = httpx.Response(200, json={
            "gpu_worker_active": False, "active_task_ids": [],
            "runtime_identity": {"pod_uid": "replacement"}})
        self.executor.retire_idle_engine(self.service, rollback=True)
        self.executor.gpu.scale_engine.assert_not_called()

    def test_idle_requires_wall_and_monotonic_intervals(self):
        from unittest.mock import patch
        since = self.executor.engine_journal.generation(self.service)["last_active"]
        with patch("app.executor.time.monotonic") as monotonic, patch("app.executor.time.time") as wall:
            monotonic.return_value = 10
            wall.return_value = since + 3600
            self.executor.retire_idle_engine(self.service)
            monotonic.return_value = 609
            self.executor.retire_idle_engine(self.service)
            monotonic.return_value = 610
            wall.return_value = since + 599
            self.executor.retire_idle_engine(self.service)
            self.executor.gpu.scale_engine.assert_not_called()
            wall.return_value = since + 600
            self.executor.retire_idle_engine(self.service)
            self.executor.gpu.scale_engine.assert_called_once_with(self.service, 0)

    def test_confirmed_unbound_rollback_rotates_station_attempt(self):
        task = self.failed_task("rollback-before-local-slot")
        lease = {"lease_id": "grant", "state": "leased", "fencing_token": 1, "version": 1}
        task = self.executor.tasks.update(task, provider_state={
            "_gpu_lease": lease, "_gpu_attempt_id": "first", "gpu_dispatched": False})
        self.executor.scheduler.release.side_effect = TimeoutError()
        unresolved = self.executor.rollback_unstarted_scheduler_grant(task)
        self.assertEqual(unresolved.provider_state["_gpu_attempt_id"], "first")
        self.assertEqual(unresolved.provider_state["_gpu_lease"], lease)
        self.executor.scheduler.release.side_effect = None
        renewed = self.executor.rollback_unstarted_scheduler_grant(unresolved)
        self.assertNotEqual(renewed.provider_state["_gpu_attempt_id"], "first")
        self.assertNotIn("_gpu_lease", renewed.provider_state)
        self.assertEqual(self.executor.tasks.get(task.video_task_id).provider_state, renewed.provider_state)

    def test_wake_int8_recovers_replaced_pod_when_replicas_already_one(self):
        """Regression test for bug_20261124_video-lipsync-mcp-runtime-not-ready:
        When an engine Pod is upgraded/replaced in-place while replicas==1 and all prior tasks are settled,
        wake_int8 must advance engine_scale_state to the new pod_uid instead of looping until deadline.
        """
        from app.executor import ExecutionError

        self.executor.engine_journal.observed(self.service, "woken_pod_uid", "pod-v013")
        self.executor.engine_journal.observed(self.service, "bound_pod_uid", "pod-v013")
        self.executor.engine_journal.observed(self.service, "dispatched_pod_uid", "pod-v013")

        new_identity = {
            "pod_uid": "pod-v014",
            "container_id": "container-v014",
            "boot_id": "boot-v014",
            "visible_gpu_uuids": ["GPU-798fb508-830f-b538-6733-72ed6d210d40"],
        }
        self.executor.client = Mock()
        self.executor.client.get.return_value = httpx.Response(
            200,
            json={
                "ready": True,
                "capabilities": {"backend": self.executor.processing[self.service].backend},
                "runtime_identity": new_identity,
            },
        )

        woken = self.executor.wake_int8(self.service, "http://engine", {})
        self.assertEqual(woken["pod_uid"], "pod-v014")
        gen = self.executor.engine_journal.generation(self.service)
        self.assertEqual(gen["generation"], 2)
        self.assertEqual(gen["woken_pod_uid"], "pod-v014")
        self.assertIsNone(gen["bound_pod_uid"])
        self.assertIsNone(gen["dispatched_pod_uid"])

        # Subsequent bound and dispatched observations on the new pod must succeed cleanly
        self.executor.engine_journal.observed(self.service, "bound_pod_uid", "pod-v014")
        self.executor.engine_journal.observed(self.service, "dispatched_pod_uid", "pod-v014")

        # Conversely, if an active lease still exists for the service during pod replacement,
        # wake_int8 must fail fast with engine_generation_requires_reconciliation.
        self.executor.scheduler.list_leases.return_value = [{"stage_id": self.service, "state": "granted"}]
        self.executor.client.get.return_value = httpx.Response(
            200,
            json={
                "ready": True,
                "capabilities": {"backend": self.executor.processing[self.service].backend},
                "runtime_identity": {**new_identity, "pod_uid": "pod-v015"},
            },
        )
        with self.assertRaisesRegex(ExecutionError, "engine_generation_requires_reconciliation"):
            self.executor.wake_int8(self.service, "http://engine", {})
