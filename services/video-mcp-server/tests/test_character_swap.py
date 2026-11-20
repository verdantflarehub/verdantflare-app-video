from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import httpx

from app.artifacts import ArtifactStore
from app.character_swap import EngineJournal, prepare_request
from app.executor import ExecutionError, VideoExecutor
from app.delivery import Delivery


class CapabilityTests(unittest.TestCase):
    def test_metadata_with_missing_or_truncated_image(self):
        with TemporaryDirectory() as tmp:
            store = ArtifactStore(Path(tmp))
            source, image = [store.create_from_chunks(project_id="pilot", operation="video.import",
                filename=name, media_type=mime, chunks=[b"content"]) for name, mime in
                (("source.mp4", "video/mp4"), ("look.png", "image/png"))]
            path = store.content_path(image)
            for code in ("character_image_integrity_failed", "character_image_not_found"):
                if path.stat().st_size:
                    path.write_bytes(b"")
                else:
                    path.unlink()
                with self.assertRaisesRegex(ValueError, code):
                    prepare_request(store, "pilot", source.artifact_id,
                                    {"character_image_artifact_id": image.artifact_id, "prompt": "Replace the man."})

    def setUp(self):
        env = patch.dict("os.environ", VIDEO_PROJECT_SERVICE_URL="http://project", VIDEO_PROJECT_SERVICE_TOKEN="test")
        env.start()
        self.addCleanup(env.stop)
        self.adapter = Mock()
        self.adapter._runtime_target.return_value = ("http://runtime", {}, "test")
        self.gpu = Mock()
        self.delivery = object.__new__(Delivery)
        self.delivery.executor = SimpleNamespace(gpu=self.gpu, processing={
            "character_swap": self.adapter, "regenerate_2k": self.adapter})

    def test_cold_and_control_plane_failure_never_probe_or_scale(self):
        self.gpu.engine_replicas.return_value = 0
        result = self.delivery.character_swap_capability()
        self.assertEqual((result["reason"], result["wake_on_task"], result["backend_ready"]),
                         ("scaled_to_zero", True, False))
        for status, expected in ((403, "rbac_denied"), (503, "k8s_unreachable")):
            response = httpx.Response(status, request=httpx.Request("GET", "http://cluster/scale"))
            self.gpu.engine_replicas.side_effect = httpx.HTTPStatusError("scale", request=response.request, response=response)
            result = self.delivery.character_swap_capability()
            self.assertEqual((result["reason"], result["orchestration_alive"], result["wake_on_task"]),
                             (expected, False, False))
        self.adapter.client.get.assert_not_called()
        self.gpu.scale_engine.assert_not_called()

    def test_hot_profile_matches_requested_weights(self):
        self.gpu.engine_replicas.return_value = 1
        self.adapter.client.get.return_value = httpx.Response(200, json={"ready": True, "capabilities": {
            "character_swap_profiles": ["h3_character_swap_lora"], "character_lora_versions": ["v1"],
            "turbo_variants": ["fl2v_turbo_8step"]}})
        self.assertEqual(self.delivery.character_swap_capability()["readiness"], "not_ready")
        self.assertEqual(self.delivery.character_swap_capability(character_lora_version="v1")["readiness"], "ready")
        self.assertEqual(self.delivery.regenerate_capability()["readiness"], "not_ready")


class PendingAcquireTests(unittest.TestCase):
    def setUp(self):
        self.executor = object.__new__(VideoExecutor)
        self.executor.scheduler = Mock()
        self.executor.scheduler.release.side_effect = lambda lid, **kwargs: {"lease_id": lid, "state": "released"}
        self.record = SimpleNamespace(video_task_id="task1", status="queued", provider_state={})
        self.live = dict(holder_task_id="task1", stage_id="character_swap", attempt_id="attempt",
                         pod_uid="current", state="granted", lease_id="new", fencing_token=2, version=1)

    def recover(self, leases):
        return self.executor.reconcile_pending_grant(
            self.record, "attempt", "character_swap", {"pod_uid": "current"}, leases, resume=True)

    def test_zero_one_many(self):
        self.assertIsNone(self.recover([]))
        self.assertEqual(self.recover([self.live]), self.live)
        older = dict(self.live, pod_uid=None, lease_id="old", fencing_token=1)
        self.assertEqual(self.recover([self.live, older]), self.live)
        self.executor.scheduler.release.assert_called_once_with(
            "old", fencing_token=1, expected_version=1, worker_settled=False, reclaim_mode="abort_unstarted")

    def test_unconfirmed_release_blocks_resume(self):
        self.executor.scheduler.release.side_effect = lambda *args, **kwargs: {}
        with self.assertRaisesRegex(ExecutionError, "reconciliation_required"):
            self.recover([self.live, dict(self.live, pod_uid=None, lease_id="old", fencing_token=1)])

    def test_restart_preserves_other_attempts_and_engines(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "coordination.sqlite3"
            journal = EngineJournal(path)
            for attempt, service in (("a", "character_swap"), ("b", "character_swap"), ("a", "regenerate_2k")):
                self.assertTrue(journal.stage_acquire("task", attempt, service, {"attempt_id": attempt})[1])
            reopened = EngineJournal(path)
            reopened.confirm_acquire("task", "a", "character_swap")
            self.assertEqual(reopened.stage_acquire("task", "b", "character_swap", {}), ({"attempt_id": "b"}, False))
            self.assertEqual(reopened.stage_acquire("task", "a", "regenerate_2k", {}), ({"attempt_id": "a"}, False))

    def test_generation_observations_follow_same_pod_through_restart(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "coordination.sqlite3"
            journal = EngineJournal(path)
            gpu = Mock()
            gpu.engine_replicas.return_value = 0
            journal.wake(gpu, "character_swap", 10)
            journal.observed("character_swap", "woken_pod_uid", "pod-a")
            journal = EngineJournal(path)
            for phase, uid in (("bound_pod_uid", "pod-b"), ("dispatched_pod_uid", "pod-a")):
                with self.assertRaisesRegex(ValueError, "reconciliation"):
                    journal.observed("character_swap", phase, uid)
            journal.observed("character_swap", "bound_pod_uid", "pod-a")
            journal.observed("character_swap", "dispatched_pod_uid", "pod-a")
            self.assertEqual(journal.generation("character_swap")["dispatched_pod_uid"], "pod-a")
