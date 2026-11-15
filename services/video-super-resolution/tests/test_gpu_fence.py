"""Unit tests for video-super-resolution 6-tuple LeaseFenceGuard, R2-H1 atomic reclaim, and R2-H3 durable persistence."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import unittest

from video_sr.gpu_guard import FencePersistenceError, LeaseFenceGuard, StaleFenceError


class VideoSRLeaseFenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._old = os.environ.get("VIDEO_GPU_GRANT_POLICY")

    def tearDown(self) -> None:
        if self._old is None:
            os.environ.pop("VIDEO_GPU_GRANT_POLICY", None)
        else:
            os.environ["VIDEO_GPU_GRANT_POLICY"] = self._old

    def _ident(self) -> dict[str, object]:
        return {
            "pod_uid": "pod-sr-1",
            "container_id": "cid-sr-1",
            "boot_id": "boot-1",
            "supervisor_pid": os.getpid(),
            "supervisor_starttime_ticks": 81000,
            "visible_gpu_uuids": ["GPU-11111111-2222-3333-4444-555555555555"],
        }

    def test_require_policy_enforces_six_tuple_monotonic_fencing_and_durable_seal(self) -> None:
        os.environ["VIDEO_GPU_GRANT_POLICY"] = "require"
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "guard_state.json"
            guard = LeaseFenceGuard(state_path=state_path)
            ident = self._ident()

            with self.assertRaises(PermissionError):
                guard.extract_and_validate({}, task_id="t1", request_digest="d1", identity=ident)

            headers_10 = {
                "x-vf-gpu-lease-id": "lease-1",
                "x-vf-gpu-fencing-token": "10",
                "x-vf-gpu-attempt-id": "att-1",
                "x-vf-gpu-stage-id": "postprocess_sr",
                "x-vf-gpu-parent-task-id": "parent-1",
                "x-vf-gpu-pod-uid": "pod-sr-1",
                "x-vf-gpu-container-id": "cid-sr-1",
                "x-vf-gpu-boot-id": "boot-1",
                "x-vf-gpu-uuid": "GPU-11111111-2222-3333-4444-555555555555",
                "x-vf-gpu-expires-at": "2099-01-01T00:00:00Z",
            }

            with self.assertRaises(PermissionError):
                guard.extract_and_validate(
                    {k: v for k, v in headers_10.items() if k != "x-vf-gpu-expires-at"},
                    task_id="t1",
                    request_digest="d1",
                    identity=ident,
                )
            with self.assertRaises(PermissionError):
                guard.extract_and_validate(
                    headers_10,
                    task_id="t1",
                    request_digest="d1",
                    identity={**ident, "pod_uid": "unknown-pod"},
                )

            grant, replay = guard.extract_and_validate(headers_10, task_id="t1", request_digest="d1", identity=ident)
            self.assertIsNone(replay)
            self.assertEqual(grant["fencing_token"], 10)

            _, replay2 = guard.extract_and_validate(headers_10, task_id="t2", request_digest="d1", identity=ident)
            self.assertEqual(replay2, "t1")
            with self.assertRaises(ValueError):
                guard.extract_and_validate(headers_10, task_id="t2", request_digest="d_mismatch", identity=ident)

            with self.assertRaises(StaleFenceError):
                guard.extract_and_validate(
                    {**headers_10, "x-vf-gpu-attempt-id": "att-replay"},
                    task_id="t_eq",
                    request_digest="d_eq",
                    identity=ident,
                )

            with self.assertRaises(StaleFenceError):
                guard.extract_and_validate(
                    {**headers_10, "x-vf-gpu-lease-id": "lease-0", "x-vf-gpu-fencing-token": "9", "x-vf-gpu-attempt-id": "att-0"},
                    task_id="t_stale",
                    request_digest="d_stale",
                    identity=ident,
                )

            affected = guard.revoke_lease("lease-1", 10)
            self.assertIn("t1", affected)
            with self.assertRaises(StaleFenceError):
                guard.verify_before_inference("t1")

            # While t1 is still inflight (not yet sealed by cancelled queue), reclaim refuses reclaimed=True
            unsettled_proof = guard.reclaim_and_seal(lease_id="lease-1", fencing_token=10, active_tasks=0, settled=True, identity=ident)
            self.assertFalse(unsettled_proof["reclaimed"])
            self.assertGreaterEqual(unsettled_proof["active_tasks"], 1)

            guard.seal_task("t1")
            proof = guard.reclaim_and_seal(lease_id="lease-1", fencing_token=10, active_tasks=0, settled=True, identity=ident)
            self.assertTrue(proof["reclaimed"])
            self.assertEqual(proof["proof_source"], "receiver_guard")
            self.assertTrue(proof["token_sealed"])
            self.assertEqual(proof["revoked_lease_id"], "lease-1")
            self.assertEqual(proof["consumed_fencing_token"], 10)

            restarted = LeaseFenceGuard(state_path=state_path)
            with self.assertRaises(StaleFenceError):
                restarted.extract_and_validate(
                    {**headers_10, "x-vf-gpu-attempt-id": "att-after-restart"},
                    task_id="t_restart",
                    request_digest="d_restart",
                    identity=ident,
                )

    def test_r2_h1_concurrent_reclaim_vs_inference_barrier_never_allows_both(self) -> None:
        os.environ["VIDEO_GPU_GRANT_POLICY"] = "require"
        ident = self._ident()
        for iteration in range(25):
            with tempfile.TemporaryDirectory() as tmpdir:
                guard = LeaseFenceGuard(state_path=Path(tmpdir) / "state.json")
                tok = 100 + iteration
                headers = {
                    "x-vf-gpu-lease-id": f"lease-{tok}",
                    "x-vf-gpu-fencing-token": str(tok),
                    "x-vf-gpu-attempt-id": f"att-{tok}",
                    "x-vf-gpu-stage-id": "sr",
                    "x-vf-gpu-pod-uid": "pod-sr-1",
                    "x-vf-gpu-container-id": "cid-sr-1",
                    "x-vf-gpu-boot-id": "boot-1",
                    "x-vf-gpu-uuid": "GPU-11111111-2222-3333-4444-555555555555",
                    "x-vf-gpu-expires-at": "2099-01-01T00:00:00Z",
                }
                barrier = threading.Barrier(2)
                outcomes: dict[str, object] = {}

                def worker_submit_and_infer() -> None:
                    barrier.wait()
                    try:
                        guard.extract_and_validate(headers, task_id=f"t-{tok}", request_digest="d", identity=ident)
                        guard.verify_before_inference(f"t-{tok}")
                        outcomes["infer_ok"] = True
                    except StaleFenceError:
                        outcomes["infer_ok"] = False

                def caller_reclaim() -> None:
                    barrier.wait()
                    proof = guard.reclaim_and_seal(
                        lease_id=f"lease-{tok}",
                        fencing_token=tok,
                        active_tasks=0,
                        settled=True,
                        identity=ident,
                    )
                    outcomes["reclaimed"] = bool(proof and proof["reclaimed"])

                t_a = threading.Thread(target=worker_submit_and_infer)
                t_b = threading.Thread(target=caller_reclaim)
                t_a.start()
                t_b.start()
                t_a.join()
                t_b.join()
                self.assertFalse(
                    bool(outcomes.get("infer_ok")) and bool(outcomes.get("reclaimed")),
                    f"R2-H1 violation on iteration {iteration}: {outcomes}",
                )

    def test_r2_h3_corrupted_state_short_writes_and_commit_failure_quarantine(self) -> None:
        os.environ["VIDEO_GPU_GRANT_POLICY"] = "require"
        ident = self._ident()
        headers = {
            "x-vf-gpu-lease-id": "lease-h3",
            "x-vf-gpu-fencing-token": "55",
            "x-vf-gpu-attempt-id": "att-h3",
            "x-vf-gpu-stage-id": "sr",
            "x-vf-gpu-pod-uid": "pod-sr-1",
            "x-vf-gpu-container-id": "cid-sr-1",
            "x-vf-gpu-boot-id": "boot-1",
            "x-vf-gpu-uuid": "GPU-11111111-2222-3333-4444-555555555555",
            "x-vf-gpu-expires-at": "2099-01-01T00:00:00Z",
        }

        # 1. Existing corrupted JSON file fails closed (never resets to 0)
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupt_path = Path(tmpdir) / "corrupt.json"
            corrupt_path.write_text("{truncated-json", encoding="utf-8")
            corrupt_guard = LeaseFenceGuard(state_path=corrupt_path)
            with self.assertRaises(FencePersistenceError):
                corrupt_guard.extract_and_validate(headers, task_id="t1", request_digest="d1", identity=ident)
            with self.assertRaises(FencePersistenceError):
                corrupt_guard.reclaim_and_seal(lease_id="lease-h3", fencing_token=55, identity=ident)

        # 2. Short writes (3 bytes at a time) loop to completion and persist valid JSON + dead-PID inflight sealing on restart
        with tempfile.TemporaryDirectory() as tmpdir:
            short_path = Path(tmpdir) / "short.json"
            write_calls = {"n": 0}
            dir_syncs = {"n": 0}

            def short_write(fd: int, buf) -> int:
                write_calls["n"] += 1
                return os.write(fd, bytes(buf[:3]))

            def count_dir_sync(_p: Path) -> None:
                dir_syncs["n"] += 1

            g1 = LeaseFenceGuard(
                state_path=short_path,
                write_fn=short_write,
                fsync_dir_fn=count_dir_sync,
            )
            g1.extract_and_validate(
                headers,
                task_id="t-inflight-crash",
                request_digest="d1",
                identity={**ident, "supervisor_pid": 999999, "supervisor_starttime_ticks": 777777},
            )
            self.assertGreater(write_calls["n"], 5)
            self.assertEqual(dir_syncs["n"], 1)

            # Simulate restart under a new supervisor PID: dead-PID inflight task is automatically sealed
            g2 = LeaseFenceGuard(state_path=short_path)
            with self.assertRaises(StaleFenceError):
                g2.verify_before_inference("t-inflight-crash")
            proof = g2.reclaim_and_seal(lease_id="lease-h3", fencing_token=55, active_tasks=0, settled=True, identity=ident)
            self.assertTrue(proof["reclaimed"])

        # 3. Failing fsync/replace quarantines guard and does NOT mutate in-memory state prior to failure
        with tempfile.TemporaryDirectory() as tmpdir:
            fail_path = Path(tmpdir) / "fail.json"

            def boom_fsync(_fd: int) -> None:
                raise OSError("disk_io_error")

            g_fail = LeaseFenceGuard(state_path=fail_path, fsync_fn=boom_fsync)
            with self.assertRaises(FencePersistenceError):
                g_fail.extract_and_validate(headers, task_id="t-fail", request_digest="d1", identity=ident)
            with self.assertRaises(FencePersistenceError):
                g_fail.verify_before_inference("t-fail")
            self.assertFalse(fail_path.exists())


if __name__ == "__main__":
    unittest.main()
