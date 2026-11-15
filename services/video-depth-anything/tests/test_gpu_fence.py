"""Unit tests for video-depth-anything 6-tuple LeaseFenceGuard, R2-H1 atomic reclaim, and R2-H3 persistence."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import threading
import unittest

from gpu_guard import FencePersistenceError, LeaseFenceGuard, StaleFenceError


class VideoDepthLeaseFenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._old = os.environ.get("DEPTH_GPU_GRANT_POLICY")

    def tearDown(self) -> None:
        if self._old is None:
            os.environ.pop("DEPTH_GPU_GRANT_POLICY", None)
        else:
            os.environ["DEPTH_GPU_GRANT_POLICY"] = self._old

    def test_accept_vs_require_atomic_reclaim_and_fail_closed_persistence(self) -> None:
        ident = {
            "pod_uid": "pod-depth-1",
            "container_id": "cid-depth-1",
            "boot_id": "boot-1",
            "supervisor_pid": os.getpid(),
            "supervisor_starttime_ticks": 83000,
            "visible_gpu_uuids": ["GPU-12345678-1234-1234-1234-1234567890ab"],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "depth_state.json"
            os.environ["DEPTH_GPU_GRANT_POLICY"] = "accept"
            guard = LeaseFenceGuard(state_path=state_path, write_fn=lambda fd, b: os.write(fd, bytes(b[:4])))
            grant, _ = guard.extract_and_validate({}, task_id="depth-legacy", request_digest="d0", identity=ident)
            self.assertIsNone(grant)

            os.environ["DEPTH_GPU_GRANT_POLICY"] = "require"
            with self.assertRaises(PermissionError):
                guard.extract_and_validate({}, task_id="depth-strict", request_digest="d1", identity=ident)

            headers = {
                "x-vf-gpu-lease-id": "lease-depth-1",
                "x-vf-gpu-fencing-token": "31",
                "x-vf-gpu-attempt-id": "att-depth-1",
                "x-vf-gpu-stage-id": "postprocess_depth",
                "x-vf-gpu-pod-uid": "pod-depth-1",
                "x-vf-gpu-container-id": "cid-depth-1",
                "x-vf-gpu-boot-id": "boot-1",
                "x-vf-gpu-uuid": "GPU-12345678-1234-1234-1234-1234567890ab",
                "x-vf-gpu-expires-at": "2099-01-01T00:00:00Z",
            }
            grant, _ = guard.extract_and_validate(headers, task_id="depth-t1", request_digest="d1", identity=ident)
            self.assertEqual(grant["fencing_token"], 31)

            # Unsettled inflight blocks reclaim_and_seal while sealing token 31
            p_busy = guard.reclaim_and_seal(lease_id="lease-depth-1", fencing_token=31, active_tasks=0, settled=True, identity=ident)
            self.assertFalse(p_busy["reclaimed"])
            self.assertTrue(p_busy["token_sealed"])
            with self.assertRaises(StaleFenceError):
                guard.verify_before_inference("depth-t1")

            guard.seal_task("depth-t1")
            p_ok = guard.reclaim_and_seal(lease_id="lease-depth-1", fencing_token=31, active_tasks=0, settled=True, identity=ident)
            self.assertTrue(p_ok["reclaimed"])

            # Barrier race test
            barrier = threading.Barrier(2)
            out: dict[str, bool] = {}
            hdr35 = {**headers, "x-vf-gpu-lease-id": "lease-depth-35", "x-vf-gpu-fencing-token": "35", "x-vf-gpu-attempt-id": "att-35"}

            def run_infer() -> None:
                barrier.wait()
                try:
                    guard.extract_and_validate(hdr35, task_id="depth-t35", request_digest="d35", identity=ident)
                    guard.verify_before_inference("depth-t35")
                    out["infer"] = True
                except StaleFenceError:
                    out["infer"] = False

            def run_reclaim() -> None:
                barrier.wait()
                pr = guard.reclaim_and_seal(lease_id="lease-depth-35", fencing_token=35, active_tasks=0, settled=True, identity=ident)
                out["reclaimed"] = bool(pr and pr["reclaimed"])

            t1, t2 = threading.Thread(target=run_infer), threading.Thread(target=run_reclaim)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            self.assertFalse(out.get("infer") and out.get("reclaimed"))

            # Corrupted state fails closed
            bad_path = Path(tmpdir) / "corrupt.json"
            bad_path.write_text("[]", encoding="utf-8")
            bad = LeaseFenceGuard(state_path=bad_path)
            with self.assertRaises(FencePersistenceError):
                bad.extract_and_validate(hdr35, task_id="x", request_digest="d", identity=ident)


if __name__ == "__main__":
    unittest.main()
