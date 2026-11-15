"""Unit tests for video-minimax-h3-singularity EpochLeaseFenceGuard, R2-H1 atomic reclaim, R2-H3 persistence, and /dev/shm cleanup."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import threading
import unittest

from h3_singularity.fence import EpochLeaseFenceGuard, FencePersistenceError, StaleFenceError, cleanup_task_shm


class SingularityEpochFenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._old_policy = os.environ.get("VIDEO_GPU_GRANT_POLICY")
        self._old_shm = os.environ.get("VF_DEV_SHM_ROOT")

    def tearDown(self) -> None:
        if self._old_policy is None:
            os.environ.pop("VIDEO_GPU_GRANT_POLICY", None)
        else:
            os.environ["VIDEO_GPU_GRANT_POLICY"] = self._old_policy
        if self._old_shm is None:
            os.environ.pop("VF_DEV_SHM_ROOT", None)
        else:
            os.environ["VF_DEV_SHM_ROOT"] = self._old_shm

    def test_epoch_fence_atomic_reclaim_fail_closed_persistence_and_shm_cleanup(self) -> None:
        os.environ["VIDEO_GPU_GRANT_POLICY"] = "require"
        with tempfile.TemporaryDirectory() as state_dir:
            state_path = Path(state_dir) / "singularity_guard.json"
            guard = EpochLeaseFenceGuard(
                state_path=state_path,
                write_fn=lambda fd, b: os.write(fd, bytes(b[:6])),
            )
            ident = {
                "pod_uid": "pod-singularity",
                "container_id": "cid-singularity",
                "boot_id": "boot-1",
                "supervisor_pid": os.getpid(),
                "supervisor_starttime_ticks": 91000,
                "visible_gpu_uuids": ["GPU-99999999-8888-7777-6666-555555555555"],
            }
            headers = {
                "x-vf-gpu-lease-id": "lease-s-1",
                "x-vf-gpu-fencing-token": "41",
                "x-vf-gpu-attempt-id": "att-s-1",
                "x-vf-gpu-stage-id": "p1_singularity",
                "x-vf-gpu-pod-uid": "pod-singularity",
                "x-vf-gpu-container-id": "cid-singularity",
                "x-vf-gpu-boot-id": "boot-1",
                "x-vf-gpu-uuid": "GPU-99999999-8888-7777-6666-555555555555",
                "x-vf-gpu-expires-at": "2099-01-01T00:00:00Z",
            }
            grant, replay = guard.extract_and_validate(headers, key_id="idem-1", request_digest="dig-1", identity=ident)
            self.assertIsNone(replay)
            self.assertEqual(grant["lease_id"], "lease-s-1")
            guard.bind_task_id("idem-1", "singularity_00000000000000000000000000000001")
            epoch = guard.allocate_worker_epoch("singularity_00000000000000000000000000000001")
            self.assertEqual(epoch, 1)

            p_busy = guard.reclaim_and_seal(lease_id="lease-s-1", fencing_token=41, active_tasks=0, settled=True, identity=ident)
            self.assertFalse(p_busy["reclaimed"])
            self.assertTrue(p_busy["token_sealed"])

            guard.seal_task_epoch("singularity_00000000000000000000000000000001")
            proof = guard.reclaim_and_seal(lease_id="lease-s-1", fencing_token=41, active_tasks=0, settled=True, identity=ident)
            self.assertTrue(proof["reclaimed"])
            self.assertEqual(proof["proof_source"], "receiver_guard")
            self.assertTrue(proof["token_sealed"])
            self.assertEqual(proof["sealed_gpu_epoch"], 1)
            self.assertEqual(proof["consumed_fencing_token"], 41)

            restarted = EpochLeaseFenceGuard(state_path=state_path)
            self.assertEqual(restarted.current_epoch, 1)
            with self.assertRaises(StaleFenceError):
                restarted.extract_and_validate(
                    {**headers, "x-vf-gpu-attempt-id": "att-s-replay"},
                    key_id="idem-2",
                    request_digest="dig-2",
                    identity=ident,
                )

            barrier = threading.Barrier(2)
            res: dict[str, bool] = {}
            hdr45 = {**headers, "x-vf-gpu-lease-id": "lease-s-45", "x-vf-gpu-fencing-token": "45", "x-vf-gpu-attempt-id": "att-45"}

            def t_spawn() -> None:
                barrier.wait()
                try:
                    guard.extract_and_validate(hdr45, key_id="idem-45", request_digest="dig-45", identity=ident)
                    guard.bind_task_id("idem-45", "singularity_00000000000000000000000000000045")
                    guard.allocate_worker_epoch("singularity_00000000000000000000000000000045")
                    res["spawned"] = True
                except StaleFenceError:
                    res["spawned"] = False

            def t_reclaim() -> None:
                barrier.wait()
                pr = guard.reclaim_and_seal(lease_id="lease-s-45", fencing_token=45, active_tasks=0, settled=True, identity=ident)
                res["reclaimed"] = bool(pr and pr["reclaimed"])

            th1, th2 = threading.Thread(target=t_spawn), threading.Thread(target=t_reclaim)
            th1.start()
            th2.start()
            th1.join()
            th2.join()
            self.assertFalse(res.get("spawned") and res.get("reclaimed"))

            bad_path = Path(state_dir) / "corrupt.json"
            bad_path.write_text("{broken", encoding="utf-8")
            bad_guard = EpochLeaseFenceGuard(state_path=bad_path)
            with self.assertRaises(FencePersistenceError):
                _ = bad_guard.current_epoch

        with tempfile.TemporaryDirectory() as shm_dir:
            os.environ["VF_DEV_SHM_ROOT"] = shm_dir
            (Path(shm_dir) / "vf_singularity_01_tensor").write_bytes(b"abc")
            (Path(shm_dir) / "vf_other_task").write_bytes(b"keep")
            removed = cleanup_task_shm("singularity_01")
            self.assertEqual(removed, 1)
            self.assertFalse((Path(shm_dir) / "vf_singularity_01_tensor").exists())
            self.assertTrue((Path(shm_dir) / "vf_other_task").exists())


if __name__ == "__main__":
    unittest.main()
