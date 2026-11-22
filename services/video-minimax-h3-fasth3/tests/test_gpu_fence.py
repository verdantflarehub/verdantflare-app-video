"""Unit tests for video-minimax-h3-fasth3 EpochLeaseFenceGuard, R2-H1 atomic reclaim, R2-H3 persistence, and /dev/shm cleanup."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import threading
import unittest

from h3_fasth3.fence import EpochLeaseFenceGuard, FencePersistenceError, StaleFenceError, cleanup_task_shm


class FastH3EpochFenceTest(unittest.TestCase):
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
            state_path = Path(state_dir) / "fasth3_guard.json"
            guard = EpochLeaseFenceGuard(
                state_path=state_path,
                write_fn=lambda fd, b: os.write(fd, bytes(b[:6])),
            )
            ident = {
                "pod_uid": "pod-fasth3",
                "container_id": "cid-fasth3",
                "boot_id": "boot-1",
                "supervisor_pid": os.getpid(),
                "supervisor_starttime_ticks": 91000,
                "visible_gpu_uuids": ["GPU-99999999-8888-7777-6666-555555555555"],
            }
            headers = {
                "x-vf-gpu-lease-id": "lease-h3-1",
                "x-vf-gpu-fencing-token": "41",
                "x-vf-gpu-attempt-id": "att-h3-1",
                "x-vf-gpu-stage-id": "p1_fasth3",
                "x-vf-gpu-pod-uid": "pod-fasth3",
                "x-vf-gpu-container-id": "cid-fasth3",
                "x-vf-gpu-boot-id": "boot-1",
                "x-vf-gpu-uuid": "GPU-99999999-8888-7777-6666-555555555555",
                "x-vf-gpu-expires-at": "2099-01-01T00:00:00Z",
            }
            grant, replay = guard.extract_and_validate(headers, key_id="idem-1", request_digest="dig-1", identity=ident)
            self.assertIsNone(replay)
            self.assertEqual(grant["lease_id"], "lease-h3-1")
            guard.bind_task_id("idem-1", "task_fasth3_01")
            epoch = guard.allocate_worker_epoch("task_fasth3_01")
            self.assertEqual(epoch, 1)

            # While worker epoch is active, reclaim_and_seal seals token 41 but refuses reclaimed=True
            p_busy = guard.reclaim_and_seal(lease_id="lease-h3-1", fencing_token=41, active_tasks=0, settled=True, identity=ident)
            self.assertFalse(p_busy["reclaimed"])
            self.assertTrue(p_busy["token_sealed"])

            guard.seal_task_epoch("task_fasth3_01")
            proof = guard.reclaim_and_seal(lease_id="lease-h3-1", fencing_token=41, active_tasks=0, settled=True, identity=ident)
            self.assertTrue(proof["reclaimed"])
            self.assertEqual(proof["proof_source"], "receiver_guard")
            self.assertTrue(proof["token_sealed"])
            self.assertEqual(proof["sealed_gpu_epoch"], 1)
            self.assertEqual(proof["consumed_fencing_token"], 41)

            restarted = EpochLeaseFenceGuard(state_path=state_path)
            self.assertEqual(restarted.current_epoch, 1)
            with self.assertRaises(StaleFenceError):
                restarted.extract_and_validate(
                    {**headers, "x-vf-gpu-attempt-id": "att-h3-replay"},
                    key_id="idem-2",
                    request_digest="dig-2",
                    identity=ident,
                )

            # Barrier race: concurrent allocate_worker_epoch vs reclaim_and_seal
            barrier = threading.Barrier(2)
            res: dict[str, bool] = {}
            hdr45 = {**headers, "x-vf-gpu-lease-id": "lease-h3-45", "x-vf-gpu-fencing-token": "45", "x-vf-gpu-attempt-id": "att-45"}

            def t_spawn() -> None:
                barrier.wait()
                try:
                    guard.extract_and_validate(hdr45, key_id="idem-45", request_digest="dig-45", identity=ident)
                    guard.bind_task_id("idem-45", "fasth3_45")
                    guard.allocate_worker_epoch("fasth3_45")
                    res["spawned"] = True
                except StaleFenceError:
                    res["spawned"] = False

            def t_reclaim() -> None:
                barrier.wait()
                pr = guard.reclaim_and_seal(lease_id="lease-h3-45", fencing_token=45, active_tasks=0, settled=True, identity=ident, expected_pod_uid=ident["pod_uid"], expected_container_id=ident["container_id"], expected_boot_id=ident["boot_id"])
                res["reclaimed"] = bool(pr and pr["reclaimed"])

            th1, th2 = threading.Thread(target=t_spawn), threading.Thread(target=t_reclaim)
            th1.start()
            th2.start()
            th1.join()
            th2.join()
            self.assertEqual(set(res), {"spawned", "reclaimed"})
            self.assertFalse(res.get("spawned") and res.get("reclaimed"))

            # Corrupted state file fails closed
            bad_path = Path(state_dir) / "corrupt.json"
            bad_path.write_text("{broken", encoding="utf-8")
            bad_guard = EpochLeaseFenceGuard(state_path=bad_path)
            with self.assertRaises(FencePersistenceError):
                _ = bad_guard.current_epoch

        with tempfile.TemporaryDirectory() as shm_dir:
            os.environ["VF_DEV_SHM_ROOT"] = shm_dir
            (Path(shm_dir) / "vf_task_fasth3_01_tensor").write_bytes(b"abc")
            (Path(shm_dir) / "vf_other_task").write_bytes(b"keep")
            removed = cleanup_task_shm("task_fasth3_01")
            self.assertEqual(removed, 1)
            self.assertFalse((Path(shm_dir) / "vf_task_fasth3_01_tensor").exists())
            self.assertTrue((Path(shm_dir) / "vf_other_task").exists())


if __name__ == "__main__":
    unittest.main()
