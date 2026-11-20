"""Unit tests for video-minimax-h3-character-swap EpochLeaseFenceGuard, R2-H1 atomic reclaim, R2-H3 persistence, and /dev/shm cleanup."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import threading
import unittest

from h3_character_swap.fence import EpochLeaseFenceGuard, FencePersistenceError, StaleFenceError, cleanup_task_shm


class CharacterSwapEpochFenceTest(unittest.TestCase):
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
            state_path = Path(state_dir) / "swap_guard.json"
            guard = EpochLeaseFenceGuard(
                state_path=state_path,
                write_fn=lambda fd, b: os.write(fd, bytes(b[:7])),
            )
            ident = {
                "pod_uid": "pod-swap",
                "container_id": "cid-swap",
                "boot_id": "boot-1",
                "supervisor_pid": os.getpid(),
                "supervisor_starttime_ticks": 92000,
                "visible_gpu_uuids": ["GPU-99999999-8888-7777-6666-555555555555"],
            }
            headers = {
                "x-vf-gpu-lease-id": "lease-2k-1",
                "x-vf-gpu-fencing-token": "51",
                "x-vf-gpu-attempt-id": "att-2k-1",
                "x-vf-gpu-stage-id": "p2_character_swap",
                "x-vf-gpu-pod-uid": "pod-swap",
                "x-vf-gpu-container-id": "cid-swap",
                "x-vf-gpu-boot-id": "boot-1",
                "x-vf-gpu-uuid": "GPU-99999999-8888-7777-6666-555555555555",
                "x-vf-gpu-expires-at": "2099-01-01T00:00:00Z",
            }
            grant, _ = guard.extract_and_validate(headers, key_id="idem-2k-1", request_digest="dig-2k-1", identity=ident)
            self.assertEqual(grant["stage_id"], "p2_character_swap")
            guard.bind_task_id("idem-2k-1", "task_2k_01")
            epoch = guard.allocate_worker_epoch("task_2k_01")
            self.assertEqual(epoch, 1)

            p_busy = guard.reclaim_and_seal(lease_id="lease-2k-1", fencing_token=51, active_tasks=0, settled=True, identity=ident)
            self.assertFalse(p_busy["reclaimed"])
            self.assertTrue(p_busy["token_sealed"])

            guard.seal_task_epoch("task_2k_01")
            proof = guard.reclaim_and_seal(lease_id="lease-2k-1", fencing_token=51, active_tasks=0, settled=True, identity=ident)
            self.assertTrue(proof["reclaimed"])
            self.assertEqual(proof["sealed_gpu_epoch"], 1)

            restarted = EpochLeaseFenceGuard(state_path=state_path)
            with self.assertRaises(StaleFenceError):
                restarted.extract_and_validate(
                    {**headers, "x-vf-gpu-attempt-id": "att-2k-replay"},
                    key_id="idem-2k-2",
                    request_digest="dig-2k-2",
                    identity=ident,
                )

            # Barrier race
            barrier = threading.Barrier(2)
            out: dict[str, bool] = {}
            hdr55 = {**headers, "x-vf-gpu-lease-id": "lease-2k-55", "x-vf-gpu-fencing-token": "55", "x-vf-gpu-attempt-id": "att-55"}

            def t_spawn() -> None:
                barrier.wait()
                try:
                    guard.extract_and_validate(hdr55, key_id="idem-55", request_digest="dig-55", identity=ident)
                    guard.bind_task_id("idem-55", "task_2k_55")
                    guard.allocate_worker_epoch("task_2k_55")
                    out["spawned"] = True
                except StaleFenceError:
                    out["spawned"] = False

            def t_reclaim() -> None:
                barrier.wait()
                pr = guard.reclaim_and_seal(lease_id="lease-2k-55", fencing_token=55, active_tasks=0, settled=True, identity=ident)
                out["reclaimed"] = bool(pr and pr["reclaimed"])

            t1, t2 = threading.Thread(target=t_spawn), threading.Thread(target=t_reclaim)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            self.assertFalse(out.get("spawned") and out.get("reclaimed"))

            bad_path = Path(state_dir) / "corrupt.json"
            bad_path.write_text('{"min_fencing_token": -1}', encoding="utf-8")
            bad = EpochLeaseFenceGuard(state_path=bad_path)
            with self.assertRaises(FencePersistenceError):
                bad.reclaim_and_seal(lease_id="lease-2k-99", fencing_token=99, identity=ident)

        with tempfile.TemporaryDirectory() as shm_dir:
            os.environ["VF_DEV_SHM_ROOT"] = shm_dir
            (Path(shm_dir) / "vf_task_2k_01_latent").write_bytes(b"xyz")
            self.assertEqual(cleanup_task_shm("task_2k_01"), 1)


if __name__ == "__main__":
    unittest.main()
