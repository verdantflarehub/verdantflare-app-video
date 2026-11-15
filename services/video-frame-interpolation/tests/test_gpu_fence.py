"""Unit tests for video-frame-interpolation 6-tuple LeaseFenceGuard, R2-H1 atomic reclaim, and R2-H3 persistence."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import threading
import unittest

from video_interpolation.gpu_guard import FencePersistenceError, LeaseFenceGuard, StaleFenceError


class VideoInterpolationLeaseFenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._old = os.environ.get("VIDEO_GPU_GRANT_POLICY")

    def tearDown(self) -> None:
        if self._old is None:
            os.environ.pop("VIDEO_GPU_GRANT_POLICY", None)
        else:
            os.environ["VIDEO_GPU_GRANT_POLICY"] = self._old

    def test_require_policy_six_tuple_atomic_reclaim_and_fail_closed_persistence(self) -> None:
        os.environ["VIDEO_GPU_GRANT_POLICY"] = "require"
        ident = {
            "pod_uid": "pod-rife-1",
            "container_id": "cid-rife-1",
            "boot_id": "boot-1",
            "supervisor_pid": os.getpid(),
            "supervisor_starttime_ticks": 82000,
            "visible_gpu_uuids": ["GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"],
        }
        headers = {
            "x-vf-gpu-lease-id": "lease-rife-1",
            "x-vf-gpu-fencing-token": "21",
            "x-vf-gpu-attempt-id": "att-rife-1",
            "x-vf-gpu-stage-id": "postprocess_rife",
            "x-vf-gpu-pod-uid": "pod-rife-1",
            "x-vf-gpu-container-id": "cid-rife-1",
            "x-vf-gpu-boot-id": "boot-1",
            "x-vf-gpu-uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "x-vf-gpu-expires-at": "2099-01-01T00:00:00Z",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "rife_state.json"
            guard = LeaseFenceGuard(state_path=state_path, write_fn=lambda fd, b: os.write(fd, bytes(b[:5])))
            grant, replay = guard.extract_and_validate(headers, task_id="rife-t1", request_digest="dig1", identity=ident)
            self.assertIsNone(replay)
            self.assertEqual(grant["stage_id"], "postprocess_rife")
            guard.verify_before_inference("rife-t1")

            with self.assertRaises(PermissionError):
                guard.extract_and_validate(
                    {**headers, "x-vf-gpu-fencing-token": "22", "x-vf-gpu-attempt-id": "att-cid", "x-vf-gpu-container-id": "cid-rife"},
                    task_id="rife-cid",
                    request_digest="dig-cid",
                    identity=ident,
                )

            # While rife-t1 is executing, reclaim_and_seal seals token 21 FIRST but reports reclaimed=False
            busy_proof = guard.reclaim_and_seal(lease_id="lease-rife-1", fencing_token=21, active_tasks=0, settled=True, identity=ident)
            self.assertFalse(busy_proof["reclaimed"])
            self.assertTrue(busy_proof["token_sealed"])
            self.assertGreaterEqual(busy_proof["active_tasks"], 1)

            guard.seal_task("rife-t1")
            with self.assertRaises(StaleFenceError):
                guard.verify_before_inference("rife-t1")
            proof = guard.reclaim_and_seal(lease_id="lease-rife-1", fencing_token=21, active_tasks=0, settled=True, identity=ident)
            self.assertTrue(proof["reclaimed"])

            # Barrier race check
            barrier = threading.Barrier(2)
            res: dict[str, bool] = {}
            hdr25 = {**headers, "x-vf-gpu-lease-id": "lease-rife-25", "x-vf-gpu-fencing-token": "25", "x-vf-gpu-attempt-id": "att-25"}

            def t_infer() -> None:
                barrier.wait()
                try:
                    guard.extract_and_validate(hdr25, task_id="rife-t25", request_digest="d25", identity=ident)
                    guard.verify_before_inference("rife-t25")
                    res["infer"] = True
                except StaleFenceError:
                    res["infer"] = False

            def t_reclaim() -> None:
                barrier.wait()
                p = guard.reclaim_and_seal(lease_id="lease-rife-25", fencing_token=25, active_tasks=0, settled=True, identity=ident)
                res["reclaimed"] = bool(p and p["reclaimed"])

            th1, th2 = threading.Thread(target=t_infer), threading.Thread(target=t_reclaim)
            th1.start()
            th2.start()
            th1.join()
            th2.join()
            self.assertFalse(res.get("infer") and res.get("reclaimed"))

            # Corrupted state file fails closed
            corrupt = Path(tmpdir) / "bad.json"
            corrupt.write_bytes(b"\xff\xfe{bad")
            bad_guard = LeaseFenceGuard(state_path=corrupt)
            with self.assertRaises(FencePersistenceError):
                bad_guard.reclaim_and_seal(lease_id="lease-x", fencing_token=99, identity=ident)

    def test_t41_pc_rife_lost_response_stage_replay_single_gpu_spawn(self) -> None:
        import hashlib
        import json
        from fastapi import HTTPException
        from video_interpolation.runtime import Queue
        from video_interpolation.schemas import InterpolationRequest

        os.environ["VIDEO_GPU_GRANT_POLICY"] = "require"
        ident = {
            "pod_uid": "pod-rife-pc",
            "container_id": "c" * 64,
            "boot_id": "boot-rife-pc",
            "supervisor_pid": os.getpid(),
            "supervisor_starttime_ticks": 95000,
            "visible_gpu_uuids": ["GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "queue"
            art_root = Path(tmpdir) / "art_root"
            art_id = "art_" + "1" * 32
            src_dir = art_root / "artifacts" / art_id
            src_dir.mkdir(parents=True, exist_ok=True)
            mp4_bytes = b"\x00\x00\x00\x18ftypmp42stage_replay_payload"
            src_sha = hashlib.sha256(mp4_bytes).hexdigest()
            (src_dir / "clip.mp4").write_bytes(mp4_bytes)
            (src_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "project_id": "proj_pc",
                        "artifact_id": art_id,
                        "filename": "clip.mp4",
                        "media_type": "video/mp4",
                        "size": len(mp4_bytes),
                        "sha256": src_sha,
                    }
                ),
                encoding="utf-8",
            )

            guard = LeaseFenceGuard(state_path=Path(tmpdir) / "guard.json")
            q = Queue(root=root, artifact_root=art_root, engine=object(), fence_guard=guard)
            q.identity = ident
            try:
                headers = {
                    "x-vf-gpu-lease-id": "lease-p3-rife",
                    "x-vf-gpu-fencing-token": "41",
                    "x-vf-gpu-parent-task-id": "parent_video_task_001",
                    "x-vf-gpu-attempt-id": "attempt-p3-1",
                    "x-vf-gpu-stage-id": "postprocess_interp",
                    "x-vf-gpu-pod-uid": ident["pod_uid"],
                    "x-vf-gpu-container-id": ident["container_id"],
                    "x-vf-gpu-boot-id": ident["boot_id"],
                    "x-vf-gpu-uuid": ident["visible_gpu_uuids"][0],
                    "x-vf-gpu-expires-at": "2099-01-01T00:00:00Z",
                }
                req1 = InterpolationRequest(
                    project_id="proj_pc",
                    idempotency_key="video_task_" + "a" * 32,
                    source_artifact_id=art_id,
                    source_sha256=src_sha,
                    target_fps_num=48,
                    target_fps_den=1,
                )
                sub1 = q.submit(req1, headers=headers)
                self.assertEqual(sub1["video_task_id"], req1.idempotency_key)

                # Simulate lost HTTP response: upstream retries the SAME (parent_task_id, attempt_id, stage_id)
                # first with identical idempotency_key AND then with a regenerated idempotency_key!
                sub_same_key = q.submit(req1, headers=headers)
                self.assertEqual(sub_same_key["video_task_id"], req1.idempotency_key)

                req2_new_key = InterpolationRequest(
                    project_id="proj_pc",
                    idempotency_key="video_task_" + "b" * 32,
                    source_artifact_id=art_id,
                    source_sha256=src_sha,
                    target_fps_num=48,
                    target_fps_den=1,
                )
                sub_stage_replay = q.submit(req2_new_key, headers=headers)
                self.assertEqual(sub_stage_replay["video_task_id"], req1.idempotency_key)

                # Verify SQLite queue contains strictly 1 row (zero second RIFE GPU worker enqueued/spawned)!
                total_rows = q.db.execute("SELECT count(*) FROM tasks").fetchone()[0]
                self.assertEqual(total_rows, 1)

                # Mutated stage parameter under the same (parent_task_id, attempt_id, stage_id) MUST fail 409 attempt_stage_conflict!
                req_mutated = InterpolationRequest(
                    project_id="proj_pc",
                    idempotency_key="video_task_" + "c" * 32,
                    source_artifact_id=art_id,
                    source_sha256=src_sha,
                    target_fps_num=60,
                    target_fps_den=1,
                )
                with self.assertRaises(HTTPException) as ctx:
                    q.submit(req_mutated, headers=headers)
                self.assertEqual(ctx.exception.status_code, 409)
            finally:
                q.close()


if __name__ == "__main__":
    unittest.main()
