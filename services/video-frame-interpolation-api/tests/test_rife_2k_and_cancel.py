"""Unit tests for RIFE 2K (<=3072) geometry, preview_geometry, aac_copy_if_aligned, and DELETE cancel."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np

from video_interpolation.integrity import sha256
from video_interpolation.media import MediaError, preview_geometry, probe, process, validate_request
from video_interpolation.runtime import Queue, create_app
from video_interpolation.schemas import InterpolationRequest


class _FakeEngine:
    def __init__(self, *, frame_delay: float = 0.0):
        self.frame_delay = frame_delay
        self.begin_calls = 0
        self.end_calls = 0
        self.midpoint_calls = 0
        self.in_midpoint_event = threading.Event()
        self.metadata = {
            "backend": "rife",
            "code_revision": "5d8adbdd40e12c2c8f91930eff838aebe561c086",
            "weights_revision": "sha256:" + "6" * 64,
            "manifest_sha256": "a" * 64,
            "weights_sha256": {"flownet.pkl": "2" * 64},
            "runtime_version": "video-frame-interpolation-api-v0.2.5",
        }

    def begin_task(self):
        self.begin_calls += 1
        return {
            "free_bytes_before": 24 * 1024**3,
            "total_bytes": 32 * 1024**3,
            "minimum_free_bytes": 8 * 1024**3,
        }

    def end_task(self):
        self.end_calls += 1
        return 512 * 1024**2

    def midpoint(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        self.midpoint_calls += 1
        self.in_midpoint_event.set()
        if self.frame_delay > 0:
            time.sleep(self.frame_delay)
        return ((left.astype(np.uint16) + right.astype(np.uint16)) // 2).astype(np.uint8)


def _write_artifact_clip(
    artifact_root: Path,
    *,
    project_id: str = "proj_alpha",
    artifact_id: str,
    width: int = 320,
    height: int = 192,
    frames: int = 6,
    fps: int = 24,
    audio_mode: str = "aac_aligned",
) -> tuple[Path, str]:
    art_dir = artifact_root / "artifacts" / artifact_id
    art_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = art_dir / "clip.mp4"
    video_dur = frames / float(fps)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size={width}x{height}:rate={fps}",
    ]
    if audio_mode != "none":
        cmd.extend(["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"])
    cmd.extend(
        [
            "-frames:v",
            str(frames),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-colorspace",
            "bt709",
        ]
    )
    if audio_mode == "aac_aligned":
        cmd.extend(["-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "128k", "-t", f"{video_dur:.6f}"])
    elif audio_mode == "mp3":
        cmd.extend(["-map", "0:v:0", "-map", "1:a:0", "-c:a", "libmp3lame", "-b:a", "128k", "-t", f"{video_dur:.6f}"])
    elif audio_mode == "aac_misaligned":
        # Audio 0.6s longer than video (> 0.1s threshold -> must transcode & trim)
        cmd.extend(["-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "128k", "-t", f"{video_dur + 0.6:.6f}"])
    else:
        cmd.extend(["-an"])
    cmd.extend(["-movflags", "+faststart", str(mp4_path)])
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    digest = sha256(mp4_path)
    meta = {
        "schema_version": "1.0",
        "artifact_id": artifact_id,
        "project_id": project_id,
        "filename": "clip.mp4",
        "media_type": "video/mp4",
        "size": mp4_path.stat().st_size,
        "sha256": digest,
    }
    (art_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return mp4_path, digest


class Rife2KAndCancelTests(unittest.TestCase):
    def test_preview_geometry_golden_table_and_max_edge_validation(self):
        golden = {
            (768, 1344): (1536, 1344),
            (2464, 1408): (3840, 1096),
            (1408, 2464): (2468, 2160),
            (1536, 2688): (2468, 2160),
            (3072, 1344): (3840, 840),
        }
        for (w, h), expected in golden.items():
            self.assertEqual(preview_geometry(w, h), expected)

        req = InterpolationRequest(
            project_id="proj_alpha",
            idempotency_key="video_task_" + "a" * 32,
            source_artifact_id="art_" + "b" * 32,
            source_sha256="c" * 64,
            target_fps_num=48,
            target_fps_den=1,
        )
        for w, h in ((2464, 1408), (1408, 2464), (1536, 2688), (3072, 1344)):
            media = {"width": w, "height": h, "frames": 124, "frame_rate": "24/1", "duration_seconds": 124 / 24, "video_codec": "h264", "audio": True}
            validated = validate_request(req, media)
            self.assertEqual((validated["width"], validated["height"], validated["frames"]), (w, h, 248))

        over_limit = {"width": 3200, "height": 1344, "frames": 124, "frame_rate": "24/1", "duration_seconds": 124 / 24, "video_codec": "h264", "audio": True}
        with self.assertRaisesRegex(MediaError, "input_exceeds_resource_limit"):
            validate_request(req, over_limit)

    def test_2k_landscape_and_portrait_processing_with_audio_copy_transcode_and_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_root = root / "source"
            engine = _FakeEngine()

            # 1) 2464x1408 (2K 16:9) with aligned AAC -> audio_mode == 'copy', preview == 3840x1096
            art_1 = "art_" + "1" * 32
            src_1, sha_1 = _write_artifact_clip(
                artifact_root,
                artifact_id=art_1,
                width=2464,
                height=1408,
                frames=4,
                fps=24,
                audio_mode="aac_aligned",
            )
            req_1 = InterpolationRequest(
                project_id="proj_alpha",
                idempotency_key="video_task_" + "1" * 32,
                source_artifact_id=art_1,
                source_sha256=sha_1,
            )
            out_dir_1 = root / "out1"
            out_dir_1.mkdir()
            res_1 = process(engine, src_1, out_dir_1, req_1)
            self.assertEqual((res_1["media"]["width"], res_1["media"]["height"], res_1["media"]["frames"]), (2464, 1408, 8))
            self.assertEqual((res_1["preview_media"]["width"], res_1["preview_media"]["height"]), (3840, 1096))
            self.assertEqual(res_1["media"]["audio_mode"], "copy")
            self.assertEqual(res_1["parameters"]["audio"], "aac_copy_if_aligned")

            # 2) 1408x2464 (2K 9:16, h > 2048!) with MP3 audio -> audio_mode == 'transcode', preview == 2468x2160
            art_2 = "art_" + "2" * 32
            src_2, sha_2 = _write_artifact_clip(
                artifact_root,
                artifact_id=art_2,
                width=1408,
                height=2464,
                frames=4,
                fps=24,
                audio_mode="mp3",
            )
            req_2 = InterpolationRequest(
                project_id="proj_alpha",
                idempotency_key="video_task_" + "2" * 32,
                source_artifact_id=art_2,
                source_sha256=sha_2,
            )
            out_dir_2 = root / "out2"
            out_dir_2.mkdir()
            res_2 = process(engine, src_2, out_dir_2, req_2)
            self.assertEqual((res_2["media"]["width"], res_2["media"]["height"], res_2["media"]["frames"]), (1408, 2464, 8))
            self.assertEqual((res_2["preview_media"]["width"], res_2["preview_media"]["height"]), (2468, 2160))
            self.assertEqual(res_2["media"]["audio_mode"], "transcode")

            # 3) Silent clip -> audio_mode == 'none'
            art_3 = "art_" + "3" * 32
            src_3, sha_3 = _write_artifact_clip(
                artifact_root,
                artifact_id=art_3,
                width=320,
                height=192,
                frames=4,
                fps=24,
                audio_mode="none",
            )
            req_3 = InterpolationRequest(
                project_id="proj_alpha",
                idempotency_key="video_task_" + "3" * 32,
                source_artifact_id=art_3,
                source_sha256=sha_3,
            )
            out_dir_3 = root / "out3"
            out_dir_3.mkdir()
            res_3 = process(engine, src_3, out_dir_3, req_3)
            self.assertEqual(res_3["media"]["audio_mode"], "none")
            self.assertFalse(res_3["media"]["audio"])
            self.assertEqual((res_3["preview_media"]["width"], res_3["preview_media"]["height"]), (640, 192))

    def test_delete_cancel_queued_running_idempotent_and_terminal_409(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proc_root = root / "processing"
            artifact_root = root / "source"
            art_id = "art_" + "4" * 32
            _, digest = _write_artifact_clip(
                artifact_root,
                artifact_id=art_id,
                width=160,
                height=96,
                frames=10,
                fps=24,
                audio_mode="aac_aligned",
            )
            slow_engine = _FakeEngine(frame_delay=0.25)

            with patch.dict(
                os.environ,
                {
                    "VIDEO_PROCESSING_ROOT": str(proc_root),
                    "VIDEO_ARTIFACT_ROOT": str(artifact_root),
                },
            ):
                app = create_app(lambda: slow_engine)
                with TestClient(app) as client:
                    # Submit task A (starts running) and task B (stays queued behind A)
                    task_a = "video_task_" + "a" * 32
                    task_b = "video_task_" + "b" * 32
                    r_a = client.post(
                        "/v1/interpolate",
                        json={
                            "project_id": "proj_alpha",
                            "idempotency_key": task_a,
                            "source_artifact_id": art_id,
                            "source_sha256": digest,
                        },
                    )
                    self.assertEqual(r_a.status_code, 202)
                    r_b = client.post(
                        "/v1/interpolate",
                        json={
                            "project_id": "proj_alpha",
                            "idempotency_key": task_b,
                            "source_artifact_id": art_id,
                            "source_sha256": digest,
                        },
                    )
                    self.assertEqual(r_b.status_code, 202)

                    # Wait until task A enters midpoint inference
                    self.assertTrue(slow_engine.in_midpoint_event.wait(timeout=5.0))
                    health = client.get("/health").json()
                    self.assertTrue(health["gpu_worker_active"])
                    self.assertEqual(health["active_task_ids"], [task_a])

                    # Cancel queued task B immediately -> 200 cancelled, settled
                    del_b = client.delete(f"/v1/interpolate/{task_b}")
                    self.assertEqual(del_b.status_code, 200)
                    self.assertEqual(del_b.json()["status"], "cancelled")
                    self.assertTrue(del_b.json()["worker_settled"])

                    # Cancel running task A -> waits for engine.end_task() and settles before returning 200
                    del_a = client.delete(f"/v1/interpolate/{task_a}")
                    self.assertEqual(del_a.status_code, 200)
                    body_a = del_a.json()
                    self.assertEqual(body_a["status"], "cancelled")
                    self.assertTrue(body_a["worker_settled"])
                    self.assertIsNotNone(body_a["started_at"])
                    self.assertEqual(slow_engine.end_calls, 1)

                    # Repeat DELETE on already-cancelled task -> idempotent 200
                    del_a_again = client.delete(f"/v1/interpolate/{task_a}")
                    self.assertEqual(del_a_again.status_code, 200)
                    self.assertEqual(del_a_again.json()["status"], "cancelled")

                    # Health reports idle again
                    health_after = client.get("/health").json()
                    self.assertFalse(health_after["gpu_worker_active"])
                    self.assertEqual(health_after["active_task_ids"], [])

                    # Submit fast task C -> let it succeed -> DELETE returns 409 task_already_terminal
                    slow_engine.frame_delay = 0.0
                    task_c = "video_task_" + "c" * 32
                    client.post(
                        "/v1/interpolate",
                        json={
                            "project_id": "proj_alpha",
                            "idempotency_key": task_c,
                            "source_artifact_id": art_id,
                            "source_sha256": digest,
                        },
                    )
                    deadline = time.time() + 15.0
                    while time.time() < deadline:
                        st = client.get(f"/v1/interpolate/{task_c}").json()
                        if st["status"] == "succeeded":
                            break
                        time.sleep(0.1)
                    self.assertEqual(st["status"], "succeeded")
                    self.assertTrue(st["worker_settled"])
                    del_c = client.delete(f"/v1/interpolate/{task_c}")
                    self.assertEqual(del_c.status_code, 409)
                    self.assertEqual(del_c.json()["detail"]["code"], "task_already_terminal")


if __name__ == "__main__":
    unittest.main()
