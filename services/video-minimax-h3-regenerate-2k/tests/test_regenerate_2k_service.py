"""Offline unit tests for video-minimax-h3-regenerate-2k (h3_regenerate_2k)."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tracemalloc
import unittest
from unittest.mock import MagicMock, patch
import weakref

from fastapi.testclient import TestClient
import numpy as np

from h3_regenerate_2k import api as api_mod
from h3_regenerate_2k.api import create_app
from h3_regenerate_2k.contract import (
    check_weights_health,
    preview_geometry,
    resolve_2k_target_geometry,
    temporal_windows,
    validate_regenerate_request,
)
from h3_regenerate_2k.errors import RuntimeErrorCode
from h3_regenerate_2k.queue import Queue
from h3_regenerate_2k.regenerate import (
    probe_audio_packets_sha256,
    render_side_by_side_preview,
    sha256_file,
    stitch_window_frames,
)
from h3_regenerate_2k.worker import check_free_vram


STANDARD_H3_768P_CANVASES = {
    "21:9": (1536, 672),
    "16:9": (1344, 768),
    "4:3": (1024, 768),
    "1:1": (768, 768),
    "3:4": (768, 1024),
    "9:16": (768, 1344),
}


def _populate_p2_sparse_weights(base_root: Path) -> None:
    exact_files = [
        (
            base_root
            / "candidates"
            / "shared"
            / "text_encoders"
            / "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            15_687_142_551,
        ),
        (base_root / "candidates" / "shared" / "vae" / "minimax_h3_video_vae_fp16.safetensors", 5_207_808_496),
        (
            base_root
            / "candidates"
            / "a2_vsa_turbo_v4"
            / "diffusion_models"
            / "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            20_970_379_616,
        ),
        (
            base_root
            / "candidates"
            / "p2_open_h3_2k_ic"
            / "comfyui"
            / "minimax_h3_hyperflow_8step_v1.0_comfyui_bf16.safetensors",
            3_926_338_288,
        ),
        (
            base_root
            / "candidates"
            / "p2_open_h3_2k_ic"
            / "official"
            / "lora"
            / "minimax_h3_lms_v1.0_r64.safetensors",
            1_239_664_536,
        ),
        (
            base_root
            / "candidates"
            / "p2_open_h3_2k_ic"
            / "comfyui"
            / "minimax_h3_open_regenerate_2k_ic_comfy_bf16.safetensors",
            5_868_430_560,
        ),
        (
            base_root
            / "candidates"
            / "p2_open_h3_2k_ic"
            / "official"
            / "latent_upsampler"
            / "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
            690_592_992,
        ),
        (
            base_root
            / "candidates"
            / "p2_open_h3_2k_ic"
            / "official"
            / "assets"
            / "silence_audio_124f.safetensors",
            26_768,
        ),
    ]
    for path, size in exact_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        if os.name == "nt":
            import ctypes
            import msvcrt

            subprocess.run(["fsutil", "sparse", "setflag", str(path)], check=True, capture_output=True)
            subprocess.run(["fsutil", "sparse", "setrange", str(path), "0", str(size)], check=True, capture_output=True)
            with path.open("r+b") as handle:
                hfile = ctypes.c_void_p(msvcrt.get_osfhandle(handle.fileno()))
                new_pos = ctypes.c_longlong(0)
                ctypes.windll.kernel32.SetFilePointerEx(hfile, ctypes.c_longlong(size), ctypes.byref(new_pos), 0)
                ctypes.windll.kernel32.SetEndOfFile(hfile)
        else:
            with path.open("wb") as handle:
                handle.truncate(size)


def _write_artifact_video(
    artifact_root: Path,
    *,
    project_id: str,
    artifact_id: str,
    width: int = 1344,
    height: int = 768,
    frames: int = 107,
    fps: int = 24,
    audio_codec: str | None = "aac",
) -> tuple[Path, str]:
    art_dir = artifact_root / "artifacts" / artifact_id
    art_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = art_dir / "clip.mp4"
    duration = frames / float(fps)
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
    if audio_codec is not None:
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
            "24",
            "-pix_fmt",
            "yuv420p",
        ]
    )
    if audio_codec == "aac":
        cmd.extend(["-c:a", "aac", "-b:a", "128k", "-t", f"{duration:.6f}"])
    elif audio_codec == "mp3":
        cmd.extend(["-c:a", "libmp3lame", "-b:a", "128k", "-t", f"{duration:.6f}"])
    cmd.extend(["-movflags", "+faststart", str(mp4_path)])
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    digest = sha256_file(mp4_path)
    metadata = {
        "schema_version": "1.0",
        "artifact_id": artifact_id,
        "project_id": project_id,
        "operation": "video.create",
        "kind": "video",
        "filename": "clip.mp4",
        "media_type": "video/mp4",
        "size": mp4_path.stat().st_size,
        "sha256": digest,
        "created_at": "2026-11-14T00:00:00Z",
    }
    (art_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return mp4_path, digest


class Regenerate2KServiceTests(unittest.TestCase):
    def test_api_module_has_zero_torch_import_and_zero_singularity_env(self):
        pkg_dir = Path(__file__).resolve().parents[1] / "src" / "h3_regenerate_2k"
        src_dir = str(pkg_dir.parent)
        env = os.environ.copy()
        env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
        code = (
            "import sys, h3_regenerate_2k.api; "
            "assert 'torch' not in sys.modules, f'torch leaked: {list(sys.modules)}'; "
            "assert not any(k.startswith('comfy') for k in sys.modules), 'comfy leaked'"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)

        for py_file in pkg_dir.glob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            self.assertNotIn("SINGULARITY_", content, f"SINGULARITY_* leaked into {py_file.name}")
            self.assertNotIn("/v1/videos", content, f"/v1/videos leaked into {py_file.name}")

    def test_cross_service_endpoints_404_and_queue_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = Queue(Path(tmp) / "tasks")
            headers = {"Authorization": "Bearer test-token"}
            try:
                cols = {row[1] for row in queue.db.execute("PRAGMA table_info(tasks)").fetchall()}
                self.assertNotIn("kind", cols)
                app = create_app(
                    queue,
                    "test-token",
                    health_checker=lambda **kw: {
                        "ready": True,
                        "capabilities": {
                            "regenerate_2k_profiles": ["open_h3_2k_ic"],
                            "regenerate_2k_grid_modes": ["budget_1440p", "exact_2x"],
                        },
                        **kw,
                    },
                )
                with TestClient(app) as client:
                    self.assertEqual(client.post("/v1/videos", json={}, headers=headers).status_code, 404)
                    self.assertEqual(
                        client.get("/v1/videos/fasth3_" + "a" * 32, headers=headers).status_code,
                        404,
                    )
                    self.assertEqual(
                        client.get("/v1/videos/by-idempotency/video_task_" + "a" * 32, headers=headers).status_code,
                        404,
                    )
            finally:
                queue.close()

    def test_offline_source_admission_http_lifecycle_and_audio_packet_digest(self):
        from h3_regenerate_2k.regenerate import probe_and_validate_source_video, resolve_source_artifact_path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_root = root / "source_video"
            art_id = "art_" + "1" * 32
            art_dir = artifact_root / "artifacts" / art_id
            art_dir.mkdir(parents=True, exist_ok=True)
            mp4_path = art_dir / "clip.mp4"
            mp4_bytes = b"fake-mp4-binary-payload"
            mp4_path.write_bytes(mp4_bytes)
            digest = sha256_file(mp4_path)
            (art_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "artifact_id": art_id,
                        "project_id": "proj-xie",
                        "kind": "video",
                        "filename": "clip.mp4",
                        "media_type": "video/mp4",
                        "size": len(mp4_bytes),
                        "sha256": digest,
                    }
                ),
                encoding="utf-8",
            )

            req = validate_regenerate_request(
                {
                    "project_id": "proj-xie",
                    "idempotency_key": "video_task_" + "1" * 32,
                    "source_artifact_id": art_id,
                    "source_sha256": digest,
                }
            )
            self.assertEqual(resolve_source_artifact_path(req, artifact_root), mp4_path)

            # Integrity mismatch -> source_integrity_failed
            bad_req = dict(req, source_sha256="f" * 64)
            with self.assertRaisesRegex(RuntimeErrorCode, "source_integrity_failed"):
                resolve_source_artifact_path(bad_req, artifact_root)

            # Audio packet SHA-256 deterministic check
            with patch("h3_regenerate_2k.regenerate.subprocess.check_output", return_value=b"aac-raw-packets"):
                self.assertEqual(
                    probe_audio_packets_sha256(mp4_path),
                    probe_audio_packets_sha256(mp4_path),
                )

            # Non-AAC audio rejected by probe_and_validate_source_video
            with patch(
                "h3_regenerate_2k.regenerate.probe_mp4_summary",
                return_value={
                    "width": 1344,
                    "height": 768,
                    "frames": 107,
                    "frame_rate": "24/1",
                    "r_frame_rate": "24/1",
                    "duration_seconds": 107 / 24.0,
                    "video_codec": "h264",
                    "audio": True,
                    "audio_codec": "mp3",
                },
            ):
                with self.assertRaisesRegex(RuntimeErrorCode, "invalid_source_media"):
                    probe_and_validate_source_video(mp4_path, "budget_1440p")

            # Already-2K rejected by probe_and_validate_source_video
            with patch(
                "h3_regenerate_2k.regenerate.probe_mp4_summary",
                return_value={
                    "width": 2464,
                    "height": 1408,
                    "frames": 107,
                    "frame_rate": "24/1",
                    "r_frame_rate": "24/1",
                    "duration_seconds": 107 / 24.0,
                    "video_codec": "h264",
                    "audio": True,
                    "audio_codec": "aac",
                },
            ):
                with self.assertRaisesRegex(RuntimeErrorCode, "source_already_2k"):
                    probe_and_validate_source_video(mp4_path, "budget_1440p")

            # Full HTTP 202 -> succeeded -> /content -> /preview -> DELETE 409 lifecycle
            fast_worker = root / "fast_regen_worker.py"
            fast_worker.write_text(
                "import pathlib, sys\n"
                "from h3_regenerate_2k.queue import Queue\n"
                "from h3_regenerate_2k.regenerate import sha256_file\n"
                "task_id, task_root = sys.argv[1], pathlib.Path(sys.argv[2])\n"
                "q = Queue(task_root, acquire_lock=False)\n"
                "tdir = task_root / task_id\n"
                "tdir.mkdir(parents=True, exist_ok=True)\n"
                "(tdir / 'output.mp4').write_bytes(b'out2k')\n"
                "(tdir / 'preview.mp4').write_bytes(b'prev2k')\n"
                "q.update(task_id, status='succeeded', stage='succeeded', result={'output_sha256': sha256_file(tdir / 'output.mp4'), 'preview_sha256': sha256_file(tdir / 'preview.mp4')}, worker_settled=0)\n"
                "q.close()\n",
                encoding="utf-8",
            )
            src_dir = str(Path(__file__).resolve().parents[1] / "src")
            queue = Queue(root / "tasks")
            headers = {"Authorization": "Bearer test-token"}
            try:
                with patch.dict(
                    os.environ,
                    {
                        "REGENERATE_2K_ARTIFACT_ROOT": str(artifact_root),
                        "PYTHONPATH": src_dir + os.pathsep + os.environ.get("PYTHONPATH", ""),
                    },
                ), patch(
                    "h3_regenerate_2k.api.probe_and_validate_source_video",
                    return_value=({}, {}, [(0, 107)]),
                ):
                    app = create_app(
                        queue,
                        "test-token",
                        worker_cmd_builder=lambda tid, troot: [sys.executable, str(fast_worker), tid, str(troot)],
                        health_checker=lambda **kw: {"ready": True, **kw},
                    )
                    with TestClient(app) as client:
                        tid = "video_task_" + "9" * 32
                        r_ok = client.post(
                            "/v1/regenerate-2k",
                            json={
                                "project_id": "proj-xie",
                                "idempotency_key": tid,
                                "source_artifact_id": art_id,
                                "source_sha256": digest,
                                "profile": "open_h3_2k_ic",
                                "grid_mode": "budget_1440p",
                                "prompt": "courtyard",
                                "seed": 7,
                            },
                            headers=headers,
                        )
                        self.assertEqual(r_ok.status_code, 202)
                        deadline = time.monotonic() + 5.0
                        while time.monotonic() < deadline:
                            st = client.get(f"/v1/regenerate-2k/{tid}", headers=headers).json()
                            if st["status"] == "succeeded" and st["worker_settled"]:
                                break
                            time.sleep(0.05)
                        self.assertEqual(st["status"], "succeeded")
                        self.assertTrue(st["worker_settled"])
                        self.assertEqual(
                            client.get(f"/v1/regenerate-2k/{tid}/content", headers=headers).content,
                            b"out2k",
                        )
                        self.assertEqual(
                            client.get(f"/v1/regenerate-2k/{tid}/preview", headers=headers).content,
                            b"prev2k",
                        )
                        del_done = client.delete(f"/v1/regenerate-2k/{tid}", headers=headers)
                        self.assertEqual(del_done.status_code, 409)
                        self.assertEqual(del_done.json()["detail"]["code"], "task_already_terminal")
            finally:
                queue.close()

    def test_six_canvases_two_grid_modes_and_rejection_boundaries(self):
        expected_budget = {
            "21:9": (2560, 1120, [[2560, 1120]]),
            "16:9": (2464, 1408, [[2464, 1408]]),
            "4:3": (1920, 1440, [[1920, 1440]]),
            "1:1": (1440, 1440, [[1440, 1440]]),
            "3:4": (1440, 1920, [[1440, 1920]]),
            "9:16": (1408, 2464, [[1408, 2464]]),
        }
        expected_exact = {
            "21:9": (3072, 1344, [[2560, 1120], [3072, 1344]]),
            "16:9": (2688, 1536, [[2464, 1408], [2688, 1536]]),
            "4:3": (2048, 1536, [[1920, 1440], [2048, 1536]]),
            "1:1": (1536, 1536, [[1440, 1440], [1536, 1536]]),
            "3:4": (1536, 2048, [[1440, 1920], [1536, 2048]]),
            "9:16": (1536, 2688, [[1408, 2464], [1536, 2688]]),
        }
        for ratio, (src_w, src_h) in STANDARD_H3_768P_CANVASES.items():
            b = resolve_2k_target_geometry(src_w, src_h, "budget_1440p")
            self.assertEqual((b["target_w"], b["target_h"], b["levels"]), expected_budget[ratio])
            self.assertEqual(src_w * b["target_h"], b["target_w"] * src_h)
            e = resolve_2k_target_geometry(src_w, src_h, "exact_2x")
            self.assertEqual((e["target_w"], e["target_h"], e["levels"]), expected_exact[ratio])
            self.assertEqual(src_w * e["target_h"], e["target_w"] * src_h)

        adaptive_prime = resolve_2k_target_geometry(1312, 768, "budget_1440p")
        self.assertEqual((adaptive_prime["target_w"], adaptive_prime["target_h"]), (2624, 1536))

        for w, h in ((2464, 1408), (1408, 2464), (2560, 1120), (1440, 1440), (2688, 1536)):
            for mode in ("budget_1440p", "exact_2x"):
                with self.assertRaisesRegex(RuntimeErrorCode, "source_already_2k"):
                    resolve_2k_target_geometry(w, h, mode)

        with self.assertRaisesRegex(RuntimeErrorCode, "invalid_source_media"):
            resolve_2k_target_geometry(1280, 720, "budget_1440p")
        with self.assertRaisesRegex(RuntimeErrorCode, "invalid_source_media"):
            resolve_2k_target_geometry(1600, 768, "exact_2x")

    def test_all_17k_plus_5_buckets_crossfade_and_single_frame_streaming_memory(self):
        buckets = [5 + 17 * k for k in range(6, 22)]
        for n_frames in buckets:
            windows = temporal_windows(n_frames)
            self.assertEqual(windows[0][0], 0)
            self.assertEqual(windows[-1][1], n_frames)
            win_arrays = [
                np.full((we - ws, 8, 8, 3), fill_value=100, dtype=np.uint8)
                for ws, we in windows
            ]
            stitched = list(stitch_window_frames(iter(win_arrays), windows))
            self.assertEqual(len(stitched), n_frames)
            for frame in stitched:
                self.assertEqual(frame.shape, (8, 8, 3))
                self.assertTrue(np.all(frame == 100))

        windows_141 = temporal_windows(141)
        self.assertEqual(windows_141, [(0, 124), (17, 141)])
        h, w = 128, 224
        weak_refs: list[weakref.ReferenceType] = []
        prev_alive_on_next_window: list[bool] = []

        def gen_windows():
            for idx, (ws, we) in enumerate(windows_141):
                gc.collect()
                if idx > 0:
                    prev_alive_on_next_window.append(weak_refs[idx - 1]() is not None)
                arr = np.full((we - ws, h, w, 3), 50 if idx == 0 else 200, dtype=np.uint8)
                weak_refs.append(weakref.ref(arr))
                box = [arr]
                del arr
                yield box
                del box

        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            count = 0
            frame = None
            for frame in stitch_window_frames(gen_windows(), windows_141):
                count += 1
                if count == 20:
                    self.assertTrue(50 < int(frame[0, 0, 0]) < 200)
                frame = None
            _, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        self.assertEqual(count, 141)
        self.assertEqual(prev_alive_on_next_window, [False])
        full_batch_f32_bytes = 107 * h * w * 3 * 4 * 3
        self.assertLess(peak_bytes, full_batch_f32_bytes // 2)

    def test_preview_geometry_golden_table_and_health_contract(self):
        golden = {
            (768, 1344): (1536, 1344),
            (2464, 1408): (3840, 1096),
            (1408, 2464): (2468, 2160),
            (1536, 2688): (2468, 2160),
            (3072, 1344): (3840, 840),
        }
        for (src_w, src_h), expected in golden.items():
            self.assertEqual(preview_geometry(src_w, src_h), expected)

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _populate_p2_sparse_weights(base)
            health = check_weights_health(
                model_root=base,
                gpu_worker_active=True,
                active_task_ids=["video_task_" + "a" * 32],
            )
            self.assertTrue(health["ready"])
            self.assertTrue(health["gpu_worker_active"])
            self.assertEqual(health["active_task_ids"], ["video_task_" + "a" * 32])
            self.assertEqual(health["capabilities"]["regenerate_2k_profiles"], ["open_h3_2k_ic"])
            self.assertEqual(health["capabilities"]["regenerate_2k_grid_modes"], ["budget_1440p", "exact_2x"])
            self.assertNotIn("default_quality_profile", health)
            self.assertNotIn("quality_profiles", health["capabilities"])

            bad_file = base / "candidates" / "p2_open_h3_2k_ic" / "official" / "assets" / "silence_audio_124f.safetensors"
            bad_file.write_bytes(b"short")
            with self.assertRaisesRegex(RuntimeErrorCode, "weight_size_mismatch:p2_silence_audio_124f"):
                check_weights_health(model_root=base)

        gpu_uuid = "GPU-798fb508-830f-b538-6733-72ed6d210d40"
        with patch.dict(os.environ, {"NVIDIA_VISIBLE_DEVICES": gpu_uuid}):
            with patch("h3_regenerate_2k.worker.subprocess.check_output", return_value=f"{gpu_uuid}, 31500\n"):
                self.assertEqual(check_free_vram(28672), (gpu_uuid, 31500))
            with patch("h3_regenerate_2k.worker.subprocess.check_output", return_value=f"{gpu_uuid}, 21000\n"):
                with self.assertRaisesRegex(RuntimeErrorCode, "gpu_memory_occupancy_changed"):
                    check_free_vram(28672)

    @unittest.skipIf(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, "ffmpeg/ffprobe required")
    def test_http_endpoints_cross_route_404_audio_copy_preview_and_reaper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_root = root / "source_video"
            valid_art_id = "art_" + "1" * 32
            valid_mp4, valid_sha = _write_artifact_video(
                artifact_root,
                project_id="proj-xie",
                artifact_id=valid_art_id,
                width=1344,
                height=768,
                frames=107,
                fps=24,
                audio_codec="aac",
            )
            already_2k_id = "art_" + "2" * 32
            _, already_2k_sha = _write_artifact_video(
                artifact_root,
                project_id="proj-xie",
                artifact_id=already_2k_id,
                width=2464,
                height=1408,
                frames=107,
                fps=24,
                audio_codec="aac",
            )
            mp3_art_id = "art_" + "3" * 32
            _, mp3_sha = _write_artifact_video(
                artifact_root,
                project_id="proj-xie",
                artifact_id=mp3_art_id,
                width=1344,
                height=768,
                frames=107,
                fps=24,
                audio_codec="mp3",
            )

            # Verify side-by-side preview + lossless AAC audio packet copy
            silent_2k = root / "silent_2k.mp4"
            muxed_2k = root / "muxed_2k.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-nostdin",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=size=2464x1408:rate=24",
                    "-frames:v",
                    "107",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-pix_fmt",
                    "yuv420p",
                    str(silent_2k),
                ],
                check=True,
                capture_output=True,
                timeout=120,
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(silent_2k),
                    "-i",
                    str(valid_mp4),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "copy",
                    "-movflags",
                    "+faststart",
                    str(muxed_2k),
                ],
                check=True,
                capture_output=True,
                timeout=120,
            )
            self.assertEqual(probe_audio_packets_sha256(valid_mp4), probe_audio_packets_sha256(muxed_2k))

            preview_out = root / "preview_check.mp4"
            render_side_by_side_preview(valid_mp4, muxed_2k, preview_out, 2464, 1408)
            probe_prev = json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=width,height",
                        "-of",
                        "json",
                        str(preview_out),
                    ],
                    timeout=60,
                )
            )["streams"][0]
            self.assertEqual((probe_prev["width"], probe_prev["height"]), (3840, 1096))

            fast_worker = root / "fast_regen_worker.py"
            fast_worker.write_text(
                "import pathlib, shutil, sys\n"
                "from h3_regenerate_2k.queue import Queue\n"
                "from h3_regenerate_2k.regenerate import sha256_file\n"
                "task_id, task_root, sample_mp4 = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])\n"
                "q = Queue(task_root, acquire_lock=False)\n"
                "tdir = task_root / task_id\n"
                "tdir.mkdir(parents=True, exist_ok=True)\n"
                "shutil.copy2(sample_mp4, tdir / 'output.mp4')\n"
                "shutil.copy2(sample_mp4, tdir / 'preview.mp4')\n"
                "digest = sha256_file(tdir / 'output.mp4')\n"
                "q.update(task_id, status='succeeded', stage='succeeded', result={'output_sha256': digest, 'preview_sha256': digest}, worker_settled=0)\n"
                "q.close()\n",
                encoding="utf-8",
            )

            src_dir = str(Path(__file__).resolve().parents[1] / "src")
            queue = Queue(root / "tasks")
            headers = {"Authorization": "Bearer test-token"}
            try:
                cols = {row[1] for row in queue.db.execute("PRAGMA table_info(tasks)").fetchall()}
                self.assertNotIn("kind", cols)
                with patch.dict(
                    os.environ,
                    {
                        "REGENERATE_2K_ARTIFACT_ROOT": str(artifact_root),
                        "PYTHONPATH": src_dir + os.pathsep + os.environ.get("PYTHONPATH", ""),
                    },
                ):
                    app = create_app(
                        queue,
                        "test-token",
                        worker_cmd_builder=lambda tid, troot: [
                            sys.executable,
                            str(fast_worker),
                            tid,
                            str(troot),
                            str(muxed_2k),
                        ],
                        health_checker=lambda **kw: {
                            "ready": True,
                            "capabilities": {
                                "regenerate_2k_profiles": ["open_h3_2k_ic"],
                                "regenerate_2k_grid_modes": ["budget_1440p", "exact_2x"],
                            },
                            **kw,
                        },
                    )
                    with TestClient(app) as client:
                        # Cross-service endpoints MUST return 404
                        self.assertEqual(client.post("/v1/videos", json={}, headers=headers).status_code, 404)
                        self.assertEqual(
                            client.get("/v1/videos/fasth3_" + "a" * 32, headers=headers).status_code,
                            404,
                        )
                        self.assertEqual(
                            client.get("/v1/videos/by-idempotency/video_task_" + "a" * 32, headers=headers).status_code,
                            404,
                        )

                        # Reject already-2K input with 422 source_already_2k
                        r_already = client.post(
                            "/v1/regenerate-2k",
                            json={
                                "project_id": "proj-xie",
                                "idempotency_key": "video_task_" + "c" * 32,
                                "source_artifact_id": already_2k_id,
                                "source_sha256": already_2k_sha,
                                "profile": "open_h3_2k_ic",
                                "grid_mode": "budget_1440p",
                                "prompt": "courtyard",
                                "seed": 7,
                            },
                            headers=headers,
                        )
                        self.assertEqual(r_already.status_code, 422)
                        self.assertEqual(r_already.json()["detail"]["code"], "source_already_2k")

                        # Reject non-AAC source audio with 422 invalid_source_media
                        r_mp3 = client.post(
                            "/v1/regenerate-2k",
                            json={
                                "project_id": "proj-xie",
                                "idempotency_key": "video_task_" + "d" * 32,
                                "source_artifact_id": mp3_art_id,
                                "source_sha256": mp3_sha,
                            },
                            headers=headers,
                        )
                        self.assertEqual(r_mp3.status_code, 422)
                        self.assertEqual(r_mp3.json()["detail"]["code"], "invalid_source_media")

                        # Valid submission -> 202 Accepted -> succeeded -> content/preview 200 -> DELETE 409
                        regen_id = "video_task_" + "e" * 32
                        r_ok = client.post(
                            "/v1/regenerate-2k",
                            json={
                                "project_id": "proj-xie",
                                "idempotency_key": regen_id,
                                "source_artifact_id": valid_art_id,
                                "source_sha256": valid_sha,
                                "profile": "open_h3_2k_ic",
                                "grid_mode": "budget_1440p",
                                "prompt": "courtyard scene",
                                "seed": 7,
                            },
                            headers=headers,
                        )
                        self.assertEqual(r_ok.status_code, 202)
                        self.assertEqual(r_ok.json()["video_task_id"], regen_id)

                        deadline = time.monotonic() + 6.0
                        while time.monotonic() < deadline:
                            st = client.get(f"/v1/regenerate-2k/{regen_id}", headers=headers).json()
                            if st["status"] == "succeeded" and st["worker_settled"]:
                                break
                            time.sleep(0.05)
                        st = client.get(f"/v1/regenerate-2k/{regen_id}", headers=headers).json()
                        self.assertEqual(st["status"], "succeeded")
                        self.assertTrue(st["worker_settled"])
                        self.assertEqual(
                            client.get(f"/v1/regenerate-2k/{regen_id}/content", headers=headers).status_code,
                            200,
                        )
                        self.assertEqual(
                            client.get(f"/v1/regenerate-2k/{regen_id}/preview", headers=headers).status_code,
                            200,
                        )
                        del_done = client.delete(f"/v1/regenerate-2k/{regen_id}", headers=headers)
                        self.assertEqual(del_done.status_code, 409)
                        self.assertEqual(del_done.json()["detail"]["code"], "task_already_terminal")
            finally:
                queue.close()

    def test_posix_reaper_kills_surviving_descendant_and_cancels_running_task(self):
        fake_proc = MagicMock()
        fake_proc.pid = 42424
        poll_states = [None, 0, 0, 0]
        fake_proc.poll.side_effect = lambda: poll_states.pop(0) if poll_states else 0
        fake_proc.wait.return_value = 0

        signals_sent: list[tuple[int, int]] = []
        descendant_alive = [True]

        def fake_killpg(pgid: int, sig: int):
            if sig == 0:
                if descendant_alive[0]:
                    return None
                raise ProcessLookupError("no such process group")
            signals_sent.append((pgid, sig))
            if sig == getattr(api_mod.signal, "SIGKILL", 9):
                descendant_alive[0] = False

        with patch.object(api_mod.os, "name", "posix"), patch.object(
            api_mod.os, "killpg", side_effect=fake_killpg, create=True
        ), patch.object(api_mod.signal, "SIGKILL", 9, create=True):
            reaped = api_mod._terminate_and_reap_process_group(
                fake_proc,
                grace_seconds=0.12,
                kill_wait_seconds=0.15,
            )
        self.assertTrue(reaped)
        self.assertIn((42424, int(api_mod.signal.SIGTERM)), signals_sent)
        self.assertIn((42424, 9), signals_sent)

        # Also test live subprocess DELETE reaping on /v1/regenerate-2k/{id}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = Queue(root / "tasks")
            ready_marker = root / "worker_started.flag"
            worker_script = root / "stubborn_worker.py"
            worker_script.write_text(
                "import pathlib, signal, sys, time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "if hasattr(signal, 'SIGBREAK'):\n"
                "    signal.signal(signal.SIGBREAK, signal.SIG_IGN)\n"
                "pathlib.Path(sys.argv[1]).write_text('started', encoding='utf-8')\n"
                "while True:\n"
                "    time.sleep(0.2)\n",
                encoding="utf-8",
            )
            headers = {"Authorization": "Bearer test-token"}
            try:
                with patch(
                    "h3_regenerate_2k.api.resolve_source_artifact_path",
                    return_value=root / "dummy.mp4",
                ), patch(
                    "h3_regenerate_2k.api.probe_and_validate_source_video",
                    return_value=({}, {}, [(0, 107)]),
                ):
                    app = create_app(
                        queue,
                        "test-token",
                        worker_cmd_builder=lambda tid, _r: [
                            sys.executable,
                            str(worker_script),
                            str(ready_marker),
                            tid,
                        ],
                        health_checker=lambda **kw: {"ready": True, **kw},
                        kill_grace_seconds=0.25,
                    )
                    with TestClient(app) as client:
                        tid = "video_task_" + "f" * 32
                        resp = client.post(
                            "/v1/regenerate-2k",
                            json={
                                "project_id": "proj-xie",
                                "idempotency_key": tid,
                                "source_artifact_id": "art_" + "1" * 32,
                                "source_sha256": "a" * 64,
                            },
                            headers=headers,
                        )
                        self.assertEqual(resp.status_code, 202)
                        deadline = time.monotonic() + 5.0
                        while time.monotonic() < deadline and not ready_marker.is_file():
                            time.sleep(0.05)
                        self.assertTrue(ready_marker.is_file())

                        cancel_resp = client.delete(f"/v1/regenerate-2k/{tid}", headers=headers)
                        self.assertEqual(cancel_resp.status_code, 200)
                        self.assertEqual(cancel_resp.json()["status"], "cancelled")
                        self.assertTrue(cancel_resp.json()["worker_settled"])

                        # Repeat DELETE is idempotent 200
                        repeat = client.delete(f"/v1/regenerate-2k/{tid}", headers=headers)
                        self.assertEqual(repeat.status_code, 200)
                        self.assertEqual(repeat.json()["status"], "cancelled")
            finally:
                queue.close()


if __name__ == "__main__":
    unittest.main()
