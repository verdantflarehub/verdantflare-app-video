"""Offline unit tests for M-A zero-CUDA Supervisor, 2K IC geometry/windowing, and /v1/regenerate-2k."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np

from h3_singularity.api import create_app
from h3_singularity.contract import (
    ASPECT_RATIOS,
    check_weights_health,
    preview_geometry,
    resolve_2k_target_geometry,
    resolve_geometry,
    temporal_windows,
    validate_regenerate_request,
    validate_request,
)
from h3_singularity.errors import RuntimeErrorCode
from h3_singularity.queue import Queue
from h3_singularity.regenerate import (
    probe_and_validate_source_video,
    render_side_by_side_preview,
    resolve_source_artifact_path,
    sha256_file,
    stitch_window_frames,
)
from h3_singularity.worker import check_free_vram


BASE_URL = "http://video-mcp-server:8000"


def _make_video_payload(suffix: str = "a", profile: str = "fasth3-nvfp4", steps: int = 8) -> dict:
    return {
        "idempotency_key": "video_task_" + (suffix * 32)[:32],
        "model": "MiniMaxAI/MiniMax-H3",
        "task": "ref2va",
        "prompt": "A dramatic courtyard sword duel",
        "seconds": 5,
        "quality_profile": profile,
        "num_inference_steps": steps,
        "conditions": [
            {
                "type": "image",
                "role": "reference",
                "uri": f"{BASE_URL}/runtime-artifacts/art_{'b' * 32}/content",
                "size": 3,
                "sha256": hashlib.sha256(b"abc").hexdigest(),
            }
        ],
        "target": {"aspect_ratio": "16:9"},
    }


def _populate_sparse_weights(base_root: Path) -> None:
    singularity_root = base_root / "singularity"
    exact_files = [
        (singularity_root / "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors", 20_967_647_456),
        (singularity_root / "loras" / "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", 1_956_193_000),
        (singularity_root / "vae" / "minimax_h3_video_vae_int8_convrot.safetensors", 2_811_065_184),
        (singularity_root / "vae" / "minimax_h3_audio_vae_fp32.safetensors", 605_254_808),
        (base_root / "candidates" / "a1_fasth3_nvfp4" / "fasth3_omniref_qad_nvfp4_comfy.safetensors", 13_773_998_700),
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

    for rel in (
        "model-evidence.json",
        "component-model-evidence.json",
        "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
        "clip_projections/mmh3-4b-ClipProj-v3.1-mlp.safetensors",
    ):
        target = singularity_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"{}")


def _write_artifact_video(
    artifact_root: Path,
    *,
    project_id: str,
    artifact_id: str,
    width: int = 1344,
    height: int = 768,
    frames: int = 124,
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


class SupervisorAndRegenerate2KTests(unittest.TestCase):
    def test_api_module_has_zero_torch_import_in_fresh_process(self):
        src_dir = str(Path(__file__).resolve().parents[1] / "src")
        env = os.environ.copy()
        env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
        code = (
            "import sys, h3_singularity.api; "
            "assert 'torch' not in sys.modules, f'torch leaked into supervisor: {list(sys.modules)}'"
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

    def test_dual_quality_profiles_and_nfe_validation(self):
        validate_request(_make_video_payload("1", profile="fasth3-nvfp4", steps=8), BASE_URL)
        validate_request(_make_video_payload("2", profile="du-0", steps=4), BASE_URL)
        with self.assertRaisesRegex(RuntimeErrorCode, "fasth3_nvfp4_requires_8_nfe"):
            validate_request(_make_video_payload("3", profile="fasth3-nvfp4", steps=4), BASE_URL)
        with self.assertRaisesRegex(RuntimeErrorCode, "singularity_requires_4_nfe"):
            validate_request(_make_video_payload("4", profile="du-0", steps=8), BASE_URL)
        with self.assertRaisesRegex(RuntimeErrorCode, "quality_profile_not_available"):
            validate_request(_make_video_payload("5", profile="a2-turbo", steps=6), BASE_URL)

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
        for ratio in ASPECT_RATIOS:
            if ratio == "adaptive":
                continue
            native = resolve_geometry(ratio)
            src_w, src_h = native["width"], native["height"]
            b = resolve_2k_target_geometry(src_w, src_h, "budget_1440p")
            self.assertEqual((b["target_w"], b["target_h"], b["levels"]), expected_budget[ratio])
            self.assertEqual(src_w * b["target_h"], b["target_w"] * src_h)
            e = resolve_2k_target_geometry(src_w, src_h, "exact_2x")
            self.assertEqual((e["target_w"], e["target_h"], e["levels"]), expected_exact[ratio])
            self.assertEqual(src_w * e["target_h"], e["target_w"] * src_h)

        # Prime adaptive ratio (41:24 = 1312x768) upgrades cleanly to exact 2x (2624x1536)
        adaptive_prime = resolve_2k_target_geometry(1312, 768, "budget_1440p")
        self.assertEqual((adaptive_prime["target_w"], adaptive_prime["target_h"]), (2624, 1536))

        # Already-2K inputs reject with source_already_2k
        for w, h in ((2464, 1408), (1408, 2464), (2560, 1120), (1440, 1440), (2688, 1536)):
            for mode in ("budget_1440p", "exact_2x"):
                with self.assertRaisesRegex(RuntimeErrorCode, "source_already_2k"):
                    resolve_2k_target_geometry(w, h, mode)

        # Non-32-aligned or >3072 target rejects with invalid_source_media
        with self.assertRaisesRegex(RuntimeErrorCode, "invalid_source_media"):
            resolve_2k_target_geometry(1280, 720, "budget_1440p")
        with self.assertRaisesRegex(RuntimeErrorCode, "invalid_source_media"):
            resolve_2k_target_geometry(1600, 768, "exact_2x")

    def test_all_17k_plus_5_buckets_preserve_exact_frame_count_and_crossfade(self):
        buckets = [5 + 17 * k for k in range(6, 22)]  # 107..362 inclusive
        self.assertEqual(buckets[0], 107)
        self.assertEqual(buckets[-1], 362)
        for n_frames in buckets:
            windows = temporal_windows(n_frames)
            self.assertEqual(windows[0][0], 0)
            self.assertEqual(windows[-1][1], n_frames)
            # Constant-value synthetic frames verify both exact frame count and unbiased linear crossfade
            win_arrays = [
                np.full((we - ws, 8, 8, 3), fill_value=100, dtype=np.uint8)
                for ws, we in windows
            ]
            stitched = list(stitch_window_frames(iter(win_arrays), windows))
            self.assertEqual(len(stitched), n_frames)
            for frame in stitched:
                self.assertEqual(frame.shape, (8, 8, 3))
                self.assertTrue(np.all(frame == 100))

    def test_preview_geometry_golden_table(self):
        golden = {
            (768, 1344): (1536, 1344),
            (2464, 1408): (3840, 1096),
            (1408, 2464): (2468, 2160),
            (1536, 2688): (2468, 2160),
            (3072, 1344): (3840, 840),
        }
        for (src_w, src_h), expected in golden.items():
            self.assertEqual(preview_geometry(src_w, src_h), expected)

    def test_zero_cuda_weights_health_and_vram_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _populate_sparse_weights(base)
            with patch.dict(os.environ, {"SINGULARITY_DEFAULT_QUALITY_PROFILE": "fasth3-nvfp4"}):
                health = check_weights_health(
                    base_root=base,
                    singularity_root=base / "singularity",
                    component_root=base / "singularity",
                    gpu_worker_active=True,
                    active_task_ids=["video_task_" + "a" * 32],
                )
            self.assertTrue(health["ready"])
            self.assertEqual(health["default_quality_profile"], "fasth3-nvfp4")
            self.assertEqual(health["nfe"], 8)
            self.assertTrue(health["gpu_worker_active"])
            self.assertEqual(health["capabilities"]["quality_profiles"], ["fasth3-nvfp4", "du-0"])
            self.assertEqual(health["capabilities"]["regenerate_2k_profiles"], ["open_h3_2k_ic"])

            # Corrupt size of one weight file -> fails fast without CUDA
            bad_file = base / "candidates" / "p2_open_h3_2k_ic" / "official" / "assets" / "silence_audio_124f.safetensors"
            bad_file.write_bytes(b"short")
            with self.assertRaisesRegex(RuntimeErrorCode, "weight_size_mismatch:p2_silence_audio_124f"):
                check_weights_health(
                    base_root=base,
                    singularity_root=base / "singularity",
                    component_root=base / "singularity",
                )

        gpu_uuid = "GPU-29a56c24-484e-d51d-fac9-b141be9e24c5"
        with patch.dict(os.environ, {"NVIDIA_VISIBLE_DEVICES": gpu_uuid}):
            with patch("h3_singularity.worker.subprocess.check_output", return_value=f"{gpu_uuid}, 30500\n"):
                self.assertEqual(check_free_vram(28672), (gpu_uuid, 30500))
            with patch("h3_singularity.worker.subprocess.check_output", return_value=f"{gpu_uuid}, 21000\n"):
                with self.assertRaisesRegex(RuntimeErrorCode, "gpu_memory_occupancy_changed"):
                    check_free_vram(28672)
            other_uuid = "GPU-11111111-1111-1111-1111-111111111111"
            with patch("h3_singularity.worker.subprocess.check_output", return_value=f"{other_uuid}, 31000\n"):
                with self.assertRaisesRegex(RuntimeErrorCode, "gpu_memory_occupancy_changed"):
                    check_free_vram(28672)

    def test_supervisor_reaps_sigterm_ignoring_subprocess_before_settling(self):
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

            def build_cmd(task_id: str, _task_root: Path) -> list[str]:
                return [sys.executable, str(worker_script), str(ready_marker), task_id]

            def healthy(**kwargs):
                return {
                    "ready": True,
                    "default_quality_profile": "fasth3-nvfp4",
                    "gpu_worker_active": kwargs.get("gpu_worker_active", False),
                    "active_task_ids": kwargs.get("active_task_ids", []),
                    "capabilities": {"quality_profiles": ["fasth3-nvfp4", "du-0"]},
                }

            headers = {"Authorization": "Bearer test-token"}
            try:
                app = create_app(
                    queue,
                    None,
                    "test-token",
                    worker_cmd_builder=build_cmd,
                    health_checker=healthy,
                    kill_grace_seconds=0.25,
                )
                with TestClient(app) as client:
                    resp = client.post(
                        "/v1/videos",
                        json=_make_video_payload("9", profile="fasth3-nvfp4", steps=8),
                        headers=headers,
                    )
                    self.assertEqual(resp.status_code, 200)
                    task_id = resp.json()["id"]

                    deadline = time.monotonic() + 5.0
                    while time.monotonic() < deadline and not ready_marker.is_file():
                        time.sleep(0.05)
                    self.assertTrue(ready_marker.is_file(), "worker subprocess did not start")

                    active_health = client.get("/health").json()
                    self.assertTrue(active_health["gpu_worker_active"])
                    self.assertEqual(active_health["active_task_ids"], [task_id])

                    cancel_resp = client.delete(f"/v1/videos/{task_id}", headers=headers)
                    self.assertEqual(cancel_resp.status_code, 200)
                    cancelled = cancel_resp.json()
                    self.assertEqual(cancelled["status"], "cancelled")
                    self.assertTrue(cancelled["worker_settled"])

                    # Repeated DELETE on cancelled is idempotent 200
                    repeat_resp = client.delete(f"/v1/videos/{task_id}", headers=headers)
                    self.assertEqual(repeat_resp.status_code, 200)
                    self.assertEqual(repeat_resp.json()["status"], "cancelled")

                    after_health = client.get("/health").json()
                    self.assertFalse(after_health["gpu_worker_active"])
                    self.assertEqual(after_health["active_task_ids"], [])
                    self.assertEqual(client.get("/live").status_code, 200)
            finally:
                queue.close()

    @unittest.skipIf(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, "ffmpeg/ffprobe required")
    def test_regenerate_2k_admission_kind_isolation_preview_and_terminal_delete(self):
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

            # Verify side-by-side preview renderer produces exact preview_geometry(2464, 1408) == (3840, 1096)
            fake_out = root / "fake_2k.mp4"
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
                    "12",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-pix_fmt",
                    "yuv420p",
                    str(fake_out),
                ],
                check=True,
                capture_output=True,
                timeout=120,
            )
            preview_out = root / "preview_check.mp4"
            render_side_by_side_preview(valid_mp4, fake_out, preview_out, 2464, 1408)
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

            # Verify HTTP /v1/regenerate-2k admission, kind isolation, and worker completion
            fast_worker = root / "fast_regen_worker.py"
            fast_worker.write_text(
                "import json, pathlib, shutil, sys\n"
                "from h3_singularity.queue import Queue\n"
                "from h3_singularity.regenerate import sha256_file\n"
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
                with patch.dict(
                    os.environ,
                    {
                        "SINGULARITY_ARTIFACT_ROOT": str(artifact_root),
                        "PYTHONPATH": src_dir + os.pathsep + os.environ.get("PYTHONPATH", ""),
                    },
                ):
                    app = create_app(
                        queue,
                        None,
                        "test-token",
                        worker_cmd_builder=lambda tid, troot: [
                            sys.executable,
                            str(fast_worker),
                            tid,
                            str(troot),
                            str(fake_out),
                        ],
                        health_checker=lambda **kw: {"ready": True, **kw},
                    )
                    with TestClient(app) as client:
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

                        # Valid submission returns 202 Accepted and is strictly isolated from /v1/videos/by-idempotency
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

                        # /v1/videos/by-idempotency filters kind='video' -> 404
                        self.assertEqual(
                            client.get(f"/v1/videos/by-idempotency/{regen_id}", headers=headers).status_code,
                            404,
                        )

                        # Wait for completion + worker_settled
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
                        # DELETE on succeeded returns 409 task_already_terminal
                        del_done = client.delete(f"/v1/regenerate-2k/{regen_id}", headers=headers)
                        self.assertEqual(del_done.status_code, 409)
                        self.assertEqual(del_done.json()["detail"]["code"], "task_already_terminal")
            finally:
                queue.close()

    def test_delete_in_take_before_spawn_window_persists_and_skips_spawn(self):
        import threading

        with tempfile.TemporaryDirectory() as tmp:
            q_root = Path(tmp) / "queue"
            queue = Queue(q_root)
            try:
                taken_event = threading.Event()
                release_take = threading.Event()
                orig_take = queue.take
                spawned_pids: list[int] = []

                def hooked_take():
                    task = orig_take()
                    if task is not None:
                        taken_event.set()
                        release_take.wait(timeout=5.0)
                    return task

                def track_cmd(tid: str, t_root: Path) -> list[str]:
                    spawned_pids.append(1)
                    return [sys.executable, "-c", "import sys; sys.exit(0)"]

                queue.take = hooked_take
                app = create_app(
                    queue,
                    engine_factory=None,
                    token="race-token",
                    worker_cmd_builder=track_cmd,
                    health_checker=lambda **kw: {"ready": True, **kw},
                    kill_grace_seconds=0.3,
                )
                headers = {"Authorization": "Bearer race-token"}
                vid_payload = {
                    "idempotency_key": "video_task_" + "c" * 32,
                    "model": "MiniMaxAI/MiniMax-H3",
                    "task": "ref2va",
                    "prompt": "race cancel",
                    "seconds": 5,
                    "conditions": [
                        {
                            "type": "image",
                            "role": "reference",
                            "uri": "http://video-mcp-server:8000/runtime-artifacts/art_" + "d" * 32 + "/content",
                            "size": 10,
                            "sha256": "e" * 64,
                        }
                    ],
                    "target": {"aspect_ratio": "16:9"},
                }
                with TestClient(app) as client:
                    created = client.post("/v1/videos", json=vid_payload, headers=headers)
                    self.assertEqual(created.status_code, 200)
                    tid = created.json()["id"]
                    self.assertTrue(taken_event.wait(timeout=3.0))

                    # Issue DELETE while supervisor is paused between queue.take() and spawn!
                    del_result: list[dict] = []

                    def _do_delete():
                        r = client.delete(f"/v1/videos/{tid}", headers=headers)
                        del_result.append((r.status_code, r.json()))

                    t_del = threading.Thread(target=_do_delete)
                    t_del.start()
                    # Wait until cancel_requested=1 is persisted in SQLite before releasing take
                    deadline = time.monotonic() + 2.0
                    while time.monotonic() < deadline and not queue.is_cancel_requested(tid):
                        time.sleep(0.01)
                    self.assertTrue(queue.is_cancel_requested(tid))
                    release_take.set()
                    t_del.join(timeout=3.0)

                    self.assertEqual(len(del_result), 1)
                    code, body = del_result[0]
                    self.assertEqual(code, 200)
                    self.assertEqual(body["status"], "cancelled")
                    self.assertTrue(body["worker_settled"])
                    self.assertEqual(spawned_pids, [])
            finally:
                queue.close()

    def test_posix_reaper_kills_surviving_descendant_when_group_leader_exits_early(self):
        from h3_singularity import api as api_mod
        from unittest.mock import MagicMock

        fake_proc = MagicMock()
        fake_proc.pid = 42424
        # Leader exits immediately upon SIGTERM
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
        self.assertFalse(descendant_alive[0])

    def test_posix_reaper_surviving_sigkill_keeps_unsettled_and_blocks_next_dispatch(self):
        from h3_singularity import api as api_mod
        from unittest.mock import MagicMock
        import threading

        # 1. Direct reaper check: when descendant survives SIGKILL past kill_wait_seconds,
        # _terminate_and_reap_process_group MUST return False rather than claiming reaped.
        stubborn_proc = MagicMock()
        stubborn_proc.pid = 51515
        stubborn_proc.poll.return_value = 0
        stubborn_proc.wait.return_value = 0
        still_alive = [True]

        def stubborn_killpg(pgid: int, sig: int):
            if sig == 0:
                if still_alive[0]:
                    return None
                raise ProcessLookupError("gone")
            return None

        with patch.object(api_mod.os, "name", "posix"), patch.object(
            api_mod.os, "killpg", side_effect=stubborn_killpg, create=True
        ), patch.object(api_mod.signal, "SIGKILL", 9, create=True):
            reaped_early = api_mod._terminate_and_reap_process_group(
                stubborn_proc,
                grace_seconds=0.06,
                kill_wait_seconds=0.08,
            )
        self.assertFalse(reaped_early)

        # 2. Supervisor end-to-end check: while Task 1's process group survives SIGKILL past
        # kill_wait_seconds and cancel_endpoint_wait_seconds, Supervisor MUST NOT report
        # worker_settled=True, MUST mark /health ready=False (reap_stalled=True), and MUST NOT
        # dispatch queued Task 2 until the stubborn group finally disappears.
        with tempfile.TemporaryDirectory() as tmp:
            q_root = Path(tmp) / "queue"
            queue = Queue(q_root)
            try:
                spawned_tasks: list[str] = []
                task1_spawned = threading.Event()
                pgid_alive_map: dict[int, bool] = {}
                mproc_map: dict[int, MagicMock] = {}
                next_pid = [61000]

                def fake_spawn(cmd: list[str], env: dict[str, str]):
                    tid = cmd[-1]
                    spawned_tasks.append(tid)
                    pid = next_pid[0]
                    next_pid[0] += 1
                    mproc = MagicMock()
                    mproc.pid = pid
                    mproc_map[pid] = mproc
                    if len(spawned_tasks) == 1:
                        pgid_alive_map[pid] = True
                        leader_running = [True]
                        mproc.poll.side_effect = lambda: None if leader_running[0] else 0

                        def _wait(*args, **kwargs):
                            leader_running[0] = False
                            return -9

                        mproc.wait.side_effect = _wait
                        mproc._leader_running = leader_running
                        task1_spawned.set()
                    else:
                        # Task 2 succeeds quickly and cleanly
                        pgid_alive_map[pid] = False
                        queue.update(tid, status="completed", stage="completed", result={"ok": True})
                        mproc.poll.return_value = 0
                        mproc.wait.return_value = 0
                    return mproc

                def supervisor_killpg(pgid: int, sig: int):
                    if sig == 0:
                        if pgid_alive_map.get(pgid, False):
                            return None
                        raise ProcessLookupError("reaped")
                    # Leader exits immediately on SIGTERM/SIGKILL, while non-leader descendant in pgid 61000 survives!
                    if pgid in mproc_map and hasattr(mproc_map[pgid], "_leader_running"):
                        mproc_map[pgid]._leader_running[0] = False
                    return None

                app = create_app(
                    queue,
                    engine_factory=None,
                    token="reap-token",
                    worker_cmd_builder=lambda tid, _r: ["fake-worker", tid],
                    health_checker=lambda **kw: {"ready": True, **kw},
                    kill_grace_seconds=0.06,
                    kill_wait_seconds=0.08,
                    cancel_endpoint_wait_seconds=0.22,
                )
                headers = {"Authorization": "Bearer reap-token"}

                def _make_payload(key_char: str) -> dict:
                    return {
                        "idempotency_key": "video_task_" + key_char * 32,
                        "model": "MiniMaxAI/MiniMax-H3",
                        "task": "ref2va",
                        "prompt": f"task {key_char}",
                        "seconds": 5,
                        "conditions": [
                            {
                                "type": "image",
                                "role": "reference",
                                "uri": "http://video-mcp-server:8000/runtime-artifacts/art_" + "d" * 32 + "/content",
                                "size": 10,
                                "sha256": "e" * 64,
                            }
                        ],
                        "target": {"aspect_ratio": "16:9"},
                    }

                with patch.object(api_mod.os, "name", "posix"), patch.object(
                    api_mod, "_spawn_worker_process", side_effect=fake_spawn
                ), patch.object(
                    api_mod.os, "killpg", side_effect=supervisor_killpg, create=True
                ), patch.object(
                    api_mod.signal, "SIGKILL", 9, create=True
                ):
                    with TestClient(app) as client:
                        r1 = client.post("/v1/videos", json=_make_payload("1"), headers=headers)
                        self.assertEqual(r1.status_code, 200)
                        t1_id = r1.json()["id"]
                        self.assertTrue(task1_spawned.wait(timeout=3.0))

                        # Enqueue Task 2 while Task 1 is active (before reap stalls)
                        r2 = client.post("/v1/videos", json=_make_payload("2"), headers=headers)
                        self.assertEqual(r2.status_code, 200)
                        t2_id = r2.json()["id"]

                        # Cancel Task 1; leader exits immediately on SIGTERM while descendant survives SIGKILL
                        del_resp = client.delete(f"/v1/videos/{t1_id}", headers=headers)

                        self.assertEqual(del_resp.status_code, 200)
                        del_body = del_resp.json()
                        self.assertEqual(del_body["status"], "cancelled")
                        self.assertFalse(del_body["worker_settled"])

                        # Poll status & health: STILL unsettled, slot occupied, Task 2 NOT spawned!
                        st1 = client.get(f"/v1/videos/{t1_id}", headers=headers).json()
                        self.assertEqual(st1["status"], "cancelled")
                        self.assertFalse(st1["worker_settled"])

                        st2 = client.get(f"/v1/videos/{t2_id}", headers=headers).json()
                        self.assertEqual(st2["status"], "queued")
                        self.assertEqual(spawned_tasks, [t1_id])

                        hlth = client.get("/health").json()
                        self.assertFalse(hlth["ready"])
                        self.assertTrue(hlth["gpu_worker_active"])
                        self.assertTrue(hlth.get("reap_stalled"))
                        self.assertEqual(hlth["active_task_ids"], [t1_id])

                        # New submission while reap_stalled MUST be rejected with 503 not_ready
                        r3 = client.post("/v1/videos", json=_make_payload("3"), headers=headers)
                        self.assertEqual(r3.status_code, 503)

                        # Now let the stubborn descendant finally exit the kernel!
                        pgid_alive_map[61000] = False

                        deadline = time.monotonic() + 3.0
                        while time.monotonic() < deadline:
                            st1_after = client.get(f"/v1/videos/{t1_id}", headers=headers).json()
                            st2_after = client.get(f"/v1/videos/{t2_id}", headers=headers).json()
                            if st1_after["worker_settled"] and st2_after["status"] == "completed":
                                break
                            time.sleep(0.05)

                        self.assertTrue(st1_after["worker_settled"])
                        self.assertEqual(st2_after["status"], "completed")
                        self.assertTrue(st2_after["worker_settled"])
                        self.assertEqual(spawned_tasks, [t1_id, t2_id])

                        hlth_after = client.get("/health").json()
                        self.assertTrue(hlth_after["ready"])
                        self.assertFalse(hlth_after["gpu_worker_active"])
            finally:
                queue.close()

    def test_stitch_window_frames_frees_previous_window_and_uses_single_frame_float32_blend(self):
        import gc
        import tracemalloc
        import weakref
        import numpy as np
        from h3_singularity.contract import temporal_windows
        from h3_singularity.regenerate import stitch_window_frames

        # N=141 -> 2 windows [0, 124) and [17, 141) with O=107 overlap frames!
        windows = temporal_windows(141)
        self.assertEqual(windows, [(0, 124), (17, 141)])
        h, w = 128, 224  # representative scaled spatial grid for fast deterministic tracemalloc check
        weak_refs: list[weakref.ReferenceType] = []
        prev_alive_on_next_window: list[bool] = []

        def gen_windows():
            for idx, (ws, we) in enumerate(windows):
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
            for frame in stitch_window_frames(gen_windows(), windows):
                count += 1
                if count == 20:
                    # First overlap frame (index 17 + 2): blended between 50 and 200
                    self.assertTrue(50 < int(frame[0, 0, 0]) < 200)
                frame = None
            _, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        self.assertEqual(count, 141)
        self.assertEqual(prev_alive_on_next_window, [False])
        # Full 107-frame float32 batch at 128x224x3 would be > 36 MiB on top of 2x uint8 windows;
        # single-frame float32 blend keeps peak Python allocation well below 20 MiB (tail + 1 win + 2 f32 frames).
        full_batch_f32_bytes = 107 * h * w * 3 * 4 * 3
        self.assertLess(peak_bytes, full_batch_f32_bytes // 2)


if __name__ == "__main__":
    unittest.main()
