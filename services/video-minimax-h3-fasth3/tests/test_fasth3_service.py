"""Offline unit tests for video-minimax-h3-fasth3 (Zero-CUDA Supervisor, pure FastH3 NVFP4)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import httpx

from h3_fasth3 import api as api_mod
from h3_fasth3.api import MAX_REQUEST_BYTES, _download_references, create_app
from h3_fasth3.contract import (
    ASPECT_RATIOS,
    check_weights_health,
    resolve_geometry,
    resolve_timing,
    validate_request,
)
from h3_fasth3.errors import RuntimeErrorCode
from h3_fasth3.queue import Queue
from h3_fasth3.worker import check_free_vram


BASE_URL = "http://video-mcp-server:8000"


def _make_video_payload(
    suffix: str = "a",
    profile: str | None = "fasth3-nvfp4",
    steps: int | None = 8,
) -> dict:
    payload: dict = {
        "idempotency_key": "video_task_" + (suffix * 32)[:32],
        "model": "MiniMaxAI/MiniMax-H3",
        "task": "ref2va",
        "prompt": "A dramatic courtyard sword duel",
        "seconds": 5,
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
    if profile is not None:
        payload["quality_profile"] = profile
    if steps is not None:
        payload["num_inference_steps"] = steps
    return payload


def _populate_fasth3_sparse_weights(base_root: Path) -> None:
    exact_files = [
        (
            base_root / "candidates" / "a1_fasth3_nvfp4" / "fasth3_omniref_qad_nvfp4_comfy.safetensors",
            13_773_998_700,
        ),
        (
            base_root
            / "candidates"
            / "shared"
            / "text_encoders"
            / "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            15_687_142_551,
        ),
        (
            base_root / "candidates" / "shared" / "vae" / "minimax_h3_video_vae_fp16.safetensors",
            5_207_808_496,
        ),
        (
            base_root / "singularity" / "vae" / "minimax_h3_audio_vae_fp32.safetensors",
            605_254_808,
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


class FastH3ServiceTests(unittest.TestCase):
    def test_api_module_has_zero_torch_import_in_fresh_process(self):
        src_dir = str(Path(__file__).resolve().parents[1] / "src")
        env = os.environ.copy()
        env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
        code = (
            "import sys, h3_fasth3.api; "
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

    def test_static_isolation_no_du0_regenerate_or_singularity_env_in_source(self):
        src_dir = Path(__file__).resolve().parents[1] / "src" / "h3_fasth3"
        for py_file in src_dir.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            self.assertNotIn("du-0", content, f"du-0 found in {py_file}")
            self.assertNotIn("regenerate", content.lower(), f"regenerate found in {py_file}")
            self.assertNotIn("SINGULARITY_", content, f"SINGULARITY_ env found in {py_file}")

    def test_pure_fasth3_profile_and_8_nfe_contract(self):
        validate_request(_make_video_payload("1", profile="fasth3-nvfp4", steps=8), BASE_URL)
        validate_request(_make_video_payload("2", profile=None, steps=None), BASE_URL)
        with self.assertRaisesRegex(RuntimeErrorCode, "quality_profile_not_available"):
            validate_request(_make_video_payload("3", profile="du-0", steps=4), BASE_URL)
        with self.assertRaisesRegex(RuntimeErrorCode, "fasth3_nvfp4_requires_8_nfe"):
            validate_request(_make_video_payload("4", profile="fasth3-nvfp4", steps=4), BASE_URL)

    def test_canvases_timing_and_reference_download_security(self):
        expected = {
            "21:9": (1536, 672),
            "16:9": (1344, 768),
            "4:3": (1024, 768),
            "1:1": (768, 768),
            "3:4": (768, 1024),
            "9:16": (768, 1344),
        }
        for ratio, size in expected.items():
            res = resolve_geometry(ratio)
            self.assertEqual((res["width"], res["height"]), size)
        self.assertEqual(resolve_timing(15)["frames"], 362)

        for uri in (
            "http://169.254.169.254/latest",
            BASE_URL + "/health",
            _make_video_payload()["conditions"][0]["uri"] + "?token=x",
        ):
            bad = _make_video_payload()
            bad["conditions"][0]["uri"] = uri
            with self.assertRaisesRegex(RuntimeErrorCode, "source_not_allowed"):
                validate_request(bad, BASE_URL)

        with tempfile.TemporaryDirectory() as directory:
            client = httpx.Client(
                transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"abc")),
                follow_redirects=False,
            )
            with patch("h3_fasth3.api.httpx.Client", return_value=client):
                files = _download_references(Path(directory), _make_video_payload()["conditions"])
                self.assertEqual(files["images"][0].read_bytes(), b"abc")

    def test_health_checks_only_fasth3_weights_and_has_no_regenerate_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _populate_fasth3_sparse_weights(base)
            health = check_weights_health(
                model_root=base,
                component_root=base / "singularity",
                gpu_worker_active=True,
                active_task_ids=["fasth3_" + "a" * 32],
            )
            self.assertTrue(health["ready"])
            self.assertEqual(health["default_quality_profile"], "fasth3-nvfp4")
            self.assertEqual(health["nfe"], 8)
            self.assertEqual(health["capabilities"]["quality_profiles"], ["fasth3-nvfp4"])
            for key in health["capabilities"]:
                self.assertFalse(key.startswith("regenerate_2k"), f"unexpected key {key}")

            bad_file = base / "candidates" / "a1_fasth3_nvfp4" / "fasth3_omniref_qad_nvfp4_comfy.safetensors"
            bad_file.write_bytes(b"short")
            with self.assertRaisesRegex(RuntimeErrorCode, "weight_size_mismatch:fasth3_nvfp4_diffusion"):
                check_weights_health(model_root=base, component_root=base / "singularity")

        gpu_uuid = "GPU-798fb508-830f-b538-6733-72ed6d210d40"
        with patch.dict(os.environ, {"NVIDIA_VISIBLE_DEVICES": gpu_uuid}):
            with patch("h3_fasth3.worker.subprocess.check_output", return_value=f"{gpu_uuid}, 30500\n"):
                self.assertEqual(check_free_vram(28672), (gpu_uuid, 30500))
            with patch("h3_fasth3.worker.subprocess.check_output", return_value=f"{gpu_uuid}, 21000\n"):
                with self.assertRaisesRegex(RuntimeErrorCode, "gpu_memory_occupancy_changed"):
                    check_free_vram(28672)

    def test_cross_endpoint_404_task_id_prefix_no_kind_column_and_cancel_reap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = Queue(root / "tasks")
            cols = {row["name"] for row in queue.db.execute("PRAGMA table_info(tasks)").fetchall()}
            self.assertNotIn("kind", cols)

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
                app = create_app(
                    queue,
                    None,
                    "test-token",
                    worker_cmd_builder=lambda tid, _r: [
                        sys.executable,
                        str(worker_script),
                        str(ready_marker),
                        tid,
                    ],
                    health_checker=lambda **kw: {
                        "ready": True,
                        "default_quality_profile": "fasth3-nvfp4",
                        "capabilities": {"quality_profiles": ["fasth3-nvfp4"]},
                        **kw,
                    },
                    kill_grace_seconds=0.25,
                )
                with TestClient(app) as client:
                    # Cross-service regenerate-2k endpoints must return 404
                    self.assertEqual(
                        client.post("/v1/regenerate-2k", json={}, headers=headers).status_code,
                        404,
                    )
                    self.assertEqual(
                        client.get("/v1/regenerate-2k/video_task_" + "a" * 32, headers=headers).status_code,
                        404,
                    )
                    # Reject du-0 with 422 quality_profile_not_available
                    r_du0 = client.post(
                        "/v1/videos",
                        json=_make_video_payload("d", profile="du-0", steps=4),
                        headers=headers,
                    )
                    self.assertEqual(r_du0.status_code, 422)
                    self.assertEqual(r_du0.json()["detail"], "quality_profile_not_available")

                    # Valid submission creates fasth3_<32hex>
                    resp = client.post("/v1/videos", json=_make_video_payload("9"), headers=headers)
                    self.assertEqual(resp.status_code, 200)
                    task_id = resp.json()["id"]
                    self.assertRegex(task_id, r"^fasth3_[0-9a-f]{32}$")

                    deadline = time.monotonic() + 5.0
                    while time.monotonic() < deadline and not ready_marker.is_file():
                        time.sleep(0.05)
                    self.assertTrue(ready_marker.is_file())

                    cancel_resp = client.delete(f"/v1/videos/{task_id}", headers=headers)
                    self.assertEqual(cancel_resp.status_code, 200)
                    self.assertEqual(cancel_resp.json()["status"], "cancelled")
                    self.assertTrue(cancel_resp.json()["worker_settled"])
                    self.assertFalse(client.get("/health").json()["gpu_worker_active"])
            finally:
                queue.close()


if __name__ == "__main__":
    unittest.main()
