"""Offline unit tests for pure Singularity (du-0) zero-CUDA Supervisor (v0.1.40)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from h3_singularity import api as api_mod
from h3_singularity.api import create_app
from h3_singularity.contract import check_weights_health, validate_request
from h3_singularity.errors import RuntimeErrorCode
from h3_singularity.queue import Queue
from h3_singularity.worker import check_free_vram


BASE_URL = "http://video-mcp-server:8000"


def _make_video_payload(suffix: str = "a", profile: str | None = "du-0", steps: int | None = 4) -> dict:
    payload = {
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


def _populate_du0_sparse_weights(base_root: Path) -> None:
    singularity_root = base_root / "singularity"
    exact_files = [
        (singularity_root / "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors", 20_967_647_456),
        (singularity_root / "loras" / "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", 1_956_193_000),
        (singularity_root / "vae" / "minimax_h3_video_vae_int8_convrot.safetensors", 2_811_065_184),
        (singularity_root / "vae" / "minimax_h3_audio_vae_fp32.safetensors", 605_254_808),
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


class PureSingularitySupervisorTests(unittest.TestCase):
    def test_zero_torch_in_api_and_zero_fasth3_or_regenerate_in_src(self):
        pkg_dir = Path(__file__).resolve().parents[1] / "src" / "h3_singularity"
        src_dir = str(pkg_dir.parent)
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

        for py_file in pkg_dir.rglob("*.py"):
            text = py_file.read_text(encoding="utf-8").lower()
            self.assertNotIn("fasth3", text, f"fasth3 leaked into {py_file.name}")
            self.assertNotIn("regenerate", text, f"regenerate leaked into {py_file.name}")

    def test_pure_du0_profile_validation_and_cross_endpoint_404(self):
        validate_request(_make_video_payload("1", profile=None, steps=None), BASE_URL)
        validate_request(_make_video_payload("2", profile="du-0", steps=4), BASE_URL)
        with self.assertRaisesRegex(RuntimeErrorCode, "quality_profile_not_available"):
            validate_request(_make_video_payload("3", profile="fasth3-nvfp4", steps=8), BASE_URL)
        with self.assertRaisesRegex(RuntimeErrorCode, "singularity_requires_4_nfe"):
            validate_request(_make_video_payload("4", profile="du-0", steps=8), BASE_URL)

        with tempfile.TemporaryDirectory() as tmp:
            queue = Queue(Path(tmp) / "tasks")
            headers = {"Authorization": "Bearer test-token"}
            try:
                app = create_app(
                    queue,
                    None,
                    "test-token",
                    health_checker=lambda **kw: {
                        "ready": True,
                        "default_quality_profile": "du-0",
                        "nfe": 4,
                        "capabilities": {"quality_profiles": ["du-0"]},
                        **kw,
                    },
                )
                with TestClient(app) as client:
                    self.assertEqual(client.post("/v1/regenerate-2k", json={}, headers=headers).status_code, 404)
                    self.assertEqual(
                        client.get("/v1/regenerate-2k/video_task_" + "a" * 32, headers=headers).status_code,
                        404,
                    )
                    bad_prof = client.post(
                        "/v1/videos",
                        json=_make_video_payload("5", profile="fasth3-nvfp4", steps=8),
                        headers=headers,
                    )
                    self.assertEqual(bad_prof.status_code, 422)
                    self.assertEqual(bad_prof.json()["detail"], "quality_profile_not_available")
            finally:
                queue.close()

    def test_zero_cuda_du0_weights_health_and_vram_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _populate_du0_sparse_weights(base)
            health = check_weights_health(
                base_root=base,
                singularity_root=base / "singularity",
                component_root=base / "singularity",
                gpu_worker_active=True,
                active_task_ids=["singularity_" + "a" * 32],
            )
            self.assertTrue(health["ready"])
            self.assertEqual(health["default_quality_profile"], "du-0")
            self.assertEqual(health["nfe"], 4)
            self.assertTrue(health["gpu_worker_active"])
            self.assertEqual(health["capabilities"]["quality_profiles"], ["du-0"])
            self.assertNotIn("regenerate_2k_profiles", health["capabilities"])

            bad_file = base / "singularity" / "vae" / "minimax_h3_audio_vae_fp32.safetensors"
            bad_file.write_bytes(b"short")
            with self.assertRaisesRegex(RuntimeErrorCode, "weight_size_mismatch:shared_audio_vae"):
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

    def test_supervisor_reaps_sigterm_ignoring_subprocess_and_handles_stalled_reap(self):
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
                app = create_app(
                    queue,
                    None,
                    "test-token",
                    worker_cmd_builder=lambda tid, _r: [sys.executable, str(worker_script), str(ready_marker), tid],
                    health_checker=lambda **kw: {"ready": True, "default_quality_profile": "du-0", **kw},
                    kill_grace_seconds=0.25,
                )
                with TestClient(app) as client:
                    resp = client.post(
                        "/v1/videos",
                        json=_make_video_payload("9", profile="du-0", steps=4),
                        headers=headers,
                    )
                    self.assertEqual(resp.status_code, 200)
                    task_id = resp.json()["id"]
                    self.assertTrue(task_id.startswith("singularity_"))

                    deadline = time.monotonic() + 5.0
                    while time.monotonic() < deadline and not ready_marker.is_file():
                        time.sleep(0.05)
                    self.assertTrue(ready_marker.is_file())

                    cancel_resp = client.delete(f"/v1/videos/{task_id}", headers=headers)
                    self.assertEqual(cancel_resp.status_code, 200)
                    self.assertEqual(cancel_resp.json()["status"], "cancelled")
                    self.assertTrue(cancel_resp.json()["worker_settled"])
            finally:
                queue.close()

        # Verify POSIX surviving-descendant SIGKILL reaper
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

        # Verify take-before-spawn DELETE window skips spawning
        with tempfile.TemporaryDirectory() as tmp2:
            queue2 = Queue(Path(tmp2) / "queue")
            try:
                taken_event = threading.Event()
                release_take = threading.Event()
                orig_take = queue2.take
                spawned: list[int] = []

                def hooked_take():
                    task = orig_take()
                    if task is not None:
                        taken_event.set()
                        release_take.wait(timeout=5.0)
                    return task

                queue2.take = hooked_take
                app2 = create_app(
                    queue2,
                    None,
                    "race-token",
                    worker_cmd_builder=lambda _tid, _r: (spawned.append(1) or [sys.executable, "-c", "pass"]),
                    health_checker=lambda **kw: {"ready": True, **kw},
                    kill_grace_seconds=0.2,
                )
                headers2 = {"Authorization": "Bearer race-token"}
                with TestClient(app2) as client2:
                    created = client2.post("/v1/videos", json=_make_video_payload("c"), headers=headers2)
                    tid = created.json()["id"]
                    self.assertTrue(taken_event.wait(timeout=3.0))
                    del_res: list[tuple[int, dict]] = []
                    t_del = threading.Thread(
                        target=lambda: del_res.append(
                            (
                                (r := client2.delete(f"/v1/videos/{tid}", headers=headers2)).status_code,
                                r.json(),
                            )
                        )
                    )
                    t_del.start()
                    deadline = time.monotonic() + 2.0
                    while time.monotonic() < deadline and not queue2.is_cancel_requested(tid):
                        time.sleep(0.01)
                    release_take.set()
                    t_del.join(timeout=3.0)
                    self.assertEqual(del_res[0][0], 200)
                    self.assertEqual(del_res[0][1]["status"], "cancelled")
                    self.assertTrue(del_res[0][1]["worker_settled"])
                    self.assertEqual(spawned, [])
            finally:
                queue2.close()


if __name__ == "__main__":
    unittest.main()
