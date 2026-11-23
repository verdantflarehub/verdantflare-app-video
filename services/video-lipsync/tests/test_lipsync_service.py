"""Unit and HTTP contract tests for video-lipsync (Zero-CUDA supervisor + isolated worker)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

from fastapi.testclient import TestClient
import pytest

from video_lipsync.api import create_app
from video_lipsync.contract import check_weights_health, preview_geometry, validate_lipsync_request
from video_lipsync.errors import RuntimeErrorCode
from video_lipsync.lipsync import probe_and_validate_lipsync_inputs
from video_lipsync.queue import Queue
from video_lipsync.worker import check_free_vram


VALID_REQ = {
    "project_id": "proj-mv-01",
    "idempotency_key": "video_task_" + "a" * 32,
    "source_video_artifact_id": "art_" + "1" * 32,
    "source_video_sha256": "b" * 64,
    "audio_artifact_id": "art_" + "2" * 32,
    "audio_sha256": "c" * 64,
}


def test_supervisor_imports_no_cuda_stack():
    """S1: Zero-CUDA supervisor must never import torch/diffusers/onnxruntime."""
    for mod in ("torch", "diffusers", "onnxruntime"):
        assert mod not in sys.modules


def test_preview_geometry_golden_table():
    assert preview_geometry(768, 1344) == (1536, 1344)
    assert preview_geometry(2464, 1408) == (3840, 1096)
    assert preview_geometry(1408, 2464) == (2468, 2160)
    assert preview_geometry(1536, 2688) == (2468, 2160)
    assert preview_geometry(3072, 1344) == (3840, 840)


def test_validate_request_strict_fields():
    norm = validate_lipsync_request(VALID_REQ)
    assert norm["inference_steps"] == 20
    assert norm["guidance_scale"] == 1.5
    assert norm["seed"] == 1247
    assert norm["audio_mix_mode"] == "driver_wav"

    # Unknown field (e.g. truncate) must be rejected
    bad = {**VALID_REQ, "truncate": True}
    with pytest.raises(RuntimeErrorCode) as exc:
        validate_lipsync_request(bad)
    assert exc.value.code == "invalid_request_fields"

    # Out-of-range steps
    with pytest.raises(RuntimeErrorCode) as exc2:
        validate_lipsync_request({**VALID_REQ, "inference_steps": 99})
    assert exc2.value.code == "invalid_inference_steps"


def test_duration_diff_boundary_2_0_seconds(tmp_path: Path, monkeypatch):
    """S4: diff == 2.0s passes, diff > 2.0s raises duration_mismatch."""
    art_root = tmp_path / "video"
    for art_id, kind, fname, content in [
        (VALID_REQ["source_video_artifact_id"], "video", "content.mp4", b"fake-mp4"),
        (VALID_REQ["audio_artifact_id"], "audio", "content.wav", b"fake-wav"),
    ]:
        d = art_root / "artifacts" / art_id
        d.mkdir(parents=True)
        (d / fname).write_bytes(content)
        sha = hashlib.sha256(content).hexdigest()
        (d / "metadata.json").write_text(
            json.dumps({"project_id": VALID_REQ["project_id"], "kind": kind, "filename": fname, "size": len(content), "sha256": sha}),
            encoding="utf-8",
        )

    req = {
        **VALID_REQ,
        "source_video_sha256": hashlib.sha256(b"fake-mp4").hexdigest(),
        "audio_sha256": hashlib.sha256(b"fake-wav").hexdigest(),
    }

    monkeypatch.setattr("video_lipsync.lipsync.probe_video_file", lambda p: {
        "width": 768, "height": 1344, "fps_num": 24, "fps_den": 1, "fps_str": "24/1",
        "duration_seconds": 10.0, "has_audio": True, "audio_codec": "aac",
    })
    monkeypatch.setattr("video_lipsync.lipsync.probe_audio_file", lambda p: {
        "codec": "pcm_s16le", "sample_rate": 48000, "channels": 1, "duration_seconds": 12.0,
    })
    probed = probe_and_validate_lipsync_inputs(req, art_root)
    assert probed["effective_duration_seconds"] == 10.0

    monkeypatch.setattr("video_lipsync.lipsync.probe_audio_file", lambda p: {
        "codec": "pcm_s16le", "sample_rate": 48000, "channels": 1, "duration_seconds": 12.01,
    })
    with pytest.raises(RuntimeErrorCode) as exc:
        probe_and_validate_lipsync_inputs(req, art_root)
    assert exc.value.code == "duration_mismatch"


def test_check_free_vram_threshold_minus_one_mib(monkeypatch):
    """S3: free_mib == threshold - 1 fails with gpu_memory_occupancy_changed without CUDA import."""
    gpu_uuid = "GPU-798fb508-830f-b538-6733-72ed6d210d40"
    monkeypatch.setenv("LIPSYNC_MIN_FREE_VRAM_MIB", "28672")
    monkeypatch.setattr("video_lipsync.worker._assigned_gpu_from_environ", lambda: gpu_uuid)
    monkeypatch.setattr("subprocess.check_output", lambda *a, **kw: f"{gpu_uuid}, 28671\n")
    with pytest.raises(RuntimeErrorCode) as exc:
        check_free_vram()
    assert exc.value.code == "gpu_memory_occupancy_changed"
    assert exc.value.runtime_metrics["free_vram_mib_before"] == 28671
    assert exc.value.runtime_metrics["min_free_vram_mib"] == 28672
    assert "torch" not in sys.modules

    # Exact threshold passes
    monkeypatch.setattr("subprocess.check_output", lambda *a, **kw: f"{gpu_uuid}, 28672\n")
    actual_uuid, free_mib = check_free_vram()
    assert actual_uuid == gpu_uuid
    assert free_mib == 28672


def test_http_lifecycle_auth_and_cancel(tmp_path: Path, monkeypatch):
    """S1/S4: Auth rejection, 202 submit, worker execution, content/preview download, and cancel."""
    monkeypatch.setenv("VIDEO_GPU_GRANT_POLICY", "disabled")
    q = Queue(tmp_path / "tasks")
    worker_script = tmp_path / "fake_worker.py"
    worker_script.write_text(
        """
import os, sys, time
from pathlib import Path
from video_lipsync.queue import Queue

tid = sys.argv[1]
root = Path(os.environ["LIPSYNC_TASK_ROOT"])
q = Queue(root, acquire_lock=False)
if tid.endswith("bbbb"):
    time.sleep(30)
task_dir = root / tid
task_dir.mkdir(parents=True, exist_ok=True)
(task_dir / "output.mp4").write_bytes(b"mp4-out")
(task_dir / "preview.mp4").write_bytes(b"mp4-prev")
q.update(tid, status="succeeded", stage="completed", result={"output_sha256": "0"*64}, worker_settled=True)
q.close()
""",
        encoding="utf-8",
    )

    def fake_health(**kw):
        return {"status": "ok", "service": "video-lipsync", "ready": True, "missing_weights": [], **kw}

    app = create_app(
        q,
        token="test-secret-token",
        worker_cmd_builder=lambda tid, root: [sys.executable, str(worker_script), tid],
        health_checker=fake_health,
        input_validator=lambda req, root: {"ok": True},
        kill_grace_seconds=0.2,
        kill_wait_seconds=1.0,
    )
    headers = {"Authorization": "Bearer test-secret-token"}
    with TestClient(app) as client:
        # No bearer -> 401
        assert client.post("/v1/lipsync", json=VALID_REQ).status_code == 401
        assert client.get("/live").status_code == 200
        assert client.get("/health").status_code == 200

        # Submit -> 202 -> wait succeeded -> download content & preview
        r = client.post("/v1/lipsync", json=VALID_REQ, headers=headers)
        assert r.status_code == 202
        tid = r.json()["video_task_id"]
        for _ in range(50):
            s = client.get(f"/v1/lipsync/{tid}", headers=headers).json()
            if s["status"] == "succeeded" and s["worker_settled"]:
                break
            time.sleep(0.05)
        assert s["status"] == "succeeded"
        assert s["worker_settled"] is True
        assert client.get(f"/v1/lipsync/{tid}/content", headers=headers).content == b"mp4-out"
        assert client.get(f"/v1/lipsync/{tid}/preview", headers=headers).content == b"mp4-prev"

        # Cancel running task
        cancel_req = {**VALID_REQ, "idempotency_key": "video_task_" + "a" * 28 + "bbbb"}
        r2 = client.post("/v1/lipsync", json=cancel_req, headers=headers)
        assert r2.status_code == 202
        tid2 = r2.json()["video_task_id"]
        time.sleep(0.25)
        rc = client.delete(f"/v1/lipsync/{tid2}", headers=headers)
        assert rc.status_code == 200
        assert rc.json()["status"] == "cancelled"
        assert rc.json()["worker_settled"] is True
    q.close()
