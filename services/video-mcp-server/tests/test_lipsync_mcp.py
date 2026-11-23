"""Unit tests for video-mcp-server lipsync validation, presentation, and GPU scheduler specs."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import types as pytypes
from types import SimpleNamespace

import pytest

if "mcp" not in sys.modules:
    _mcp = pytypes.ModuleType("mcp")
    _mcp.types = pytypes.ModuleType("mcp.types")
    sys.modules["mcp"] = _mcp
    sys.modules["mcp.types"] = _mcp.types

from app.dashboard import task_channel, task_presentation
from app.gpu_lease import GpuCoordinator, SCHEDULER_CONSUMER_SPECS
from app.lipsync import prepare_request, validate_lipsync_request
from app.titles import OPERATIONS


def test_lipsync_validate_and_prepare_request(tmp_path: Path):
    v_bytes = b"fake-video-bytes"
    a_bytes = b"fake-wav-bytes"
    v_path = tmp_path / "v.mp4"
    a_path = tmp_path / "a.wav"
    v_path.write_bytes(v_bytes)
    a_path.write_bytes(a_bytes)

    v_art = SimpleNamespace(
        artifact_id="art_" + "1" * 32,
        media_type="video/mp4",
        size=len(v_bytes),
        sha256=hashlib.sha256(v_bytes).hexdigest(),
    )
    a_art = SimpleNamespace(
        artifact_id="art_" + "2" * 32,
        media_type="audio/wav",
        size=len(a_bytes),
        sha256=hashlib.sha256(a_bytes).hexdigest(),
    )

    class FakeStore:
        def get(self, aid, pid):
            return v_art if aid == v_art.artifact_id else a_art

        def content_path(self, art):
            return v_path if art.artifact_id == v_art.artifact_id else a_path

    req = prepare_request(FakeStore(), "proj-mv", v_art.artifact_id, {"audio_artifact_id": a_art.artifact_id})
    assert req["service"] == "lipsync"
    assert req["model"] == "latentsync-1.6"
    assert req["inference_steps"] == 20
    assert req["guidance_scale"] == 1.5
    assert req["seed"] == 1247
    assert req["audio_mix_mode"] == "driver_wav"
    assert req["source_video_sha256"] == v_art.sha256
    assert req["audio_sha256"] == a_art.sha256

    with pytest.raises(ValueError, match="invalid_request_fields"):
        validate_lipsync_request({**req, "idempotency_key": "video_task_" + "0" * 32, "truncate": True})


def test_lipsync_scheduler_and_dashboard_presentation():
    spec = SCHEDULER_CONSUMER_SPECS["lipsync"]
    assert spec["consumer_kind"] == "video.lipsync"
    assert spec["stage_id"] == "lipsync"
    assert spec["reclaim_mode"] == "epoch_unload"
    assert GpuCoordinator.engine_deployment("lipsync") == "video-lipsync"
    assert OPERATIONS["lipsync"] == "视频口型同步"

    rec = SimpleNamespace(service="lipsync", runtime_route=None, request={"model": "latentsync-1.6"})
    assert task_channel(rec) == "local"
    pres = task_presentation(rec)
    assert pres["channel_id"] == "local"
    assert pres["channel_label"] == "本地"
    assert pres["model_label"] == "LatentSync 1.6"
    assert pres["generation_code"] == "LipSync"
