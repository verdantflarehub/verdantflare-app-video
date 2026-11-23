"""CPU-only input contract, preview geometry, and weight health checks for LatentSync 1.6."""

from __future__ import annotations

import os
from pathlib import Path
import re

from .errors import RuntimeErrorCode


IDENTIFIERS = {
    "project_id": r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}",
    "idempotency_key": r"video_task_[0-9a-f]{32}",
    "source_video_artifact_id": r"art_[0-9a-f]{32}",
    "source_video_sha256": r"[0-9a-f]{64}",
    "audio_artifact_id": r"art_[0-9a-f]{32}",
    "audio_sha256": r"[0-9a-f]{64}",
}

AUDIO_MIX_MODES = ("driver_wav", "source_audio", "mute")

REQUIRED_WEIGHT_FILES = (
    "checkpoints/latentsync_unet.pt",
    "checkpoints/whisper/tiny.pt",
    "sd-vae-ft-mse/config.json",
)


def preview_geometry(w: int, h: int) -> tuple[int, int]:
    """Cross-repo integer rational pure function for side-by-side preview (<= 3840x2160)."""
    if w <= 1920 and h <= 2160:
        panel_w = max(2, (w // 2) * 2)
        panel_h = max(2, (h // 2) * 2)
        return (panel_w * 2, panel_h)
    if 1920 * h <= 2160 * w:
        panel_w = 1920
        panel_h = max(2, ((h * 1920) // (w * 2)) * 2)
        return (panel_w * 2, panel_h)
    panel_w = max(2, ((w * 2160) // (h * 2)) * 2)
    panel_h = 2160
    return (panel_w * 2, panel_h)


def validate_lipsync_request(payload: dict) -> dict:
    """Validate and normalize POST /v1/lipsync request payload (additionalProperties: false)."""
    required = set(IDENTIFIERS)
    allowed = required | {"inference_steps", "guidance_scale", "seed", "audio_mix_mode"}
    if not isinstance(payload, dict) or required - payload.keys() or payload.keys() - allowed:
        raise RuntimeErrorCode("invalid_request_fields")

    request = payload.copy()
    for key, pattern in IDENTIFIERS.items():
        value = request[key]
        if not isinstance(value, str) or not re.fullmatch(pattern, value):
            raise RuntimeErrorCode(f"invalid_{key}")

    steps = request.setdefault("inference_steps", 20)
    if type(steps) is not int or not 10 <= steps <= 50:
        raise RuntimeErrorCode("invalid_inference_steps")

    guidance = request.setdefault("guidance_scale", 1.5)
    if isinstance(guidance, bool) or not isinstance(guidance, (int, float)) or not 1.0 <= float(guidance) <= 3.0:
        raise RuntimeErrorCode("invalid_guidance_scale")
    request["guidance_scale"] = round(float(guidance), 4)

    seed = request.setdefault("seed", 1247)
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        raise RuntimeErrorCode("invalid_seed")

    mix_mode = request.setdefault("audio_mix_mode", "driver_wav")
    if not isinstance(mix_mode, str) or mix_mode not in AUDIO_MIX_MODES:
        raise RuntimeErrorCode("unsupported_audio_mix_mode")

    return request


def check_weights_health(
    *,
    model_root: str | Path | None = None,
    gpu_worker_active: bool = False,
    active_task_ids: list[str] | None = None,
) -> dict:
    root = Path(model_root or os.environ.get("LIPSYNC_MODEL_ROOT", "/models/LatentSync-1.6"))
    missing = [rel for rel in REQUIRED_WEIGHT_FILES if not (root / rel).is_file()]
    return {
        "status": "ok" if not missing else "not_ready",
        "service": "video-lipsync",
        "ready": not missing,
        "missing_weights": missing,
        "capabilities": {
            "model_id": "latentsync-1.6",
            "backend": "latentsync_1_6",
            "audio_mix_modes": list(AUDIO_MIX_MODES),
            "inference_steps_range": [10, 50],
            "guidance_scale_range": [1.0, 3.0],
            "max_duration_seconds": 60.0,
            "max_duration_diff_seconds": 2.0,
            "min_free_vram_mib": max(28672, int(os.environ.get("LIPSYNC_MIN_FREE_VRAM_MIB", "28672"))),
        },
        "gpu_worker_active": bool(gpu_worker_active),
        "active_task_ids": list(active_task_ids or []),
    }
