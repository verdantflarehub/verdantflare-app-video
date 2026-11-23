"""Pure lip-sync validation shared by preflight and submission."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess

from .artifacts import ArtifactNotFound


IDENTIFIERS = {
    "project_id": r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}",
    "idempotency_key": r"video_task_[0-9a-f]{32}",
    "source_video_artifact_id": r"art_[0-9a-f]{32}",
    "source_video_sha256": r"[0-9a-f]{64}",
    "audio_artifact_id": r"art_[0-9a-f]{32}",
    "audio_sha256": r"[0-9a-f]{64}",
}

AUDIO_MIX_MODES = ("driver_wav", "source_audio", "mute")
WAV_MEDIA_TYPES = {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}


def validate_lipsync_request(payload: dict) -> dict:
    required = set(IDENTIFIERS)
    allowed = required | {"inference_steps", "guidance_scale", "seed", "audio_mix_mode"}
    if not isinstance(payload, dict) or required - payload.keys() or payload.keys() - allowed:
        raise ValueError("invalid_request_fields")
    request = payload.copy()
    for key, pattern in IDENTIFIERS.items():
        value = request[key]
        if not isinstance(value, str) or not re.fullmatch(pattern, value):
            raise ValueError(f"invalid_{key}")
    steps = request.setdefault("inference_steps", 20)
    if type(steps) is not int or not 10 <= steps <= 50:
        raise ValueError("invalid_inference_steps")
    guidance = request.setdefault("guidance_scale", 1.5)
    if isinstance(guidance, bool) or not isinstance(guidance, (int, float)) or not 1.0 <= float(guidance) <= 3.0:
        raise ValueError("invalid_guidance_scale")
    request["guidance_scale"] = round(float(guidance), 4)
    seed = request.setdefault("seed", 1247)
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("invalid_seed")
    mix_mode = request.setdefault("audio_mix_mode", "driver_wav")
    if not isinstance(mix_mode, str) or mix_mode not in AUDIO_MIX_MODES:
        raise ValueError("unsupported_audio_mix_mode")
    return request


def _probe_duration_seconds(path) -> float | None:
    ffprobe_bin = os.environ.get("FFPROBE_BIN") or shutil.which("ffprobe")
    if not ffprobe_bin:
        return None
    try:
        raw = subprocess.check_output(
            [ffprobe_bin, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
        val = float((json.loads(raw).get("format") or {}).get("duration") or 0.0)
        return val if val > 0 else None
    except Exception:
        return None


def prepare_request(store, project_id: str, source_video_artifact_id: str, parameters: dict) -> dict:
    source_video = store.get(source_video_artifact_id, project_id)
    audio_id = parameters.get("audio_artifact_id", "")
    audio = store.get(audio_id, project_id)
    if not source_video.media_type.startswith("video/"):
        raise ValueError("invalid_source_media")
    if audio.media_type not in WAV_MEDIA_TYPES:
        raise ValueError("invalid_audio_media")
    for role, artifact in (("source", source_video), ("audio", audio)):
        try:
            with store.content_path(artifact).open("rb") as stream:
                size = os.fstat(stream.fileno()).st_size
        except (OSError, ArtifactNotFound) as exc:
            raise ValueError(role + "_not_found") from exc
        if size != artifact.size:
            raise ValueError(role + "_integrity_failed")
    v_dur = _probe_duration_seconds(store.content_path(source_video))
    a_dur = _probe_duration_seconds(store.content_path(audio))
    if v_dur is not None and a_dur is not None and abs(v_dur - a_dur) > 2.0 + 1e-6:
        raise ValueError("duration_mismatch")
    validated = validate_lipsync_request({
        **parameters,
        "project_id": project_id,
        "source_video_artifact_id": source_video_artifact_id,
        "source_video_sha256": source_video.sha256,
        "audio_sha256": audio.sha256,
        "idempotency_key": "video_task_" + "0" * 32,
    })
    del validated["idempotency_key"]
    return {
        **validated,
        "source_artifact_id": source_video_artifact_id,
        "service": "lipsync",
        "model": "latentsync-1.6",
    }
