"""Artifact resolution, media probing, and LatentSync 1.6 execution pipeline."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .contract import preview_geometry
from .errors import RuntimeErrorCode


PCM_WAV_CODECS = {"pcm_s16le", "pcm_s24le", "pcm_f32le"}
MAX_DURATION_DIFF_SECONDS = 2.0


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def resolve_artifact_path(
    artifact_root: Path,
    *,
    project_id: str,
    artifact_id: str,
    expected_sha256: str,
    expected_kind: str,
) -> Path:
    """Zero-copy verified lookup of a project artifact on shared PVC (rejects symlinks)."""
    not_found_code = "source_not_found" if expected_kind == "video" else "audio_not_found"
    integrity_code = "source_integrity_failed" if expected_kind == "video" else "audio_integrity_failed"

    art_dir = artifact_root / "artifacts" / artifact_id
    if not art_dir.exists() or art_dir.is_symlink() or not art_dir.is_dir():
        raise RuntimeErrorCode(not_found_code)

    meta_path = art_dir / "metadata.json"
    if not meta_path.is_file() or meta_path.is_symlink():
        raise RuntimeErrorCode(not_found_code)

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeErrorCode(integrity_code) from exc

    if meta.get("project_id") != project_id:
        raise RuntimeErrorCode(not_found_code)
    if meta.get("kind") != expected_kind:
        raise RuntimeErrorCode("invalid_source_media" if expected_kind == "video" else "invalid_audio_media")

    filename = meta.get("filename") or ("content.mp4" if expected_kind == "video" else "content.wav")
    content_path = art_dir / filename
    if not content_path.is_file() or content_path.is_symlink():
        raise RuntimeErrorCode(not_found_code)

    actual_size = content_path.stat().st_size
    if meta.get("size") is not None and int(meta["size"]) != actual_size:
        raise RuntimeErrorCode(integrity_code)
    actual_sha = _sha256_file(content_path)
    if actual_sha != expected_sha256 or (meta.get("sha256") and meta["sha256"] != actual_sha):
        raise RuntimeErrorCode(integrity_code)
    return content_path


def _ffprobe_json(path: Path) -> dict:
    ffprobe_bin = os.environ.get("FFPROBE_BIN", "ffprobe")
    try:
        raw = subprocess.check_output(
            [
                ffprobe_bin,
                "-v", "error",
                "-show_format",
                "-show_streams",
                "-of", "json",
                str(path),
            ],
            text=True,
            timeout=20,
        )
        return json.loads(raw)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise RuntimeErrorCode("invalid_source_media") from exc


def _parse_fps(raw: str) -> Fraction:
    try:
        frac = Fraction(raw)
        if frac <= 0:
            raise ValueError("non-positive fps")
        return frac
    except Exception as exc:
        raise RuntimeErrorCode("invalid_source_fps") from exc


def probe_video_file(path: Path) -> dict:
    info = _ffprobe_json(path)
    fmt = info.get("format") or {}
    fmt_names = {s.strip().lower() for s in str(fmt.get("format_name", "")).split(",") if s.strip()}
    if not (fmt_names & {"mp4", "mov", "m4a", "3gp", "3g2", "mj2"}):
        raise RuntimeErrorCode("invalid_source_media")

    vstreams = [s for s in info.get("streams") or [] if s.get("codec_type") == "video"]
    if len(vstreams) != 1:
        raise RuntimeErrorCode("invalid_source_media")
    vs = vstreams[0]
    w = int(vs.get("width") or 0)
    h = int(vs.get("height") or 0)
    if w <= 0 or h <= 0 or w % 2 != 0 or h % 2 != 0 or min(w, h) < 256 or max(w, h) > 2048:
        raise RuntimeErrorCode("invalid_source_dimensions")
    if str(vs.get("pix_fmt") or "") != "yuv420p":
        raise RuntimeErrorCode("invalid_source_media")

    r_fps = _parse_fps(str(vs.get("r_frame_rate") or "0/1"))
    avg_fps = _parse_fps(str(vs.get("avg_frame_rate") or str(r_fps)))
    if not (15 <= float(r_fps) <= 60):
        raise RuntimeErrorCode("invalid_source_fps")
    if abs(float(r_fps) - float(avg_fps)) > 0.05:
        raise RuntimeErrorCode("invalid_source_media")

    duration = float(vs.get("duration") or fmt.get("duration") or 0.0)
    if not (1.0 <= duration <= 60.0):
        raise RuntimeErrorCode("duration_out_of_range")

    astreams = [s for s in info.get("streams") or [] if s.get("codec_type") == "audio"]
    return {
        "width": w,
        "height": h,
        "fps_num": r_fps.numerator,
        "fps_den": r_fps.denominator,
        "fps_str": f"{r_fps.numerator}/{r_fps.denominator}",
        "duration_seconds": round(duration, 6),
        "has_audio": len(astreams) > 0,
        "audio_codec": astreams[0].get("codec_name") if astreams else None,
    }


def probe_audio_file(path: Path) -> dict:
    try:
        info = _ffprobe_json(path)
    except RuntimeErrorCode as exc:
        raise RuntimeErrorCode("invalid_audio_media") from exc

    fmt = info.get("format") or {}
    fmt_names = {s.strip().lower() for s in str(fmt.get("format_name", "")).split(",") if s.strip()}
    if "wav" not in fmt_names:
        raise RuntimeErrorCode("invalid_audio_media")

    astreams = [s for s in info.get("streams") or [] if s.get("codec_type") == "audio"]
    if len(astreams) != 1:
        raise RuntimeErrorCode("invalid_audio_media")
    as_ = astreams[0]
    codec = str(as_.get("codec_name") or "").lower()
    if codec not in PCM_WAV_CODECS:
        raise RuntimeErrorCode("invalid_audio_media")
    sr = int(as_.get("sample_rate") or 0)
    ch = int(as_.get("channels") or 0)
    if not (16000 <= sr <= 96000) or ch not in (1, 2):
        raise RuntimeErrorCode("invalid_audio_media")

    duration = float(as_.get("duration") or fmt.get("duration") or 0.0)
    if not (1.0 <= duration <= 60.0):
        raise RuntimeErrorCode("duration_out_of_range")
    return {
        "codec": codec,
        "sample_rate": sr,
        "channels": ch,
        "duration_seconds": round(duration, 6),
    }


def probe_and_validate_lipsync_inputs(request: dict, artifact_root: Path) -> dict:
    """Synchronous pre-enqueue validation of video/audio artifacts and duration alignment."""
    video_path = resolve_artifact_path(
        artifact_root,
        project_id=request["project_id"],
        artifact_id=request["source_video_artifact_id"],
        expected_sha256=request["source_video_sha256"],
        expected_kind="video",
    )
    audio_path = resolve_artifact_path(
        artifact_root,
        project_id=request["project_id"],
        artifact_id=request["audio_artifact_id"],
        expected_sha256=request["audio_sha256"],
        expected_kind="audio",
    )
    v_meta = probe_video_file(video_path)
    a_meta = probe_audio_file(audio_path)

    diff = abs(v_meta["duration_seconds"] - a_meta["duration_seconds"])
    if diff > MAX_DURATION_DIFF_SECONDS + 1e-6:
        raise RuntimeErrorCode("duration_mismatch")

    return {
        "video_path": video_path,
        "audio_path": audio_path,
        "video": v_meta,
        "audio": a_meta,
        "effective_duration_seconds": round(min(v_meta["duration_seconds"], a_meta["duration_seconds"]), 6),
    }


def run_lipsync(
    request: dict,
    *,
    artifact_root: Path,
    task_dir: Path,
    gpu_uuid: str,
    free_vram_mib_before: int,
) -> dict:
    """Execute ByteDance LatentSync 1.6 in isolated worker, restore source CFR, and render preview."""
    t_start = time.monotonic()
    task_dir.mkdir(parents=True, exist_ok=True)
    probed = probe_and_validate_lipsync_inputs(request, artifact_root)
    video_path: Path = probed["video_path"]
    audio_path: Path = probed["audio_path"]
    v_meta = probed["video"]
    eff_dur = probed["effective_duration_seconds"]

    latentsync_root = Path(os.environ.get("LATENTSYNC_ROOT", "/opt/latentsync"))
    model_root = Path(os.environ.get("LIPSYNC_MODEL_ROOT", "/models/LatentSync-1.6"))
    ffmpeg_bin = os.environ.get("FFMPEG_BIN", "ffmpeg")
    raw_out = task_dir / "_latentsync_raw_25fps.mp4"
    final_out = task_dir / "output.mp4"
    preview_out = task_dir / "preview.mp4"

    t_inf0 = time.monotonic()
    cmd = [
        sys.executable,
        "-m", "scripts.inference",
        "--unet_config_path", str(latentsync_root / "configs/unet/stage2_512.yaml"),
        "--inference_ckpt_path", str(model_root / "checkpoints/latentsync_unet.pt"),
        "--inference_steps", str(request["inference_steps"]),
        "--guidance_scale", str(request["guidance_scale"]),
        "--seed", str(request["seed"]),
        "--enable_deepcache",
        "--video_path", str(video_path),
        "--audio_path", str(audio_path),
        "--video_out_path", str(raw_out),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{latentsync_root}:{env.get('PYTHONPATH', '')}"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    proc = subprocess.run(cmd, cwd=str(latentsync_root), env=env, check=False)
    if proc.returncode != 0 or not raw_out.is_file():
        raise RuntimeErrorCode("inference_failed")
    inference_seconds = round(time.monotonic() - t_inf0, 3)

    # Restore exact source CFR and mix requested audio track
    t_enc0 = time.monotonic()
    mix_mode = request["audio_mix_mode"]
    enc_cmd = [
        ffmpeg_bin, "-y", "-v", "error",
        "-i", str(raw_out),
    ]
    if mix_mode == "driver_wav":
        enc_cmd += [
            "-i", str(audio_path),
            "-map", "0:v:0", "-map", "1:a:0",
            "-r", v_meta["fps_str"],
            "-t", f"{eff_dur:.6f}",
            "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
            str(final_out),
        ]
    elif mix_mode == "source_audio" and v_meta["has_audio"]:
        enc_cmd += [
            "-i", str(video_path),
            "-map", "0:v:0", "-map", "1:a:0",
            "-r", v_meta["fps_str"],
            "-t", f"{eff_dur:.6f}",
            "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-c:a", "copy", "-movflags", "+faststart",
            str(final_out),
        ]
    else:
        enc_cmd += [
            "-map", "0:v:0", "-an",
            "-r", v_meta["fps_str"],
            "-t", f"{eff_dur:.6f}",
            "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(final_out),
        ]
    if subprocess.run(enc_cmd, check=False).returncode != 0 or not final_out.is_file():
        raise RuntimeErrorCode("output_media_mismatch")

    # Side-by-side preview matching preview_geometry(w, h)
    prev_w, prev_h = preview_geometry(v_meta["width"], v_meta["height"])
    panel_w = prev_w // 2
    prev_cmd = [
        ffmpeg_bin, "-y", "-v", "error",
        "-i", str(video_path),
        "-i", str(final_out),
        "-filter_complex",
        f"[0:v]scale={panel_w}:{prev_h},setsar=1[l];[1:v]scale={panel_w}:{prev_h},setsar=1[r];[l][r]hstack=inputs=2[v]",
        "-map", "[v]",
        "-map", "1:a:0?" if mix_mode != "mute" else "0:a:0?",
        "-r", v_meta["fps_str"],
        "-t", f"{eff_dur:.6f}",
        "-c:v", "libx264", "-crf", "20", "-preset", "fast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
        str(preview_out),
    ]
    if subprocess.run(prev_cmd, check=False).returncode != 0 or not preview_out.is_file():
        raise RuntimeErrorCode("preview_media_mismatch")

    try:
        raw_out.unlink(missing_ok=True)
    except OSError:
        pass

    encode_seconds = round(time.monotonic() - t_enc0, 3)
    out_meta = probe_video_file(final_out)
    prev_meta = probe_video_file(preview_out)
    total_seconds = round(time.monotonic() - t_start, 3)

    return {
        "input_media": v_meta,
        "audio_media": probed["audio"],
        "media": {
            **out_meta,
            "audio_mode": mix_mode,
        },
        "preview_media": prev_meta,
        "parameters": {
            "inference_steps": request["inference_steps"],
            "guidance_scale": request["guidance_scale"],
            "seed": request["seed"],
            "audio_mix_mode": mix_mode,
            "video_codec": "h264",
            "pixel_format": "yuv420p",
            "crf": 16,
        },
        "model": {
            "backend": "latentsync_1_6",
            "model_id": "latentsync-1.6",
            "unet_config": "configs/unet/stage2_512.yaml",
        },
        "input_video_sha256": request["source_video_sha256"],
        "input_audio_sha256": request["audio_sha256"],
        "output_sha256": _sha256_file(final_out),
        "preview_sha256": _sha256_file(preview_out),
        "runtime_metrics": {
            "inference_seconds": inference_seconds,
            "postprocess_encode_seconds": encode_seconds,
            "runtime_total_seconds": total_seconds,
            "gpu_uuid": gpu_uuid,
            "free_vram_mib_before": free_vram_mib_before,
        },
    }
