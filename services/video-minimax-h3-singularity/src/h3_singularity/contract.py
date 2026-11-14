"""Native H3 canvas, temporal grids, and zero-CUDA health contracts for Singularity (du-0)."""

from __future__ import annotations

import math
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

from . import __version__
from .errors import RuntimeErrorCode


ASPECT_RATIOS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "adaptive")
QUALITY_PROFILES = ("du-0",)
DEFAULT_QUALITY_PROFILE = "du-0"
DEFAULT_NFE = 4

MAX_PIXELS = 768 * 1344
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def default_quality_profile() -> str:
    return DEFAULT_QUALITY_PROFILE


def resolve_geometry(aspect_ratio: str, source_size: tuple[int, int] | None = None) -> dict:
    if aspect_ratio not in ASPECT_RATIOS:
        raise RuntimeErrorCode("unsupported_aspect_ratio")
    if aspect_ratio == "adaptive":
        if source_size is None or min(source_size) <= 0:
            raise RuntimeErrorCode("adaptive_requires_visual_reference")
        numerator, denominator = source_size
    else:
        numerator, denominator = map(int, aspect_ratio.split(":"))
    ratio = numerator / denominator
    width, height = (768 * ratio, 768) if ratio >= 1 else (768, 768 / ratio)
    scale = min(1, math.sqrt(MAX_PIXELS / (width * height)))
    width, height = max(32, round(width * scale / 32) * 32), max(32, round(height * scale / 32) * 32)
    divisor = math.gcd(width, height)
    return {
        "requested_aspect_ratio": aspect_ratio,
        "width": width,
        "height": height,
        "actual_aspect_ratio": f"{width // divisor}:{height // divisor}",
        "canvas_basis": "first_visual_reference" if aspect_ratio == "adaptive" else "requested_aspect_ratio",
    }


def resolve_timing(seconds: int) -> dict:
    if type(seconds) is not int or not 4 <= seconds <= 15:
        raise RuntimeErrorCode("seconds_must_be_integer_4_to_15")
    frames = seconds * 24
    frames += (5 - frames) % 17
    return {"requested_seconds": seconds, "frames": frames, "fps": 24, "video_duration_seconds": frames / 24}


def validate_request(payload: dict, reference_base_url: str):
    required = {"idempotency_key", "model", "task", "prompt", "seconds", "conditions", "target"}
    allowed = required | {
        "num_inference_steps",
        "num_outputs_per_prompt",
        "flow_shift",
        "audio_flow_shift",
        "seed",
        "quality_profile",
        "audio_guidance",
        "image_guidance",
    }
    if not isinstance(payload, dict) or not required <= payload.keys() or payload.keys() - allowed:
        raise RuntimeErrorCode("invalid_request_fields")
    if payload["model"] != "MiniMaxAI/MiniMax-H3" or payload["task"] != "ref2va":
        raise RuntimeErrorCode("unsupported_model_or_task")
    image_guidance = payload.get("image_guidance", "reference")
    if not isinstance(image_guidance, str) or image_guidance not in {"reference", "first_frame"}:
        raise RuntimeErrorCode("unsupported_image_guidance")
    audio_guidance = payload.get("audio_guidance", "reference")
    if not isinstance(audio_guidance, str) or audio_guidance not in {"reference", "timeline"}:
        raise RuntimeErrorCode("unsupported_audio_guidance")
    if not isinstance(payload["prompt"], str) or not payload["prompt"].strip():
        raise RuntimeErrorCode("prompt_required")
    resolve_timing(payload["seconds"])
    quality_profile = payload.get("quality_profile", DEFAULT_QUALITY_PROFILE)
    if not isinstance(quality_profile, str) or quality_profile not in QUALITY_PROFILES:
        raise RuntimeErrorCode("quality_profile_not_available")
    steps = payload.get("num_inference_steps", DEFAULT_NFE)
    if type(steps) is not int or steps != DEFAULT_NFE:
        raise RuntimeErrorCode("singularity_requires_4_nfe")
    if type(payload.get("num_outputs_per_prompt", 1)) is not int or payload.get("num_outputs_per_prompt", 1) != 1:
        raise RuntimeErrorCode("single_output_required")
    for name, expected in (("flow_shift", 12.0), ("audio_flow_shift", 3.0)):
        value = payload.get(name, expected)
        if type(value) not in (int, float) or value != expected:
            raise RuntimeErrorCode("unsupported_" + name)
    seed = payload.get("seed", 7)
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        raise RuntimeErrorCode("invalid_seed")
    target = payload["target"]
    if not isinstance(target, dict) or target.keys() - {
        "aspect_ratio",
        "short_edge",
        "duration_seconds",
        "width",
        "height",
    }:
        raise RuntimeErrorCode("invalid_target")
    ratio = target.get("aspect_ratio")
    if ratio not in ASPECT_RATIOS:
        raise RuntimeErrorCode("unsupported_aspect_ratio")
    if target.get("duration_seconds", payload["seconds"]) != payload["seconds"]:
        raise RuntimeErrorCode("inconsistent_target_duration")
    if target.get("short_edge", 768) != 768:
        raise RuntimeErrorCode("unsupported_short_edge")
    if "width" in target or "height" in target:
        if ratio == "adaptive":
            raise RuntimeErrorCode("adaptive_canvas_is_resolved_from_reference")
        geometry = resolve_geometry(ratio)
        if (target.get("width"), target.get("height")) != (geometry["width"], geometry["height"]):
            raise RuntimeErrorCode("inconsistent_target_geometry")
    refs = payload["conditions"]
    if not isinstance(refs, list) or not 1 <= len(refs) <= 12:
        raise RuntimeErrorCode("reference_count_must_be_1_to_12")
    counts = {"image": 0, "video": 0, "audio": 0}
    for item in refs:
        validate_reference(item, reference_base_url)
        counts[item["type"]] += 1
    if counts["image"] > 9 or counts["video"] > 3 or counts["audio"] > 3:
        raise RuntimeErrorCode("too_many_references_of_type")
    if image_guidance == "first_frame" and not counts["image"]:
        raise RuntimeErrorCode("first_frame_requires_image")
    if audio_guidance == "timeline" and (counts["audio"] != 1 or counts["video"]):
        raise RuntimeErrorCode("timeline_requires_one_audio_and_no_video")
    if ratio == "adaptive" and not counts["image"] and not counts["video"]:
        raise RuntimeErrorCode("adaptive_requires_visual_reference")


def validate_reference(item: dict, reference_base_url: str):
    if not isinstance(item, dict) or set(item) != {"type", "role", "uri", "size", "sha256"}:
        raise RuntimeErrorCode("invalid_reference_fields")
    if item["type"] not in {"image", "video", "audio"} or item["role"] != "reference":
        raise RuntimeErrorCode("invalid_reference_type_or_role")
    if type(item["size"]) is not int or not 0 < item["size"] <= 1024**3:
        raise RuntimeErrorCode("invalid_reference_size")
    if not isinstance(item["sha256"], str) or not SHA256_RE.fullmatch(item["sha256"]):
        raise RuntimeErrorCode("invalid_reference_hash")
    uri = item["uri"]
    if not isinstance(uri, str):
        raise RuntimeErrorCode("invalid_reference_uri")
    parsed, base = urlsplit(uri), urlsplit(reference_base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or (parsed.scheme, parsed.netloc) != (base.scheme, base.netloc)
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or not re.fullmatch(r"/runtime-artifacts/art_[0-9a-f]{32}/content", parsed.path)
    ):
        raise RuntimeErrorCode("reference_source_not_allowed")


def check_weights_health(
    *,
    base_root: Path | None = None,
    singularity_root: Path | None = None,
    component_root: Path | None = None,
    gpu_worker_active: bool = False,
    active_task_ids: list[str] | None = None,
) -> dict:
    """Zero-CUDA file existence and locked byte-size verification for GET /health (pure du-0)."""
    resolved_base = Path(base_root or os.environ.get("H3_BASE_MODEL_ROOT", "/models/MiniMax-H3"))
    resolved_singularity = Path(
        singularity_root
        or os.environ.get("SINGULARITY_MODEL_ROOT", str(resolved_base / "singularity"))
    )
    resolved_component = Path(
        component_root or os.environ.get("H3_COMPONENT_MODEL_ROOT", str(resolved_singularity))
    )

    diffusion_name = os.environ.get(
        "SINGULARITY_DIFFUSION_NAME",
        "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors",
    )
    exact_size_checks: list[tuple[str, Path, int]] = [
        ("du0_diffusion", resolved_singularity / diffusion_name, 20_967_647_456),
        (
            "du0_lora",
            resolved_singularity
            / "loras"
            / os.environ.get(
                "SINGULARITY_LORA_NAME",
                "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
            ),
            1_956_193_000,
        ),
        (
            "du0_video_vae",
            resolved_component
            / "vae"
            / os.environ.get("H3_VIDEO_VAE_NAME", "minimax_h3_video_vae_int8_convrot.safetensors"),
            2_811_065_184,
        ),
        (
            "shared_audio_vae",
            resolved_component
            / "vae"
            / os.environ.get("H3_AUDIO_VAE_NAME", "minimax_h3_audio_vae_fp32.safetensors"),
            605_254_808,
        ),
    ]
    nonempty_checks: list[tuple[str, Path]] = [
        ("du0_model_evidence", resolved_singularity / "model-evidence.json"),
        ("du0_component_evidence", resolved_component / "component-model-evidence.json"),
        (
            "du0_clip",
            resolved_component
            / "text_encoders"
            / os.environ.get("H3_CLIP_NAME", "qwen3vl_4b_fp8_scaled.safetensors"),
        ),
        (
            "du0_clip_proj",
            resolved_singularity
            / "clip_projections"
            / os.environ.get("H3_CLIP_PROJECTION_NAME", "mmh3-4b-ClipProj-v3.1-mlp.safetensors"),
        ),
    ]

    for label, path, expected_bytes in exact_size_checks:
        if not path.is_file():
            raise RuntimeErrorCode(f"weight_missing:{label}")
        if path.stat().st_size != expected_bytes:
            raise RuntimeErrorCode(f"weight_size_mismatch:{label}")
    for label, path in nonempty_checks:
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeErrorCode(f"weight_missing:{label}")

    version = os.environ.get("SINGULARITY_RUNTIME_VERSION", f"video-minimax-h3-singularity-v{__version__}")
    return {
        "ready": True,
        "runtime_version": version,
        "architecture": "m-a-subprocess-supervisor",
        "default_quality_profile": DEFAULT_QUALITY_PROFILE,
        "model": diffusion_name,
        "nfe": DEFAULT_NFE,
        "gpu_worker_active": bool(gpu_worker_active),
        "active_task_ids": list(active_task_ids or []),
        "capabilities": {
            "aspect_ratios": list(ASPECT_RATIOS),
            "quality_profiles": list(QUALITY_PROFILES),
            "audio_guidance_modes": ["reference", "timeline"],
            "image_guidance_modes": ["reference", "first_frame"],
            "first_frame_experimental": True,
            "first_frame_gpu_verified": False,
            "timeline_audio_experimental": True,
            "timeline_audio_gpu_verified": False,
            "duration_seconds": {"minimum": 4, "maximum": 15},
            "frame_grid": "17k+5",
            "fps": 24,
            "canvases": {ratio: resolve_geometry(ratio) for ratio in ASPECT_RATIOS if ratio != "adaptive"},
        },
    }
