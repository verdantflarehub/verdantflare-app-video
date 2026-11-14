"""Native H3 canvas, temporal grids, 2K IC geometry, and zero-CUDA health contracts."""

from __future__ import annotations

import fractions
import math
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

from . import __version__
from .errors import RuntimeErrorCode


ASPECT_RATIOS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "adaptive")
QUALITY_PROFILES = ("fasth3-nvfp4", "du-0")
PROFILE_NFE = {"fasth3-nvfp4": 8, "du-0": 4}
REGENERATE_2K_PROFILES = ("open_h3_2k_ic",)
REGENERATE_2K_GRID_MODES = ("budget_1440p", "exact_2x")

MAX_PIXELS = 768 * 1344
BUDGET_1440P_MAX_PIXELS = 2560 * 1440  # 3,686,400
RIFE_2K_MAX_EDGE = 3072
LATENT_STAGE_SHORT_MAX = 90  # 1440 // 16
LATENT_STAGE_LONG_MAX = 160  # 2560 // 16
IC_WINDOW_FRAMES = 124

PROJECT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
VIDEO_TASK_ID_RE = re.compile(r"video_task_[0-9a-f]{32}")
ARTIFACT_ID_RE = re.compile(r"art_[0-9a-f]{32}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")

LOCKED_P2_CODE_REVISION = "54a7104b204fb46486cf85c2f600942118186d57"
LOCKED_P2_WEIGHTS_SHA256 = {
    "minimax_h3_ref2va_pruned_int8_convrot.safetensors": (
        "9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779"
    ),
    "minimax_h3_hyperflow_8step_v1.0_comfyui_bf16.safetensors": (
        "9d08ffe79fd252a5feec3b53f074fbc12bfea9bb486c84af21a5db46854d0198"
    ),
    "minimax_h3_lms_v1.0_r64.safetensors": (
        "16f3195bc6bffa431c0598dc2031e520ba058b4fcbc9f104db9e3fdfdeacf60c"
    ),
    "minimax_h3_open_regenerate_2k_ic_comfy_bf16.safetensors": (
        "a0c93dce50d9034cc084073b6c095d09ff25ad02351ac4d4f10113d67ae64ab3"
    ),
    "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors": (
        "4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6"
    ),
    "silence_audio_124f.safetensors": (
        "dc90d8b8505f7160110b4122fb78348b232877e2928f7049c6833dcebde3f18c"
    ),
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": (
        "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3f2cba62feb"
    ),
}
# Exact verified SHA-256 digests from h3_5090_candidate_weights_v1.json
LOCKED_P2_WEIGHTS_SHA256["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"] = (
    "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6"
)
LOCKED_P2_WEIGHTS_SHA256["minimax_h3_video_vae_fp16.safetensors"] = (
    "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522"
)


def default_quality_profile() -> str:
    profile = os.environ.get("SINGULARITY_DEFAULT_QUALITY_PROFILE", "du-0").strip()
    if profile not in QUALITY_PROFILES:
        raise RuntimeErrorCode("quality_profile_not_available")
    return profile


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


def is_valid_h3_frame_bucket(frames: int) -> bool:
    return type(frames) is int and 107 <= frames <= 362 and (frames - 5) % 17 == 0


def temporal_windows(frames: int) -> list[tuple[int, int]]:
    """Deterministic 2K IC temporal window split preserving exact frame count N after overlap crossfade."""
    if not is_valid_h3_frame_bucket(frames):
        raise RuntimeErrorCode("invalid_source_media")
    if frames <= IC_WINDOW_FRAMES:
        return [(0, frames)]
    if frames <= 243:
        return [(0, IC_WINDOW_FRAMES), (frames - IC_WINDOW_FRAMES, frames)]
    s_mid = (frames - IC_WINDOW_FRAMES) // 2
    return [(0, IC_WINDOW_FRAMES), (s_mid, s_mid + IC_WINDOW_FRAMES), (frames - IC_WINDOW_FRAMES, frames)]


def preview_geometry(w: int, h: int) -> tuple[int, int]:
    """Byte-identical rational integer side-by-side preview geometry bounded to <= 3840x2160."""
    if type(w) is not int or type(h) is not int or w <= 0 or h <= 0:
        raise ValueError("invalid_preview_dimensions")
    if w <= 1920 and h <= 2160:
        panel_w = max(2, (w // 2) * 2)
        panel_h = max(2, (h // 2) * 2)
    elif 1920 * h <= 2160 * w:
        panel_w = 1920
        panel_h = max(2, ((h * 1920) // (w * 2)) * 2)
    else:
        panel_w = max(2, ((w * 2160) // (h * 2)) * 2)
        panel_h = 2160
    return (panel_w * 2, panel_h)


def _plan_latent_upscale_levels(
    src_w: int,
    src_h: int,
    target_w: int,
    target_h: int,
) -> list[list[int]]:
    tgt_lat_short = min(target_w, target_h) // 16
    tgt_lat_long = max(target_w, target_h) // 16
    if tgt_lat_short <= LATENT_STAGE_SHORT_MAX and tgt_lat_long <= LATENT_STAGE_LONG_MAX:
        return [[target_w, target_h]]

    g = math.gcd(src_w, src_h)
    rw, rh = src_w // g, src_h // g
    step_k = math.lcm(32 // math.gcd(32, rw), 32 // math.gcd(32, rh))
    unit_w, unit_h = rw * step_k, rh * step_k
    stage1: list[int] | None = None
    m = 1
    while True:
        sw, sh = unit_w * m, unit_h * m
        if min(sw, sh) // 16 > LATENT_STAGE_SHORT_MAX or max(sw, sh) // 16 > LATENT_STAGE_LONG_MAX:
            break
        if sw > src_w and sw < target_w:
            stage1 = [sw, sh]
        m += 1

    if stage1 is not None:
        return [stage1, [target_w, target_h]]
    return [[target_w, target_h]]


def resolve_2k_target_geometry(
    src_w: int,
    src_h: int,
    grid_mode: str = "budget_1440p",
) -> dict:
    """Resolve strict aspect-preserving, 32-aligned 2K target canvas under budget_1440p or exact_2x."""
    if type(src_w) is not int or type(src_h) is not int or src_w < 32 or src_h < 32:
        raise RuntimeErrorCode("invalid_source_media")
    if src_w % 32 != 0 or src_h % 32 != 0:
        raise RuntimeErrorCode("invalid_source_media")
    if grid_mode not in REGENERATE_2K_GRID_MODES:
        raise RuntimeErrorCode("unsupported_grid_mode")

    g = math.gcd(src_w, src_h)
    rw, rh = src_w // g, src_h // g
    step_k = math.lcm(32 // math.gcd(32, rw), 32 // math.gcd(32, rh))
    unit_w, unit_h = rw * step_k, rh * step_k

    max_budget: tuple[int, int] | None = None
    m = 1
    while True:
        cand_w, cand_h = unit_w * m, unit_h * m
        if cand_w * cand_h > BUDGET_1440P_MAX_PIXELS:
            break
        if min(cand_w, cand_h) // 16 <= LATENT_STAGE_SHORT_MAX and max(cand_w, cand_h) // 16 <= LATENT_STAGE_LONG_MAX:
            max_budget = (cand_w, cand_h)
        m += 1

    # Detect inputs that have already reached or exceeded the 2K target grid.
    already_2k_canvas = (
        min(src_w, src_h) >= 1408
        or max(src_w, src_h) >= 2464
        or src_w * src_h >= 1440 * 1440
        or (max_budget is not None and unit_w < src_w and max_budget[0] <= src_w)
    )
    if already_2k_canvas:
        raise RuntimeErrorCode("source_already_2k")

    if grid_mode == "exact_2x":
        target_w, target_h = src_w * 2, src_h * 2
    else:
        if max_budget is not None and max_budget[0] > src_w:
            target_w, target_h = max_budget
        else:
            # Prime adaptive ratio with no intermediate exact-ratio multiple below 1440p budget.
            target_w, target_h = src_w * 2, src_h * 2

    if target_w <= src_w or target_h <= src_h:
        raise RuntimeErrorCode("source_already_2k")
    if target_w % 32 != 0 or target_h % 32 != 0 or src_w * target_h != target_w * src_h:
        raise RuntimeErrorCode("invalid_source_media")
    if max(target_w, target_h) > RIFE_2K_MAX_EDGE:
        raise RuntimeErrorCode("invalid_source_media")

    levels = _plan_latent_upscale_levels(src_w, src_h, target_w, target_h)
    scale = fractions.Fraction(target_w, src_w)
    divisor = math.gcd(target_w, target_h)
    return {
        "source_w": src_w,
        "source_h": src_h,
        "target_w": target_w,
        "target_h": target_h,
        "grid_mode": grid_mode,
        "levels": levels,
        "scale_fraction": f"{scale.numerator}/{scale.denominator}",
        "actual_aspect_ratio": f"{target_w // divisor}:{target_h // divisor}",
    }


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
    quality_profile = payload.get("quality_profile", default_quality_profile())
    if not isinstance(quality_profile, str) or quality_profile not in QUALITY_PROFILES:
        raise RuntimeErrorCode("quality_profile_not_available")
    expected_nfe = PROFILE_NFE[quality_profile]
    steps = payload.get("num_inference_steps", expected_nfe)
    if type(steps) is not int or steps != expected_nfe:
        raise RuntimeErrorCode(
            "singularity_requires_4_nfe" if quality_profile == "du-0" else "fasth3_nvfp4_requires_8_nfe"
        )
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
    if not isinstance(target, dict) or target.keys() - {"aspect_ratio", "short_edge", "duration_seconds", "width", "height"}:
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


def validate_regenerate_request(payload: dict) -> dict:
    """Validate internal /v1/regenerate-2k payload from Video MCP ProcessingExecutor."""
    required = {"project_id", "idempotency_key", "source_artifact_id", "source_sha256"}
    allowed = required | {"profile", "grid_mode", "prompt", "seed"}
    if not isinstance(payload, dict) or not required <= payload.keys() or payload.keys() - allowed:
        raise RuntimeErrorCode("invalid_request_fields")

    project_id = payload["project_id"]
    if not isinstance(project_id, str) or not PROJECT_ID_RE.fullmatch(project_id):
        raise RuntimeErrorCode("invalid_project_id")
    idempotency_key = payload["idempotency_key"]
    if not isinstance(idempotency_key, str) or not VIDEO_TASK_ID_RE.fullmatch(idempotency_key):
        raise RuntimeErrorCode("invalid_idempotency_key")
    source_artifact_id = payload["source_artifact_id"]
    if not isinstance(source_artifact_id, str) or not ARTIFACT_ID_RE.fullmatch(source_artifact_id):
        raise RuntimeErrorCode("invalid_source_artifact_id")
    source_sha256 = payload["source_sha256"]
    if not isinstance(source_sha256, str) or not SHA256_RE.fullmatch(source_sha256):
        raise RuntimeErrorCode("invalid_source_sha256")

    profile = payload.get("profile", "open_h3_2k_ic")
    if not isinstance(profile, str) or profile not in REGENERATE_2K_PROFILES:
        raise RuntimeErrorCode("unsupported_regenerate_profile")
    grid_mode = payload.get("grid_mode", "budget_1440p")
    if not isinstance(grid_mode, str) or grid_mode not in REGENERATE_2K_GRID_MODES:
        raise RuntimeErrorCode("unsupported_grid_mode")
    prompt = payload.get("prompt", "")
    if not isinstance(prompt, str) or len(prompt) > 16384:
        raise RuntimeErrorCode("invalid_prompt")
    seed = payload.get("seed", 7)
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        raise RuntimeErrorCode("invalid_seed")

    return {
        "project_id": project_id,
        "idempotency_key": idempotency_key,
        "source_artifact_id": source_artifact_id,
        "source_sha256": source_sha256,
        "profile": profile,
        "grid_mode": grid_mode,
        "prompt": prompt,
        "seed": seed,
    }


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
    """Zero-CUDA file existence and locked byte-size verification for GET /health."""
    resolved_base = Path(base_root or os.environ.get("H3_BASE_MODEL_ROOT", "/models/MiniMax-H3"))
    resolved_singularity = Path(
        singularity_root
        or os.environ.get("SINGULARITY_MODEL_ROOT", str(resolved_base / "singularity"))
    )
    resolved_component = Path(
        component_root or os.environ.get("H3_COMPONENT_MODEL_ROOT", str(resolved_singularity))
    )
    default_profile = default_quality_profile()

    exact_size_checks: list[tuple[str, Path, int]] = [
        (
            "du0_diffusion",
            resolved_singularity
            / os.environ.get(
                "SINGULARITY_DIFFUSION_NAME",
                "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors",
            ),
            20_967_647_456,
        ),
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
        (
            "fasth3_nvfp4_diffusion",
            resolved_base / "candidates" / "a1_fasth3_nvfp4" / "fasth3_omniref_qad_nvfp4_comfy.safetensors",
            13_773_998_700,
        ),
        (
            "shared_qwen32b_nvfp4_awq",
            resolved_base
            / "candidates"
            / "shared"
            / "text_encoders"
            / "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            15_687_142_551,
        ),
        (
            "shared_video_vae_fp16",
            resolved_base / "candidates" / "shared" / "vae" / "minimax_h3_video_vae_fp16.safetensors",
            5_207_808_496,
        ),
        (
            "p2_convrot_dit",
            resolved_base
            / "candidates"
            / "a2_vsa_turbo_v4"
            / "diffusion_models"
            / "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            20_970_379_616,
        ),
        (
            "p2_hyperflow_lora",
            resolved_base
            / "candidates"
            / "p2_open_h3_2k_ic"
            / "comfyui"
            / "minimax_h3_hyperflow_8step_v1.0_comfyui_bf16.safetensors",
            3_926_338_288,
        ),
        (
            "p2_lms_lora",
            resolved_base
            / "candidates"
            / "p2_open_h3_2k_ic"
            / "official"
            / "lora"
            / "minimax_h3_lms_v1.0_r64.safetensors",
            1_239_664_536,
        ),
        (
            "p2_open_2k_ic_lora",
            resolved_base
            / "candidates"
            / "p2_open_h3_2k_ic"
            / "comfyui"
            / "minimax_h3_open_regenerate_2k_ic_comfy_bf16.safetensors",
            5_868_430_560,
        ),
        (
            "p2_latent_upscaler_3d",
            resolved_base
            / "candidates"
            / "p2_open_h3_2k_ic"
            / "official"
            / "latent_upsampler"
            / "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
            690_592_992,
        ),
        (
            "p2_silence_audio_124f",
            resolved_base
            / "candidates"
            / "p2_open_h3_2k_ic"
            / "official"
            / "assets"
            / "silence_audio_124f.safetensors",
            26_768,
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
        "default_quality_profile": default_profile,
        "model": (
            "fasth3_omniref_qad_nvfp4_comfy.safetensors"
            if default_profile == "fasth3-nvfp4"
            else os.environ.get(
                "SINGULARITY_DIFFUSION_NAME",
                "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors",
            )
        ),
        "nfe": PROFILE_NFE[default_profile],
        "gpu_worker_active": bool(gpu_worker_active),
        "active_task_ids": list(active_task_ids or []),
        "capabilities": {
            "aspect_ratios": list(ASPECT_RATIOS),
            "quality_profiles": list(QUALITY_PROFILES),
            "regenerate_2k_profiles": list(REGENERATE_2K_PROFILES),
            "regenerate_2k_grid_modes": list(REGENERATE_2K_GRID_MODES),
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
