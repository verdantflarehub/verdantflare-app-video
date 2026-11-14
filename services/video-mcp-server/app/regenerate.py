"""Pure 2K IC geometry, windowing, preview bounds, and locked digest contracts.

Keep pure functions byte-identical in behavior with:
- services/video-minimax-h3-singularity/src/h3_singularity/contract.py
- services/video-frame-interpolation-api/src/video_interpolation/media.py
"""

from __future__ import annotations

import fractions
import math
import os


SINGULARITY_QUALITY_PROFILES = ("fasth3-nvfp4", "du-0")
SINGULARITY_PROFILE_NFE = {"fasth3-nvfp4": 8, "du-0": 4}
REGENERATE_2K_PROFILES = ("open_h3_2k_ic",)
REGENERATE_2K_GRID_MODES = ("budget_1440p", "exact_2x")
REGENERATE_2K_MODEL_ID = "open-h3-regenerate-2k-ic"

BUDGET_1440P_MAX_PIXELS = 2560 * 1440  # 3,686,400
RIFE_2K_MAX_EDGE = 3072
LATENT_STAGE_SHORT_MAX = 90  # 1440 // 16
LATENT_STAGE_LONG_MAX = 160  # 2560 // 16
IC_WINDOW_FRAMES = 124

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
        "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6"
    ),
    "minimax_h3_video_vae_fp16.safetensors": (
        "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522"
    ),
}


def default_singularity_quality_profile() -> str:
    profile = os.environ.get("H3_SINGULARITY_DEFAULT_QUALITY_PROFILE", "du-0").strip() or "du-0"
    if profile not in SINGULARITY_QUALITY_PROFILES:
        raise ValueError("quality_profile_not_available")
    return profile


def is_valid_h3_frame_bucket(frames: int) -> bool:
    return type(frames) is int and 107 <= frames <= 362 and (frames - 5) % 17 == 0


def temporal_windows(frames: int) -> list[tuple[int, int]]:
    """Deterministic 2K IC temporal window split preserving exact frame count N after overlap crossfade."""
    if not is_valid_h3_frame_bucket(frames):
        raise ValueError("invalid_source_media")
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
        raise ValueError("invalid_source_media")
    if src_w % 32 != 0 or src_h % 32 != 0:
        raise ValueError("invalid_source_media")
    if grid_mode not in REGENERATE_2K_GRID_MODES:
        raise ValueError("unsupported_grid_mode")

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

    already_2k_canvas = (
        min(src_w, src_h) >= 1408
        or max(src_w, src_h) >= 2464
        or src_w * src_h >= 1440 * 1440
        or (max_budget is not None and unit_w < src_w and max_budget[0] <= src_w)
    )
    if already_2k_canvas:
        raise ValueError("source_already_2k")

    if grid_mode == "exact_2x":
        target_w, target_h = src_w * 2, src_h * 2
    else:
        if max_budget is not None and max_budget[0] > src_w:
            target_w, target_h = max_budget
        else:
            target_w, target_h = src_w * 2, src_h * 2

    if target_w <= src_w or target_h <= src_h:
        raise ValueError("source_already_2k")
    if target_w % 32 != 0 or target_h % 32 != 0 or src_w * target_h != target_w * src_h:
        raise ValueError("invalid_source_media")
    if max(target_w, target_h) > RIFE_2K_MAX_EDGE:
        raise ValueError("invalid_source_media")

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


def is_2k_canvas(width: int | float | None, height: int | float | None) -> bool:
    """Return True when dimensions represent a 2K canvas (and not 4K)."""
    if not width or not height:
        return False
    w, h = int(width), int(height)
    if w <= 0 or h <= 0:
        return False
    short_side, long_side = min(w, h), max(w, h)
    if short_side >= 2160:
        return False
    return short_side >= 1408 or long_side >= 2464
