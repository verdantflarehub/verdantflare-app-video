"""CPU-only input contract and 124-frame temporal planning."""

from dataclasses import dataclass
import os
from pathlib import Path
import re

from .errors import RuntimeErrorCode


@dataclass(frozen=True)
class SwapWindow:
    start: int
    end: int
    history: int
    bucket: int

    def pad(self, frames):
        valid = frames[self.start:self.end]
        if len(valid) != self.end - self.start:
            raise RuntimeErrorCode("duration_out_of_range")
        return list(valid) + [valid[-1]] * (self.bucket - len(valid))

    def trim(self, decoded):
        if len(decoded) != self.bucket:
            raise RuntimeErrorCode("frame_count_mismatch")
        return decoded[self.history:self.end - self.start]


def plan_swap_windows(frames: int) -> list[SwapWindow]:
    if type(frames) is not int or not 24 <= frames <= 362:
        raise RuntimeErrorCode("duration_out_of_range")
    windows = []
    start = 0
    while True:
        end = min(start + 124, frames)
        bucket = 107 if end - start <= 107 else 124
        windows.append(SwapWindow(start, end, 18 if start else 0, bucket))
        if end == frames:
            return windows
        start += 106


TURBO_VARIANTS = ("fl2v_turbo_8step", "pika_h3_flex_turbo_8step_hd_ref2va")
IDENTIFIERS = {
    "project_id": r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}",
    "idempotency_key": r"video_task_[0-9a-f]{32}",
    "source_artifact_id": r"art_[0-9a-f]{32}",
    "character_image_artifact_id": r"art_[0-9a-f]{32}",
    "source_sha256": r"[0-9a-f]{64}",
    "character_image_sha256": r"[0-9a-f]{64}",
}
OPTIONS = {
    "character_lora_version": ("v1.5", "v1"),
    "turbo_variant": TURBO_VARIANTS,
    "resolution_preset": ("768p", "480p"),
    "seed_mode": ("fixed",),
}


def validate_swap_request(payload: dict) -> dict:
    required = set(IDENTIFIERS) | {"prompt"}
    allowed = required | set(OPTIONS) | {"seed", "lora_strength"}
    if not isinstance(payload, dict) or required - payload.keys() or payload.keys() - allowed:
        raise RuntimeErrorCode("invalid_request_fields")
    request = payload.copy()
    for key, pattern in IDENTIFIERS.items():
        value = request[key]
        if not isinstance(value, str) or not re.fullmatch(pattern, value):
            raise RuntimeErrorCode(f"invalid_{key}")
    prompt = request["prompt"]
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16384:
        raise RuntimeErrorCode("invalid_prompt")
    if any(int(n) != 1 for n in re.findall(r"<(?:Picture|Image|Video)\s+(\d+)>", prompt, re.I)):
        raise RuntimeErrorCode("multi_character_single_pass_unsupported")
    for key, choices in OPTIONS.items():
        if request.setdefault(key, choices[0]) not in choices:
            raise RuntimeErrorCode(f"unsupported_{key}")
    seed = request.setdefault("seed", 7)
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        raise RuntimeErrorCode("invalid_seed")
    strength = request.setdefault("lora_strength", 1.0)
    if type(strength) not in (int, float) or not 0.5 <= strength <= 1.2:
        raise RuntimeErrorCode("invalid_lora_strength")
    return request


def check_weights_health(*, model_root=None, gpu_worker_active=False, active_task_ids=None):
    base = Path(model_root or os.getenv("CHARACTER_SWAP_MODEL_ROOT", "/models/MiniMax-H3"))
    missing = []
    for relative, size, _ in WEIGHTS:
        path = base / relative
        if not path.is_file() or path.stat().st_size != size:
            missing.append(relative)
    return {
        "ready": not missing,
        "missing_weights": missing,
        "runtime_version": os.getenv("CHARACTER_SWAP_RUNTIME_VERSION", "video-minimax-h3-character-swap-v0.1.0"),
        "architecture": "m-a-subprocess-supervisor",
        "model": Path(WEIGHTS[0][0]).name,
        "gpu_worker_active": gpu_worker_active,
        "active_task_ids": active_task_ids or [],
        "capabilities": {
            "character_swap_profiles": ["h3_character_swap_lora"],
            "character_lora_versions": list(OPTIONS["character_lora_version"]),
            "resolution_presets": list(OPTIONS["resolution_preset"]),
            "turbo_variants": list(TURBO_VARIANTS),
        },
    }


# Relative PVC path, exact bytes, SHA-256 from the approved contract.
WEIGHTS = (
    ("candidates/a2_vsa_turbo_v4/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
     20_970_379_616, "9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779"),
    ("candidates/shared/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
     15_687_142_551, "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6"),
    ("candidates/shared/vae/minimax_h3_video_vae_fp16.safetensors",
     5_207_808_496, "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522"),
    ("singularity/vae/minimax_h3_audio_vae_fp32.safetensors",
     605_254_808, "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48"),
    ("character_swap/loras/h3_character_swap_v1.5.safetensors",
     155_110_792, "8f2c3c3eba161df8d0c6099598b4730f641f5ce01e04cd16d2427cff8a35218f"),
    ("character_swap/loras/h3_character_swap_pro4500_1000.safetensors",
     155_110_320, "4b2a3f420ae804c0aa3422761ff84dbd1bf52eef6900ffab6d2e66df63cb4e79"),
    ("character_swap/loras/minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
     1_956_193_000, "08cfe946033af7d27719b964b6e0a0e50c32138daabbd6ce4137e23df6bf9980"),
    ("character_swap/loras/pika_h3_flex_turbo_8step_hd_ref2va.safetensors",
     3_912_311_400, "dfe46d972408d1094053ff1f30c720bf16176ec334962009767ab0fff29e6de2"),
)
