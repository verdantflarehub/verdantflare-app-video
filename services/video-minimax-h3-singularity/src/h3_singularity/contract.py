"""Native du-0 canvas and temporal grids, independent of CUDA imports."""

import math
import re
from urllib.parse import urlsplit

from .errors import RuntimeErrorCode


ASPECT_RATIOS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "adaptive")
MAX_PIXELS = 768 * 1344


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
    return {"requested_aspect_ratio": aspect_ratio, "width": width, "height": height,
            "actual_aspect_ratio": f"{width // divisor}:{height // divisor}",
            "canvas_basis": "first_visual_reference" if aspect_ratio == "adaptive" else "requested_aspect_ratio"}


def resolve_timing(seconds: int) -> dict:
    if type(seconds) is not int or not 4 <= seconds <= 15:
        raise RuntimeErrorCode("seconds_must_be_integer_4_to_15")
    frames = seconds * 24
    frames += (5 - frames) % 17
    return {"requested_seconds": seconds, "frames": frames, "fps": 24, "video_duration_seconds": frames / 24}


def validate_request(payload: dict, reference_base_url: str):
    required = {"idempotency_key", "model", "task", "prompt", "seconds", "conditions", "target"}
    allowed = required | {"num_inference_steps", "num_outputs_per_prompt", "flow_shift", "audio_flow_shift", "seed", "quality_profile"}
    if not isinstance(payload, dict) or not required <= payload.keys() or payload.keys() - allowed:
        raise RuntimeErrorCode("invalid_request_fields")
    if payload["model"] != "MiniMaxAI/MiniMax-H3" or payload["task"] != "ref2va":
        raise RuntimeErrorCode("unsupported_model_or_task")
    if not isinstance(payload["prompt"], str) or not payload["prompt"].strip():
        raise RuntimeErrorCode("prompt_required")
    resolve_timing(payload["seconds"])
    if type(payload.get("num_inference_steps", 4)) is not int or payload.get("num_inference_steps", 4) != 4:
        raise RuntimeErrorCode("singularity_requires_4_nfe")
    if payload.get("quality_profile", "du-0") != "du-0":
        raise RuntimeErrorCode("quality_profile_not_available")
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
    if ratio == "adaptive" and not counts["image"] and not counts["video"]:
        raise RuntimeErrorCode("adaptive_requires_visual_reference")


def validate_reference(item: dict, reference_base_url: str):
    if not isinstance(item, dict) or set(item) != {"type", "role", "uri", "size", "sha256"}:
        raise RuntimeErrorCode("invalid_reference_fields")
    if item["type"] not in {"image", "video", "audio"} or item["role"] != "reference":
        raise RuntimeErrorCode("invalid_reference_type_or_role")
    if type(item["size"]) is not int or not 0 < item["size"] <= 1024**3:
        raise RuntimeErrorCode("invalid_reference_size")
    if not isinstance(item["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
        raise RuntimeErrorCode("invalid_reference_hash")
    uri = item["uri"]
    if not isinstance(uri, str):
        raise RuntimeErrorCode("invalid_reference_uri")
    parsed, base = urlsplit(uri), urlsplit(reference_base_url)
    if (parsed.scheme not in {"http", "https"} or (parsed.scheme, parsed.netloc) != (base.scheme, base.netloc)
        or parsed.username or parsed.password or parsed.query or parsed.fragment
        or not re.fullmatch(r"/runtime-artifacts/art_[0-9a-f]{32}/content", parsed.path)):
        raise RuntimeErrorCode("reference_source_not_allowed")
