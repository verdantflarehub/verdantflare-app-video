"""Public du-0 output facts; keep in parity with the runtime contract tests."""

import math


ASPECT_RATIOS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "adaptive")


def output_spec(aspect_ratio, seconds, visual=None):
    if aspect_ratio == "adaptive":
        if not visual:
            raise ValueError("adaptive requires an image or video reference")
        ratio = visual["width"] / visual["height"]
    else:
        a, b = map(int, aspect_ratio.split(":"))
        ratio = a / b
    width, height = (768 * ratio, 768) if ratio >= 1 else (768, 768 / ratio)
    scale = min(1, math.sqrt((768 * 1344) / (width * height)))
    width, height = max(32, round(width * scale / 32) * 32), max(32, round(height * scale / 32) * 32)
    divisor = math.gcd(width, height)
    frames = seconds * 24 + (5 - seconds * 24) % 17
    return {"requested_aspect_ratio": aspect_ratio, "width": width, "height": height,
            "actual_aspect_ratio": f"{width // divisor}:{height // divisor}",
            "canvas_basis": "first_visual_reference" if aspect_ratio == "adaptive" else "requested_aspect_ratio",
            "requested_seconds": seconds, "frames": frames, "fps": 24, "video_duration_seconds": frames / 24}
