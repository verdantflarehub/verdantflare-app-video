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


def validate_timeline_wav(path, frames):
    """Read exact PCM sample counts; container duration alone is insufficient."""
    import wave
    try:
        with wave.open(str(path), 'rb') as wav:
            rate, count = wav.getframerate(), wav.getnframes()
            channels, width = wav.getnchannels(), wav.getsampwidth()
            if wav.getcomptype() != 'NONE' or channels not in (1, 2) or width not in (2, 3, 4) or rate <= 0:
                raise ValueError('timeline_audio_requires_pcm_wav')
            if count <= 0 or abs(count * 24 - frames * rate) > 24:
                raise ValueError('timeline_audio_duration_mismatch')
            if len(wav.readframes(count + 1)) != count * channels * width:
                raise ValueError('timeline_audio_truncated')
    except (wave.Error, EOFError, OSError) as exc:
        raise ValueError('timeline_audio_requires_pcm_wav') from exc
    return {'sample_rate': rate, 'sample_count': count, 'start_frame': 0, 'target_frames': frames}


def check_audio_guidance(audio_guidance, service):
    if audio_guidance is None:
        return
    if service != 'h3-singularity':
        raise ValueError('audio_guidance_requires_h3_singularity')
    if not isinstance(audio_guidance, str) or audio_guidance not in {'reference', 'timeline'}:
        raise ValueError('unsupported_audio_guidance')


def runtime_supports_timeline(health):
    capabilities = health.get('capabilities', {}) if isinstance(health, dict) else {}
    modes = capabilities.get('audio_guidance_modes', []) if isinstance(capabilities, dict) else []
    return isinstance(modes, list) and 'timeline' in modes


def check_image_guidance(mode, service):
    if mode is None:
        return
    if service != 'h3-singularity':
        raise ValueError('image_guidance_requires_h3_singularity')
    if not isinstance(mode, str) or mode not in {'reference', 'first_frame'}:
        raise ValueError('unsupported_image_guidance')


def runtime_supports_first_frame(health):
    capabilities = health.get('capabilities', {}) if isinstance(health, dict) else {}
    modes = capabilities.get('image_guidance_modes', []) if isinstance(capabilities, dict) else []
    return isinstance(modes, list) and 'first_frame' in modes
