"""Opt-in target-timeline conditioning; never truncate approved dialogue.

This provides conditioning, not guaranteed audio preservation or lip sync.
No model imports are needed for the duration and payload checks.
"""

from .errors import RuntimeErrorCode


def validate_timeline_pcm(audio: dict, frames: int) -> dict:
    waveform = audio["waveform"]
    rate = audio["sample_rate"]
    if (type(rate) is not int or rate <= 0 or waveform.ndim != 3
            or waveform.shape[0] != 1 or waveform.shape[1] not in (1, 2)):
        raise RuntimeErrorCode("invalid_timeline_audio")
    samples = waveform.shape[-1]
    # Compare rational durations, allowing only one PCM sample of rounding.
    if samples <= 0 or abs(samples * 24 - frames * rate) > 24:
        raise RuntimeErrorCode("timeline_audio_duration_mismatch")
    return {"sample_rate": rate, "sample_count": samples,
            "target_frames": frames, "start_frame": 0}


def anchor_encoded_audio(conditioning: list, audio_ticks: int) -> list:
    """Reuse the already encoded standalone reference at target frame zero.

    Unlike AddGuide's default overflow behavior, reject any length mismatch.
    Copy metadata containers, preserving tensor identity and original inputs.
    """
    if not conditioning or type(audio_ticks) is not int or audio_ticks <= 0:
        raise RuntimeErrorCode("invalid_timeline_conditioning")
    result = []
    for embedding, metadata in conditioning:
        refs = metadata.get("minimax_refs", [])
        audio_refs = [ref for ref in refs if ref.get("kind") == "audio"]
        if len(audio_refs) != 1 or any(ref.get("kind") == "video_audio" for ref in refs):
            raise RuntimeErrorCode("timeline_requires_one_standalone_audio")
        encoded = audio_refs[0].get("audio_latent")
        if encoded is None or tuple(encoded.shape) != (1, 32, 2, audio_ticks):
            raise RuntimeErrorCode("timeline_audio_latent_length_mismatch")
        keyframes = list(metadata.get("minimax_keyframes", []))
        if any(k.get("audio_latent") is not None for k in keyframes):
            raise RuntimeErrorCode("conflicting_timeline_audio")
        keyframes.append({"resolved_frame_index": 0, "audio_latent": encoded})
        result.append([embedding, {**metadata, "minimax_keyframes": keyframes}])
    return result
