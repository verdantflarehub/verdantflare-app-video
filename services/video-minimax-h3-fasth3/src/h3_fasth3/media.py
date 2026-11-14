"""Reference decoding and generated MP4/AAC muxing."""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import wave

import av
import numpy as np
import torch
from PIL import Image, ImageOps


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        if min(image.size) < 32 or image.width * image.height > 16 * 1024 * 1024 or getattr(image, "n_frames", 1) != 1:
            raise ValueError("image_decode_budget_exceeded")
        image = ImageOps.exif_transpose(image).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array)[None]


def load_video(path: Path) -> tuple[torch.Tensor, dict | None, float]:
    """Sample the source presentation timeline at H3's 24 FPS.

    Decode sound in a fresh container: the video iterator has already reached
    EOF. Stream objects, rather than absolute stream indices, select audio.
    """
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream_side_data=rotation",
        "-of", "json", str(path)], timeout=30))
    rotation = float(next((item["rotation"] for stream in probe.get("streams", [])
                          for item in stream.get("side_data_list", []) if "rotation" in item), 0))
    if not math.isfinite(rotation) or abs(rotation / 90 - round(rotation / 90)) > 1e-6:
        raise ValueError("video_rotation_must_be_right_angle")
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("video_has_no_stream")
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.base_rate or 24)
        if not 0 < fps <= 120 or stream.width * stream.height > 4096 * 2160:
            raise ValueError("video_decode_budget_exceeded")
        frames, timestamps = [], []
        last_duration = 1 / fps
        for frame in container.decode(stream):
            timestamp = float(frame.time) if frame.time is not None else len(frames) / fps
            if ((len(frames) + 1) * frame.width * frame.height > 1_000_000_000
                or timestamps and timestamp - timestamps[0] > 15.001):
                raise ValueError("video_decode_budget_exceeded")
            if timestamps and timestamp <= timestamps[-1]:
                raise ValueError("video_nonmonotonic_timestamps")
            timestamps.append(timestamp)
            pixels = frame.to_ndarray(format="rgb24")
            frames.append(np.rot90(pixels, round(rotation / 90)).copy() if rotation else pixels)
            last_duration = float(frame.duration * frame.time_base) if frame.duration and frame.time_base else 1 / fps
        has_audio = bool(container.streams.audio)
    if not frames:
        raise ValueError("video_has_no_frame")
    start = timestamps[0]
    duration = timestamps[-1] - start + last_duration
    count = max(1, math.ceil(duration * 24 - 1e-6))
    indices = np.searchsorted(np.asarray(timestamps) - start, np.arange(count) / 24 + 1e-8, side="right") - 1
    sampled = np.stack([frames[max(0, int(index))] for index in indices])
    audio = load_audio(path, start_seconds=start, duration_seconds=count / 24) if has_audio else None
    return torch.from_numpy(sampled.astype(np.float32) / 255.0), audio, 24.0


def align_video_reference(frames: torch.Tensor, audio: dict | None, output_frames: int):
    """Match the pinned node's downward reference bucket, including its sound."""
    count = min(len(frames), output_frames)
    count -= (count - 5) % 17
    if count < 5:
        raise ValueError("reference_video_too_short")
    if audio is not None:
        audio = {**audio, "waveform": audio["waveform"][..., :round(count / 24 * audio["sample_rate"])]}
    return frames[:count], audio


def load_audio(path: Path, *, start_seconds: float | None = None, duration_seconds: float | None = None) -> dict:
    """Normalize PCM through FFmpeg to planar float, retaining channel order.

    A paired video supplies its timeline origin and duration. Preserve audio
    offsets with silence and trim only samples outside that video interval.
    """
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError("audio_has_no_stream")
        stream = container.streams.audio[0]
        rate = stream.codec_context.sample_rate or 32000
        if not 1 <= len(stream.layout.channels) <= 2 or not 0 < rate <= 192000:
            raise ValueError("audio_decode_budget_exceeded")
        resampler = av.AudioResampler(format="fltp", layout=stream.layout.name, rate=rate)
        chunks = []
        cursor = 0
        origin = start_seconds

        def append(frame):
            nonlocal origin, cursor
            data = frame.to_ndarray()
            timestamp = float(frame.time) if frame.time is not None else (origin or 0) + cursor / rate
            if origin is None:
                origin = timestamp
            offset = round((timestamp - origin) * rate)
            if offset + data.shape[1] > math.ceil(15.1 * rate):
                raise ValueError("audio_decode_budget_exceeded")
            # Discard encoder priming or overlapping samples before the origin.
            skip = max(0, cursor - offset)
            data = data[:, skip:]
            offset += skip
            if data.shape[1] == 0:
                return
            if offset > cursor:
                chunks.append(torch.zeros((data.shape[0], offset - cursor)))
            chunks.append(torch.from_numpy(data.copy()))
            cursor = offset + data.shape[1]

        for frame in container.decode(stream):
            for converted in resampler.resample(frame):
                append(converted)
        for converted in resampler.resample(None):
            append(converted)
    if not chunks:
        raise ValueError("audio_has_no_frame")
    waveform = torch.cat(chunks, dim=1)
    if duration_seconds is not None:
        length = round(duration_seconds * rate)
        waveform = waveform[:, :length]
        if waveform.shape[1] < length:
            waveform = torch.nn.functional.pad(waveform, (0, length - waveform.shape[1]))
    return {"waveform": waveform[None], "sample_rate": int(rate)}


def _write_wav(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    waveform = waveform.detach().float().cpu().clamp(-1, 1)
    if waveform.ndim == 3:
        waveform = waveform[0]
    pcm = (waveform.T * 32767).round().to(torch.int16).numpy().tobytes()
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(int(waveform.shape[0]))
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm)


def mux_mp4(frames: torch.Tensor, audio: torch.Tensor | dict, output: Path, fps: int = 24, sample_rate: int = 32000) -> dict:
    """Mux decoded H3 video/audio and return media facts after ffprobe."""
    if frames.ndim != 4:
        raise ValueError(f"invalid_video_shape:{tuple(frames.shape)}")
    if frames.shape[-1] != 3:
        raise ValueError(f"invalid_video_channels:{tuple(frames.shape)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    wav = output.with_suffix(".wav")
    # Comfy VAEDecodeAudio returns an AUDIO object. Preserve its source rate
    # in the WAV header; ffmpeg resamples to the requested output rate below.
    waveform = audio["waveform"] if isinstance(audio, dict) else audio
    source_rate = int(audio["sample_rate"]) if isinstance(audio, dict) else sample_rate
    _write_wav(wav, waveform, source_rate)
    command = [
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{frames.shape[2]}x{frames.shape[1]}", "-r", str(fps), "-i", "pipe:0",
        "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
        "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", str(sample_rate),
        "-ac", "2", "-movflags", "+faststart", str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            process.stdin.write((frame.clamp(0, 1).mul(255).round().to(torch.uint8).numpy()).tobytes())
        process.stdin.close()
        stderr = process.stderr.read()
        if process.wait() != 0:
            raise RuntimeError(stderr.decode(errors="replace"))
    except BaseException as exc:
        # Preserve ffmpeg's actual diagnostic in the Python error. The
        # runtime queue records the exception type, so without this context a
        # media failure is indistinguishable from a sampler failure.
        if isinstance(exc, BrokenPipeError):
            detail = process.stderr.read().decode(errors="replace").strip()
            process.kill()
            process.wait()
            raise RuntimeError(f"ffmpeg_pipe_failed:{detail}") from exc
        process.kill()
        process.wait()
        raise
    finally:
        process.stdin.close()
        process.stderr.close()
        wav.unlink(missing_ok=True)
    probe = subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-show_entries",
        "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate,nb_read_frames,sample_rate,channels",
        "-of", "json", str(output),
    ])
    import json
    data = json.loads(probe)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    sound = next(s for s in data["streams"] if s["codec_type"] == "audio")
    return {
        "width": int(video["width"]), "height": int(video["height"]),
        "frames": int(video["nb_read_frames"]), "frame_rate": video["r_frame_rate"],
        "duration_seconds": float(data["format"]["duration"]), "video_codec": video["codec_name"],
        "audio_codec": sound["codec_name"], "audio_sample_rate": int(sound["sample_rate"]),
        "audio_channels": int(sound["channels"]), "sha256": sha256(output), "bytes": output.stat().st_size,
    }
