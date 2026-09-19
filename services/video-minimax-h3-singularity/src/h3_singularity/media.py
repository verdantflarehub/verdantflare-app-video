"""Reference decoding and generated MP4/AAC muxing."""

from __future__ import annotations

import hashlib
import io
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
        image = ImageOps.exif_transpose(image).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array)[None]


def load_video(path: Path) -> tuple[torch.Tensor, dict | None, float]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.base_rate or 24)
        frames = [torch.from_numpy(frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0) for frame in container.decode(stream)]
        audio_stream = container.streams.audio[0] if container.streams.audio else None
        audio = None
        if audio_stream is not None:
            chunks = []
            rate = audio_stream.codec_context.sample_rate or 32000
            for frame in container.decode(audio=audio_stream.index):
                data = frame.to_ndarray()
                if data.ndim == 1:
                    data = data[None, :]
                elif data.shape[0] != audio_stream.channels:
                    data = data.reshape(-1, audio_stream.channels).T
                scale = 32768.0 if data.dtype.kind in "iu" else 1.0
                chunks.append(torch.from_numpy(data.astype(np.float32) / scale))
            if chunks:
                audio = {"waveform": torch.cat(chunks, dim=1)[None], "sample_rate": int(rate)}
    if not frames:
        raise ValueError("video_has_no_frame")
    return torch.stack(frames), audio, fps


def load_audio(path: Path) -> dict:
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError("audio_has_no_stream")
        stream = container.streams.audio[0]
        rate = stream.codec_context.sample_rate or 32000
        chunks = []
        for frame in container.decode(audio=stream.index):
            data = frame.to_ndarray()
            if data.ndim == 1:
                data = data[None, :]
            elif data.shape[0] != stream.channels:
                data = data.reshape(-1, stream.channels).T
            scale = 32768.0 if data.dtype.kind in "iu" else 1.0
            chunks.append(torch.from_numpy(data.astype(np.float32) / scale))
    if not chunks:
        raise ValueError("audio_has_no_frame")
    return {"waveform": torch.cat(chunks, dim=1)[None], "sample_rate": int(rate)}


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


def mux_mp4(frames: torch.Tensor, audio: torch.Tensor, output: Path, fps: int = 24, sample_rate: int = 32000) -> dict:
    """Mux decoded H3 video/audio and return media facts after ffprobe."""
    if frames.ndim != 4:
        raise ValueError(f"invalid_video_shape:{tuple(frames.shape)}")
    if frames.shape[-1] != 3:
        raise ValueError(f"invalid_video_channels:{tuple(frames.shape)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    wav = output.with_suffix(".wav")
    _write_wav(wav, audio, sample_rate)
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
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
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
