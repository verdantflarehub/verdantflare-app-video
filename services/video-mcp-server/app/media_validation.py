"""Decode imported media before publishing a generation input."""

from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
from PIL import Image


MAX_IMAGE_PIXELS = 16 * 1024 * 1024
MAX_VIDEO_PIXELS = 4096 * 2160
MAX_DECODED_VIDEO_PIXELS = 1_000_000_000


def validate_reference_media(path: Path, media_type: str) -> dict:
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path),
        ], check=True, capture_output=True, text=True, timeout=30)
        probe = json.loads(result.stdout)
        videos = [x for x in probe["streams"] if x["codec_type"] == "video" and not x.get("disposition", {}).get("attached_pic")]
        audios = [x for x in probe["streams"] if x["codec_type"] == "audio"]
        kind = media_type.split("/")[0]
        media = {"type": kind, "has_audio": bool(audios)}
        if kind in {"image", "video"}:
            if len(videos) != 1:
                raise ValueError("reference_requires_one_visual_stream")
            video = videos[0]
            width, height = int(video["width"]), int(video["height"])
            if width <= 0 or height <= 0:
                raise ValueError("reference_media_decode_failed")
            if min(width, height) < 32 or width * height > (MAX_IMAGE_PIXELS if kind == "image" else MAX_VIDEO_PIXELS):
                raise ValueError("reference_pixel_budget_exceeded")
            media.update(width=width, height=height, video_codec=video["codec_name"])
        if kind == "image":
            allowed = {"image/png": "png", "image/jpeg": "mjpeg", "image/webp": "webp"}
            if video["codec_name"] != allowed.get(media_type) or audios:
                raise ValueError("reference_image_format_mismatch")
            with Image.open(path) as image:
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("reference_image_must_be_still")
                if image.getexif().get(274, 1) in {5, 6, 7, 8}:
                    media["width"], media["height"] = height, width
        else:
            duration = float(probe["format"]["duration"])
            if not 2 <= duration <= 15.001:
                raise ValueError("reference_duration_must_be_2_to_15_seconds")
            media["duration_seconds"] = duration
            if kind == "video":
                if video["codec_name"] not in {"h264", "hevc"} or "mp4" not in probe["format"]["format_name"]:
                    raise ValueError("reference_video_requires_mp4_h264_or_hevc")
                fps = float(Fraction(video["avg_frame_rate"]))
                if not 0 < fps <= 120 or width * height * fps * duration > MAX_DECODED_VIDEO_PIXELS:
                    raise ValueError("reference_video_decode_budget_exceeded")
                media["source_fps"] = fps
                media["conditioning_fps"] = 24
                video_duration = float(video.get("duration", duration))
                if not 2 <= video_duration <= 15.001:
                    raise ValueError("reference_video_duration_must_be_2_to_15_seconds")
                media["video_duration_seconds"] = video_duration
                media["resampled_frames"] = math.ceil(video_duration * 24 - 1e-6)
                rotation = float(next((item["rotation"] for item in video.get("side_data_list", []) if "rotation" in item), 0))
                if not math.isfinite(rotation) or abs(rotation / 90 - round(rotation / 90)) > 1e-6:
                    raise ValueError("reference_video_rotation_must_be_right_angle")
                media["display_rotation_degrees"] = rotation
                if round(rotation / 90) % 2:
                    media["width"], media["height"] = height, width
            elif kind == "audio":
                if videos or len(audios) != 1:
                    raise ValueError("reference_requires_one_audio_stream")
                codec = audios[0]["codec_name"]
                if not (media_type == "audio/mpeg" and codec == "mp3" or
                        media_type == "audio/wav" and codec.startswith("pcm_") and probe["format"]["format_name"] == "wav"):
                    raise ValueError("reference_audio_requires_wav_pcm_or_mp3")
            else:
                raise ValueError("unsupported_reference_type")
        if audios:
            if len(audios) != 1 or not 1 <= int(audios[0]["channels"]) <= 2:
                raise ValueError("reference_audio_requires_mono_or_stereo")
            media.update(audio_codec=audios[0]["codec_name"], sample_rate=int(audios[0]["sample_rate"]),
                         channels=int(audios[0]["channels"]))
            if not 0 < media["sample_rate"] <= 192000:
                raise ValueError("reference_audio_sample_rate_exceeded")
        subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(path),
                        "-map", "0:v?", "-map", "0:a?", "-f", "null", "-"],
                       check=True, capture_output=True, timeout=120)
        return media
    except (subprocess.SubprocessError, OSError, KeyError, json.JSONDecodeError, ZeroDivisionError) as exc:
        raise ValueError("reference_media_decode_failed") from exc
