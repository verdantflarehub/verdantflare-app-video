"""Verify local media before admission and execute native Ref2VA sampling in the worker."""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

from .contract import IDENTIFIERS, WEIGHTS, plan_swap_windows
from .errors import RuntimeErrorCode
import re


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_mp4_summary(path: Path) -> dict:
    """Probe MP4 streams and exact CFR frame count via ffprobe."""
    try:
        raw = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-f",
                "mov",
                "-show_streams",
                "-of",
                "json",
                str(path),
            ],
            timeout=120,
        )
        data = json.loads(raw)
    except (subprocess.SubprocessError, ValueError, KeyError, OSError) as exc:
        raise RuntimeErrorCode("invalid_source_media") from exc

    videos = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    audios = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    if len(audios) > 1 or (audios and audios[0].get("codec_name") != "aac"):
        raise RuntimeErrorCode("unsupported_source_audio_codec")
    if len(videos) != 1:
        raise RuntimeErrorCode("invalid_source_media")
    stream = videos[0]
    try:
        avg_fps = Fraction(stream["avg_frame_rate"])
        r_fps = Fraction(stream.get("r_frame_rate", stream["avg_frame_rate"]))
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        raise RuntimeErrorCode("invalid_source_media") from exc

    if not 128 <= width <= 3840 or not 128 <= height <= 3840 or width % 2 != 0 or height % 2 != 0:
        raise RuntimeErrorCode("invalid_source_dimensions")
    if stream.get("color_transfer") not in (None, "unknown", "bt709") or stream.get("pix_fmt") not in (
        "yuv420p",
        "yuvj420p",
    ):
        raise RuntimeErrorCode("invalid_source_dimensions")
    if stream.get("sample_aspect_ratio") not in (None, "N/A", "1:1") or any(
        s.get("rotation", 0) for s in stream.get("side_data_list", [])
    ):
        raise RuntimeErrorCode("invalid_source_dimensions")
    if not 12 <= avg_fps <= 120:
        raise RuntimeErrorCode("invalid_source_fps")
    if any(abs(float(s.get("start_time", 0))) > 0.001 for s in [stream, *audios]):
        raise RuntimeErrorCode("invalid_source_media")

    count = 0
    previous: float | None = None
    with tempfile.TemporaryFile() as err_file:
        proc = subprocess.Popen(
            [
                "ffprobe",
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-f",
                "mov",
                "-select_streams",
                "v:0",
                "-show_frames",
                "-show_entries",
                "frame=best_effort_timestamp_time:side_data=",
                "-of",
                "compact=p=0:nk=0",
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=err_file,
            text=True,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                fields = dict(part.split("=", 1) for part in line.strip().split("|") if "=" in part)
                if "best_effort_timestamp_time" not in fields:
                    continue
                ts = float(fields["best_effort_timestamp_time"])
                if (
                    not math.isfinite(ts)
                    or (previous is None and abs(ts) > 0.001)
                    or (previous is not None and abs(ts - previous - float(1 / avg_fps)) > 0.0005)
                ):
                    raise RuntimeErrorCode("invalid_source_media")
                count += 1
                previous = ts
            if proc.wait(timeout=180) != 0 or count <= 0:
                raise RuntimeErrorCode("invalid_source_media")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            if proc.stdout is not None:
                proc.stdout.close()

    audio_codec = audios[0].get("codec_name") if audios else None
    return {
        "width": width,
        "height": height,
        "frames": count,
        "frame_rate": str(avg_fps),
        "r_frame_rate": str(r_fps),
        "duration_seconds": float(count / avg_fps),
        "video_codec": str(stream.get("codec_name", "")),
        "audio": bool(audios),
        "audio_codec": audio_codec,
    }


def probe_audio_packets_sha256(mp4_path: Path) -> str:
    """Compute deterministic SHA-256 over raw demuxed audio stream bytes."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(mp4_path),
        "-map",
        "0:a:0",
        "-c:a",
        "copy",
        "-f",
        "adts",
        "pipe:1",
    ]
    raw = subprocess.check_output(cmd, stderr=subprocess.PIPE, timeout=180)
    return hashlib.sha256(raw).hexdigest()


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


def render_side_by_side_preview(
    source_mp4: Path,
    output_mp4: Path,
    preview_mp4: Path,
    target_w: int,
    target_h: int,
    fps: str = "24/1",
    timeout: int = 300,
) -> None:
    """Render side-by-side comparison preview bounded by rational preview_geometry(target_w, target_h)."""
    pw, ph = preview_geometry(target_w, target_h)
    panel_w = pw // 2
    filter_graph = (
        f"[0:v]scale={panel_w}:{ph}:flags=bicubic,fps={fps},setsar=1[left];"
        f"[1:v]scale={panel_w}:{ph}:flags=bicubic,fps={fps},setsar=1[right];"
        f"[left][right]hstack=inputs=2:shortest=1[v]"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source_mp4),
            "-i",
            str(output_mp4),
            "-filter_complex_threads",
            "1",
            "-filter_complex",
            filter_graph,
            "-map",
            "[v]",
            "-map",
            "1:a:0?",
            "-c:v",
            "libx264",
            "-threads",
            "4",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(preview_mp4),
        ],
        check=True,
        capture_output=True,
        timeout=timeout,
    )


def _resolve_artifact(request, root, role, kind):
    field = role + "_artifact_id"
    ident = request[field]
    if not isinstance(ident, str) or not re.fullmatch(IDENTIFIERS[field], ident):
        raise RuntimeErrorCode("invalid_" + field)
    artifacts = root / "artifacts"
    directory = artifacts / ident
    metadata = directory / "metadata.json"
    if any(p.is_symlink() for p in (root, artifacts, directory, metadata)) or not metadata.is_file():
        raise RuntimeErrorCode(role + "_not_found")
    try:
        data = json.loads(metadata.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        raise RuntimeErrorCode("invalid_" + role + "_media") from None
    if not isinstance(data, dict) or data.get("project_id") != request["project_id"] or data.get("artifact_id") != ident:
        raise RuntimeErrorCode(role + "_not_found")
    media_type = str(data.get("media_type", ""))
    valid_type = media_type.startswith("video/") if kind == "video" else media_type in {"image/png", "image/jpeg", "image/webp"}
    name = data.get("filename")
    if data.get("kind", kind) != kind or not valid_type or not isinstance(name, str) or not name or name in {".", ".."} or Path(name).name != name:
        raise RuntimeErrorCode("invalid_" + role + "_media")
    path = directory / name
    if path.is_symlink() or not path.is_file() or path.resolve().parent != directory.resolve():
        raise RuntimeErrorCode(role + "_not_found")
    expected = request[role + "_sha256"]
    if data.get("size") != path.stat().st_size or data.get("sha256") != expected or sha256_file(path) != expected:
        raise RuntimeErrorCode(role + "_integrity_failed")
    return path


def resolve_source_artifact_path(request, root):
    return _resolve_artifact(request, Path(root), "source", "video")


def resolve_character_image_artifact_path(request, root):
    return _resolve_artifact(request, Path(root), "character_image", "image")


def probe_and_validate_swap_inputs(request, root):
    from PIL import Image
    source = resolve_source_artifact_path(request, root)
    image = resolve_character_image_artifact_path(request, root)
    try:
        with Image.open(image, formats=["JPEG", "PNG", "WEBP"]) as ref:
            ref.verify()
    except (ValueError, SyntaxError, OSError, Image.DecompressionBombError) as exc:
        raise RuntimeErrorCode("invalid_character_image_media") from exc
    summary = probe_mp4_summary(source)
    plan_swap_windows(round(summary["duration_seconds"] * 24))
    return summary


def _canvas(width, height, preset):
    canvases = [(1344, 768), (768, 1344), (1024, 1024), (1152, 864), (864, 1152)]
    if preset == "480p":
        canvases = [(864, 480), (480, 864)]
    return min(canvases, key=lambda shape: abs(math.log(width * shape[1] / (height * shape[0]))))


def _normalize(source, directory, summary):
    if summary["frame_rate"] == summary["r_frame_rate"] == "24":
        return source, summary
    normalized = directory / "source_24fps.mp4"
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(source),
                        "-vf", "fps=24:round=near", "-an", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", str(normalized)], check=True, timeout=240)
        result = probe_mp4_summary(normalized)
    except (subprocess.SubprocessError, RuntimeErrorCode) as exc:
        raise RuntimeErrorCode("fps_normalization_failed") from exc
    plan_swap_windows(result["frames"])
    return normalized, result


def _sample_window(request, video, reference, history, width, height, model_paths, peaks):
    import nodes
    import torch
    import comfy.model_management as mm
    from comfy_extras import nodes_custom_sampler as sampler
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo, MiniMaxH3AddGuide

    def record_peak(phase):
        previous = peaks.get(phase, {})
        peaks[phase] = {
            "allocated_bytes": max(previous.get("allocated_bytes", 0), torch.cuda.max_memory_allocated()),
            "reserved_bytes": max(previous.get("reserved_bytes", 0), torch.cuda.max_memory_reserved()),
        }

    torch.cuda.reset_peak_memory_stats()
    clip = nodes.CLIPLoader().load_clip(model_paths["clip"], type="minimax")[0]
    vae = nodes.VAELoader().load_vae(model_paths["vae"])[0]
    prompt = ("Replace the specified person in <Video 1> with the person in <Picture 1>. "
              "Preserve action, camera, background and timing. " + request["prompt"])
    positive, latent = MiniMaxH3ReferenceToVideo.execute(
        clip, prompt, width, height, len(video), "match", vae, None,
        {"ref_image_0": reference}, {"ref_video_0": video}, {}, {},
    )
    if history is not None:
        # Native H3 accepts clips of 5 + 17*k or single images. Cover each of
        # the 18 timestamps exactly once, without extending into new frames.
        for start, stop in ((0, 5), (5, 10), (10, 15), (15, 16), (16, 17), (17, 18)):
            positive = MiniMaxH3AddGuide.execute(
                positive, latent, start, vae=vae, image=history[start:stop],
            )[0]
    record_peak("reference_encoding")
    del clip, vae
    mm.unload_all_models()
    torch.cuda.reset_peak_memory_stats()
    model = nodes.UNETLoader().load_unet(model_paths["base"], "default")[0]
    loader = nodes.LoraLoaderModelOnly()
    model = loader.load_lora_model_only(model, model_paths["turbo"], 1.0)[0]
    model = loader.load_lora_model_only(model, model_paths["character"], request["lora_strength"])[0]
    noise = sampler.RandomNoise.execute(request["seed"])[0]
    guider = sampler.CFGGuider.execute(model, positive, positive, 1.0)[0]
    algorithm = sampler.KSamplerSelect.execute("er_sde")[0]
    sigmas = sampler.BasicScheduler.execute(model, "normal", 8, 1.0)[0]
    sampled = sampler.SamplerCustomAdvanced.execute(noise, guider, algorithm, sigmas, latent)[0]
    video_samples = sampled["samples"].tensors[0].to("cpu")
    record_peak("sampling")
    del model, guider, positive, sampled, latent
    mm.unload_all_models()
    torch.cuda.reset_peak_memory_stats()
    vae = nodes.VAELoader().load_vae(model_paths["vae"])[0]
    decoded = nodes.VAEDecode().decode(vae, {"samples": video_samples})[0]
    decoded = decoded.detach().to("cpu", dtype=torch.float32)
    record_peak("video_decode")
    del vae
    mm.unload_all_models()
    return decoded


def run_character_swap(request, source, directory, set_stage, *, gpu_uuid, free_vram_mib_before):
    import folder_paths
    import numpy as np
    from PIL import Image
    import torch

    started = time.monotonic()
    root = Path(os.environ.get("CHARACTER_SWAP_MODEL_ROOT", "/models/MiniMax-H3"))
    inputs = Path(os.environ.get("CHARACTER_SWAP_ARTIFACT_ROOT", "/data/projects/video"))
    summary = probe_and_validate_swap_inputs(request, inputs)
    normalized, media = _normalize(source, directory, summary)
    windows = plan_swap_windows(media["frames"])
    width, height = _canvas(media["width"], media["height"], request["resolution_preset"])
    paths = [root / relative for relative, _, _ in WEIGHTS]
    for path, (_, size, sha) in zip(paths, WEIGHTS):
        if path.stat().st_size != size or sha256_file(path) != sha:
            raise RuntimeErrorCode("model_integrity_failed")
    for kind, index in (("diffusion_models", 0), ("text_encoders", 1), ("vae", 2), ("loras", 4)):
        folder_paths.add_model_folder_path(kind, str(paths[index].parent), is_default=True)
    models = dict(base=paths[0].name, clip=paths[1].name, vae=paths[2].name,
                  character=paths[4 if request["character_lora_version"] == "v1.5" else 5].name,
                  turbo=paths[6 if request["turbo_variant"] == "fl2v_turbo_8step" else 7].name)
    with Image.open(resolve_character_image_artifact_path(request, inputs)) as im:
        reference = torch.from_numpy(np.array(im.convert("RGB"))).float().div_(255).unsqueeze(0)
    raw = directory / "generated.rgb"
    history = None
    peaks = {}
    with raw.open("xb") as stream, torch.inference_mode():
        for index, win in enumerate(windows):
            set_stage(f"sampling_window_{index + 1}")
            vf = (f"trim=start_frame={win.start}:end_frame={win.end},"
                  f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                  f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2")
            pixels = subprocess.check_output(["ffmpeg", "-v", "error", "-nostdin", "-i",
                str(normalized), "-vf", vf, "-an", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], timeout=180)
            frames = np.frombuffer(pixels, np.uint8).reshape(-1, height, width, 3)
            length = win.end - win.start
            if len(frames) != length:
                raise RuntimeErrorCode("source_decode_failed")
            batch = np.pad(frames, ((0, win.bucket-length), (0,0), (0,0), (0,0)), mode="edge")
            decoded = _sample_window(request, torch.from_numpy(batch).float().div_(255), reference,
                                     history, width, height, models, peaks)
            for frame in win.trim(decoded):
                stream.write(frame.clamp(0, 1).mul(255).round().byte().numpy().tobytes())
            history = decoded[length-18:length].clone()
            del batch, pixels, frames, decoded
    output = directory / "output.mp4"
    args = ["ffmpeg", "-v", "error", "-nostdin", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", "24", "-i", str(raw), "-i", str(source),
            "-map", "0:v", "-map", "1:a:0?", "-c:v", "libx264", "-crf", "16",
            "-pix_fmt", "yuv420p", "-c:a", "copy", str(output)]
    try:
        subprocess.run(args, check=True, timeout=300)
    finally:
        raw.unlink(missing_ok=True)
    result = probe_mp4_summary(output)
    if (result["frames"], result["audio"]) != (media["frames"], summary["audio"]):
        raise RuntimeErrorCode("output_media_mismatch")
    audio_hash = probe_audio_packets_sha256(source) if summary["audio"] else None
    output_audio_hash = probe_audio_packets_sha256(output) if result["audio"] else None
    if output_audio_hash != audio_hash:
        raise RuntimeErrorCode("audio_stream_copy_mismatch")
    preview = directory / "preview.mp4"
    render_side_by_side_preview(normalized, output, preview, width, height)
    preview_media = probe_mp4_summary(preview)
    if preview_media["frames"] != result["frames"]:
        raise RuntimeErrorCode("preview_media_mismatch")
    metrics = dict(seed=request["seed"], seed_mode=request["seed_mode"],
        window_seeds=[request["seed"] for _ in windows], window_buckets=[w.bucket for w in windows],
        gpu_uuid=gpu_uuid, free_vram_mib_before=free_vram_mib_before,
        gpu_peak_allocated_bytes=max(p["allocated_bytes"] for p in peaks.values()),
        gpu_peak_reserved_bytes=max(p["reserved_bytes"] for p in peaks.values()),
        stage_vram_peaks=peaks, elapsed_seconds=time.monotonic()-started,
        audio_mode="copy" if audio_hash else "none", audio_preserved=True, source_audio_sha256=audio_hash,
        output_audio_sha256=output_audio_hash, source_original_fps=summary["frame_rate"],
        fps_normalized_to_24=normalized != source, source_frames=media["frames"], output_frames=result["frames"])
    if not re.search(r"\b(?:woman|man|girl|boy)\b|[男女]", request["prompt"], re.I):
        metrics["prompt_warning"] = "generic_subject_term_used: prefer explicit woman/man over generic character"
    return dict(input_media=summary, media=result, preview_media=preview_media, runtime_metrics=metrics,
                windows=[vars(w) for w in windows], model={"backend": "h3_character_swap_lora",
                    "weights_sha256": {Path(p).name: sha for p, _, sha in WEIGHTS}}, parameters=request, input_sha256=request["source_sha256"],
                output_sha256=sha256_file(output), preview_sha256=sha256_file(preview))
