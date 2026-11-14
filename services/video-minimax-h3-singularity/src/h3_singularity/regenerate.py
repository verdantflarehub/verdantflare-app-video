"""Open-H3-Regenerate-2K-IC read-only source validation and single-window disk-streaming pipeline."""

from __future__ import annotations

from fractions import Fraction
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np

from .contract import (
    LOCKED_P2_CODE_REVISION,
    LOCKED_P2_WEIGHTS_SHA256,
    is_valid_h3_frame_bucket,
    preview_geometry,
    resolve_2k_target_geometry,
    temporal_windows,
)
from .errors import RuntimeErrorCode


IC_TASK_PROMPT = (
    "Enhance <Video 1> into a high-quality version of the same video. "
    "Preserve the scene, objects, camera motion, timing, and composition of every frame. "
    "Restore fine details and remove degradation. Do not continue or change the scene."
)
UP3_SIGMAS = [0.834942, 0.696852, 0.468753, 0.0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_vm_hwm_bytes() -> int:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def resolve_source_artifact_path(request: dict, artifact_root: Path) -> Path:
    """Resolve and verify read-only Artifact MP4 under /source/video (zero HTTP download)."""
    root = Path(artifact_root)
    artifacts_dir = root / "artifacts"
    if root.is_symlink() or artifacts_dir.is_symlink():
        raise RuntimeErrorCode("source_not_found")
    directory = artifacts_dir / request["source_artifact_id"]
    metadata_path = directory / "metadata.json"
    if directory.is_symlink() or metadata_path.is_symlink() or not metadata_path.is_file():
        raise RuntimeErrorCode("source_not_found")
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("metadata must be object")
    except (ValueError, OSError):
        raise RuntimeErrorCode("invalid_source_media") from None

    if data.get("project_id") != request["project_id"] or data.get("artifact_id") != request["source_artifact_id"]:
        raise RuntimeErrorCode("source_not_found")
    if data.get("kind", "video") != "video" or not str(data.get("media_type", "")).startswith("video/"):
        raise RuntimeErrorCode("invalid_source_media")
    filename = data.get("filename", "")
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or filename in {".", ".."}
    ):
        raise RuntimeErrorCode("invalid_source_media")
    path = directory / filename
    if path.is_symlink() or not path.is_file() or path.resolve().parent != directory.resolve():
        raise RuntimeErrorCode("source_not_found")
    if (
        path.stat().st_size != data.get("size")
        or data.get("sha256") != request["source_sha256"]
        or sha256_file(path) != request["source_sha256"]
    ):
        raise RuntimeErrorCode("source_integrity_failed")
    return path


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
    if len(videos) != 1 or len(audios) > 1:
        raise RuntimeErrorCode("invalid_source_media")
    stream = videos[0]
    try:
        avg_fps = Fraction(stream["avg_frame_rate"])
        r_fps = Fraction(stream.get("r_frame_rate", stream["avg_frame_rate"]))
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        raise RuntimeErrorCode("invalid_source_media") from exc

    if width <= 0 or height <= 0 or width % 2 != 0 or height % 2 != 0 or avg_fps <= 0:
        raise RuntimeErrorCode("invalid_source_media")
    if stream.get("color_transfer") not in (None, "unknown", "bt709") or stream.get("pix_fmt") not in (
        "yuv420p",
        "yuvj420p",
    ):
        raise RuntimeErrorCode("invalid_source_media")
    if stream.get("sample_aspect_ratio") not in (None, "N/A", "1:1") or any(
        s.get("rotation", 0) for s in stream.get("side_data_list", [])
    ):
        raise RuntimeErrorCode("invalid_source_media")
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


def probe_and_validate_source_video(source_mp4: Path, grid_mode: str) -> tuple[dict, dict, list[tuple[int, int]]]:
    """Enforce 2K IC source admission rules: 32-aligned, 24/1 FPS, 17k+5 bucket, AAC-or-silent, not already 2K."""
    probed = probe_mp4_summary(source_mp4)
    width = probed["width"]
    height = probed["height"]
    if width < 32 or height < 32 or width % 32 != 0 or height % 32 != 0:
        raise RuntimeErrorCode("invalid_source_media")
    if Fraction(probed["frame_rate"]) != Fraction(24, 1) or Fraction(probed["r_frame_rate"]) != Fraction(24, 1):
        raise RuntimeErrorCode("invalid_source_media")
    if not is_valid_h3_frame_bucket(probed["frames"]):
        raise RuntimeErrorCode("invalid_source_media")
    if probed["audio"] and probed["audio_codec"] != "aac":
        raise RuntimeErrorCode("invalid_source_media")

    geometry = resolve_2k_target_geometry(width, height, grid_mode)
    windows = temporal_windows(probed["frames"])
    input_media = {
        "width": width,
        "height": height,
        "frames": probed["frames"],
        "frame_rate": "24/1",
        "duration_seconds": probed["duration_seconds"],
        "video_codec": probed["video_codec"],
        "audio": probed["audio"],
    }
    return input_media, geometry, windows


def plan_two_tile_spatial_grid(lat_h: int, lat_w: int) -> dict:
    """Plan 2-tile spatial split along longest axis with multiples-of-4 tile size & even second-tile offset."""
    if lat_w >= lat_h:
        tc = int(math.ceil(((lat_w + 8) / 2.0) / 4.0) * 4)
        c1 = lat_w - tc
        ovw = 2 * tc - lat_w
        if c1 <= 0 or c1 % 2 != 0 or ovw < 8:
            raise RuntimeErrorCode("invalid_source_media")
        return {
            "axis": "width",
            "rows": [0],
            "cols": [0, c1],
            "trows": [lat_h],
            "tcols": [tc, tc],
            "row_ovl": [0],
            "col_ovl": [0, ovw],
            "fh": 0,
            "fw": ovw // 2,
            "seam_cap": 0.75,
        }
    tr = int(math.ceil(((lat_h + 8) / 2.0) / 4.0) * 4)
    r1 = lat_h - tr
    ovh = 2 * tr - lat_h
    if r1 <= 0 or r1 % 2 != 0 or ovh < 8:
        raise RuntimeErrorCode("invalid_source_media")
    return {
        "axis": "height",
        "rows": [0, r1],
        "cols": [0],
        "trows": [tr, tr],
        "tcols": [lat_w],
        "row_ovl": [0, ovh],
        "col_ovl": [0],
        "fh": ovh // 2,
        "fw": 0,
        "seam_cap": 0.75,
    }


def stitch_window_frames(window_frames_iter, windows: list[tuple[int, int]]):
    """Stream decoded uint8 window arrays [W_len, H, W, 3] with full-overlap linear crossfade deduplication.

    Frees each full window array BEFORE requesting the next window and blends overlap 1 frame at a time
    so peak float32 allocation is <= 2 single frames and host VmHWM stays strictly < 28 GiB.
    """
    tail: np.ndarray | None = None
    num_windows = len(windows)
    it = iter(window_frames_iter)
    for idx in range(num_windows):
        try:
            raw_win = next(it)
        except StopIteration:
            raise RuntimeErrorCode("decoded_window_frame_mismatch") from None
        win_u8 = raw_win.pop() if isinstance(raw_win, list) else raw_win
        del raw_win

        ws, we = windows[idx]
        expected_len = we - ws
        if win_u8.shape[0] != expected_len:
            del win_u8
            raise RuntimeErrorCode("decoded_window_frame_mismatch")

        if idx == 0:
            if num_windows == 1:
                for f_idx in range(expected_len):
                    yield win_u8[f_idx].copy()
                del win_u8
            else:
                next_ws = windows[1][0]
                overlap_next = we - next_ws
                body_end = expected_len - overlap_next
                for f_idx in range(body_end):
                    yield win_u8[f_idx].copy()
                tail = win_u8[body_end:].copy()
                del win_u8
        else:
            assert tail is not None
            overlap_prev = tail.shape[0]
            inv_overlap = np.float32(1.0 / float(overlap_prev))
            for f_idx in range(overlap_prev):
                a = np.float32(f_idx + 0.5) * inv_overlap
                blended_f32 = (np.float32(1.0) - a) * tail[f_idx].astype(np.float32) + a * win_u8[f_idx].astype(
                    np.float32
                )
                frame_out = np.clip(np.rint(blended_f32), 0.0, 255.0).astype(np.uint8)
                del blended_f32
                yield frame_out
                del frame_out
            del tail
            tail = None

            if idx < num_windows - 1:
                next_ws = windows[idx + 1][0]
                overlap_next = we - next_ws
                body_end = expected_len - overlap_next
                for f_idx in range(overlap_prev, body_end):
                    yield win_u8[f_idx].copy()
                tail = win_u8[body_end:].copy()
                del win_u8
            else:
                for f_idx in range(overlap_prev, expected_len):
                    yield win_u8[f_idx].copy()
                del win_u8
    # Drain iterator end to verify no extra unexpected windows exist.
    try:
        extra = next(it)
        del extra
        raise RuntimeErrorCode("decoded_window_frame_mismatch")
    except StopIteration:
        pass


def decode_rgb24_window(video_path: Path, width: int, height: int, start_frame: int, end_frame: int) -> np.ndarray:
    """Decode only [start_frame, end_frame) RGB24 frames of source MP4."""
    expected = end_frame - start_frame
    frame_bytes = width * height * 3
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(video_path),
        "-vf",
        f"select='between(n\\,{start_frame}\\,{end_frame - 1})',setpts=N/24/TB",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frames: list[np.ndarray] = []
    try:
        assert proc.stdout is not None
        while True:
            buf = b""
            while len(buf) < frame_bytes:
                chunk = proc.stdout.read(frame_bytes - len(buf))
                if not chunk:
                    break
                buf += chunk
            if not buf:
                break
            if len(buf) != frame_bytes:
                raise RuntimeErrorCode("source_decode_failed")
            frames.append(np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3)).copy())
        if proc.wait(timeout=300) != 0 or len(frames) != expected:
            raise RuntimeErrorCode("source_decode_failed")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()
    return np.stack(frames, axis=0)


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


def run_regenerate_2k(
    request: dict,
    source_mp4: Path,
    task_dir: Path,
    set_stage,
    *,
    gpu_uuid: str = "",
    free_vram_mib_before: int = 0,
) -> dict:
    """Execute P2a..P2d single-window disk-streaming Open-H3-Regenerate-2K-IC inference."""
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file

    import comfy.lora
    import comfy.model_management as model_management
    import comfy.nested_tensor
    import comfy.sample
    import comfy.samplers
    import comfy.utils
    import comfy_extras.nodes_custom_sampler as custom_sampler
    import comfy_extras.nodes_minimax_h3 as h3_nodes
    import folder_paths
    import nodes

    from .latent_resizer_3d import load_resizer_3d
    from .lowvram import make_h3_lowvram_attention

    models_root = Path(os.environ.get("H3_BASE_MODEL_ROOT", "/models/MiniMax-H3"))
    folder_paths.add_model_folder_path(
        "diffusion_models",
        str(models_root / "candidates" / "a2_vsa_turbo_v4" / "diffusion_models"),
    )
    folder_paths.add_model_folder_path(
        "loras",
        str(models_root / "candidates" / "p2_open_h3_2k_ic" / "comfyui"),
    )
    folder_paths.add_model_folder_path(
        "loras",
        str(models_root / "candidates" / "p2_open_h3_2k_ic" / "official" / "lora"),
    )
    folder_paths.add_model_folder_path(
        "text_encoders",
        str(models_root / "candidates" / "shared" / "text_encoders"),
    )
    folder_paths.add_model_folder_path(
        "vae",
        str(models_root / "candidates" / "shared" / "vae"),
    )

    input_media, geometry, windows = probe_and_validate_source_video(source_mp4, request["grid_mode"])
    src_w, src_h = geometry["source_w"], geometry["source_h"]
    target_w, target_h = geometry["target_w"], geometry["target_h"]
    levels = geometry["levels"]
    lat_h, lat_w = target_h // 16, target_w // 16
    tile_grid = plan_two_tile_spatial_grid(lat_h, lat_w)

    win_dir = task_dir / "_windows"
    win_dir.mkdir(parents=True, exist_ok=True)
    peak_reserved_bytes = 0

    def hard_unload(clip_obj=None) -> None:
        nonlocal peak_reserved_bytes
        if torch.cuda.is_available():
            peak_reserved_bytes = max(peak_reserved_bytes, int(torch.cuda.max_memory_reserved()))
        if clip_obj is not None:
            cpu_dev = torch.device("cpu")
            base = getattr(clip_obj, "_base", clip_obj)
            patcher = getattr(base, "patcher", None)
            is_dyn = bool(patcher is not None and getattr(patcher, "is_dynamic", lambda: False)())
            if not is_dyn and patcher is not None:
                patcher.offload_device = cpu_dev
                m = getattr(patcher, "model", None)
                if m is not None and hasattr(m, "to"):
                    m.to(device=cpu_dev)
        model_management.unload_all_models()
        gc.collect()
        model_management.soft_empty_cache(force=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def move_tree_to_cpu(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().to("cpu")
        if getattr(obj, "is_nested", False) and hasattr(obj, "tensors"):
            return comfy.nested_tensor.NestedTensor(tuple(move_tree_to_cpu(t) for t in obj.tensors))
        if isinstance(obj, dict):
            return {k: move_tree_to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [move_tree_to_cpu(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(move_tree_to_cpu(v) for v in obj)
        return obj

    def spatial_fade_mask(tile_h, tile_w, ovh, ovw, done_top, done_left, fade_h=0, fade_w=0, seam_cap=0.75):
        mask = torch.ones(tile_h, tile_w, dtype=torch.float32)

        def profile(n, ov, fade):
            p = torch.ones(n, dtype=torch.float32)
            f = min(fade, ov)
            frozen = ov - f
            p[:frozen] = 0.0
            if f > 0:
                p[frozen:ov] = torch.linspace(0.0, seam_cap, f)
            ramp = min(ov, n - ov)
            if ramp > 0 and seam_cap < 1.0:
                start = seam_cap if f > 0 else 0.0
                p[ov : ov + ramp] = torch.linspace(start, 1.0, ramp)
            return p

        if done_left and ovw > 0:
            mask = torch.minimum(mask, profile(tile_w, ovw, fade_w)[None, :])
        if done_top and ovh > 0:
            mask = torch.minimum(mask, profile(tile_h, ovh, fade_h)[:, None])
        return mask

    def dc_correct(new_t, refs, clamp=0.05, min_samples=256):
        pairs = [p for p in refs if p is not None and p[0] is not None and p[0].numel() >= min_samples]
        if not pairs:
            return new_t
        pa = torch.cat([a.float().permute(0, 2, 3, 4, 1).reshape(-1, a.shape[1]) for a, _ in pairs])
        pb = torch.cat([b.float().permute(0, 2, 3, 4, 1).reshape(-1, b.shape[1]) for _, b in pairs])
        dc = (pa - pb).median(dim=0).values.clamp(-clamp, clamp)
        return new_t - dc.view(1, -1, 1, 1, 1).to(new_t.device, new_t.dtype)

    def grade_pin(chunk, ref, clamp=0.05):
        a = chunk.float().permute(0, 2, 3, 4, 1).reshape(-1, chunk.shape[1])
        b = ref.float().permute(0, 2, 3, 4, 1).reshape(-1, ref.shape[1])
        dc = (a - b).median(dim=0).values.clamp(-clamp, clamp)
        return chunk - dc.view(1, -1, 1, 1, 1).to(chunk.device, chunk.dtype)

    try:
        # 1. Stage P2a: Single-window decode & Qwen3-VL / VAE conditioning serialized to disk
        set_stage("encoding_condition")
        p2a_t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

        clip = nodes.CLIPLoader().load_clip("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", type="minimax")[0]
        vae = nodes.VAELoader().load_vae("minimax_h3_video_vae_fp16.safetensors")[0]
        vae.device = torch.device("cuda:0")
        vae.patcher.load_device = torch.device("cuda:0")
        vae.output_device = torch.device("cpu")
        video_vae_inner = getattr(vae, "first_stage_model", None)
        if video_vae_inner is not None and getattr(video_vae_inner, "comfy_has_chunked_io", False):
            video_vae_inner.tile_size = 256
            video_vae_inner.tile_overlap_min = 64

            def encode_temporal_cpu(x, device):
                chunks = []
                for idx in range(math.ceil(x.shape[2] / video_vae_inner.clip_length)):
                    clip_x = x[:, :, idx * video_vae_inner.clip_length : (idx + 1) * video_vae_inner.clip_length].to(
                        device
                    )
                    if clip_x.shape[2] < video_vae_inner.clip_length:
                        pad_f = clip_x[:, :, -1:].repeat(
                            1, 1, video_vae_inner.clip_length - clip_x.shape[2], 1, 1
                        )
                        clip_x = torch.cat([clip_x, pad_f], dim=2)
                    chunks.append(video_vae_inner._adaptive_encode(video_vae_inner._normalize_pixels(clip_x)).to("cpu"))
                    del clip_x
                res = torch.cat(chunks, dim=2)
                if video_vae_inner.token_drop > 0:
                    res = res[:, :, : -video_vae_inner.token_drop]
                return res

            video_vae_inner.encode_temporal = encode_temporal_cpu

        scene_desc = request["prompt"].strip()
        ic_prompt = f"{IC_TASK_PROMPT}\n\nScene description: {scene_desc}" if scene_desc else IC_TASK_PROMPT
        rows, cols = tile_grid["rows"], tile_grid["cols"]
        trows, tcols = tile_grid["trows"], tile_grid["tcols"]

        with torch.inference_mode():
            for w_idx, (ws, we) in enumerate(windows):
                win_len = we - ws
                win_rgb_u8 = decode_rgb24_window(source_mp4, src_w, src_h, ws, we)
                win_f32 = torch.from_numpy(win_rgb_u8).float().div_(255.0)
                del win_rgb_u8
                z_draft = vae.encode(win_f32).detach().to("cpu", dtype=torch.float32).contiguous()

                half_h, half_w = target_h // 2, target_w // 2
                if (src_h, src_w) == (half_h, half_w):
                    half_frames = win_f32
                else:
                    half_frames = (
                        F.interpolate(
                            win_f32.permute(0, 3, 1, 2),
                            size=(half_h, half_w),
                            mode="bilinear",
                            align_corners=False,
                        )
                        .permute(0, 2, 3, 1)
                        .clamp_(0.0, 1.0)
                        .contiguous()
                    )
                del win_f32

                win_tile_conds: list[tuple[object, object]] = []
                for ri, r0 in enumerate(rows):
                    for cj, c0 in enumerate(cols):
                        tr, tc = trows[ri], tcols[cj]
                        ry0, ry1 = (r0 // 2) * 16, ((r0 + tr) // 2) * 16
                        cx0, cx1 = (c0 // 2) * 16, ((c0 + tc) // 2) * 16
                        ref_tile = half_frames[:, ry0:ry1, cx0:cx1, :].contiguous()
                        cond_out, ref_lat_out = h3_nodes.MiniMaxH3ReferenceToVideo.execute(
                            clip,
                            ic_prompt,
                            tc * 16,
                            tr * 16,
                            win_len,
                            "match",
                            vae,
                            None,
                            {},
                            {"ref_video_0": ref_tile},
                            {},
                            {},
                        )
                        win_tile_conds.append((move_tree_to_cpu(cond_out), move_tree_to_cpu(ref_lat_out)))
                        del ref_tile, cond_out, ref_lat_out
                del half_frames
                torch.save(z_draft, win_dir / f"win_{w_idx}_draft.pt")
                torch.save(win_tile_conds, win_dir / f"win_{w_idx}_cond.pt")
                del z_draft, win_tile_conds
                gc.collect()

        hard_unload(clip)
        del clip, vae, video_vae_inner
        hard_unload(None)
        p2a_sec = time.perf_counter() - p2a_t0

        # 2. Stage P2b: Single-window 3D Conv latent upscaling
        set_stage("latent_upscaling")
        p2b_t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

        upscaler_ckpt = (
            models_root
            / "candidates"
            / "p2_open_h3_2k_ic"
            / "official"
            / "latent_upsampler"
            / "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"
        )
        resizer, norm_mean, norm_std = load_resizer_3d(upscaler_ckpt, device="cuda:0", dtype=torch.bfloat16)

        with torch.inference_mode():
            for w_idx in range(len(windows)):
                draft_path = win_dir / f"win_{w_idx}_draft.pt"
                cur = torch.load(draft_path, map_location="cpu", weights_only=False)
                draft_path.unlink(missing_ok=True)
                for pass_w, pass_h in levels:
                    eff_scale = (pass_w / (cur.shape[-1] * 16.0) + pass_h / (cur.shape[-2] * 16.0)) / 2.0
                    s_norm = (cur.to("cuda:0", dtype=torch.bfloat16) - norm_mean) / norm_std
                    out_norm = resizer(
                        s_norm,
                        scale=eff_scale,
                        target_size=(cur.shape[2], pass_h // 16, pass_w // 16),
                        enable_chunking=True,
                    )
                    cur = (out_norm * norm_std + norm_mean).float().to("cpu").contiguous()
                    del s_norm, out_norm
                if not torch.isfinite(cur).all():
                    raise RuntimeErrorCode("nonfinite_upscaled_latent")
                torch.save(cur, win_dir / f"win_{w_idx}_up.pt")
                del cur
                gc.collect()

        del resizer, norm_mean, norm_std
        hard_unload(None)
        p2b_sec = time.perf_counter() - p2b_t0

        # 3. Stage P2c: Single-window up3 3-step 2-tile DiT refinement
        set_stage("refining_2k")
        p2c_t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

        dit_model = nodes.UNETLoader().load_unet("minimax_h3_ref2va_pruned_int8_convrot.safetensors", "default")[0]
        sd_keys = list(dit_model.model.state_dict().keys())
        lora_key_map: dict[str, str] = {}
        for k in sd_keys:
            if k.endswith(".weight"):
                stem = k[: -len(".weight")]
                lora_key_map[stem] = k
                if stem.startswith("diffusion_model."):
                    lora_key_map[stem[len("diffusion_model.") :]] = k

        lora_specs = [
            (
                models_root
                / "candidates"
                / "p2_open_h3_2k_ic"
                / "comfyui"
                / "minimax_h3_hyperflow_8step_v1.0_comfyui_bf16.safetensors",
                208,
            ),
            (
                models_root
                / "candidates"
                / "p2_open_h3_2k_ic"
                / "official"
                / "lora"
                / "minimax_h3_lms_v1.0_r64.safetensors",
                258,
            ),
            (
                models_root
                / "candidates"
                / "p2_open_h3_2k_ic"
                / "comfyui"
                / "minimax_h3_open_regenerate_2k_ic_comfy_bf16.safetensors",
                208,
            ),
        ]
        for lpath, min_expected in lora_specs:
            l_sd = comfy.utils.load_torch_file(str(lpath), safe_load=True)
            loaded = comfy.lora.load_lora(l_sd, lora_key_map, log_missing=False)
            patched_keys = dit_model.add_patches(loaded, 1.0)
            del l_sd, loaded
            if len(patched_keys) < min_expected:
                raise RuntimeErrorCode("lora_patch_incomplete")

        dit_model = h3_nodes.MiniMaxH3SigmaShift.execute(dit_model, 12.0, 3.0)[0]
        attn_metrics: dict[str, object] = {}
        dit_model.set_model_optimized_attention(make_h3_lowvram_attention(attn_metrics))

        silence_path = (
            models_root / "candidates" / "p2_open_h3_2k_ic" / "official" / "assets" / "silence_audio_124f.safetensors"
        )
        silence_raw = load_file(str(silence_path), device="cpu")["audio_latents"]
        silence_full = silence_raw.permute(1, 0, 2).unsqueeze(0).float().contiguous()

        sampler = custom_sampler.KSamplerSelect.execute("euler")[0]
        sigmas = torch.tensor(UP3_SIGMAS, dtype=torch.float32)
        row_ovl, col_ovl = tile_grid["row_ovl"], tile_grid["col_ovl"]
        fh, fw, seam_cap = tile_grid["fh"], tile_grid["fw"], tile_grid["seam_cap"]
        seed = int(request["seed"])

        with torch.inference_mode():
            for w_idx in range(len(windows)):
                up_path = win_dir / f"win_{w_idx}_up.pt"
                cond_path = win_dir / f"win_{w_idx}_cond.pt"
                z_up = torch.load(up_path, map_location="cpu", weights_only=False)
                win_conds = torch.load(cond_path, map_location="cpu", weights_only=False)
                up_path.unlink(missing_ok=True)
                cond_path.unlink(missing_ok=True)

                first_ref_samples = win_conds[0][1]["samples"]
                audio_ticks = (
                    int(first_ref_samples.tensors[1].shape[-1])
                    if getattr(first_ref_samples, "is_nested", False)
                    else int(silence_full.shape[-1])
                )
                silence_lat = silence_full[..., :audio_ticks].contiguous()

                win_seed = int(seed + w_idx)
                gen = torch.Generator("cpu").manual_seed(win_seed)
                noise_v = torch.randn(z_up.shape, generator=gen, dtype=torch.float32)
                noise_a = torch.randn(silence_lat.shape, generator=gen, dtype=torch.float32)

                chunk_out = z_up.clone()
                flat_tile_idx = 0
                for ri, r0 in enumerate(rows):
                    for cj, c0 in enumerate(cols):
                        tr, tc = trows[ri], tcols[cj]
                        ovh, ovw = row_ovl[ri], col_ovl[cj]
                        cond_t, ref_lat_t = win_conds[flat_tile_idx]
                        tile_in = chunk_out[:, :, :, r0 : r0 + tr, c0 : c0 + tc].clone()
                        m = spatial_fade_mask(
                            tr,
                            tc,
                            ovh,
                            ovw,
                            done_top=(ri > 0),
                            done_left=(cj > 0),
                            fade_h=fh,
                            fade_w=fw,
                            seam_cap=seam_cap,
                        )
                        mv = m[None, None, None].to(tile_in.device)
                        ma = torch.zeros_like(silence_lat, dtype=torch.float32)

                        piece = dict(ref_lat_t)
                        piece["samples"] = comfy.nested_tensor.NestedTensor((tile_in, silence_lat.clone()))
                        piece["noise_mask"] = comfy.nested_tensor.NestedTensor((mv, ma))
                        tile_noise = comfy.nested_tensor.NestedTensor(
                            (
                                noise_v[:, :, :, r0 : r0 + tr, c0 : c0 + tc].contiguous(),
                                noise_a.contiguous(),
                            )
                        )
                        guider = comfy.samplers.CFGGuider(dit_model)
                        guider.inner_set_conds({"positive": cond_t})
                        latent_img = comfy.sample.fix_empty_latent_channels(
                            model=guider.model_patcher.model,
                            latent_image=piece["samples"],
                        )
                        out_nt = guider.sample(
                            tile_noise,
                            latent_img,
                            sampler,
                            sigmas,
                            denoise_mask=piece["noise_mask"],
                            disable_pbar=True,
                            seed=win_seed,
                        )
                        tile_v = (
                            (out_nt.tensors[0] if out_nt.is_nested else out_nt)
                            .detach()
                            .to("cpu", dtype=torch.float32)
                        )
                        del out_nt, guider, latent_img, piece, tile_noise

                        region = chunk_out[:, :, :, r0 : r0 + tr, c0 : c0 + tc].clone()
                        tile_v = dc_correct(
                            tile_v,
                            [
                                (tile_v[:, :, :, :, :ovw], region[:, :, :, :, :ovw]) if (cj > 0 and ovw > 0) else None,
                                (tile_v[:, :, :, :ovh, :], region[:, :, :, :ovh, :]) if (ri > 0 and ovh > 0) else None,
                                (tile_v, z_up[:, :, :, r0 : r0 + tr, c0 : c0 + tc]) if (ri == 0 and cj == 0) else None,
                            ],
                        )
                        if cj > 0 and ovw > 0:
                            wts = torch.linspace(0.0, 1.0, ovw, dtype=torch.float32).view(1, 1, 1, 1, ovw)
                            region[:, :, :, :, :ovw] = (
                                region[:, :, :, :, :ovw] * (1.0 - wts) + tile_v[:, :, :, :, :ovw] * wts
                            )
                        if ri > 0 and ovh > 0:
                            wts = torch.linspace(0.0, 1.0, ovh, dtype=torch.float32).view(1, 1, 1, ovh, 1)
                            region[:, :, :, :ovh, :] = (
                                region[:, :, :, :ovh, :] * (1.0 - wts) + tile_v[:, :, :, :ovh, :] * wts
                            )
                        band = torch.zeros((1, 1, 1, tr, tc), dtype=torch.bool)
                        if cj > 0 and ovw > 0:
                            band[:, :, :, :, :ovw] = True
                        if ri > 0 and ovh > 0:
                            band[:, :, :, :ovh, :] = True
                        region = torch.where(band, region, tile_v)
                        chunk_out[:, :, :, r0 : r0 + tr, c0 : c0 + tc] = region
                        flat_tile_idx += 1
                        model_management.soft_empty_cache()

                chunk_out = grade_pin(chunk_out, z_up)
                if not torch.isfinite(chunk_out).all():
                    raise RuntimeErrorCode("nonfinite_refined_latent")
                torch.save(chunk_out.contiguous(), win_dir / f"win_{w_idx}_refined.pt")
                del chunk_out, z_up, win_conds, noise_v, noise_a
                gc.collect()

        del dit_model, silence_full, silence_raw
        hard_unload(None)
        p2c_sec = time.perf_counter() - p2c_t0

        # 4. Stage P2d: Single-window FP16 Video VAE decode streaming directly into ffmpeg H.264 pipe
        set_stage("decoding_and_muxing")
        p2d_t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

        vae_dec = nodes.VAELoader().load_vae("minimax_h3_video_vae_fp16.safetensors")[0]
        vae_dec.device = torch.device("cuda:0")
        vae_dec.patcher.load_device = torch.device("cuda:0")
        vae_dec.output_device = torch.device("cpu")
        v_inner = getattr(vae_dec, "first_stage_model", None)
        if v_inner is not None and getattr(v_inner, "comfy_has_chunked_io", False):
            v_inner.tile_size = 256
            v_inner.tile_overlap_min = 64
            if hasattr(v_inner, "decoder_tile_size"):
                v_inner.decoder_tile_size = 256
            if hasattr(v_inner, "decoder_tile_overlap_min"):
                v_inner.decoder_tile_overlap_min = 64

        silent_mp4 = task_dir / "silent_2k.mp4"
        output_mp4 = task_dir / "output.mp4"
        preview_mp4 = task_dir / "preview.mp4"

        def iter_decoded_windows():
            with torch.inference_mode():
                for w_idx, (ws, we) in enumerate(windows):
                    expected_len = we - ws
                    ref_path = win_dir / f"win_{w_idx}_refined.pt"
                    z_ref = torch.load(ref_path, map_location="cpu", weights_only=False)
                    ref_path.unlink(missing_ok=True)
                    dec_t = vae_dec.decode(z_ref)
                    del z_ref
                    if dec_t.ndim == 5:
                        if dec_t.shape[-1] == 3:
                            dec_t = dec_t.reshape(-1, dec_t.shape[-3], dec_t.shape[-2], 3)
                        elif dec_t.shape[1] == 3:
                            dec_t = dec_t[0].permute(1, 2, 3, 0)
                    elif dec_t.ndim == 4 and dec_t.shape[-1] != 3 and dec_t.shape[1] == 3:
                        dec_t = dec_t.permute(0, 2, 3, 1)
                    dec_t = dec_t[:expected_len, :target_h, :target_w, :3]
                    u8_slices = []
                    for s0 in range(0, dec_t.shape[0], 16):
                        sl = (
                            dec_t[s0 : s0 + 16]
                            .float()
                            .clamp_(0.0, 1.0)
                            .mul_(255.0)
                            .round_()
                            .to(torch.uint8)
                            .cpu()
                            .numpy()
                        )
                        u8_slices.append(sl)
                    win_u8 = np.concatenate(u8_slices, axis=0)
                    del dec_t, u8_slices
                    gc.collect()
                    box = [win_u8]
                    del win_u8
                    yield box
                    del box
                    gc.collect()

        log_path = silent_mp4.with_suffix(".ffmpeg.log")
        with log_path.open("wb") as log_f:
            proc = subprocess.Popen(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-nostdin",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    f"{target_w}x{target_h}",
                    "-framerate",
                    "24",
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "8",
                    "-preset",
                    "fast",
                    "-crf",
                    "16",
                    "-pix_fmt",
                    "yuv420p",
                    "-color_primaries",
                    "bt709",
                    "-color_trc",
                    "bt709",
                    "-colorspace",
                    "bt709",
                    str(silent_mp4),
                ],
                stdin=subprocess.PIPE,
                stderr=log_f,
            )
            try:
                assert proc.stdin is not None
                frame = None
                for frame in stitch_window_frames(iter_decoded_windows(), windows):
                    proc.stdin.write(np.ascontiguousarray(frame).tobytes())
                    frame = None
                proc.stdin.close()
                if proc.wait(timeout=600) != 0:
                    raise RuntimeErrorCode("video_encode_failed")
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                if proc.stdin is not None and not proc.stdin.closed:
                    proc.stdin.close()
        log_path.unlink(missing_ok=True)

        del vae_dec, v_inner
        hard_unload(None)
    finally:
        shutil.rmtree(win_dir, ignore_errors=True)

    media_timeout = max(300, math.ceil(input_media["duration_seconds"] * 12))
    if input_media["audio"]:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(silent_mp4),
                "-i",
                str(source_mp4),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                str(output_mp4),
            ],
            check=True,
            capture_output=True,
            timeout=media_timeout,
        )
        audio_mode = "copy"
        source_audio_sha256 = probe_audio_packets_sha256(source_mp4)
        output_audio_sha256 = probe_audio_packets_sha256(output_mp4)
        if source_audio_sha256 != output_audio_sha256:
            raise RuntimeErrorCode("audio_stream_copy_mismatch")
    else:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(silent_mp4),
                "-map",
                "0:v:0",
                "-c:v",
                "copy",
                "-movflags",
                "+faststart",
                str(output_mp4),
            ],
            check=True,
            capture_output=True,
            timeout=media_timeout,
        )
        audio_mode = "none"
        output_audio_sha256 = None

    silent_mp4.unlink(missing_ok=True)
    render_side_by_side_preview(
        source_mp4,
        output_mp4,
        preview_mp4,
        target_w,
        target_h,
        fps="24/1",
        timeout=media_timeout,
    )

    out_probe = probe_mp4_summary(output_mp4)
    prev_probe = probe_mp4_summary(preview_mp4)
    expected_pw, expected_ph = preview_geometry(target_w, target_h)
    if (
        (out_probe["width"], out_probe["height"]) != (target_w, target_h)
        or out_probe["frames"] != input_media["frames"]
        or Fraction(out_probe["frame_rate"]) != Fraction(24, 1)
        or out_probe["audio"] != input_media["audio"]
    ):
        raise RuntimeErrorCode("output_media_mismatch")
    if (
        (prev_probe["width"], prev_probe["height"]) != (expected_pw, expected_ph)
        or prev_probe["frames"] != input_media["frames"]
        or Fraction(prev_probe["frame_rate"]) != Fraction(24, 1)
        or prev_probe["audio"] != input_media["audio"]
    ):
        raise RuntimeErrorCode("preview_media_mismatch")

    for candidate_mp4 in (output_mp4, preview_mp4):
        subprocess.run(
            ["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-i", str(candidate_mp4), "-f", "null", "-"],
            check=True,
            capture_output=True,
            timeout=media_timeout,
        )

    p2d_sec = time.perf_counter() - p2d_t0
    vm_hwm_bytes = read_vm_hwm_bytes()

    media_payload = {
        "width": out_probe["width"],
        "height": out_probe["height"],
        "frames": out_probe["frames"],
        "frame_rate": out_probe["frame_rate"],
        "duration_seconds": out_probe["duration_seconds"],
        "video_codec": out_probe["video_codec"],
        "audio": out_probe["audio"],
        "audio_mode": audio_mode,
    }
    if output_audio_sha256 is not None:
        media_payload["audio_sha256"] = output_audio_sha256

    preview_payload = {
        "width": prev_probe["width"],
        "height": prev_probe["height"],
        "frames": prev_probe["frames"],
        "frame_rate": prev_probe["frame_rate"],
        "duration_seconds": prev_probe["duration_seconds"],
        "video_codec": prev_probe["video_codec"],
        "audio": prev_probe["audio"],
    }

    manifest_sha256 = hashlib.sha256(
        json.dumps(LOCKED_P2_WEIGHTS_SHA256, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    prompt_sha256 = hashlib.sha256(request["prompt"].encode("utf-8")).hexdigest()
    runtime_metrics = {
        "p2a_cond_seconds": p2a_sec,
        "p2b_upscale_seconds": p2b_sec,
        "p2c_refine_seconds": p2c_sec,
        "p2d_decode_encode_seconds": p2d_sec,
        "vm_hwm_bytes": vm_hwm_bytes,
        "gpu_peak_reserved_bytes": peak_reserved_bytes,
        "gpu_uuid": gpu_uuid,
        "free_vram_mib_before": free_vram_mib_before,
    }
    return {
        "input_media": input_media,
        "media": media_payload,
        "preview_media": preview_payload,
        "geometry": {
            "source_w": src_w,
            "source_h": src_h,
            "target_w": target_w,
            "target_h": target_h,
            "levels": levels,
        },
        "windows": [[ws, we] for ws, we in windows],
        "parameters": {
            "profile": "open_h3_2k_ic",
            "grid_mode": request["grid_mode"],
            "seed": seed,
            "prompt_sha256": prompt_sha256,
            "audio": audio_mode,
            "video_codec": "h264",
            "pixel_format": "yuv420p",
            "crf": 16,
            "tiles": 2,
        },
        "model": {
            "backend": "open_h3_2k_ic",
            "profile": "open_h3_2k_ic",
            "code_revision": LOCKED_P2_CODE_REVISION,
            "weights_revision": f"sha256:{manifest_sha256}",
            "manifest_sha256": manifest_sha256,
            "weights_sha256": dict(LOCKED_P2_WEIGHTS_SHA256),
        },
        "input_sha256": request["source_sha256"],
        "output_sha256": sha256_file(output_mp4),
        "preview_sha256": sha256_file(preview_mp4),
        "runtime_metrics": runtime_metrics,
    }
