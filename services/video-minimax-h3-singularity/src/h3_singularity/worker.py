"""Isolated single-task GPU worker subprocess invoked by the zero-CUDA M-A Supervisor."""

from __future__ import annotations

import ctypes
import gc
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

from .contract import default_quality_profile, resolve_geometry, resolve_timing
from .errors import RuntimeErrorCode
from .queue import Queue
from .regenerate import (
    read_vm_hwm_bytes,
    resolve_source_artifact_path,
    run_regenerate_2k,
)


GPU_UUID_RE = re.compile(r"GPU-[0-9a-f-]{36}")


def _assigned_gpu_from_environ() -> str | None:
    try:
        entries = [
            entry.split(b"=", 1)[1].decode("ascii", errors="ignore")
            for entry in Path("/proc/self/environ").read_bytes().split(b"\0")
            if entry.startswith(b"NVIDIA_VISIBLE_DEVICES=")
        ]
    except OSError:
        entries = [os.environ.get("NVIDIA_VISIBLE_DEVICES", "")]
    concrete = [v.strip() for v in entries if v.strip() and v.strip() not in {"all", "void", "none"}]
    if len(concrete) == 1 and GPU_UUID_RE.fullmatch(concrete[0]):
        return concrete[0]
    if len(concrete) > 1:
        raise RuntimeErrorCode("gpu_memory_occupancy_changed")
    return None


def check_free_vram(min_free_mib: int | None = None) -> tuple[str, int]:
    """Verify assigned physical GPU UUID and enforce free VRAM gate before loading CUDA weights."""
    required_mib = (
        int(os.environ.get("SINGULARITY_MIN_FREE_VRAM_MIB", "28672"))
        if min_free_mib is None
        else int(min_free_mib)
    )
    expected_uuid = _assigned_gpu_from_environ()
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeErrorCode("gpu_memory_occupancy_changed") from exc

    rows: list[tuple[str, int]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2 or not GPU_UUID_RE.fullmatch(parts[0]):
            raise RuntimeErrorCode("gpu_memory_occupancy_changed")
        try:
            free_val = int(parts[1])
        except ValueError as exc:
            raise RuntimeErrorCode("gpu_memory_occupancy_changed") from exc
        rows.append((parts[0], free_val))

    if len(rows) != 1:
        raise RuntimeErrorCode("gpu_memory_occupancy_changed")
    actual_uuid, free_mib = rows[0]
    if expected_uuid is not None and actual_uuid != expected_uuid:
        raise RuntimeErrorCode(
            "gpu_memory_occupancy_changed",
            metrics={
                "expected_gpu_uuid": expected_uuid,
                "gpu_uuid": actual_uuid,
                "free_vram_mib_before": free_mib,
                "min_free_vram_mib": required_mib,
            },
        )
    if free_mib < required_mib:
        raise RuntimeErrorCode(
            "gpu_memory_occupancy_changed",
            metrics={
                "gpu_uuid": actual_uuid,
                "free_vram_mib_before": free_mib,
                "min_free_vram_mib": required_mib,
            },
        )
    return actual_uuid, free_mib


def _preload_sm120_cuda13_and_comfy() -> dict:
    """Preload CUDA 13 shared libraries and initialize comfy-aimdo + comfy-kitchen BEFORE torch import."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for cuda13_lib in (
        "/usr/local/cuda/lib64/libcudart.so.13",
        "/usr/local/cuda/lib64/libcublas.so.13",
        "/usr/local/cuda/lib64/libcublasLt.so.13",
    ):
        if os.path.exists(cuda13_lib):
            ctypes.CDLL(cuda13_lib, mode=ctypes.RTLD_GLOBAL)

    comfy_root = os.environ.get("COMFY_ROOT", "/opt/comfy")
    if comfy_root not in sys.path:
        sys.path.insert(0, comfy_root)

    aimdo_mod = None
    try:
        import comfy_aimdo.control as _aimdo_ctl

        _aimdo_ctl.init(simple_vram_headroom=None, nvml_pressure=True)
        aimdo_mod = _aimdo_ctl
    except Exception:
        aimdo_mod = None

    import torch  # noqa: F401
    import comfy_kitchen as ck
    import comfy.quant_ops  # noqa: F401
    import comfy.memory_management
    import comfy.model_management as model_management
    import comfy.model_patcher
    import folder_paths  # noqa: F401
    import nodes  # noqa: F401
    import comfy_extras.nodes_minimax_h3  # noqa: F401
    import comfy_extras.nodes_custom_sampler  # noqa: F401
    import comfy_extras.nodes_audio  # noqa: F401

    ck.registry.enable("cuda")
    ck_cuda_active = not ck.list_backends().get("cuda", {}).get("disabled", True)
    aimdo_active = False
    headroom_gib = float(os.environ.get("SINGULARITY_AIMDO_HEADROOM_GIB", "8.0"))
    if aimdo_mod is not None:
        aimdo_ok = aimdo_mod.init_devices(
            (d.index, int(headroom_gib * 1024**3)) for d in model_management.get_all_torch_devices()
        )
        if aimdo_ok:
            comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
            comfy.memory_management.aimdo_enabled = True
            aimdo_active = True

    return {
        "comfy_kitchen_cuda_enabled": ck_cuda_active,
        "dynamic_vram_aimdo_enabled": aimdo_active,
        "aimdo_headroom_gib": headroom_gib,
    }


def _run_fasth3_nvfp4(
    request: dict,
    files: dict[str, list[Path]],
    output_mp4: Path,
    set_stage,
    *,
    gpu_uuid: str,
    free_vram_mib_before: int,
    sm120_info: dict,
) -> dict:
    """Execute P1 primary profile: FastH3 Omni NVFP4 QAD (8 NFE, VSA 0.90, 32-step PDD fix)."""
    import torch

    import comfy.model_management as model_management
    import comfy_extras.nodes_audio as audio_nodes
    import comfy_extras.nodes_custom_sampler as custom_sampler
    import comfy_extras.nodes_minimax_h3 as h3_nodes
    import folder_paths
    import latent_preview
    import nodes

    from .audio_guidance import anchor_encoded_audio, validate_timeline_pcm
    from .image_guidance import anchor_first_image, validate_first_image
    from .media import align_video_reference, load_audio, load_image, load_video, mux_mp4, sha256
    from .vsa_patch import FastH3VSAPatch

    base_root = Path(os.environ.get("H3_BASE_MODEL_ROOT", "/models/MiniMax-H3"))
    singularity_root = Path(os.environ.get("SINGULARITY_MODEL_ROOT", str(base_root / "singularity")))
    component_root = Path(os.environ.get("H3_COMPONENT_MODEL_ROOT", str(singularity_root)))

    folder_paths.add_model_folder_path(
        "diffusion_models",
        str(base_root / "candidates" / "a1_fasth3_nvfp4"),
    )
    folder_paths.add_model_folder_path(
        "text_encoders",
        str(base_root / "candidates" / "shared" / "text_encoders"),
    )
    folder_paths.add_model_folder_path(
        "vae",
        str(base_root / "candidates" / "shared" / "vae"),
    )
    folder_paths.add_model_folder_path("vae", str(component_root / "vae"))

    def hard_unload(clip_obj=None) -> float:
        t0 = time.perf_counter()
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
        return time.perf_counter() - t0

    started = time.perf_counter()
    decode_started = time.perf_counter()
    images = [load_image(path) for path in files.get("images", [])]
    videos = []
    timing = resolve_timing(request["seconds"])
    first_video_size = None
    video_audios = {}
    video_scale = float(os.environ.get("SINGULARITY_REFERENCE_VIDEO_SCALE", "0.5"))
    if not 0.25 <= video_scale <= 1.0:
        raise RuntimeErrorCode("invalid_reference_video_scale")
    for index, path in enumerate(files.get("videos", [])):
        frames_ref, audio_ref, _fps = load_video(path)
        frames_ref, audio_ref = align_video_reference(frames_ref, audio_ref, timing["frames"])
        if first_video_size is None:
            first_video_size = (int(frames_ref.shape[2]), int(frames_ref.shape[1]))
        if video_scale < 1.0:
            scaled_height = max(16, int(frames_ref.shape[1] * video_scale) // 16 * 16)
            scaled_width = max(16, int(frames_ref.shape[2] * video_scale) // 16 * 16)
            frames_ref = (
                torch.nn.functional.interpolate(
                    frames_ref.permute(0, 3, 1, 2),
                    size=(scaled_height, scaled_width),
                    mode="bilinear",
                    align_corners=False,
                )
                .permute(0, 2, 3, 1)
                .contiguous()
            )
        videos.append(frames_ref)
        if audio_ref is not None:
            video_audios[f"ref_video_audio_{index}"] = audio_ref
    audios = [load_audio(path) for path in files.get("audios", [])]
    timeline_audio = None
    if request.get("audio_guidance", "reference") == "timeline":
        if len(audios) != 1 or videos:
            raise RuntimeErrorCode("timeline_requires_one_audio_and_no_video")
        timeline_audio = validate_timeline_pcm(audios[0], timing["frames"])
    if not images and not videos and not audios:
        raise RuntimeErrorCode("reference_required")

    target = request.get("target") or {}
    source_size = (int(images[0].shape[2]), int(images[0].shape[1])) if images else first_video_size
    geometry = resolve_geometry(target.get("aspect_ratio", "9:16"), source_size)
    width, height = geometry["width"], geometry["height"]
    if request.get("image_guidance") == "first_frame":
        validate_first_image(images, width, height)
    ref_decode_seconds = time.perf_counter() - decode_started

    set_stage("warming")
    torch.cuda.reset_peak_memory_stats()
    model_t0 = time.perf_counter()
    base_m = nodes.UNETLoader().load_unet("fasth3_omniref_qad_nvfp4_comfy.safetensors", "default")[0]
    dm = base_m.model.diffusion_model
    if dm.final_layer.video_out.weight.shape[0] > 96:
        # 32-step PDD head output dimension fix verified in T2.1
        dm.latents_dim = 24
        dm.audio_latents_dim = 32
        dm.final_layer.video_out.out_features = 96
        dm.final_layer.audio_out.out_features = 32
    model_load_seconds = time.perf_counter() - model_t0

    patch_t0 = time.perf_counter()
    dit_model = FastH3VSAPatch().patch(base_m, sparsity=0.90)[0]
    dit_model = h3_nodes.MiniMaxH3SigmaShift.execute(dit_model, 12.0, 3.0)[0]
    patch_seconds = time.perf_counter() - patch_t0

    clip = nodes.CLIPLoader().load_clip("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", type="minimax")[0]
    vae = nodes.VAELoader().load_vae("minimax_h3_video_vae_fp16.safetensors")[0]
    audio_vae = nodes.VAELoader().load_vae(
        os.environ.get("H3_AUDIO_VAE_NAME", "minimax_h3_audio_vae_fp32.safetensors")
    )[0]
    for v_obj in (vae, audio_vae):
        v_obj.device = torch.device("cuda:0")
        v_obj.patcher.load_device = torch.device("cuda:0")
        v_obj.output_device = torch.device("cpu")

    video_vae_inner = getattr(vae, "first_stage_model", None)
    if video_vae_inner is not None and getattr(video_vae_inner, "comfy_has_chunked_io", False):
        video_vae_inner.tile_size = 128
        video_vae_inner.tile_overlap_min = min(int(getattr(video_vae_inner, "tile_overlap_min", 64)), 32)
        if hasattr(video_vae_inner, "decoder_tile_size"):
            video_vae_inner.decoder_tile_size = 256
        if hasattr(video_vae_inner, "decoder_tile_overlap_min"):
            video_vae_inner.decoder_tile_overlap_min = 64

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

    seed = int(request.get("seed", 7))
    length = timing["frames"]

    with torch.no_grad():
        hard_unload(None)
        model_management.vram_state = model_management.VRAMState.NORMAL_VRAM

        orig_v_encode = vae.encode
        orig_a_encode = audio_vae.encode
        orig_c_encode = clip.encode_from_tokens_scheduled
        clip_released = False
        text_encoder_seconds = 0.0
        vae_ref_seconds = 0.0
        audio_ref_seconds = 0.0

        def ensure_clip_unloaded():
            nonlocal clip_released
            if not clip_released:
                hard_unload(clip)
                clip_released = True

        def timed_v_encode(pixels, *a, **kw):
            nonlocal vae_ref_seconds
            ensure_clip_unloaded()
            t0 = time.perf_counter()
            try:
                return orig_v_encode(pixels, *a, **kw)
            finally:
                vae_ref_seconds += time.perf_counter() - t0

        def timed_a_encode(waveform, *a, **kw):
            nonlocal audio_ref_seconds
            ensure_clip_unloaded()
            t0 = time.perf_counter()
            try:
                return orig_a_encode(waveform, *a, **kw)
            finally:
                audio_ref_seconds += time.perf_counter() - t0

        def timed_c_encode(*a, **kw):
            nonlocal text_encoder_seconds
            t0 = time.perf_counter()
            try:
                return orig_c_encode(*a, **kw)
            finally:
                torch.cuda.synchronize()
                text_encoder_seconds += time.perf_counter() - t0

        vae.encode = timed_v_encode
        audio_vae.encode = timed_a_encode
        object.__setattr__(clip, "encode_from_tokens_scheduled", timed_c_encode)
        enc_t0 = time.perf_counter()
        try:
            conditioning, latent = h3_nodes.MiniMaxH3ReferenceToVideo.execute(
                clip,
                request["prompt"],
                width,
                height,
                length,
                "match",
                vae,
                audio_vae,
                {f"ref_image_{i}": im for i, im in enumerate(images)},
                {f"ref_video_{i}": vd for i, vd in enumerate(videos)},
                video_audios,
                {f"ref_audio_{i}": au for i, au in enumerate(audios)},
            )
            if request.get("image_guidance") == "first_frame":
                conditioning = anchor_first_image(conditioning, width, height)
            if timeline_audio is not None:
                audio_ticks = int(latent["samples"].tensors[1].shape[-1])
                conditioning = anchor_encoded_audio(conditioning, audio_ticks)
        finally:
            vae.encode = orig_v_encode
            audio_vae.encode = orig_a_encode
            object.__setattr__(clip, "encode_from_tokens_scheduled", orig_c_encode)

        ref_and_text_seconds = time.perf_counter() - enc_t0
        hard_unload(clip)

        set_stage("generating")
        sampler = custom_sampler.KSamplerSelect.execute("euler")[0]
        sigmas = custom_sampler.BasicScheduler.execute(dit_model, "simple", 8, 1.0)[0]
        guider = custom_sampler.BasicGuider.execute(dit_model, conditioning)[0]
        noise = custom_sampler.RandomNoise.execute(seed)[0]

        step_times: list[float] = []
        orig_prep_cb = latent_preview.prepare_callback

        def timed_prep_cb(*args, **kwargs):
            cb = orig_prep_cb(*args, **kwargs)
            torch.cuda.synchronize()
            prev = time.perf_counter()

            def step_cb(step, x0, x, total_steps):
                nonlocal prev
                torch.cuda.synchronize()
                now_t = time.perf_counter()
                step_times.append(now_t - prev)
                prev = now_t
                return cb(step, x0, x, total_steps)

            return step_cb

        dit_t0 = time.perf_counter()
        latent_preview.prepare_callback = timed_prep_cb
        try:
            sampled = custom_sampler.SamplerCustomAdvanced.execute(
                noise,
                guider,
                sampler,
                sigmas,
                latent,
            )[0]
        finally:
            latent_preview.prepare_callback = orig_prep_cb
        dit_seconds = time.perf_counter() - dit_t0
        if len(step_times) != 8:
            raise RuntimeErrorCode(f"dit_step_metrics_unavailable:{len(step_times)}")

        peak_reserved_dit = int(torch.cuda.max_memory_reserved())
        hard_unload(None)

        decode_t0 = time.perf_counter()
        if video_vae_inner is not None and getattr(video_vae_inner, "comfy_has_chunked_io", False):
            video_vae_inner.tile_size = 256
            video_vae_inner.tile_overlap_min = 64
        model_management.load_models_gpu(
            [p for p in (getattr(vae, "patcher", None), getattr(audio_vae, "patcher", None)) if p is not None],
            force_full_load=True,
        )
        frames = nodes.VAEDecode().decode(vae, sampled)[0]
        audio_out = audio_nodes.VAEDecodeAudio.execute(audio_vae, sampled)[0]
        decode_seconds = time.perf_counter() - decode_t0

        if frames.ndim == 5:
            if frames.shape[-1] == 3:
                frames = frames.reshape(-1, frames.shape[-3], frames.shape[-2], 3)
            elif frames.shape[1] == 3:
                frames = frames[0].permute(1, 2, 3, 0)
            else:
                raise RuntimeErrorCode(f"invalid_video_decode_shape:{tuple(frames.shape)}")
        elif frames.ndim == 4 and frames.shape[-1] != 3 and frames.shape[1] == 3:
            frames = frames.permute(0, 2, 3, 1)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise RuntimeErrorCode(f"invalid_video_decode_shape:{tuple(frames.shape)}")
        frames = frames.detach().to("cpu").contiguous()

        set_stage("saving")
        mux_t0 = time.perf_counter()
        media = mux_mp4(frames, audio_out, output_mp4)
        mux_seconds = time.perf_counter() - mux_t0
        peak_alloc = int(torch.cuda.max_memory_allocated())
        peak_res = max(peak_reserved_dit, int(torch.cuda.max_memory_reserved()))

    metrics = {
        "quality_profile": "fasth3-nvfp4",
        "resolved_output": {**geometry, **timing},
        "reference_decode_seconds": ref_decode_seconds,
        "model_load_seconds": model_load_seconds,
        "lora_patch_seconds": patch_seconds,
        "reference_and_text_encoder_seconds": ref_and_text_seconds,
        "text_encoder_seconds": text_encoder_seconds,
        "vae_reference_seconds": vae_ref_seconds,
        "audio_vae_reference_seconds": audio_ref_seconds,
        "dit_seconds": dit_seconds,
        "dit_step_seconds": step_times,
        "video_audio_vae_seconds": decode_seconds,
        "video_encoding_seconds": mux_seconds,
        "runtime_total_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": peak_alloc,
        "peak_reserved_bytes": peak_res,
        "vm_hwm_bytes": read_vm_hwm_bytes(),
        "gpu_uuid": gpu_uuid,
        "free_vram_mib_before": free_vram_mib_before,
        "sequence_frames": int(media["frames"]),
        "nfe": 8,
        "seed": seed,
        **sm120_info,
    }
    return {
        "content_sha256": sha256(output_mp4),
        "media": media,
        "runtime_metrics": metrics,
        "model": "fasth3_omniref_qad_nvfp4_comfy.safetensors",
        "lora": None,
        "execution_instance_id": str(uuid.uuid4()),
    }


def run_worker_task(task_id: str, task_root: Path | None = None) -> int:
    root = Path(task_root or os.environ.get("SINGULARITY_TASK_ROOT", "/data/projects/singularity/tasks"))
    queue = Queue(root, acquire_lock=False)
    try:
        task = queue.get_any(task_id)
        kind = task["kind"]
        request = task["request"]
        directory = root / task_id

        if kind == "video":
            from .api import _download_references

            download_started = time.perf_counter()
            download_seconds = None
            try:
                directory.mkdir(parents=True, exist_ok=False)
                queue.update(task_id, stage="downloading", worker_settled=0)
                files = _download_references(directory, request.get("conditions", []))
                download_seconds = time.perf_counter() - download_started
                gpu_uuid, free_mib = check_free_vram()
                profile = request.get("quality_profile", default_quality_profile())
                queue.update(task_id, stage="warming", worker_settled=0)
                if profile == "fasth3-nvfp4":
                    sm120_info = _preload_sm120_cuda13_and_comfy()
                    result = _run_fasth3_nvfp4(
                        request,
                        files,
                        directory / "output.mp4",
                        lambda stage: queue.update(task_id, stage=stage, worker_settled=0),
                        gpu_uuid=gpu_uuid,
                        free_vram_mib_before=free_mib,
                        sm120_info=sm120_info,
                    )
                else:
                    os.environ.setdefault("H3_CLIP_DEVICE", "cuda:0")
                    os.environ.setdefault("H3_VAE_DEVICE", "cuda:0")
                    os.environ.setdefault("SINGULARITY_LOW_VRAM_MODE", "auto")
                    from .engine import Engine

                    engine = Engine()
                    result = engine.generate(
                        request,
                        files,
                        directory / "output.mp4",
                        lambda stage: queue.update(task_id, stage=stage, worker_settled=0),
                    )
                metrics = dict(result.pop("runtime_metrics", {}) or {})
                metrics["reference_download_seconds"] = download_seconds
                metrics.setdefault("gpu_uuid", gpu_uuid)
                metrics.setdefault("free_vram_mib_before", free_mib)
                metrics.setdefault("vm_hwm_bytes", read_vm_hwm_bytes())
                queue.update(
                    task_id,
                    status="completed",
                    stage="completed",
                    execution_instance_id=result.pop("execution_instance_id", str(uuid.uuid4())),
                    runtime_metrics=metrics,
                    result=result,
                    worker_settled=0,
                )
                return 0
            except Exception as exc:
                code = getattr(exc, "code", type(exc).__name__)
                metrics = dict(getattr(exc, "runtime_metrics", None) or {})
                if download_seconds is None:
                    metrics["reference_download_seconds"] = time.perf_counter() - download_started
                queue.update(
                    task_id,
                    status="failed",
                    stage="engine_failed",
                    runtime_metrics=metrics,
                    error=code,
                    worker_settled=0,
                )
                return 1

        # kind == "regenerate_2k"
        try:
            directory.mkdir(parents=True, exist_ok=False)
            artifact_root = Path(os.environ.get("SINGULARITY_ARTIFACT_ROOT", "/source/video"))
            source_mp4 = resolve_source_artifact_path(request, artifact_root)
            gpu_uuid, free_mib = check_free_vram()
            _preload_sm120_cuda13_and_comfy()
            result = run_regenerate_2k(
                request,
                source_mp4,
                directory,
                lambda stage: queue.update(task_id, stage=stage, worker_settled=0),
                gpu_uuid=gpu_uuid,
                free_vram_mib_before=free_mib,
            )
            (directory / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            queue.update(
                task_id,
                status="succeeded",
                stage="succeeded",
                runtime_metrics=result.get("runtime_metrics"),
                result=result,
                worker_settled=0,
            )
            return 0
        except Exception as exc:
            code = getattr(exc, "code", type(exc).__name__)
            metrics = dict(getattr(exc, "runtime_metrics", None) or {})
            queue.update(
                task_id,
                status="failed",
                stage="engine_failed",
                runtime_metrics=metrics or None,
                error=code,
                worker_settled=0,
            )
            return 1
    finally:
        queue.close()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        sys.stderr.write("Usage: python -m h3_singularity.worker <task_id>\n")
        return 2
    return run_worker_task(args[0])


if __name__ == "__main__":
    raise SystemExit(main())
