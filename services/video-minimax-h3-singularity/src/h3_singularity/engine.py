"""ComfyUI H3 execution adapter for the pinned Singularity Ref2VA model."""

from __future__ import annotations

import os
import math
from pathlib import Path
import time
import uuid

import torch

from .media import load_audio, load_image, load_video, mux_mp4, sha256
from .errors import RuntimeErrorCode
from . import __version__
from .hr_refine import (
    audio_range,
    h3_resize_video,
    spatial_tiles,
    temporal_windows,
)


HR_SAMPLERS = ("euler", "er_sde")
HR_SCHEDULERS = ("simple", "beta")


class Engine:
    """Load H3 components once and run one serialized Ref2VA request at a time."""

    def __init__(self):
        self.model_root = Path(os.environ.get("SINGULARITY_MODEL_ROOT", "/models/singularity"))
        self.base_root = Path(os.environ.get("H3_BASE_MODEL_ROOT", "/models/MiniMax-H3"))
        # The non-DiT H3 components are staged beside the pinned Singularity
        # weights. Keep this root explicit so a compatible-looking Diffusers
        # tree cannot be loaded by accident.
        self.component_root = Path(
            os.environ.get("H3_COMPONENT_MODEL_ROOT", str(self.model_root))
        )
        self.diffusion_name = os.environ.get(
            "SINGULARITY_DIFFUSION_NAME", "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"
        )
        self.lora_name = os.environ.get(
            "SINGULARITY_LORA_NAME", "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
        )
        self.clip_name = os.environ.get("H3_CLIP_NAME", "qwen3vl_4b_fp8_scaled.safetensors")
        self.clip_type = os.environ.get("H3_CLIP_TYPE", "auto")
        self.clip_projection_name = os.environ.get(
            "H3_CLIP_PROJECTION_NAME", "mmh3-4b-ClipProj-v3.1-mlp.safetensors"
        )
        self.clip_device = torch.device(os.environ.get("H3_CLIP_DEVICE", "cuda:1"))
        self.clip_mode = os.environ.get("H3_CLIP_MODE", "resident")
        self.vae_device = torch.device(os.environ.get("H3_VAE_DEVICE", "cuda:1"))
        self.video_vae_name = os.environ.get("H3_VIDEO_VAE_NAME", "minimax_h3_video_vae_int8_convrot.safetensors")
        self.audio_vae_name = os.environ.get("H3_AUDIO_VAE_NAME", "minimax_h3_audio_vae_fp32.safetensors")
        # Dual-sample latent refinement is an explicit diagnostic profile. It
        # must never be enabled by an ordinary 4-NFE request accidentally.
        self.dual_sample_mode = os.environ.get("SINGULARITY_DUAL_SAMPLE_MODE", "off")
        self.hr_refine_mode = os.environ.get("SINGULARITY_HR_REFINE_MODE", "off").strip().lower()
        if self.hr_refine_mode not in {"off", "hr-refine-tile-v1", "hr-refine-global-v2", "hybrid-4+3", "hybrid-4+2"}:
            raise RuntimeErrorCode("invalid_hr_refine_mode")
        self.hr_refine_steps = int(os.environ.get("SINGULARITY_HR_REFINE_STEPS", "2"))
        if self.hr_refine_steps not in {1, 2}:
            raise RuntimeErrorCode("invalid_hr_refine_steps")
        self.hr_sampler = os.environ.get("SINGULARITY_HR_SAMPLER", "euler").strip().lower()
        self.hr_scheduler = os.environ.get("SINGULARITY_HR_SCHEDULER", "simple").strip().lower()
        self._validate_hr_sampling_config(self.hr_sampler, self.hr_scheduler)
        self.lowvram_mode = os.environ.get("SINGULARITY_LOW_VRAM_MODE", "off")
        if self.lowvram_mode not in {"off", "auto"}:
            raise RuntimeErrorCode("invalid_low_vram_mode")
        self.lowvram_state = os.environ.get("SINGULARITY_LOW_VRAM_STATE", "low").strip().lower()
        if self.lowvram_state not in {"low", "low_vram", "no", "no_vram"}:
            raise RuntimeErrorCode("invalid_low_vram_state")
        self.upscaler_device = torch.device(
            os.environ.get("SINGULARITY_UPSCALER_DEVICE", str(self.vae_device))
        )
        self.upscaler_name = os.environ.get(
            "SINGULARITY_UPSCALER_NAME", "minimax_h3_latent_upscaler_3d_fp16.safetensors"
        )
        self.upscaler_root = Path(
            os.environ.get("SINGULARITY_UPSCALER_ROOT", "/models/h3-latent")
        )
        self._load_components()

    @staticmethod
    def _release_comfy_models() -> float:
        """Release all Comfy patchers before changing pipeline stages.

        Comfy keeps the most recently used patchers in its smart-memory cache.
        That is useful for interactive workflows, but leaves the 20 GiB DiT
        resident while H3 VAE reference encoding starts.  The runtime is a
        serialized worker, so a full stage boundary is safe and deterministic.
        """
        import comfy.model_management as model_management

        started = time.perf_counter()
        model_management.unload_all_models()
        model_management.soft_empty_cache()
        return time.perf_counter() - started

    def _release_clip_to_cpu(self) -> float:
        """Move every ClipProj allocation off GPU1 before VAE work.

        ``ProjectedCLIP`` is a wrapper.  Its ``patcher.model`` is only one
        owner of the Qwen tensors; the loader also registers a pinned patcher
        and keeps the projection MLP in a separate ``_gpu`` cache.  Calling
        ``model.to('cpu')`` on the wrapper therefore leaves the real CUDA
        storage resident.  Use the ClipProj release hooks first, then recurse
        through the underlying model as a compatibility fallback.
        """
        started = time.perf_counter()
        cpu = torch.device("cpu")
        try:
            from .clipproj.clipproj_pinning import release_all
            release_all()
        except Exception:
            # Older ClipProj builds do not expose the registry.  The direct
            # fallback below still handles their patcher object.
            pass
        try:
            from .clipproj.clipproj_nodes import purge_projections
            purge_projections(self.clip_device)
        except Exception:
            pass

        base = getattr(self.clip, "_base", self.clip)
        patcher = getattr(base, "patcher", None)
        if patcher is not None:
            patcher.offload_device = cpu
            model = getattr(patcher, "model", None)
            if model is not None and hasattr(model, "to"):
                model.to(device=cpu)
        for owner in (base, getattr(base, "cond_stage_model", None)):
            if owner is not None and hasattr(owner, "to"):
                owner.to(device=cpu)
        import comfy.model_management as model_management
        import gc
        gc.collect()
        model_management.soft_empty_cache(force=True)
        if torch.cuda.is_available():
            with torch.cuda.device(self.clip_device):
                torch.cuda.empty_cache()
        return time.perf_counter() - started

    @staticmethod
    def _gpu_snapshot() -> dict[str, dict[str, int]]:
        if not torch.cuda.is_available():
            return {}
        result: dict[str, dict[str, int]] = {}
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                free, total = torch.cuda.mem_get_info()
                result[str(index)] = {
                    "free_bytes": int(free),
                    "total_bytes": int(total),
                    "allocated_bytes": int(torch.cuda.memory_allocated()),
                    "reserved_bytes": int(torch.cuda.memory_reserved()),
                    "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                }
        return result

    @staticmethod
    def _configure_comfy():
        import sys
        comfy_root = os.environ.get("COMFY_ROOT", "/opt/comfy")
        if comfy_root not in sys.path:
            sys.path.insert(0, comfy_root)
        import folder_paths
        # The image contains ComfyUI but model directories are on read-only PVCs.
        folder_paths.add_model_folder_path("diffusion_models", os.environ.get("SINGULARITY_MODEL_ROOT", "/models/singularity"))
        singularity_root = os.environ.get("SINGULARITY_MODEL_ROOT", "/models/singularity")
        folder_paths.add_model_folder_path("loras", os.path.join(singularity_root, "loras"))
        folder_paths.add_model_folder_path("clip_projections", os.path.join(singularity_root, "clip_projections"))
        component_root = os.environ.get(
            "H3_COMPONENT_MODEL_ROOT",
            os.environ.get("SINGULARITY_MODEL_ROOT", "/models/singularity"),
        )
        folder_paths.add_model_folder_path("text_encoders", os.path.join(component_root, "text_encoders"))
        folder_paths.add_model_folder_path("vae", os.path.join(component_root, "vae"))
        # Register H3 V3 nodes before importing the engine nodes.
        import nodes
        import comfy_extras.nodes_minimax_h3  # noqa: F401
        import comfy_extras.nodes_custom_sampler  # noqa: F401
        import comfy_extras.nodes_audio  # noqa: F401
        from .clipproj import ClipProjLoader
        return folder_paths, nodes, ClipProjLoader

    def _load_components(self):
        if not torch.cuda.is_available():
            raise RuntimeErrorCode("cuda_unavailable")
        evidence = self.model_root / "model-evidence.json"
        if not evidence.is_file():
            raise RuntimeErrorCode("singularity_model_not_verified")
        component_evidence = self.component_root / "component-model-evidence.json"
        if not component_evidence.is_file():
            raise RuntimeErrorCode("h3_components_not_verified")
        self.folder_paths, self.nodes, self.clipproj_loader = self._configure_comfy()
        started = time.perf_counter()
        diffusion_path = self.model_root / self.diffusion_name
        lora_path = self.model_root / "loras" / self.lora_name
        for path in (diffusion_path, lora_path):
            if not path.is_file():
                raise RuntimeErrorCode(f"model_file_missing:{path.name}")
        if self.dual_sample_mode not in {"off", "du-1", "du-2", "du-3"}:
            raise RuntimeErrorCode("invalid_dual_sample_mode")
        lowvram_enabled = (
            self.lowvram_mode == "auto"
            or self.dual_sample_mode != "off"
            or self.hr_refine_mode != "off"
        )
        model_started = time.perf_counter()
        if lowvram_enabled:
            from .lowvram import CoffH3ModelLoader

            self.model, self.lowvram_loader = CoffH3ModelLoader(
                self.nodes, self.diffusion_name, torch.device("cuda:0")
            ).load("auto")
        else:
            self.model = self.nodes.UNETLoader().load_unet(self.diffusion_name, "default")[0]
            self.lowvram_loader = {
                "loader": "UNETLoader",
                "loader_implementation": "native-comfy-default",
                "backend": "default",
            }
        self.model_load_seconds = time.perf_counter() - model_started
        lora_started = time.perf_counter()
        self.model = self.nodes.LoraLoaderModelOnly().load_lora_model_only(self.model, self.lora_name, 1.0)[0]
        self.lora_patch_seconds = time.perf_counter() - lora_started
        self.clip = self.clipproj_loader().load(
            self.clip_name,
            self.clip_type,
            self.clip_projection_name,
            str(self.clip_device),
            self.clip_mode,
            unique_id="h3-singularity-clipproj",
        )[0]
        self.vae = self.nodes.VAELoader().load_vae(self.video_vae_name)[0]
        self.audio_vae = self.nodes.VAELoader().load_vae(self.audio_vae_name)[0]
        if self.dual_sample_mode != "off":
            self.upscaler_path = self.upscaler_root / self.upscaler_name
            if not self.upscaler_path.is_file():
                raise RuntimeErrorCode(f"upscaler_file_missing:{self.upscaler_path.name}")
        else:
            self.upscaler_path = None
        for vae in (self.vae, self.audio_vae):
            vae.device = self.vae_device
            vae.patcher.load_device = self.vae_device
        # Keep encoded references and decoded frames on host memory.  Comfy's
        # default intermediate device is CUDA, which makes a 15 s reference
        # video retain its full latent tensor on the 24 GiB card while the
        # next reference is encoded.  The H3 model moves reference latents to
        # its execution device when sampling, and muxing accepts CPU frames.
        cpu = torch.device("cpu")
        self.vae.output_device = cpu
        self.audio_vae.output_device = cpu
        # Encoding and decoding have different memory lifetimes.  Reference
        # encoding must stay conservative on a 24 GiB card, while decoding
        # happens after DiT and Qwen have been unloaded and can use the native
        # 256/64 tile geometry.  Sharing the 128 px encoder tile with decode
        # creates unnecessary low-context seams in the final image.
        legacy_tile = os.environ.get("SINGULARITY_VAE_TILE_SIZE")
        encoder_tile_size = int(os.environ.get("SINGULARITY_VAE_ENCODER_TILE_SIZE", legacy_tile or "128"))
        decoder_tile_size = int(os.environ.get("SINGULARITY_VAE_DECODER_TILE_SIZE", "256"))
        if encoder_tile_size < 64 or encoder_tile_size % 16:
            raise RuntimeErrorCode("invalid_vae_tile_size")
        if decoder_tile_size < 64 or decoder_tile_size % 16:
            raise RuntimeErrorCode("invalid_vae_decoder_tile_size")
        video_vae = getattr(self.vae, "first_stage_model", None)
        self._video_vae_model = video_vae
        self._vae_encoder_overlap = encoder_tile_size // 4
        self._vae_decoder_overlap = int(os.environ.get("SINGULARITY_VAE_DECODER_OVERLAP", "64"))
        if getattr(video_vae, "comfy_has_chunked_io", False):
            video_vae.tile_size = encoder_tile_size
            video_vae.tile_overlap_min = min(int(getattr(video_vae, "tile_overlap_min", 64)), self._vae_encoder_overlap)
            if hasattr(video_vae, "decoder_tile_size"):
                video_vae.decoder_tile_size = decoder_tile_size
            if hasattr(video_vae, "decoder_tile_overlap_min"):
                video_vae.decoder_tile_overlap_min = self._vae_decoder_overlap
            # The native temporal encoder keeps every 17-frame output on the
            # execution device until the final concat.  Stage each chunk on
            # CPU immediately; the wrapper and the DiT already support CPU
            # reference latents and move them back only when sampling.
            def encode_temporal_cpu(x, device):
                chunks = []
                for index in range(math.ceil(x.shape[2] / video_vae.clip_length)):
                    clip_x = x[:, :, index * video_vae.clip_length:(index + 1) * video_vae.clip_length].to(device)
                    if clip_x.shape[2] < video_vae.clip_length:
                        pad_frames = clip_x[:, :, -1:].repeat(1, 1, video_vae.clip_length - clip_x.shape[2], 1, 1)
                        clip_x = torch.cat([clip_x, pad_frames], dim=2)
                    chunks.append(video_vae._adaptive_encode(video_vae._normalize_pixels(clip_x)).to("cpu"))
                    del clip_x
                result = torch.cat(chunks, dim=2)
                if video_vae.token_drop > 0:
                    result = result[:, :, :-video_vae.token_drop]
                return result

            video_vae.encode_temporal = encode_temporal_cpu
        self.vae_tile_size = encoder_tile_size
        self.vae_decoder_tile_size = decoder_tile_size
        self.load_seconds = time.perf_counter() - started
        self.version = os.environ.get("SINGULARITY_RUNTIME_VERSION", f"video-minimax-h3-singularity-v{__version__}")
        self.execution_instance_id = str(uuid.uuid4())

    def health(self) -> dict:
        return {
            "ready": True,
            "runtime_version": self.version,
            "model": self.diffusion_name,
            "lora": self.lora_name,
            "components": {
                "root": str(self.component_root),
                "evidence": "component-model-evidence.json",
                "clip": self.clip_name,
                "clip_type": self.clip_type,
                "clip_projection": self.clip_projection_name,
                "clip_device": str(self.clip_device),
                "video_vae": self.video_vae_name,
                "audio_vae": self.audio_vae_name,
                "vae_device": str(self.vae_device),
                "vae_tile_size": self.vae_tile_size,
                "vae_decoder_tile_size": self.vae_decoder_tile_size,
            },
            "nfe": 4,
            "dual_sample_mode": self.dual_sample_mode,
            "hr_refine_mode": self.hr_refine_mode,
            "hr_refine_steps": self.hr_refine_steps,
            "hr_sampler": self.hr_sampler,
            "hr_scheduler": self.hr_scheduler,
            "low_vram_mode": self.lowvram_mode,
            "low_vram_state": self.lowvram_state,
            "low_vram_loader": self.lowvram_loader,
            "upscaler": self.upscaler_name if self.dual_sample_mode != "off" else None,
            "gpu_count": torch.cuda.device_count(),
            "cpu_offload": True,
        }

    @staticmethod
    def _validate_hr_sampling_config(sampler: str, scheduler: str) -> None:
        if sampler not in HR_SAMPLERS:
            raise RuntimeErrorCode("invalid_hr_sampler")
        if scheduler not in HR_SCHEDULERS:
            raise RuntimeErrorCode("invalid_hr_scheduler")

    @staticmethod
    def _split_dual_sigmas(sigmas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split a 4-NFE sigma path into two 2-NFE paths at one boundary."""
        if not isinstance(sigmas, torch.Tensor) or sigmas.ndim == 0:
            raise RuntimeErrorCode("invalid_dual_sample_sigmas")
        points = int(sigmas.shape[-1])
        if points != 5:
            raise RuntimeErrorCode(f"dual_sample_requires_5_sigma_points:{points}")
        return sigmas[..., :3].contiguous(), sigmas[..., 2:].contiguous()

    def _upscale_video_latent(self, video: torch.Tensor, scale: float) -> tuple[torch.Tensor, float]:
        """Run the pinned H3 3D latent upscaler on the VAE/Qwen GPU."""
        if self.upscaler_path is None:
            raise RuntimeErrorCode("dual_sample_disabled")
        if not 1.0 < scale <= 1.5:
            raise RuntimeErrorCode("invalid_dual_sample_scale")
        from safetensors.torch import load_file
        from h3_latent_upscaler.vendor import upscaler_3d

        started = time.perf_counter()
        state = load_file(str(self.upscaler_path), device="cpu")
        state = upscaler_3d._extract_upscaler_sd(state)
        cfg = upscaler_3d._detect_arch(state)
        old_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float16)
            model = upscaler_3d.LatentResizer3D(
                in_channels=cfg["in_channels"], in_blocks=cfg["in_blocks"],
                out_blocks=cfg["out_blocks"], channels=cfg["channels"],
                dropout=cfg["dropout"], attn=cfg["attn"],
                temporal_every=cfg["temporal_every"],
                temporal_kernel=cfg["temporal_kernel"],
            )
        finally:
            torch.set_default_dtype(old_dtype)
        model.load_state_dict(state, strict=True)
        del state
        model.eval().requires_grad_(False).to(self.upscaler_device)
        # The model operates on normalized H3 video latents. Preserve the
        # temporal grid and scale only the spatial latent dimensions.
        with torch.inference_mode():
            mean, std = upscaler_3d._make_norm_tensors(self.upscaler_device, torch.float16)
            normalized = (video.to(self.upscaler_device, dtype=torch.float16) - mean) / std
            target_size = (
                int(video.shape[2]),
                max(2, int(round(video.shape[3] * scale))),
                max(2, int(round(video.shape[4] * scale))),
            )
            resized = model(
                normalized,
                scale=scale,
                target_size=target_size,
                enable_chunking=True,
            )
            resized = (resized * std + mean).to("cpu").contiguous()
        del model, mean, std, normalized
        self._release_comfy_models()
        if not torch.isfinite(resized).all():
            raise RuntimeErrorCode("nonfinite_upscaled_latent")
        return resized, time.perf_counter() - started

    @staticmethod
    def _crop_hr_conditioning(conditioning, source_h: int, source_w: int,
                              top: int, left: int, height: int, width: int):
        """Crop H3 keyframe latents for a tile without re-running Qwen."""
        import torch.nn.functional as functional

        result = []
        for tensor, data in conditioning:
            copied = dict(data)
            keyframes = copied.get("minimax_keyframes")
            if keyframes:
                cropped = []
                for keyframe in keyframes:
                    item = dict(keyframe)
                    latent = item.get("latent")
                    if latent is not None:
                        if latent.shape[-2:] != (source_h, source_w):
                            b, c, t, h, w = latent.shape
                            latent = functional.interpolate(
                                latent.to(torch.float32).reshape(b * t, c, h, w),
                                size=(source_h, source_w), mode="bilinear", align_corners=False,
                            ).reshape(b, c, t, source_h, source_w).to(latent.dtype)
                        item["latent"] = latent[..., top:top + height, left:left + width].contiguous()
                    cropped.append(item)
                copied["minimax_keyframes"] = cropped
            result.append([tensor, copied])
        return result

    def _sample_hr_tiles(self, model, conditioning, sampled_low, sampler, seed,
                         sigmas, sample_stage, timings):
        """Refine a resized H3 video latent tile by tile while freezing audio."""
        import comfy.nested_tensor

        samples = sampled_low.get("samples")
        if not getattr(samples, "is_nested", False) or len(samples.tensors) != 2:
            raise RuntimeErrorCode("hr_refine_invalid_av_latent")
        low_video, low_audio = samples.tensors
        try:
            from h3_latent_upscaler.vendor import upscaler_3d

            mean, std = upscaler_3d._make_norm_tensors(low_video.device, torch.float32)
        except Exception as exc:
            raise RuntimeErrorCode("hr_refine_norm_unavailable") from exc
        target_h, target_w = 84, 48
        video = h3_resize_video(low_video, target_h, target_w, mean, std)
        audio = low_audio.to(video.device).contiguous()
        timings["hr_refine_resize_shape"] = list(video.shape)
        timings["hr_refine_audio_shape"] = list(audio.shape)
        timings["hr_refine_steps"] = self.hr_refine_steps
        timings["hr_refine_sigma"] = [0.24, 0.08, 0.0] if self.hr_refine_steps == 2 else [0.16, 0.0]
        windows = temporal_windows(int(video.shape[2]), chunk_frames=73, overlap_frames=22)
        tiles = spatial_tiles(int(video.shape[3]), int(video.shape[4]), 42, 24, 8)
        timings["hr_refine_temporal_windows"] = len(windows)
        timings["hr_refine_spatial_tiles"] = len(tiles)
        timings["hr_refine_tile_count"] = len(windows) * len(tiles)
        if self.hr_refine_steps == 2:
            refine_sigmas = torch.tensor([0.24, 0.08, 0.0], device=sigmas.device, dtype=sigmas.dtype)
        else:
            refine_sigmas = torch.tensor([0.16, 0.0], device=sigmas.device, dtype=sigmas.dtype)

        output = torch.zeros_like(video)
        weights = torch.zeros((1, 1, video.shape[2], video.shape[3], video.shape[4]),
                              device=video.device, dtype=torch.float32)
        tile_times = []
        overlap_deltas = []
        for window_index, (t0, t1, f0, f1) in enumerate(windows):
            chunk = video[:, :, t0:t1]
            a0, a1 = audio_range(f0, f1, int(audio.shape[-1]))
            chunk_audio = audio[..., a0:a1].contiguous()
            for tile_index, (r0, r1, c0, c1, overlap_h, overlap_w) in enumerate(tiles):
                self._active_stage = "dit_hr_refine"
                tile = chunk[..., r0:r1, c0:c1].contiguous()
                existing = output[:, :, t0:t1, r0:r1, c0:c1]
                existing_weights = weights[:, :, t0:t1, r0:r1, c0:c1]
                existing_mask = existing_weights > 0
                if bool(existing_mask.any()):
                    tile = torch.where(
                        existing_mask.expand_as(tile),
                        existing / existing_weights.clamp_min(1e-6).to(existing.dtype),
                        tile,
                    ).contiguous()
                tile_conditioning = self._crop_hr_conditioning(
                    conditioning, video.shape[-2], video.shape[-1], r0, c0, r1 - r0, c1 - c0,
                )
                # Previously completed tiles are the stable source in overlap
                # bands. The current tile is sampled only where no output exists.
                mask = (~existing_mask).to(dtype=torch.float32)
                audio_mask = torch.zeros((1, 32, 2, max(1, a1 - a0)),
                                         device=audio.device, dtype=torch.float32)
                stage_latent = {
                    "samples": comfy.nested_tensor.NestedTensor((tile, chunk_audio)),
                    "noise_mask": comfy.nested_tensor.NestedTensor((mask, audio_mask)),
                }
                # Keep the video tile and frozen audio in a single AV latent. The
                # explicit mask prevents any audio noise or shape conversion.
                before = time.perf_counter()
                refined, _, _ = sample_stage(
                    f"dit_hr_tile_{window_index}_{tile_index}", stage_latent,
                    refine_sigmas, self.hr_refine_steps,
                    conditioning_override=tile_conditioning,
                    seed_override=seed + window_index * 1000 + tile_index,
                )
                elapsed = time.perf_counter() - before
                tile_times.append(elapsed)
                refined_video = refined["samples"].tensors[0]
                if not torch.isfinite(refined_video).all():
                    raise RuntimeErrorCode("hr_refine_nonfinite_tile")
                if bool(existing_mask.any()):
                    overlap_deltas.append(float(
                        (refined_video - tile).abs().masked_select(existing_mask.expand_as(refined_video)).median().item()
                    ))
                blend = torch.ones_like(mask)
                if overlap_w:
                    blend[..., :, :, :overlap_w] *= torch.linspace(0.0, 1.0, overlap_w, device=video.device).view(1, 1, 1, 1, overlap_w)
                if overlap_h:
                    blend[..., :, :overlap_h, :] *= torch.linspace(0.0, 1.0, overlap_h, device=video.device).view(1, 1, 1, overlap_h, 1)
                output[:, :, t0:t1, r0:r1, c0:c1] += refined_video * blend.to(refined_video.dtype)
                weights[:, :, t0:t1, r0:r1, c0:c1] += blend
                del stage_latent, refined, refined_video, tile_conditioning, tile
                self._release_comfy_models()
            del chunk_audio
        if not torch.isfinite(weights).all() or bool((weights <= 0).any()):
            raise RuntimeErrorCode("hr_refine_incomplete_tiles")
        output = (output / weights.to(output.dtype)).contiguous()
        if overlap_deltas and max(overlap_deltas) > 0.05:
            raise RuntimeErrorCode("hr_refine_overlap_mismatch")
        timings["hr_refine_tile_seconds"] = tile_times
        timings["hr_refine_overlap_median_max"] = max(overlap_deltas, default=0.0)
        return {"samples": comfy.nested_tensor.NestedTensor((output, audio))}

    def _sample_hr_global(self, conditioning, sampled_low, refine_sigmas, sample_stage, seed,
                          timings, conditioning_override=None, profile="hr-refine-global-v2",
                          sampler_override=None):
        """One target-grid low-noise pass; the bounded replacement for tile v1."""
        import comfy.nested_tensor
        samples = sampled_low["samples"]
        low_video, low_audio = samples.tensors
        from h3_latent_upscaler.vendor import upscaler_3d
        mean, std = upscaler_3d._make_norm_tensors(low_video.device, torch.float32)
        video = h3_resize_video(low_video, 84, 48, mean, std)
        audio = low_audio.to(video.device).contiguous()
        stage = {
            "samples": comfy.nested_tensor.NestedTensor((video, audio)),
            "noise_mask": comfy.nested_tensor.NestedTensor((
                torch.ones((1, 1, video.shape[2], video.shape[3], video.shape[4]), device=video.device),
                torch.zeros((1, 32, 2, audio.shape[-1]), device=audio.device),
            )),
        }
        sampled, seconds, steps = sample_stage(
            "dit_hr_global", stage, refine_sigmas, len(refine_sigmas) - 1,
            conditioning_override=conditioning_override, seed_override=seed,
            sampler_override=sampler_override,
        )
        out_video, out_audio = sampled["samples"].tensors
        if not torch.isfinite(out_video).all() or tuple(out_audio.shape) != tuple(audio.shape):
            raise RuntimeErrorCode("hr_refine_global_invalid_output")
        timings["hr_refine_mode"] = profile
        timings["hr_refine_resize_shape"] = list(video.shape)
        timings["hr_refine_audio_shape"] = list(audio.shape)
        timings["hr_refine_tile_count"] = 1
        timings["hr_refine_tile_seconds"] = [seconds]
        timings["dit_hr_global_seconds"] = seconds
        timings["hr_refine_step_seconds"] = list(steps)
        timings["hr_refine_sigma"] = [0.16, 0.0]
        timings["hr_refine_steps"] = len(refine_sigmas) - 1
        timings["hr_refine_sigma"] = [float(value) for value in refine_sigmas.detach().cpu().tolist()]
        return sampled

    @torch.no_grad()
    def generate(self, request: dict, files: dict[str, list[Path]], output: Path, set_stage):
        # Apply in the actual worker thread. CPU copies of reference latents
        # otherwise retain autograd graphs and their CUDA activations across
        # images and temporal chunks. Use no_grad: Comfy quantized parameters
        # currently fail during CPU offload under inference_mode.
        self._active_stage = "starting"
        self._active_timings: dict[str, object] = {}
        try:
            return self._generate(request, files, output, set_stage)
        except torch.cuda.OutOfMemoryError as exc:
            stage = getattr(self, "_active_stage", "unknown")
            metrics = dict(getattr(self, "_active_timings", {}))
            metrics["failure_stage"] = stage
            metrics["failure_gpu"] = self._gpu_snapshot()
            if hasattr(self, "_active_started"):
                metrics["runtime_total_seconds"] = time.perf_counter() - self._active_started
            raise RuntimeErrorCode(f"{stage}_out_of_memory", metrics=metrics) from exc
        except Exception as exc:
            if getattr(exc, "runtime_metrics", None) is None:
                metrics = dict(getattr(self, "_active_timings", {}))
                metrics["failure_stage"] = getattr(self, "_active_stage", "unknown")
                metrics["failure_exception"] = f"{type(exc).__name__}: {exc}"
                if hasattr(self, "_active_started"):
                    metrics["runtime_total_seconds"] = time.perf_counter() - self._active_started
                setattr(exc, "runtime_metrics", metrics)
            raise
        finally:
            previous = getattr(self, "_lowvram_previous_state", None)
            if previous:
                from .lowvram import restore_comfy_vram_state

                restore_comfy_vram_state(previous)
                self._lowvram_previous_state = None

    def _generate(self, request: dict, files: dict[str, list[Path]], output: Path, set_stage):
        import comfy_extras.nodes_custom_sampler as custom_sampler
        import comfy_extras.nodes_minimax_h3 as h3_nodes
        import nodes

        if request.get("task") != "ref2va":
            raise RuntimeErrorCode("unsupported_task")
        target = request.get("target") or {}
        width = int(target.get("width") or 768)
        height = int(target.get("height") or 1344)
        if width != 768 or height != 1344:
            raise RuntimeErrorCode("unsupported_geometry")
        steps = int(request.get("num_inference_steps", 4))
        if steps != 4:
            raise RuntimeErrorCode("singularity_requires_4_nfe")
        quality_profile = str(request.get("quality_profile", "du-0"))
        if quality_profile not in {"du-0", "du-1", "du-2", "du-3", "hr-refine-tile-v1", "hr-refine-global-v2", "hybrid-4+3", "hybrid-4+2"}:
            raise RuntimeErrorCode("unsupported_quality_profile")
        if quality_profile in {"hr-refine-tile-v1", "hr-refine-global-v2", "hybrid-4+3", "hybrid-4+2"} and self.hr_refine_mode != quality_profile:
            raise RuntimeErrorCode("hr_refine_profile_unavailable")
        if quality_profile == "du-0" and self.dual_sample_mode != "off":
            raise RuntimeErrorCode("dual_sample_profile_required")
        if quality_profile not in {"du-0", "hr-refine-tile-v1", "hr-refine-global-v2", "hybrid-4+3", "hybrid-4+2"} and self.dual_sample_mode != quality_profile:
            raise RuntimeErrorCode("dual_sample_profile_unavailable")
        # DU-3 uses the 640x1120 latent grid for the first 2 NFE, then
        # upscales [70,40] to [84,48] before the final 2 NFE at 768x1344.
        generation_width = 640 if quality_profile in {"du-3", "hr-refine-tile-v1", "hr-refine-global-v2", "hybrid-4+3", "hybrid-4+2"} else width
        generation_height = 1120 if quality_profile in {"du-3", "hr-refine-tile-v1", "hr-refine-global-v2", "hybrid-4+3", "hybrid-4+2"} else height
        seed = int(request.get("seed", 7))
        if not 0 <= seed <= 0xFFFFFFFF:
            raise RuntimeErrorCode("invalid_seed")
        self._active_stage = "reference_decode"
        decode_started = time.perf_counter()
        images = [load_image(path) for path in files.get("images", [])]
        videos = []
        video_audios = {}
        video_scale = float(os.environ.get("SINGULARITY_REFERENCE_VIDEO_SCALE", "0.5"))
        if not 0.25 <= video_scale <= 1.0:
            raise RuntimeErrorCode("invalid_reference_video_scale")
        for index, path in enumerate(files.get("videos", [])):
            frames, audio, _fps = load_video(path)
            if video_scale < 1.0:
                scaled_height = max(16, int(frames.shape[1] * video_scale) // 16 * 16)
                scaled_width = max(16, int(frames.shape[2] * video_scale) // 16 * 16)
                frames = torch.nn.functional.interpolate(
                    frames.permute(0, 3, 1, 2), size=(scaled_height, scaled_width), mode="bilinear", align_corners=False
                ).permute(0, 2, 3, 1).contiguous()
            videos.append(frames)
            if audio is not None:
                video_audios[f"ref_video_audio_{index}"] = audio
        audios = [load_audio(path) for path in files.get("audios", [])]
        timings = {
            "reference_decode_seconds": time.perf_counter() - decode_started,
            "reference_image_count": len(images),
            "reference_video_count": len(videos),
            "reference_video_scale": video_scale,
            "reference_audio_count": len(audios),
            "vae_chunked_io": True,
            "vae_tile_size": self.vae_tile_size,
            "vae_decoder_tile_size": self.vae_decoder_tile_size,
        }
        self._active_timings = timings
        if not images and not videos:
            raise RuntimeErrorCode("reference_required")
        set_stage("warming")
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        self._active_started = started
        model = h3_nodes.MiniMaxH3SigmaShift.execute(self.model, 12.0, 3.0)[0]
        lowvram_state = None
        attention_metrics = {}
        if self.lowvram_mode == "auto" or quality_profile != "du-0":
            from .lowvram import configure_comfy_low_vram, make_h3_lowvram_attention

            lowvram_state = configure_comfy_low_vram(self.lowvram_state)
            self._lowvram_previous_state = lowvram_state.get("previous_vram_state")
            model.set_model_optimized_attention(make_h3_lowvram_attention(attention_metrics))
            timings["attention_backend"] = "h3_lowvram_sdpa"
            timings["low_vram"] = lowvram_state
            timings["attention_metrics"] = attention_metrics
        timings["model_load_seconds"] = self.model_load_seconds
        timings["lora_patch_seconds"] = self.lora_patch_seconds
        self._active_stage = "vae_reference_encode"
        timings["offload_before_reference_seconds"] = self._release_comfy_models()
        encoded_started = time.perf_counter()
        original_video_encode = self.vae.encode
        original_audio_encode = self.audio_vae.encode
        vae_image_seconds = 0.0
        vae_video_seconds = 0.0
        audio_vae_seconds = 0.0
        vae_stage_released = False

        def timed_video_vae_encode(pixels, *args, **kwargs):
            nonlocal vae_image_seconds, vae_video_seconds, vae_stage_released
            call_started = time.perf_counter()
            try:
                # The H3 reference node encodes text and references in one
                # graph call.  Its Qwen encoder can therefore remain in the
                # Comfy cache when the first VAE tile starts.  Conditioning
                # has already been copied to the intermediate device, so a
                # hard stage boundary is safe and keeps GPU1 available for
                # the INT8 ConvRot VAE.
                if not vae_stage_released:
                    timings["offload_before_vae_seconds"] = self._release_comfy_models()
                    timings["clip_to_cpu_seconds"] = self._release_clip_to_cpu()
                    vae_stage_released = True
                return original_video_encode(pixels, *args, **kwargs)
            finally:
                elapsed = time.perf_counter() - call_started
                if pixels.ndim >= 4 and pixels.shape[0] > 1:
                    vae_video_seconds += elapsed
                else:
                    vae_image_seconds += elapsed
                timings["vae_reference_image_seconds"] = vae_image_seconds
                timings["vae_reference_video_seconds"] = vae_video_seconds

        def timed_audio_vae_encode(waveform, *args, **kwargs):
            nonlocal audio_vae_seconds
            call_started = time.perf_counter()
            try:
                return original_audio_encode(waveform, *args, **kwargs)
            finally:
                audio_vae_seconds += time.perf_counter() - call_started
                timings["audio_vae_reference_seconds"] = audio_vae_seconds

        original_clip_encode = self.clip.encode_from_tokens_scheduled
        text_encoder_seconds = 0.0

        def timed_clip_encode(*args, **kwargs):
            nonlocal text_encoder_seconds
            self._active_stage = "text_encoder"
            call_started = time.perf_counter()
            try:
                return original_clip_encode(*args, **kwargs)
            finally:
                text_encoder_seconds += time.perf_counter() - call_started
                timings["text_encoder_seconds"] = text_encoder_seconds

        self.vae.encode = timed_video_vae_encode
        self.audio_vae.encode = timed_audio_vae_encode
        seconds = float(request.get("seconds", 15))
        # The frozen 15-second fixture is 345 frames (14.375 s), already on
        # H3's 17k+5 temporal grid. Preserve that exact comparison geometry.
        length = 345 if seconds == 15 else int(round(seconds * 24))
        try:
            # ProjectedCLIP forwards ordinary attribute writes to _base;
            # shadow the wrapper method itself so Qwen timing/stage is real.
            object.__setattr__(self.clip, "encode_from_tokens_scheduled", timed_clip_encode)
            conditioning, latent = h3_nodes.MiniMaxH3ReferenceToVideo.execute(
                self.clip, request["prompt"], generation_width, generation_height, length, "match", self.vae, self.audio_vae,
                {f"ref_image_{i}": image for i, image in enumerate(images)},
                {f"ref_video_{i}": video for i, video in enumerate(videos)},
                video_audios, {f"ref_audio_{i}": audio for i, audio in enumerate(audios)},
            )
        finally:
            self.vae.encode = original_video_encode
            self.audio_vae.encode = original_audio_encode
            object.__setattr__(self.clip, "encode_from_tokens_scheduled", original_clip_encode)
        timings["reference_and_text_encoder_seconds"] = time.perf_counter() - encoded_started
        timings["vae_reference_image_seconds"] = vae_image_seconds
        timings["vae_reference_video_seconds"] = vae_video_seconds
        timings["audio_vae_reference_seconds"] = audio_vae_seconds
        timings["text_encoder_seconds"] = text_encoder_seconds
        timings["gpu_after_reference"] = self._gpu_snapshot()
        refine_condition = None
        if quality_profile in {"hybrid-4+3", "hybrid-4+2"}:
            # CGlide's refine pass re-encodes image references at a larger
            # canvas while keeping video references at their base geometry.
            # The pinned upstream H3 node has no ref_refine_scale input, so
            # request a 1.5x internal image canvas and retain only its
            # conditioning output. The returned high-resolution empty latent
            # is deliberately discarded.
            self._active_stage = "refine_condition_encode"
            refine_started = time.perf_counter()
            refine_width = int(round(generation_width * 1.5 / 32.0) * 32)
            refine_height = int(round(generation_height * 1.5 / 32.0) * 32)
            refine_condition, _ = h3_nodes.MiniMaxH3ReferenceToVideo.execute(
                self.clip, request["prompt"], refine_width, refine_height, length, "match", self.vae,
                self.audio_vae,
                {f"ref_image_{i}": image for i, image in enumerate(images)},
                {f"ref_video_{i}": video for i, video in enumerate(videos)},
                video_audios, {f"ref_audio_{i}": audio for i, audio in enumerate(audios)},
            )
            timings["refine_condition_scale"] = 1.5
            timings["refine_condition_geometry"] = [refine_width, refine_height]
            timings["refine_condition_seconds"] = time.perf_counter() - refine_started
            timings["refine_condition_image_count"] = len(images)
            timings["refine_condition_video_count"] = len(videos)
        self._active_timings = timings
        self._active_stage = "dit"
        timings["offload_before_dit_seconds"] = self._release_comfy_models()
        set_stage("generating")
        sampler = custom_sampler.KSamplerSelect.execute("euler")[0]
        sigmas = custom_sampler.BasicScheduler.execute(model, "simple", 4, 1.0)[0]
        low_sigmas, high_sigmas = self._split_dual_sigmas(sigmas)
        import latent_preview
        import comfy.nested_tensor

        def sample_stage(stage_name, stage_latent, stage_sigmas, expected_steps,
                         conditioning_override=None, seed_override=None, sampler_override=None):
            self._active_stage = stage_name
            guider = custom_sampler.BasicGuider.execute(
                model, conditioning if conditioning_override is None else conditioning_override
            )[0]
            noise = custom_sampler.RandomNoise.execute(seed if seed_override is None else seed_override)[0]
            started_stage = time.perf_counter()
            step_times = []
            original_prepare_callback = latent_preview.prepare_callback

            def timed_prepare_callback(*args, **kwargs):
                callback = original_prepare_callback(*args, **kwargs)
                previous = time.perf_counter()

                def timed_callback(step, x0, x, total_steps):
                    nonlocal previous
                    current = time.perf_counter()
                    step_times.append(current - previous)
                    previous = current
                    return callback(step, x0, x, total_steps)

                return timed_callback

            latent_preview.prepare_callback = timed_prepare_callback
            try:
                result = custom_sampler.SamplerCustomAdvanced.execute(
                    noise,
                    guider,
                    sampler if sampler_override is None else sampler_override,
                    stage_sigmas,
                    stage_latent,
                )[0]
            finally:
                latent_preview.prepare_callback = original_prepare_callback
            if len(step_times) != expected_steps:
                raise RuntimeErrorCode(
                    f"{stage_name}_step_metrics_unavailable:{len(step_times)}"
                )
            return result, time.perf_counter() - started_stage, step_times

        if quality_profile == "du-0":
            sampled, dit_seconds, step_times = sample_stage("dit", latent, sigmas, 4)
            timings["dit_seconds"] = dit_seconds
            timings["dit_step_seconds"] = list(step_times)
            timings["dual_sample_enabled"] = False
        elif quality_profile in {"hybrid-4+3", "hybrid-4+2"}:
            # The first pass remains the proven 4-NFE low-grid path. The
            # second pass is a separate low-noise scheduler window, equivalent
            # to ComfyUI's simple/3-step/denoise=0.3 sampler. It receives the
            # larger image-reference conditioning and never resamples audio.
            sampled_low, low_seconds, low_step_times = sample_stage(
                "dit_low_resolution", latent, sigmas, 4
            )
            self._active_stage = "dit_hr_global"
            timings["offload_before_hr_refine_seconds"] = self._release_comfy_models()
            previous_q_chunk = os.environ.get("SINGULARITY_LOW_VRAM_Q_CHUNK")
            hr_q_chunk = os.environ.get("SINGULARITY_HR_Q_CHUNK", "2048")
            if hr_q_chunk not in {"1024", "2048", "4096", "8192"}:
                raise RuntimeErrorCode("invalid_hr_q_chunk")
            os.environ["SINGULARITY_LOW_VRAM_Q_CHUNK"] = hr_q_chunk
            try:
                from .lowvram import make_h3_lowvram_attention

                model.set_model_optimized_attention(
                    make_h3_lowvram_attention(timings.get("attention_metrics", {}))
                )
                hr_steps = 2 if quality_profile == "hybrid-4+2" else 3
                hr_denoise = float(os.environ.get("SINGULARITY_HR_DENOISE", "0.3"))
                if not 0.0 < hr_denoise <= 1.0:
                    raise RuntimeErrorCode("invalid_hr_denoise")
                hr_sampler = custom_sampler.KSamplerSelect.execute(self.hr_sampler)[0]
                hr_sigmas = custom_sampler.BasicScheduler.execute(
                    model, self.hr_scheduler, hr_steps, hr_denoise
                )[0]
                sampled = self._sample_hr_global(
                    conditioning, sampled_low, hr_sigmas, sample_stage, seed, timings,
                    conditioning_override=refine_condition, profile=quality_profile,
                    sampler_override=hr_sampler,
                )
            finally:
                if previous_q_chunk is None:
                    os.environ.pop("SINGULARITY_LOW_VRAM_Q_CHUNK", None)
                else:
                    os.environ["SINGULARITY_LOW_VRAM_Q_CHUNK"] = previous_q_chunk
            timings["dit_low_seconds"] = low_seconds
            timings["dit_hr_seconds"] = sum(timings.get("hr_refine_tile_seconds", []))
            timings["dit_hr_seconds"] = timings.get("dit_hr_global_seconds", timings["dit_hr_seconds"])
            timings["dit_seconds"] = low_seconds + timings["dit_hr_seconds"]
            timings["dit_step_seconds"] = list(low_step_times) + list(timings.get("hr_refine_tile_seconds", []))
            timings["dual_sample_enabled"] = False
            timings["hr_refine_enabled"] = True
            timings["quality_profile"] = quality_profile
            timings["hr_q_chunk"] = int(hr_q_chunk)
            timings["hr_refine_denoise"] = hr_denoise
            timings["hr_sampler"] = self.hr_sampler
            timings["hr_scheduler"] = self.hr_scheduler
        elif quality_profile in {"hr-refine-tile-v1", "hr-refine-global-v2"}:
            # HR profile deliberately spends the full 4 NFE budget on the
            # low-grid structure pass, then applies only low-noise tile steps
            # after deterministic H3 latent resizing.
            sampled_low, low_seconds, low_step_times = sample_stage(
                "dit_low_resolution", latent, sigmas, 4
            )
            timings["dit_low_seconds"] = low_seconds
            timings["dit_low_step_seconds"] = list(low_step_times)
            self._active_stage = "dit_hr_resize"
            timings["offload_before_hr_refine_seconds"] = self._release_comfy_models()
            previous_q_chunk = os.environ.get("SINGULARITY_LOW_VRAM_Q_CHUNK")
            os.environ["SINGULARITY_LOW_VRAM_Q_CHUNK"] = "2048"
            try:
                from .lowvram import make_h3_lowvram_attention

                model.set_model_optimized_attention(
                    make_h3_lowvram_attention(timings.get("attention_metrics", {}))
                )
                if quality_profile == "hr-refine-global-v2":
                    global_sigmas = torch.tensor([0.16, 0.0], device=sigmas.device, dtype=sigmas.dtype)
                    sampled = self._sample_hr_global(
                        conditioning, sampled_low, global_sigmas, sample_stage, seed, timings,
                    )
                else:
                    sampled = self._sample_hr_tiles(
                        model, conditioning, sampled_low, sampler, seed, sigmas,
                        sample_stage, timings,
                    )
            finally:
                if previous_q_chunk is None:
                    os.environ.pop("SINGULARITY_LOW_VRAM_Q_CHUNK", None)
                else:
                    os.environ["SINGULARITY_LOW_VRAM_Q_CHUNK"] = previous_q_chunk
            timings["dit_hr_seconds"] = sum(timings.get("hr_refine_tile_seconds", []))
            timings["dit_seconds"] = low_seconds + timings["dit_hr_seconds"]
            timings["dit_step_seconds"] = list(low_step_times) + list(timings.get("hr_refine_tile_seconds", []))
            timings["dual_sample_enabled"] = False
            timings["hr_refine_enabled"] = True
            timings["quality_profile"] = quality_profile
        else:
            sampled_low, low_seconds, low_step_times = sample_stage(
                "dit_low_resolution", latent, low_sigmas, 2
            )
            low_samples = sampled_low.get("samples")
            if not getattr(low_samples, "is_nested", False) or len(low_samples.tensors) != 2:
                raise RuntimeErrorCode("dual_sample_invalid_av_latent")
            low_video, low_audio = low_samples.tensors
            scale = {"du-1": 1.5, "du-2": 1.25, "du-3": 1.2}[quality_profile]
            timings["dual_sample_scale"] = scale
            timings["low_resolution_latent_shape"] = list(low_video.shape)
            self._active_stage = "latent_upscale"
            timings["offload_before_upscale_seconds"] = self._release_comfy_models()
            upscaled_video, upscale_seconds = self._upscale_video_latent(low_video, scale)
            timings["latent_upscale_seconds"] = upscale_seconds
            timings["upscaled_latent_shape"] = list(upscaled_video.shape)
            high_latent = {
                "samples": comfy.nested_tensor.NestedTensor((
                    upscaled_video,
                    low_audio.to(upscaled_video.device).contiguous(),
                ))
            }
            timings["offload_before_refine_seconds"] = self._release_comfy_models()
            sampled, high_seconds, high_step_times = sample_stage(
                "dit_high_resolution", high_latent, high_sigmas, 2
            )
            timings["dit_low_seconds"] = low_seconds
            timings["dit_high_seconds"] = high_seconds
            timings["dit_seconds"] = low_seconds + high_seconds
            timings["dit_step_seconds"] = list(low_step_times) + list(high_step_times)
            timings["dual_sample_enabled"] = True
            timings["dual_sample_profile"] = quality_profile
        timings["gpu_after_dit"] = self._gpu_snapshot()
        self._active_stage = "vae_decode"
        timings["offload_before_decode_seconds"] = self._release_comfy_models()
        decode_started = time.perf_counter()
        try:
            # Comfy's H3 VAE implementation routes both encode and decode
            # through the same ``tile_size`` field.  Some newer variants expose
            # separate decoder fields, but the pinned runtime ignores those
            # fields and reads ``tile_size`` from ``tiled_decode``.  Switch the
            # actual fields for the decode window and restore the memory-safe
            # encoder geometry after the frames have been produced.
            video_vae = self._video_vae_model
            previous_tile = getattr(video_vae, "tile_size", None)
            previous_overlap = getattr(video_vae, "tile_overlap_min", None)
            if getattr(video_vae, "comfy_has_chunked_io", False):
                video_vae.tile_size = self.vae_decoder_tile_size
                video_vae.tile_overlap_min = self._vae_decoder_overlap
                if hasattr(video_vae, "decoder_tile_size"):
                    video_vae.decoder_tile_size = self.vae_decoder_tile_size
                if hasattr(video_vae, "decoder_tile_overlap_min"):
                    video_vae.decoder_tile_overlap_min = self._vae_decoder_overlap
            # Under Comfy's NO_VRAM state either VAE can retain fused
            # normalization weights on CPU after the DiT stage has released
            # its patcher. Explicitly force-load both VAE patchers on GPU1
            # before decoding so weights and latents share a device. This is
            # a stage-local residency request, not a second offload system.
            import comfy.model_management as model_management

            # NO_VRAM is needed while the DiT competes with token-refiner
            # activations. Once DiT is released, VAE decoding has a separate
            # GPU1 budget; NORMAL_VRAM prevents Comfy from leaving fused VAE
            # normalization weights on CPU while their activations run on
            # cuda:1.
            vram_states = getattr(model_management, "VRAMState", None)
            if vram_states is not None and hasattr(vram_states, "NORMAL_VRAM"):
                model_management.vram_state = vram_states.NORMAL_VRAM
                timings["vae_vram_state"] = "NORMAL_VRAM"
            vae_patchers = [patcher for patcher in (
                getattr(self.vae, "patcher", None),
                getattr(self.audio_vae, "patcher", None),
            ) if patcher is not None]
            model_management.load_models_gpu(vae_patchers, force_full_load=True)
            timings["vae_force_full_load"] = True
            frames = nodes.VAEDecode().decode(self.vae, sampled)[0]
            audio_patcher = getattr(self.audio_vae, "patcher", None)
            timings["audio_vae_force_full_load"] = audio_patcher is not None
            audio = __import__("comfy_extras.nodes_audio", fromlist=["VAEDecodeAudio"]).VAEDecodeAudio.execute(self.audio_vae, sampled)[0]
        finally:
            if getattr(video_vae, "comfy_has_chunked_io", False):
                video_vae.tile_size = previous_tile
                video_vae.tile_overlap_min = previous_overlap
                if hasattr(video_vae, "decoder_tile_size"):
                    video_vae.decoder_tile_size = self.vae_decoder_tile_size
                if hasattr(video_vae, "decoder_tile_overlap_min"):
                    video_vae.decoder_tile_overlap_min = self._vae_decoder_overlap
            timings["video_audio_vae_seconds"] = time.perf_counter() - decode_started
            timings["gpu_after_decode"] = self._gpu_snapshot()
        # Comfy's VAEDecode normally flattens video batches to [T,H,W,3],
        # while older H3 nodes can return [B,C,T,H,W] or [B,T,H,W,3].
        # Normalize all supported layouts before crossing into the CPU muxer;
        # keeping this boundary explicit also prevents a CUDA tensor from
        # surviving after the VAE stage.
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
        timings["decoded_video_shape"] = list(frames.shape)
        if isinstance(audio, dict) and isinstance(audio.get("waveform"), torch.Tensor):
            timings["decoded_audio_shape"] = list(audio["waveform"].shape)
        set_stage("saving")
        mux_started = time.perf_counter()
        media = mux_mp4(frames, audio, output)
        timings["video_encoding_seconds"] = time.perf_counter() - mux_started
        timings["runtime_total_seconds"] = time.perf_counter() - started
        if torch.cuda.is_available():
            timings["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            timings["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        timings["sequence_frames"] = int(media["frames"])
        timings["nfe"] = 4
        timings["seed"] = seed
        timings["sigma_points"] = 5
        self._active_timings = timings
        return {
            "content_sha256": sha256(output),
            "media": media,
            "runtime_metrics": timings,
            "model": self.diffusion_name,
            "lora": self.lora_name,
            "execution_instance_id": self.execution_instance_id,
        }
