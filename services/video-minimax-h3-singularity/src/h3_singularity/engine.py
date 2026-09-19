"""ComfyUI H3 execution adapter for the pinned Singularity Ref2VA model."""

from __future__ import annotations

import os
from pathlib import Path
import time
import uuid

import torch

from .media import load_audio, load_image, load_video, mux_mp4, sha256


class RuntimeErrorCode(RuntimeError):
    def __init__(self, code: str, *, metrics: dict | None = None):
        super().__init__(code)
        self.code = code
        self.runtime_metrics = metrics


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
        self.clip_name = os.environ.get("H3_CLIP_NAME", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
        self.video_vae_name = os.environ.get("H3_VIDEO_VAE_NAME", "minimax_h3_video_vae_fp16.safetensors")
        self.audio_vae_name = os.environ.get("H3_AUDIO_VAE_NAME", "minimax_h3_audio_vae_fp32.safetensors")
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
        return folder_paths, nodes

    def _load_components(self):
        if not torch.cuda.is_available():
            raise RuntimeErrorCode("cuda_unavailable")
        evidence = self.model_root / "model-evidence.json"
        if not evidence.is_file():
            raise RuntimeErrorCode("singularity_model_not_verified")
        component_evidence = self.component_root / "component-model-evidence.json"
        if not component_evidence.is_file():
            raise RuntimeErrorCode("h3_components_not_verified")
        self.folder_paths, self.nodes = self._configure_comfy()
        started = time.perf_counter()
        diffusion_path = self.model_root / self.diffusion_name
        lora_path = self.model_root / "loras" / self.lora_name
        for path in (diffusion_path, lora_path):
            if not path.is_file():
                raise RuntimeErrorCode(f"model_file_missing:{path.name}")
        model_started = time.perf_counter()
        self.model = self.nodes.UNETLoader().load_unet(self.diffusion_name, "default")[0]
        self.model_load_seconds = time.perf_counter() - model_started
        lora_started = time.perf_counter()
        self.model = self.nodes.LoraLoaderModelOnly().load_lora_model_only(self.model, self.lora_name, 1.0)[0]
        self.lora_patch_seconds = time.perf_counter() - lora_started
        self.clip = self.nodes.CLIPLoader().load_clip(self.clip_name, "minimax")[0]
        self.vae = self.nodes.VAELoader().load_vae(self.video_vae_name)[0]
        self.audio_vae = self.nodes.VAELoader().load_vae(self.audio_vae_name)[0]
        self.load_seconds = time.perf_counter() - started
        self.version = os.environ.get("SINGULARITY_RUNTIME_VERSION", "video-minimax-h3-singularity-v0.1.1")
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
                "video_vae": self.video_vae_name,
                "audio_vae": self.audio_vae_name,
            },
            "nfe": 4,
            "gpu_count": torch.cuda.device_count(),
            "cpu_offload": True,
        }

    def generate(self, request: dict, files: dict[str, list[Path]], output: Path, set_stage):
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
                if hasattr(self, "_active_started"):
                    metrics["runtime_total_seconds"] = time.perf_counter() - self._active_started
                setattr(exc, "runtime_metrics", metrics)
            raise

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
        seed = int(request.get("seed", 7))
        if not 0 <= seed <= 0xFFFFFFFF:
            raise RuntimeErrorCode("invalid_seed")
        self._active_stage = "reference_decode"
        decode_started = time.perf_counter()
        images = [load_image(path) for path in files.get("images", [])]
        videos = []
        video_audios = {}
        for index, path in enumerate(files.get("videos", [])):
            frames, audio, _fps = load_video(path)
            videos.append(frames)
            if audio is not None:
                video_audios[f"ref_video_audio_{index}"] = audio
        audios = [load_audio(path) for path in files.get("audios", [])]
        timings = {
            "reference_decode_seconds": time.perf_counter() - decode_started,
            "reference_image_count": len(images),
            "reference_video_count": len(videos),
            "reference_audio_count": len(audios),
            "vae_chunked_io": True,
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

        def timed_video_vae_encode(pixels, *args, **kwargs):
            nonlocal vae_image_seconds, vae_video_seconds
            call_started = time.perf_counter()
            try:
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
            self.clip.encode_from_tokens_scheduled = timed_clip_encode
            conditioning, latent = h3_nodes.MiniMaxH3ReferenceToVideo.execute(
                self.clip, request["prompt"], width, height, length, "match", self.vae, self.audio_vae,
                {f"ref_image_{i}": image for i, image in enumerate(images)},
                {f"ref_video_{i}": video for i, video in enumerate(videos)},
                video_audios, {f"ref_audio_{i}": audio for i, audio in enumerate(audios)},
            )
        finally:
            self.vae.encode = original_video_encode
            self.audio_vae.encode = original_audio_encode
            self.clip.encode_from_tokens_scheduled = original_clip_encode
        timings["reference_and_text_encoder_seconds"] = time.perf_counter() - encoded_started
        timings["vae_reference_image_seconds"] = vae_image_seconds
        timings["vae_reference_video_seconds"] = vae_video_seconds
        timings["audio_vae_reference_seconds"] = audio_vae_seconds
        timings["text_encoder_seconds"] = text_encoder_seconds
        timings["gpu_after_reference"] = self._gpu_snapshot()
        self._active_timings = timings
        self._active_stage = "dit"
        timings["offload_before_dit_seconds"] = self._release_comfy_models()
        set_stage("generating")
        sampler = custom_sampler.KSamplerSelect.execute("euler")[0]
        sigmas = custom_sampler.BasicScheduler.execute(model, "simple", 4, 1.0)[0]
        guider = custom_sampler.BasicGuider.execute(model, conditioning)[0]
        noise = custom_sampler.RandomNoise.execute(seed)[0]
        dit_started = time.perf_counter()
        step_times = []
        import latent_preview
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
            sampled = custom_sampler.SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent)[0]
        finally:
            latent_preview.prepare_callback = original_prepare_callback
            timings["dit_seconds"] = time.perf_counter() - dit_started
            timings["dit_step_seconds"] = list(step_times)
        timings["gpu_after_dit"] = self._gpu_snapshot()
        if len(step_times) != 4:
            raise RuntimeErrorCode(f"dit_step_metrics_unavailable:{len(step_times)}")
        self._active_stage = "vae_decode"
        timings["offload_before_decode_seconds"] = self._release_comfy_models()
        decode_started = time.perf_counter()
        try:
            frames = nodes.VAEDecode().decode(self.vae, sampled)[0]
            audio = __import__("comfy_extras.nodes_audio", fromlist=["VAEDecodeAudio"]).VAEDecodeAudio.execute(self.audio_vae, sampled)[0]
        finally:
            timings["video_audio_vae_seconds"] = time.perf_counter() - decode_started
            timings["gpu_after_decode"] = self._gpu_snapshot()
        if frames.ndim == 5:
            frames = frames[0].permute(1, 2, 3, 0).contiguous()
        elif frames.ndim == 4 and frames.shape[1] == 3:
            frames = frames.permute(0, 2, 3, 1).contiguous()
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
