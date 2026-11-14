"""GPU-resident H3 latent enlargement and optional explicit H3 resampling."""
import gc
import json
import re
import time
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from .resources import ResourceError, confined_file, sha256
from .tensors import load_av, require_single_4090, decoded_video_frames
from .conditioning import load_conditioning


def configure_comfy():
    from comfy.cli_args import args
    # Comfy's ordinary smart-memory policy may stream weights. This runtime
    # explicitly requires full loading and disables the dynamic VRAM allocator.
    args.highvram = True
    args.disable_dynamic_vram = True
    args.disable_smart_memory = True
    import comfy.model_management as mm
    original = mm.load_models_gpu
    def resident(models, *positional, **kwargs):
        kwargs['force_full_load'] = True
        result = original(models, *positional, **kwargs)
        for model in models:
            if model.loaded_size() < model.model_size():
                raise ResourceError('cpu_offload_forbidden')
        return result
    mm.load_models_gpu = resident
    # The pinned VAE catches OOM and silently retries with different tiles.
    # A capacity profile must execute exactly its selected strategy.
    def reject_fallback(error):
        raise error
    mm.raise_non_oom = reject_fallback


def release_stage():
    import comfy.model_management as mm
    mm.unload_all_models()
    gc.collect()
    torch.cuda.empty_cache()


class Engine:
    def __init__(self, models, profile_path, lock_path):
        self.models = Path(models)
        self.gpu = require_single_4090()
        self.lock = json.loads(Path(lock_path).read_text())
        self.profile = json.loads(Path(profile_path).read_text())
        if self.profile.get('gpu_uuid') != self.gpu['uuid'] or self.profile.get('cpu_offload') is not False:
            raise ResourceError('capacity_profile_mismatch')
        if self.profile.get('mode') not in {'latent_only', 'latent_refine'}:
            raise ResourceError('unsupported_profile')
        if self.profile.get('backend_lock_sha256') != sha256(Path(lock_path)):
            raise ResourceError('capacity_profile_mismatch')
        if not re.fullmatch(r'[0-9a-f]{64}', self.profile.get('source_manifest_sha256', '')):
            raise ResourceError('capacity_profile_mismatch')
        self.weights = {}
        needed = ['upscaler', 'video_vae'] + (['refiner'] if self.profile['mode'] == 'latent_refine' else [])
        for name in needed:
            item = self.lock[name]
            path = confined_file(self.models, item['file'])
            if path.stat().st_size != item['size'] or sha256(path) != item['sha256']:
                raise ResourceError('model_integrity_failed')
            self.weights[name] = path
        configure_comfy()

    def validate_request(self, resources, request):
        p = self.profile
        # Initial profiles cover the exact calibrated source, including all
        # conditioning hashes. Equal frame dimensions alone prove no capacity.
        if resources.get('manifest_sha256') != p['source_manifest_sha256']:
            raise ResourceError('capacity_profile_mismatch')
        manifest = resources['manifest']
        if 'seed' in request and (type(request['seed']) is not int or not 0 <= request['seed'] <= 4294967295):
            raise ResourceError('invalid_seed')
        if request.get('profile_id', p['id']) != p['id']:
            raise ResourceError('unsupported_profile')
        media = manifest['media']
        if manifest['source_route'] not in p['source_routes']:
            raise ResourceError('incompatible_source_route')
        for key in ('width', 'height', 'frames'):
            if media[key] != p['source_' + key]:
                raise ResourceError('capacity_profile_mismatch')
        width, height = request.get('target_width', p['target_width']), request.get('target_height', p['target_height'])
        if (width, height) != (p['target_width'], p['target_height']):
            raise ResourceError('capacity_profile_mismatch')
        if width * media['height'] != height * media['width'] or width <= media['width']:
            raise ResourceError('invalid_target_geometry')
        return width, height

    def generate(self, resources, request, directory):
        from .vendor import upscaler_3d as upstream
        from .media import finish_video
        import comfy.sd
        import comfy.utils
        width, height = self.validate_request(resources, request)
        directory = Path(directory)
        video, audio = load_av(resources)
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        state = load_file(str(self.weights['upscaler']), device='cpu')
        state = upstream._extract_upscaler_sd(state)
        cfg = upstream._detect_arch(state)
        # Construct directly in desired CPU dtype before a single GPU transfer.
        old_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float16)
            model = upstream.LatentResizer3D(in_channels=cfg['in_channels'], in_blocks=cfg['in_blocks'],
                out_blocks=cfg['out_blocks'], channels=cfg['channels'], dropout=cfg['dropout'],
                attn=cfg['attn'], temporal_every=cfg['temporal_every'], temporal_kernel=cfg['temporal_kernel'])
        finally:
            torch.set_default_dtype(old_dtype)
        model.load_state_dict(state, strict=True)
        del state
        model.eval().requires_grad_(False).to('cuda')
        with torch.inference_mode():
            mean, std = upstream._make_norm_tensors('cuda', torch.float16)
            normalized = (video.to('cuda', dtype=torch.float16) - mean) / std
            resized = model(normalized, scale=width / resources['manifest']['media']['width'],
                            target_size=(video.shape[2], height // 16, width // 16),
                            enable_chunking=self.profile['upscaler_temporal_chunking'])
            resized = (resized * std + mean).cpu().contiguous()
        del model, mean, std, normalized
        release_stage()
        if not torch.isfinite(resized).all():
            raise ResourceError('nonfinite_upscaled_latent')
        save_file({'samples': resized}, directory / 'upscaled.safetensors')
        metrics = {'latent_upscale_seconds': time.monotonic() - started}
        if self.profile['mode'] == 'latent_refine':
            refined_at = time.monotonic()
            resized = self.refine(resized, audio, resources, request)
            save_file({'samples': resized}, directory / 'refined.safetensors')
            metrics['refine_seconds'] = time.monotonic() - refined_at
        decode_at = time.monotonic()
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(self.weights['video_vae']), safe_load=True))
        with torch.inference_mode():
            frames = vae.decode(resized)
        del vae
        release_stage()
        frames = decoded_video_frames(frames, count=resources['manifest']['media']['frames'],
                                      width=width, height=height)
        media = finish_video(frames, resources['files']['source_video'], directory)
        metrics['vae_and_mux_seconds'] = time.monotonic() - decode_at
        metrics['runtime_total_seconds'] = time.monotonic() - started
        metrics['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        metrics['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        return {'source_video_task_id': request['source_video_task_id'], 'media': media,
                'profile_id': self.profile['id'], 'runtime_metrics': metrics,
                'content_sha256': sha256(directory / 'output.mp4'),
                'preview_sha256': sha256(directory / 'preview.mp4')}

    def refine(self, video, audio, resources, request):
        import comfy.sd
        import comfy.samplers
        import comfy.nested_tensor
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
        from .vendor.split_upscale import MMH3SplitUpscale
        p = self.profile
        # This is an explicit, locked profile, never a silent model fallback.
        model = comfy.sd.load_diffusion_model(str(self.weights['refiner']), disable_dynamic=True)
        model = MiniMaxH3SigmaShift.execute(model, 12.0, 3.0)[0]
        conditioning = load_conditioning(resources)
        seed = request.get('seed', resources['manifest']['request']['seed'])
        class Noise:
            def __init__(self, seed):
                self.seed = seed
            def generate_noise(self, latent):
                generator = torch.Generator('cpu').manual_seed(self.seed)
                return torch.randn(latent['samples'].shape, generator=generator, dtype=torch.float32)
        latent = {'samples': comfy.nested_tensor.NestedTensor((video, audio))}
        with torch.inference_mode():
            output = MMH3SplitUpscale.execute(latent, conditioning, model, Noise(seed),
                comfy.samplers.sampler_object(p['sampler']), torch.tensor(p['sigmas'], dtype=torch.float32),
                cfg=1.0, temporal_split_param=p.get('temporal_split'), spatial_split_param=p.get('spatial_split'),
                seam_polish=p.get('seam_polish', 'off'), color_match=p.get('color_match', False))[0]['samples']
        result = output.tensors[0].cpu().contiguous()
        # Split freezes the audio mask and returns original audio chunks. Keep
        # the exact source stream for mux; still reject changed tensor geometry.
        if output.tensors[1].shape != audio.shape or result.shape != video.shape or not torch.isfinite(result).all():
            raise ResourceError('refine_geometry_or_finiteness_failed')
        del model, output
        release_stage()
        return result
