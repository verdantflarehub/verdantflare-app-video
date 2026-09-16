"""Persistent SeedVR2-3B runner with exact geometry and one-step restoration."""
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

from video_sr.integrity import verify_bundle
from video_sr.media import MediaError
from video_sr.capacity import CapacityProfile, CapacityError
from video_sr.integrity import sha256
import subprocess

LOCK = json.loads(Path(__file__).with_name('backend-lock.json').read_text())


class Engine:
    def __init__(self, *, calibration=False):
        self.profile = None if calibration else CapacityProfile(os.environ.get('VIDEO_SR_CAPACITY_PROFILE', '/config/sr-capacity.json'))
        self.root = Path(os.environ.get('VIDEO_MODEL_ROOT', '/models/seedvr2')).resolve()
        upstream = Path(os.environ.get('VIDEO_UPSTREAM_ROOT', '/opt/seedvr')).resolve()
        self.metadata = verify_bundle(self.root, LOCK, upstream)
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        import torch
        from video_sr.gpu_guard import verify_gpu
        verify_gpu(torch, os.environ.get("VIDEO_GPU_ALLOCATION", ""))
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise RuntimeError('one_bf16_cuda_device_required')
        self.torch = torch
        sys.path.insert(0, str(upstream))
        # Upstream resolves inherited model YAML paths relative to its repository.
        os.chdir(upstream)
        from common.config import load_config
        from projects.video_diffusion_sr.infer import VideoDiffusionInfer
        from projects.inference_seedvr2_3b import generation_step
        from common.seed import set_seed
        self.generation_step, self.set_seed = generation_step, set_seed
        torch.cuda.set_device(0)
        if not torch.distributed.is_initialized():
            self.rendezvous = tempfile.TemporaryDirectory(prefix='seedvr2-rendezvous-')
            torch.distributed.init_process_group('nccl', rank=0, world_size=1,
                init_method=Path(self.rendezvous.name, 'store').as_uri(), timeout=timedelta(seconds=3600))
        config = load_config(str(upstream / 'configs_3b/main.yaml'))
        # Reference Torch norms have identical parameter names/shapes; avoid Apex ABI coupling.
        for key in ('norm', 'vid_out_norm', 'txt_in_norm', 'qk_norm'):
            value = config.dit.model.get(key)
            if value in ('fusedrms', 'fusedln'):
                config.dit.model[key] = {'fusedrms': 'rms', 'fusedln': 'layer'}[value]
        config.vae.checkpoint = str(self.root / 'ema_vae.pth')
        # Temporal convolution caches otherwise accumulate on CUDA across VAE layers.
        config.vae.slicing.memory_device = 'cpu'
        config.vae.model.slicing_up_num = 2
        config.vae.memory_limit.conv_max_mem = 0.125
        config.vae.memory_limit.norm_max_mem = 0.125
        config.diffusion.cfg.scale = 1.0
        config.diffusion.cfg.rescale = 0.0
        config.diffusion.timesteps.sampling.steps = 1
        self.runner = VideoDiffusionInfer(config)
        self.runner.configure_dit_model(device='cpu', checkpoint=str(self.root / 'seedvr2_ema_3b.pth'))
        self.runner.dit.eval().requires_grad_(False)
        from types import MethodType
        from models.dit_v2.rope import NaMMRotaryEmbedding3d
        from seedvr_memory import compact_rope_freqs, mapped_dit_offload
        mapped_dit_offload(self.runner.dit)
        for module in self.runner.dit.modules():
            if isinstance(module, NaMMRotaryEmbedding3d):
                module.get_freqs = MethodType(compact_rope_freqs, module)
        self.runner.configure_vae_model()
        if hasattr(self.runner.vae, 'set_memory_limit'):
            self.runner.vae.set_memory_limit(**config.vae.memory_limit)
        self.runner.vae.to('cpu')
        self.runner.configure_diffusion()
        self.embeddings = {'texts_pos': [torch.load(self.root / 'pos_emb.pt', map_location='cpu', weights_only=True)],
                           'texts_neg': [torch.load(self.root / 'neg_emb.pt', map_location='cpu', weights_only=True)]}
        self.metadata['runtime_version'] = 'video-super-resolution-api-v0.1.0'

    def capacity_identity(self):
        from video_sr import windowing, media, capacity
        import hashlib
        import platform
        import flash_attn_2_cuda
        from importlib.metadata import version
        torch = self.torch
        devices = subprocess.run(['nvidia-smi', '--query-gpu=uuid,driver_version', '--format=csv,noheader'],
                                 check=True, capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        if len(devices) != 1:
            raise CapacityError('one_physical_gpu_required_for_capacity_identity')
        gpu_uuid, driver = [part.strip() for part in devices[0].split(',')]
        files = (Path(__file__), Path(windowing.__file__), Path(media.__file__), Path(capacity.__file__), Path(__file__).with_name('calibrate.py'), Path(__file__).with_name('seedvr_memory.py'))
        implementation = hashlib.sha256(''.join(sha256(p) for p in files).encode()).hexdigest()
        return dict(gpu_uuid=gpu_uuid, gpu_name=torch.cuda.get_device_name(0),
                    total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                    driver_version=driver, torch_version=torch.__version__, cuda_version=torch.version.cuda,
                    model_manifest_sha256=self.metadata['manifest_sha256'],
                    code_revision=self.metadata['code_revision'], implementation_sha256=implementation,
                    precision='bf16', sample_steps=1, normalization='torch_rms_layer',
                    cuda_allocator_config=os.environ.get('PYTORCH_CUDA_ALLOC_CONF', ''),
                    python_version=platform.python_version(), flash_attention_binary_sha256=sha256(flash_attn_2_cuda.__file__),
                    libraries={name: version(name) for name in ('diffusers', 'flash-attn', 'numpy', 'opencv-python-headless')})

    def select_window(self, media, req):
        if self.profile is None:
            raise CapacityError('capacity_profile_missing_or_invalid')
        policy = self.profile.select(self.capacity_identity(), media, req)
        self.torch.cuda.synchronize()
        self.torch.cuda.empty_cache()
        free, total = self.torch.cuda.mem_get_info()
        if total - free > self.profile.data['baseline_used_bytes'] + min(self.profile.data['reserve_bytes'], 64 * 1024 * 1024):
            raise CapacityError('gpu_memory_occupancy_changed')
        return policy

    def restore_window(self, frames, req):
        torch, runner = self.torch, self.runner
        self.set_seed(req.seed, same_across_ranks=True)
        with torch.inference_mode():
            video = torch.from_numpy(frames).permute(0, 3, 1, 2).float().cuda() / 255
            # Request validation preserves aspect ratio and requires divisible-by-16 dimensions.
            video = torch.nn.functional.interpolate(video, size=(req.target_height, req.target_width), mode='area')
            video = video.clamp(0, 1).mul(2).sub(1).permute(1, 0, 2, 3)
            count = video.shape[1]
            # Short shots/tails must also support the VAE's temporal slicing.
            padded_count = max(9, ((count - 1 + 3) // 4) * 4 + 1)
            pad = padded_count - count
            if pad:
                video = torch.cat((video, video[:, -1:].expand(-1, pad, -1, -1)), dim=1)
            try:
                runner.dit.to('cpu')
                runner.vae.to('cuda')
                latents = runner.vae_encode([video])
                del video
                self.clear_vae_cache()
                runner.vae.to('cpu')
                runner.dit.to('cuda')
                embeddings = {k: [v.to('cuda') for v in values] for k, values in self.embeddings.items()}
                samples = self.generation_step(runner, embeddings, latents)[0]
                if samples.shape[0] < count or not torch.isfinite(samples).all():
                    raise MediaError('invalid_seedvr2_output')
                result = samples[:count].permute(0, 2, 3, 1).clamp(-1, 1).add(1).mul(127.5).round().byte().cpu().numpy()
            finally:
                self.clear_vae_cache()
                runner.dit.to('cpu')
                runner.vae.to('cpu')
                torch.cuda.empty_cache()
        return result

    def clear_vae_cache(self):
        # Causal state is needed within an encode/decode phase, never between them.
        from models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d
        for module in self.runner.vae.modules():
            if isinstance(module, InflatedCausalConv3d):
                module.memory = None
