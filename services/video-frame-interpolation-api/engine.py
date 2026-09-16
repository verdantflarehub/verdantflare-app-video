"""Pinned ECCV2022 RIFE IFNet, loaded once per single-GPU worker."""
import json
import os
from pathlib import Path
import sys

from video_interpolation.integrity import verify_bundle

LOCK = json.loads(Path(__file__).with_name('backend-lock.json').read_text())


class Engine:
    def __init__(self):
        root = Path(os.environ.get('VIDEO_MODEL_ROOT', '/models/rife')).resolve()
        upstream = Path(os.environ.get('VIDEO_UPSTREAM_ROOT', '/opt/rife')).resolve()
        self.metadata = verify_bundle(root, LOCK, upstream)
        import torch
        from video_interpolation.gpu_guard import verify_gpu
        verify_gpu(torch, os.environ.get("VIDEO_GPU_ALLOCATION", ""))
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('exactly_one_visible_cuda_device_required')
        sys.path.insert(0, str(upstream))
        from model.IFNet import IFNet
        self.torch = torch
        self.model = IFNet().eval().cuda()
        state = torch.load(root / 'flownet.pkl', map_location='cpu', weights_only=True)
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.requires_grad_(False)
        self.metadata['runtime_version'] = 'video-frame-interpolation-api-v0.1.1'

    def midpoint(self, left, right):
        torch = self.torch
        h, w = left.shape[:2]
        def tensor(frame):
            value = torch.from_numpy(frame.copy()).permute(2, 0, 1).unsqueeze(0).cuda().float() / 255
            return torch.nn.functional.pad(value, (0, (-w) % 32, 0, (-h) % 32))
        with torch.inference_mode():
            inputs = torch.cat((tensor(left), tensor(right)), dim=1)
            merged = self.model(inputs, scale=[4, 2, 1], timestep=.5)[2][2]
            result = merged[0, :, :h, :w].permute(1, 2, 0)
            if not torch.isfinite(result).all():
                raise RuntimeError('invalid_rife_output')
            return result.clamp(0, 1).mul(255).round().byte().cpu().numpy()
