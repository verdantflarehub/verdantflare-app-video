"""Pinned temporal Video-Depth-Anything; CUDA is mandatory."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

LOCK = json.loads(Path(__file__).with_name('model-lock.json').read_text())


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checkpoint():
    return Path(os.environ.get('DEPTH_MODEL_ROOT', '/models/depth')) / LOCK['revision'] / LOCK['filename']


def verify():
    path = checkpoint()
    if not path.is_file() or path.stat().st_size != LOCK['size'] or sha256(path) != LOCK['sha256']:
        raise RuntimeError('model_integrity_failed')
    return path


class Engine:
    def __init__(self):
        import torch
        from gpu_guard import verify_gpu
        verify_gpu(torch, os.environ.get('DEPTH_GPU_ALLOCATION', ''))
        from video_depth_anything.video_depth import VideoDepthAnything
        path = verify()
        self.model = VideoDepthAnything(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
        self.model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        self.model = self.model.to('cuda').eval()
        self.gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version', '--format=csv,noheader'], text=True).strip()
        print(json.dumps({'event': 'model_ready', 'model': LOCK, 'gpu': self.gpu, 'torch': torch.__version__, 'cuda': torch.version.cuda}), flush=True)

    def infer(self, frames, fps):
        return self.model.infer_video_depth(frames, fps, input_size=518, device='cuda', fp32=False)[0]
