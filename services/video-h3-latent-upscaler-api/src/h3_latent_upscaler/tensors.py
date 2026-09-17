"""Canonical H3 AV tensors; never reconstruct latent inputs from MP4."""
import torch
from safetensors.torch import load_file
from .resources import ResourceError


def require_finite(tensor, shape, name):
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != len(shape):
        raise ResourceError(f'invalid_{name}_shape')
    if any(expected is not None and got != expected for got, expected in zip(tensor.shape, shape)):
        raise ResourceError(f'invalid_{name}_shape')
    if any(d <= 0 for d in tensor.shape) or tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ResourceError(f'invalid_{name}_dtype')
    if not torch.isfinite(tensor).all().item():
        raise ResourceError(f'nonfinite_{name}')
    return tensor


def canonical_audio_from_diffusers(audio):
    """Diffusers batches stereo channels: [2,32,T] -> [1,32,2,T]."""
    require_finite(audio, (2, 32, None), 'audio')
    return audio.permute(1, 0, 2).unsqueeze(0).contiguous()


def decoded_video_frames(frames, *, count, width, height):
    """Comfy video VAEs return [B,T,H,W,C]; media encoding takes [T,H,W,C]."""
    if frames.ndim == 5:
        if frames.shape[0] != 1:
            raise ResourceError('decoded_batch_mismatch')
        frames = frames[0]
    require_finite(frames, (count, height, width, 3), 'decoded_video')
    return frames


def load_av(resources):
    manifest = resources['manifest']
    if manifest.get('representation') != 'h3-normalized-av/v1':
        raise ResourceError('unsupported_latent_representation')
    video_data = load_file(str(resources['files']['video_latent']), device='cpu')
    audio_data = load_file(str(resources['files']['audio_latent']), device='cpu')
    if set(video_data) != {'samples'} or set(audio_data) != {'samples'}:
        raise ResourceError('invalid_tensor_keys')
    video = require_finite(video_data['samples'], (1, 24, None, None, None), 'video')
    audio = require_finite(audio_data['samples'], (1, 32, 2, None), 'audio')
    media = manifest.get('media', {})
    if media.get('fps') != 24 or media.get('width') != video.shape[-1] * 16 or media.get('height') != video.shape[-2] * 16:
        raise ResourceError('latent_media_mismatch')
    # H3 video VAE uses 17n+5 frames and 5n+2 latent temporal slices.
    if video.shape[2] < 2 or (video.shape[2] - 2) % 5:
        raise ResourceError('unsupported_temporal_geometry')
    frames = (video.shape[2] - 2) // 5 * 17 + 5
    if media.get('frames') != frames:
        raise ResourceError('latent_media_mismatch')
    return video, audio


def require_single_4090():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ResourceError('single_gpu_required')
    device = torch.cuda.get_device_properties(0)
    if '4090' not in device.name or device.total_memory < 23 * 1024**3:
        raise ResourceError('full_rtx4090_required')
    return {'name': device.name, 'uuid': str(device.uuid), 'total_memory': device.total_memory}
