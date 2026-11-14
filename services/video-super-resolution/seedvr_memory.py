"""Equivalent language RoPE coordinates without the upstream dense 1024x128x128 grid."""

def compact_rope_freqs(module, vid_shape, txt_shape):
    import torch
    video, text = [], []
    for (frames, height, width), length in zip(vid_shape.tolist(), txt_shape[:, 0].tolist()):
        if min(frames, height, width, length) < 1 or frames + length > 1024 or max(height, width) > 128:
            raise ValueError('unsupported_seedvr_rope_geometry')
        # Preserve upstream language positions: video time starts after text tokens.
        axis = module.rope(torch.arange(max(length + frames, height, width), device=module.rope.device))
        temporal = axis[length:length + frames, None, None].expand(frames, height, width, -1)
        vertical = axis[None, :height, None].expand(frames, height, width, -1)
        horizontal = axis[None, None, :width].expand(frames, height, width, -1)
        video.append(torch.cat((temporal, vertical, horizontal), dim=-1).reshape(frames * height * width, -1))
        text.append(axis[:length].repeat(1, 3))
    return torch.cat(video), torch.cat(text)


def mapped_dit_offload(model):
    """Retain checkpoint-backed CPU tensors; avoid anonymous CPU copies after inference.

    Call immediately after mmap checkpoint loading, before any dtype conversion.
    This adapter supports the pinned runner's CPU/CUDA .to(device) transitions.
    """
    import torch
    from types import MethodType
    weights = {name: parameter.detach() for name, parameter in model.named_parameters()}
    original_to = model.to

    def move(self, device):
        target = torch.device(device)
        if target.type not in ('cpu', 'cuda'):
            raise ValueError('unsupported_seedvr_offload_device')
        for name, parameter in self.named_parameters():
            parameter.data = (weights[name] if target.type == 'cpu'
                              else weights[name].to(device=target, dtype=torch.bfloat16))
        # Move non-parameter buffers while preserving their FP32 positional frequencies.
        return original_to(target)

    model.to = MethodType(move, model)
