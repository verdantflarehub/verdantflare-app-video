"""Version-scoped Qwen placement and bounded feed-forward activations.

Only the conditioner executes on GPU1. The pipeline and returned embeddings remain
on GPU0. CPU offload is still required: Qwen weights do not fit on either card.
"""
import functools
import inspect
import os
import types


def encoder_config():
    device = os.environ.get('VDN_ENCODER_DEVICE', 'cuda:1')
    chunk = int(os.environ.get('VDN_ENCODER_MLP_CHUNK', '512'))
    if device not in {'cuda:0', 'cuda:1'} or not 1 <= chunk <= 4096:
        raise ValueError('invalid VDN encoder device or MLP chunk size')
    return device, chunk


def chunked_mlp(module, x, chunk_size):
    """Token-local MLP; attention and its full sequence remain untouched."""
    import torch
    if x.shape[-2] <= chunk_size:
        return module.down_proj(module.act_fn(module.gate_proj(x)) * module.up_proj(x))
    output = torch.empty_like(x)
    for start in range(0, x.shape[-2], chunk_size):
        end = min(start + chunk_size, x.shape[-2])
        part = x[..., start:end, :]
        output[..., start:end, :] = module.down_proj(
            module.act_fn(module.gate_proj(part)) * module.up_proj(part))
    return output


def install_chunking(encoder, chunk_size):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextMLP
    targets = [m for m in encoder.modules() if type(m) is Qwen3VLTextMLP]
    if len(targets) != encoder.config.text_config.num_hidden_layers:
        raise RuntimeError('unsupported Qwen text MLP layout')
    for module in targets:
        if hasattr(module, '_vdn_original_forward'):
            raise RuntimeError('Qwen chunking already installed')
        module._vdn_original_forward = module.forward
        def forward(self, hidden_state):
            return chunked_mlp(self, hidden_state, chunk_size)
        module.forward = types.MethodType(forward, module)
    return len(targets)


def install_device_boundary():
    import torch
    from diffusers import ModularPipeline
    from diffusers.modular_pipelines.minimax_h3 import encoders
    original = encoders.get_qwen3vl_prompt_embeds
    if getattr(original, '_vdn_device_boundary', False):
        return
    signature = inspect.signature(original)
    required = {'text_encoder', 'processor', 'token_ids', 'vision_inputs',
                'text_encoder_layer', 'device', 'dtype'}
    if set(signature.parameters) != required:
        raise RuntimeError('unsupported MiniMax Qwen encoding signature')

    @functools.wraps(original)
    def encode(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        encoder = bound.arguments['text_encoder']
        device = getattr(encoder, '_vdn_encoder_device', None)
        if device is None:
            return original(*args, **kwargs)
        target = encoder._vdn_generation_device
        bound.arguments['device'] = torch.device(device)
        # Also set the current CUDA device for helpers allocating without a device.
        with torch.cuda.device(device):
            result = original(*bound.args, **bound.kwargs)
        return result.to(device=target)

    encode._vdn_device_boundary = True
    # ModularPipeline otherwise infers its device from the first offload group,
    # which could be the encoder. Do not let GPU1 become the denoising device.
    previous = ModularPipeline._execution_device
    def execution_device(self):
        for component in self.components.values():
            target = getattr(component, '_vdn_generation_device', None)
            if target is not None:
                return torch.device(target)
        return previous.fget(self)
    ModularPipeline._execution_device = property(execution_device)
    encoders.get_qwen3vl_prompt_embeds = encode


def configure_encoder(pipe, generation_device):
    from diffusers.hooks import apply_group_offloading
    device, chunk = encoder_config()
    encoder = pipe.text_encoder
    count = install_chunking(encoder, chunk)
    encoder._vdn_encoder_device = device
    encoder._vdn_generation_device = str(generation_device)
    install_device_boundary()
    # No asynchronous weight prefetch overlapping the long-sequence activations.
    apply_group_offloading(encoder, onload_device=device, offload_device='cpu',
                           offload_type='leaf_level', use_stream=False)
    encoder._vdn_encoder_profile = dict(device=device, generation_device=str(generation_device),
                                       mlp_chunk_tokens=chunk, mlp_layers=count, use_stream=False)
