"""Version-scoped Qwen placement and bounded feed-forward activations.

Only the conditioner executes on GPU1. The pipeline and returned embeddings remain
on GPU0. CPU offload is still required: Qwen weights do not fit on either card.
"""
import functools
import inspect
import json
import time
import os
import types


def chunked_rmsnorm(original, x, chunk_size):
    """Q/K are [batch, tokens, heads, head_dim]; normalize the same last axis."""
    import torch
    if x.ndim != 4:
        raise RuntimeError('unsupported Qwen Q/K norm layout')
    if x.shape[1] <= chunk_size:
        return original(x)
    first = original(x[:, :chunk_size])
    output = torch.empty_like(x, dtype=first.dtype)
    output[:, :chunk_size] = first
    del first
    for start in range(chunk_size, x.shape[1], chunk_size):
        output[:, start:start + chunk_size] = original(x[:, start:start + chunk_size])
    return output


def install_qk_norm_chunking(encoder, chunk_size):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention, Qwen3VLTextRMSNorm
    attention = [m for m in encoder.modules() if type(m) is Qwen3VLTextAttention]
    if len(attention) != encoder.config.text_config.num_hidden_layers:
        raise RuntimeError('unsupported Qwen attention layout')
    for layer in attention:
        for module in (layer.q_norm, layer.k_norm):
            if type(module) is not Qwen3VLTextRMSNorm or hasattr(module, '_vdn_original_forward'):
                raise RuntimeError('unsupported or already patched Qwen Q/K norm')
            module._vdn_original_forward = module.forward
            def forward(self, x):
                return chunked_rmsnorm(self._vdn_original_forward, x, chunk_size)
            module.forward = types.MethodType(forward, module)
    return len(attention) * 2


def selected_qwen_prompt_embeds(text_encoder, processor, token_ids, vision_inputs=None,
                               text_encoder_layer=50, device=None, dtype=None):
    """Match pinned Diffusers embeddings without retaining every decoder state.

    Transformers 5.15.0 captures decoder outputs before DeepStack injection and
    replaces only the final collected state with post-norm output. Keep exactly
    that intermediate output; run the full model, with cache/collection disabled.
    """
    import torch
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    language = text_encoder.model.language_model
    count = text_encoder.config.text_config.num_hidden_layers
    if type(language) is not Qwen3VLTextModel or len(language.layers) != count:
        raise RuntimeError('unsupported Qwen text model layout')
    if type(text_encoder_layer) is not int or not 0 <= text_encoder_layer < count:
        raise ValueError('conditioning layer must precede final post-norm state')
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    mm_ids = torch.tensor(processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device)
    vision = {name: value.to(device, text_encoder.dtype) if name.startswith('pixel_') else value.to(device)
              for name, value in (vision_inputs or {}).items()}
    captured = []
    def capture(module, args, output):
        value = args[0] if text_encoder_layer == 0 else output
        if not isinstance(value, torch.Tensor) or captured:
            raise RuntimeError('unexpected Qwen hidden-state capture')
        captured.append(value)
    hook = language.layers[max(0, text_encoder_layer - 1)].register_forward_hook(capture)
    try:
        offload_hook = getattr(text_encoder, '_hf_hook', None)
        if offload_hook is not None and hasattr(offload_hook, 'pre_forward'):
            offload_hook.pre_forward(text_encoder)
        outputs = text_encoder.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                     mm_token_type_ids=mm_ids, use_cache=False,
                                     output_hidden_states=False, **vision)
        del outputs
        if len(captured) != 1:
            raise RuntimeError('Qwen conditioning state was not captured')
        return captured[0].to(device=device, dtype=dtype)
    finally:
        hook.remove()
        captured.clear()


def encoder_config():
    device = os.environ.get('VDN_ENCODER_DEVICE', 'cuda:1')
    chunk = int(os.environ.get('VDN_ENCODER_MLP_CHUNK', '512'))
    residency = os.environ.get('VDN_ENCODER_MLP_RESIDENCY', '0')
    if device not in {'cuda:0', 'cuda:1'} or not 1 <= chunk <= 4096 or residency not in {'0', '1'}:
        raise ValueError('invalid VDN encoder device or MLP chunk size')
    return device, chunk, residency == '1'


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


def install_mlp_residency(encoder, onload_device, offload_device='cpu'):
    """Keep one Qwen MLP's three projections resident across token chunks.

    The encoder is first given leaf-level hooks for its memory bound.  For each
    text MLP, replace the three child hooks with one parent group so gate/up/down
    are transferred once per MLP forward instead of once per 512-token chunk.
    """
    import torch
    from diffusers.hooks import HookRegistry
    from diffusers.hooks.group_offloading import (GroupOffloadingConfig, GroupOffloadingHook,
                                                   ModuleGroup, _GROUP_OFFLOADING)
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextMLP

    targets = [m for m in encoder.modules() if type(m) is Qwen3VLTextMLP]
    if len(targets) != encoder.config.text_config.num_hidden_layers:
        raise RuntimeError('unsupported Qwen text MLP layout')
    onload = torch.device(onload_device)
    offload = torch.device(offload_device)
    config = GroupOffloadingConfig(onload_device=onload, offload_device=offload,
                                   offload_type='block_level', num_blocks_per_group=1,
                                   non_blocking=False, stream=None, record_stream=False,
                                   low_cpu_mem_usage=False, offload_to_disk_path=None,
                                   block_modules=None, exclude_kwargs=None, module_prefix='')
    for mlp in targets:
        projections = [mlp.gate_proj, mlp.up_proj, mlp.down_proj]
        for projection in projections:
            registry = getattr(projection, '_diffusers_hook', None)
            if registry is None or registry.get_hook(_GROUP_OFFLOADING) is None:
                raise RuntimeError('Qwen MLP projection leaf hook missing')
            registry.remove_hook(_GROUP_OFFLOADING, recurse=False)
        group = ModuleGroup(modules=projections, offload_device=offload,
                            onload_device=onload, offload_leader=mlp,
                            onload_leader=mlp, onload_self=True, group_id='qwen_mlp')
        HookRegistry.check_if_exists_or_initialize(mlp).register_hook(
            GroupOffloadingHook(group, config=config), _GROUP_OFFLOADING)
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
        started = time.monotonic()
        print(json.dumps({"stage": "reference_encoding", "state": "started",
                          "tokens": len(bound.arguments["token_ids"])}), flush=True)
        with torch.cuda.device(device):
            result = selected_qwen_prompt_embeds(*bound.args, **bound.kwargs)
        result = result.to(device=target)
        print(json.dumps({"stage": "reference_encoding", "state": "completed",
                          "seconds": time.monotonic() - started}), flush=True)
        return result

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
    device, chunk, mlp_residency = encoder_config()
    encoder = pipe.text_encoder
    count = install_chunking(encoder, chunk)
    norm_count = install_qk_norm_chunking(encoder, chunk)
    encoder._vdn_encoder_device = device
    encoder._vdn_generation_device = str(generation_device)
    install_device_boundary()
    # No asynchronous weight prefetch overlapping the long-sequence activations.
    apply_group_offloading(encoder, onload_device=device, offload_device='cpu',
                           offload_type='leaf_level', use_stream=False)
    residency_count = install_mlp_residency(encoder, device) if mlp_residency else 0
    encoder._vdn_encoder_profile = dict(device=device, generation_device=str(generation_device),
                                       mlp_chunk_tokens=chunk, mlp_layers=count, use_stream=False,
                                       mlp_residency=mlp_residency, mlp_residency_layers=residency_count,
                                       qk_norm_chunk_tokens=chunk, qk_norm_modules=norm_count,
                                       hidden_states='selected_intermediate_only')
