"""Inference-only DiT head groups and token-local projections on fixed Ada GPUs.

Only attention heads are partitioned: every group sees the original full sequence,
window plan and temporal/linear state. CPU banks assemble all heads before output
projection so FP8 row scales and the output-channel reduction remain unchanged.
"""
import types


def projection_shape(module):
    weight = getattr(module, 'weight_fp8', None)
    if weight is None:
        weight = getattr(module, 'weight', None)
    if weight is None or weight.ndim != 2:
        raise RuntimeError('unsupported DiT projection weights')
    return tuple(weight.shape)


def validate_geometry(module):
    """H3 trunk width (5376) and QKV width (56*128=7168) are independent."""
    qkv_width = module.num_heads * module.head_dim
    width = projection_shape(module.orig.to_q)[1]
    if any(projection_shape(p) != (qkv_width, width)
           for p in (module.orig.to_q, module.orig.to_k, module.orig.to_v)):
        raise RuntimeError('unsupported DiT QKV projection geometry')
    if (projection_shape(module.orig.to_out[0]) != (width, qkv_width)
            or module.linear_attention.num_heads != module.num_heads
            or module.linear_attention.head_dim != module.head_dim
            or projection_shape(module.to_out_linear) != (width, qkv_width)):
        raise RuntimeError('unsupported DiT branch projection geometry')


def sliced_projection(upstream, module, x, channels, quantized=None):
    import torch
    if isinstance(module, upstream.Fp8Linear):
        q, scale = quantized if quantized is not None else upstream.quantize_activation(x)
        weight_scale = module.weight_scale
        if weight_scale.numel() != 1:
            weight_scale = weight_scale[:, channels].contiguous()
        out = torch._scaled_mm(q, module.weight_fp8[channels].t(), scale_a=scale,
                               scale_b=weight_scale, out_dtype=x.dtype, use_fast_accum=True)
        if module.bias is not None:
            out = out + module.bias[channels]
        return out
    if type(module) is not torch.nn.Linear:
        raise RuntimeError('unsupported DiT projection type')
    return torch.nn.functional.linear(x, module.weight[channels],
                                     None if module.bias is None else module.bias[channels])


def gate_heads(gate, x, heads, tokens):
    """Run the original gate on full-width token rows, retaining only these heads."""
    import torch
    result = torch.empty((len(x), heads.stop-heads.start, gate.head_dim or 1),
                         device=x.device, dtype=x.dtype)
    for start in range(0, len(x), tokens):
        result[start:start+tokens] = gate(x[start:start+tokens])[:, heads]
    return result


def project_bank(projection, bank, device, dtype, tokens, output=None, start_row=0):
    """Project complete head rows, never independently quantize/reduce head slices."""
    import torch
    add = output is not None
    for start in range(0, len(bank), tokens):
        rows = bank[start:start+tokens].to(device=device, dtype=dtype)
        part = projection(rows)
        if output is None:
            output = torch.empty((len(bank), part.shape[-1]), device=device, dtype=part.dtype)
        if add:
            output[start_row+start:start_row+start+len(part)].add_(part)
        else:
            output[start:start+len(part)].copy_(part)
        del rows, part
    return output


def grouped_attention(upstream, self, x, rotary_emb, head_group, token_group):
    import torch
    if torch.is_grad_enabled() or not self.inference_mode or not self.hybrid_inference_mode:
        raise RuntimeError('grouped DiT attention requires inference mode')
    # Token-chunked FFN/output projections preserve rowwise scales, not a tensor scale.
    if upstream.per_tensor_gemm():
        raise RuntimeError('DiT memory profile requires rowwise FP8 (RTX 4090)')
    layout = self.layout
    bounds = self._bounds(layout) if layout is not None else None
    full = layout is None or all(lo <= 0 and hi >= layout.num_frames-1 for lo, hi in bounds)
    linear = not full and self.linear_attention_enabled
    n, width = x.shape
    heads, dim = self.num_heads, self.head_dim
    if projection_shape(self.orig.to_q) != (heads*dim, width) or self.linear_attention.head_dim != dim:
        raise RuntimeError('unsupported DiT projection geometry')
    soft_bank = torch.empty((n, heads, dim), dtype=x.dtype, device='cpu')
    linear_bank = None
    if linear:
        branch = self.linear_attention
        vs, ve = layout.video_start, layout.video_end
        xv = x[vs:ve]
        linear_bank = torch.empty((ve-vs, heads, dim), dtype=x.dtype, device='cpu')
        beta = torch.sigmoid(branch.beta_proj(xv))
        mean = xv.view(layout.num_frames, layout.tokens_per_frame, -1).mean(1, dtype=torch.float32)
        ts, te = layout.text_range if self.enable_text_state else (0, 0)
        tx = x[ts:te] if self.enable_text_state else None
        text_beta = torch.sigmoid(branch.beta_proj(tx)) if tx is not None else None
    projections = (self.orig.to_q, self.orig.to_k, self.orig.to_v)
    quantized = upstream.quantize_activation(x) if all(isinstance(p, upstream.Fp8Linear) for p in projections) else None
    for first in range(0, heads, head_group):
        hs = slice(first, min(first+head_group, heads))
        cs = slice(hs.start*dim, hs.stop*dim)
        raw = tuple(sliced_projection(upstream, p, x, cs, quantized).unflatten(-1, (hs.stop-hs.start, dim)) for p in projections)
        if rotary_emb is not None:
            query = upstream._qk_prep(raw[0], self.orig.norm_q.weight, self.orig.norm_q.eps, *rotary_emb)
            key = upstream._qk_prep(raw[1], self.orig.norm_k.weight, self.orig.norm_k.eps, *rotary_emb)
        else:
            query, key = self.orig.norm_q(raw[0]), self.orig.norm_k(raw[1])
        if full:
            soft = upstream.dispatch_attention_fn(query.unsqueeze(0), key.unsqueeze(0), raw[2].unsqueeze(0),
                     attn_mask=None, dropout_p=0.0, is_causal=False,
                     backend=getattr(type(self.orig.processor), '_attention_backend', None)).squeeze(0)
        else:
            soft = self._window_softmax(query, key, raw[2], layout, bounds, dim**-0.5, True)
        del query, key
        if self.enable_softmax_gate:
            gate = gate_heads(self.softmax_gate, x, hs, token_group)
            flat = upstream.apply_softmax_gate(soft, gate, inference=True)
            del gate, soft
        else:
            flat = soft.flatten(1)
            del soft
        soft_bank[:, hs].copy_(flat.view(n, hs.stop-hs.start, dim))
        del flat
        if linear:
            gate = gate_heads(branch.output_gate, xv, hs, token_group)
            readout = branch(None, layout.num_frames, layout.tokens_per_frame, bounds,
                             tuple(t[vs:ve] for t in raw),
                             frame_size=layout.frame_size if branch.short_conv is not None else None,
                             skip_ends=self.anchor_frames == 'both',
                             text_x=tx, text_qkv_raw=tuple(t[ts:te] for t in raw) if tx is not None else None,
                             inference=True, heads=hs, beta=beta[:, hs], gate=gate, frame_mean=mean,
                             text_beta=text_beta[:, hs] if text_beta is not None else None)
            linear_bank[:, hs].copy_(readout.view(ve-vs, hs.stop-hs.start, dim))
            del gate, readout
        del raw
    del quantized
    out = project_bank(lambda rows: self.orig.to_out[1](self.orig.to_out[0](rows)),
                       soft_bank.flatten(1), x.device, x.dtype, token_group)
    del soft_bank
    if linear:
        project_bank(self.to_out_linear, linear_bank.flatten(1), x.device, x.dtype,
                     token_group, output=out, start_row=vs)
    return out


def replace_forward(module, expected, replacement):
    """Keep the pinned Diffusers offload pre/post hooks around the replaced body."""
    registry = getattr(module, '_diffusers_hook', None)
    reference = registry._fn_refs[0] if registry is not None and registry._fn_refs else None
    current = reference.forward if reference is not None else module.forward
    if getattr(current, '__func__', None) is not expected:
        raise RuntimeError('unsupported DiT forward/offload hook layout')
    bound = types.MethodType(replacement, module)
    if reference is None:
        module.forward = bound
    else:
        reference.forward = bound


def chunked_block(upstream, self, hidden_states, temb, adaln_indices, rotary_emb,
                  attention_mask, token_group):
    import torch
    if torch.is_grad_enabled() or hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise RuntimeError('DiT block chunks require batch-one inference')
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)
    pre, post = upstream._compiled('pre', upstream._pre_ref), upstream._compiled('post', upstream._post_ref)
    normed = pre(hidden_states, self.norm1.weight, self.norm1.eps, scale_msa, shift_msa, adaln_indices)
    out = self.attn(normed, rotary_emb, attention_mask)
    del normed
    # Reuse only our own attention output. The caller's input must remain untouched.
    for start in range(0, hidden_states.shape[1], token_group):
        rows = slice(start, start+token_group)
        indices = adaln_indices[rows]
        out[:, rows].copy_(post(hidden_states[:, rows], gate_msa, indices, out[:, rows]))
    # FFN is token-local: normalize, evaluate and merge each row chunk together.
    for start in range(0, out.shape[1], token_group):
        rows = slice(start, start+token_group)
        indices = adaln_indices[rows]
        normed = pre(out[:, rows], self.norm2.weight, self.norm2.eps, scale_mlp, shift_mlp, indices)
        branch = self.ff(normed)
        del normed
        out[:, rows].copy_(post(out[:, rows], gate_mlp, indices, branch))
        del branch
    return out


def install_dit_memory(upstream, transformer, head_group=4, token_group=512):
    import torch
    if type(head_group) is not int or head_group < 1 or type(token_group) is not int or token_group < 1:
        raise ValueError('positive integer DiT group sizes required')
    hybrids = list(upstream.iter_hybrids(transformer))
    if not hybrids:
        raise RuntimeError('no DiT hybrid attention modules')
    for module in hybrids:
        validate_geometry(module)
        if hasattr(module, '_vdn_grouped'):
            raise RuntimeError('DiT memory profile already installed')
        conv = module.linear_attention.short_conv
        if conv is not None and 'q' in conv.projs:
            raise RuntimeError('Q-convolution is not supported by the pinned inference readout')
        if module.teacher_mode or not module.inference_mode:
            raise RuntimeError('DiT inference profile required')
        def forward(self, x, rotary_emb):
            return grouped_attention(upstream, self, x, rotary_emb, head_group, token_group)
        module._hybrid_forward = types.MethodType(forward, module)
        module._vdn_grouped = True
    count = 0
    for block in transformer.transformer_blocks:
        ff = block.ff
        if hasattr(ff, '_vdn_original_ff'):
            raise RuntimeError('DiT FFN already patched')
        # Patch the actual module after block offload installed its surrounding hook.
        # Calling upstream's pure FFN body keeps the parent offload wrapper untouched.
        ff._vdn_original_ff = upstream.fast_ff_forward
        def chunked(self, x):
            if torch.is_grad_enabled() or upstream.per_tensor_gemm():
                raise RuntimeError('DiT FFN chunks require rowwise inference')
            shape = x.shape
            rows = x.reshape(-1, shape[-1])
            out = torch.empty_like(rows)
            for start in range(0, len(rows), token_group):
                out[start:start+token_group] = upstream.fast_ff_forward(self, rows[start:start+token_group])
            return out.reshape(shape)
        # Group offload belongs to whole transformer blocks, not their FFN children.
        replace_forward(ff, upstream.fast_ff_forward, chunked)
        def block_forward(self, hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
            return chunked_block(upstream, self, hidden_states, temb, adaln_indices, rotary_emb,
                                 attention_mask, token_group)
        replace_forward(block, upstream.fast_block_forward, block_forward)
        count += 1
    return dict(attention_head_group=head_group, projection_token_group=token_group,
                attention_modules=len(hybrids), ffn_modules=count, branch_storage='cpu',
                residual_merge_tokens=token_group)
