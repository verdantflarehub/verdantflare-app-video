"""FastH3 VSA-H3 tile-64 attention patch for MiniMax H3 on Blackwell SM120 via comfy-kitchen."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math

import torch

log = logging.getLogger("FastH3VSA")

_RUNTIME_KEY = "fasth3_vsa_runtime_v2"
_PATCH_MARKER = "fasth3_vsa_patch_v2"
_WRAPPER_KEY = "fasth3_vsa_layout_v2"
_TILE = 64
_TILE_SHAPE = (4, 4, 4)


@dataclass
class _Geometry:
    untile_index: torch.Tensor
    block_len: torch.Tensor
    num_prefix_tiles: int
    total_rows: int
    padded_rows: int


class _VSAState:
    def __init__(self, sparsity: float):
        self.sparsity = float(sparsity)
        self._geometry: dict[tuple, _Geometry] = {}
        self.calls = 0

    def geometry(self, prefix_segments, video_grid, device) -> _Geometry:
        key = (tuple(prefix_segments), tuple(video_grid), str(device))
        hit = self._geometry.get(key)
        if hit is not None:
            return hit

        prefix_segments = tuple(int(x) for x in prefix_segments if int(x) > 0)
        vt, vh, vw = (int(x) for x in video_grid)
        if min(vt, vh, vw) <= 0:
            raise RuntimeError(f"FastH3 VSA: invalid video token grid {video_grid}")

        prefix_len = sum(prefix_segments)
        video_len = vt * vh * vw
        total_rows = prefix_len + video_len

        untile = torch.empty(total_rows, dtype=torch.long, device=device)
        live = []
        tile_id = 0
        src = 0

        for seg_len in prefix_segments:
            left = seg_len
            while left:
                n = min(_TILE, left)
                dst0 = tile_id * _TILE
                untile[src : src + n] = torch.arange(dst0, dst0 + n, device=device)
                live.append(n)
                src += n
                left -= n
                tile_id += 1
        num_prefix_tiles = tile_id

        ts_t, ts_h, ts_w = _TILE_SHAPE
        video_start = prefix_len
        for t0 in range(0, vt, ts_t):
            for h0 in range(0, vh, ts_h):
                for w0 in range(0, vw, ts_w):
                    original = []
                    for dt in range(ts_t):
                        t = t0 + dt
                        if t >= vt:
                            continue
                        for dh in range(ts_h):
                            h = h0 + dh
                            if h >= vh:
                                continue
                            for dw in range(ts_w):
                                w = w0 + dw
                                if w >= vw:
                                    continue
                                original.append(video_start + (t * vh + h) * vw + w)
                    n = len(original)
                    if not n:
                        continue
                    dst0 = tile_id * _TILE
                    idx = torch.tensor(original, dtype=torch.long, device=device)
                    untile[idx] = torch.arange(dst0, dst0 + n, device=device)
                    live.append(n)
                    tile_id += 1

        if src != prefix_len:
            raise RuntimeError(f"FastH3 VSA: prefix accounting mismatch {src} != {prefix_len}")
        if len(live) != tile_id:
            raise RuntimeError("FastH3 VSA: internal tile accounting mismatch")
        block_len = torch.tensor(live, dtype=torch.int32, device=device)
        if int(block_len.sum()) != total_rows:
            raise RuntimeError(f"FastH3 VSA: tiled live rows sum to {int(block_len.sum())}, expected {total_rows}")
        if int(torch.unique(untile).numel()) != total_rows:
            raise RuntimeError("FastH3 VSA: untile map is not injective")

        geo = _Geometry(
            untile_index=untile,
            block_len=block_len,
            num_prefix_tiles=num_prefix_tiles,
            total_rows=total_rows,
            padded_rows=tile_id * _TILE,
        )
        self._geometry[key] = geo
        log.info(
            "FastH3 VSA geometry: prefix=%s grid=%s tiles=%d (%d prefix), rows=%d -> %d padded",
            prefix_segments,
            video_grid,
            tile_id,
            num_prefix_tiles,
            total_rows,
            geo.padded_rows,
        )
        return geo


def _make_diffusion_wrapper(state: _VSAState, diffusion_model):
    def wrapper(executor, x, timestep, context, transformer_options=None, **kwargs):
        from comfy.ldm.minimax.model import PackedLayout

        if transformer_options is None:
            transformer_options = {}
        payload = kwargs.get("minimax_payload") or {}
        video_x, audio_x = x[0], x[1]
        if video_x.ndim != 5 or int(video_x.shape[0]) != 1:
            raise RuntimeError("FastH3 VSA requires native batch-1 MiniMax H3 AV latents")

        pt, ph, pw = tuple(int(v) for v in diffusion_model.patch_size)
        latent_t = int(math.ceil(video_x.shape[2] / pt) * pt)
        latent_h = int(math.ceil(video_x.shape[3] / ph) * ph)
        latent_w = int(math.ceil(video_x.shape[4] / pw) * pw)
        audio_t = int(audio_x.shape[-1])
        text_len = int(context.shape[1])

        layout = payload.get("layout")
        signature = (text_len, latent_t, latent_h, latent_w, audio_t)
        if layout is None or tuple(getattr(layout, "signature", ())) != signature:
            layout = PackedLayout(
                text_len,
                latent_t,
                latent_h,
                latent_w,
                audio_t,
                keyframes=payload.get("keyframes"),
                refs=payload.get("refs"),
            )

        segments = list(layout.segments)
        video_segments = [s for s in segments if s[2] == "video"]
        if len(video_segments) != 1:
            raise RuntimeError(f"FastH3 VSA expected one target video segment, got {video_segments}")
        va, vb, _ = video_segments[0]
        if vb != int(layout.seq_len):
            raise RuntimeError("FastH3 VSA expects generated video to be the final packed segment")

        prefix_segments = tuple(int(b - a) for a, b, kind in segments if kind != "video")
        grid = (latent_t // pt, latent_h // ph, latent_w // pw)
        expected_video = grid[0] * grid[1] * grid[2]
        if int(vb - va) != expected_video:
            raise RuntimeError(f"FastH3 VSA video rows {vb - va} do not match token grid {grid} ({expected_video})")

        runtime = {
            "prefix_segments": prefix_segments,
            "video_grid": grid,
            "layout_seq_len": int(layout.seq_len),
        }
        prior = transformer_options.get(_RUNTIME_KEY)
        transformer_options[_RUNTIME_KEY] = runtime
        try:
            return executor(x, timestep, context, transformer_options, **kwargs)
        finally:
            if prior is None:
                transformer_options.pop(_RUNTIME_KEY, None)
            else:
                transformer_options[_RUNTIME_KEY] = prior

    return wrapper


def _vsa_attention(attn, x, rope_freqs, transformer_options, state: _VSAState):
    import comfy.model_management
    import comfy.quant_ops
    import comfy_kitchen as ck

    runtime = transformer_options.get(_RUNTIME_KEY)
    if runtime is None:
        raise RuntimeError("FastH3 VSA runtime metadata is missing; diffusion wrapper did not run")
    if not hasattr(attn, "to_gate_compress"):
        raise RuntimeError("FastH3 VSA checkpoint gate is missing (attn.to_gate_compress)")

    s = int(x.shape[0])
    x_dev = x.device
    inner = int(attn.heads * attn.head_dim)
    geo = state.geometry(runtime["prefix_segments"], runtime["video_grid"], x_dev)
    if geo.total_rows != s:
        raise RuntimeError(f"FastH3 VSA geometry has {geo.total_rows} rows but attention got {s}")

    x_t = x.new_zeros((geo.padded_rows, x.shape[-1]))
    x_t[geo.untile_index] = x
    del x
    sp = geo.padded_rows

    qkv = attn.qkv_proj(x_t)
    q = qkv[:, :inner].contiguous().view(1, sp, attn.heads, attn.head_dim)
    k = qkv[:, inner : 2 * inner].contiguous().view(1, sp, attn.heads, attn.head_dim)
    v = qkv[:, 2 * inner :].contiguous().view(1, sp, attn.heads, attn.head_dim)
    del qkv

    if rope_freqs is not None:
        rope_t = rope_freqs.new_zeros((rope_freqs.shape[0], sp, *rope_freqs.shape[2:]))
        rope_t[:, geo.untile_index] = rope_freqs
        qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x_dev)
        kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x_dev)
        rot = rope_t.shape[-3] * 2
        comfy.quant_ops.ck.rms_rope_split_half_(
            q, k, rope_t, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
        )
        del rope_t, qw, kw
    else:
        q = attn.q_norm(q.view(sp, attn.heads, attn.head_dim)).unsqueeze(0)
        k = attn.k_norm(k.view(sp, attn.heads, attn.head_dim)).unsqueeze(0)

    if type(attn.to_gate_compress) is torch.nn.Linear:
        gw = attn.to_gate_compress.weight
        if gw.device != x_t.device or gw.dtype != x_t.dtype:
            gate_raw = torch.nn.functional.linear(x_t, gw.to(device=x_t.device, dtype=x_t.dtype))
        else:
            gate_raw = attn.to_gate_compress(x_t)
    else:
        gate_raw = attn.to_gate_compress(x_t)
    gate_t = gate_raw.view(1, sp, attn.heads, attn.head_dim)
    del gate_raw, x_t

    if q.device.type != "cuda" or q.dtype != torch.bfloat16 or int(attn.head_dim) != 128:
        raise RuntimeError(
            "FastH3 VSA CUDA path requires CUDA BF16 with head_dim=128; "
            f"got device={q.device}, dtype={q.dtype}, head_dim={attn.head_dim}"
        )

    keep_ratio = max(0.0, min(1.0, 1.0 - state.sparsity))
    with ck.use_backend("cuda"):
        out_t = ck.sol_attn(
            q,
            k,
            v,
            scale=attn.head_dim**-0.5,
            sink_blocks=[0, geo.num_prefix_tiles],
            sink_q=[0, geo.num_prefix_tiles],
            topk_ratio=keep_ratio,
            tail=False,
            block_len=geo.block_len,
            coarse_gate=gate_t,
        )

    del q, k, v, gate_t
    out = out_t[:, geo.untile_index]
    del out_t
    out = out.reshape(s, inner)
    state.calls += 1
    return attn.out_proj(out)


def _make_block_replace(block, state: _VSAState):
    def replace(args, extra):
        from comfy.ldm.minimax import model as mm

        x = args["img"]
        t_emb = args["t_emb"]
        mod_segments = args["mod_segments"]
        rope_freqs = args["rope_freqs"]
        transformer_options = args["transformer_options"]

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
        attn_out = _vsa_attention(
            block.attn,
            mm._mod_scale_shift(block.norm1(x), shift_msa, scale_msa, mod_segments),
            rope_freqs,
            transformer_options,
            state,
        )
        x = mm._mod_gate(x, gate_msa, attn_out, mod_segments)
        del attn_out
        h = mm._mod_scale_shift(block.norm2(x), shift_mlp, scale_mlp, mod_segments)
        x = mm._mod_gate(x, gate_mlp, block.mlp(h), mod_segments)
        del h
        return {"img": x}

    return replace


class FastH3VSAPatch:
    """FastVideo FastH3 VSA-H3 semantics on ComfyUI using comfy-kitchen Sol/VSA CUDA."""

    def patch(self, model, sparsity: float = 0.90):
        sparsity = float(sparsity)
        if not 0.0 <= sparsity < 1.0:
            raise ValueError("sparsity must be in [0,1)")

        patched = model.clone()
        options = patched.model_options
        if options.get(_PATCH_MARKER):
            raise RuntimeError("FastH3 VSA patch is already installed on this MODEL")

        transformer = options.setdefault("transformer_options", {})
        if transformer.get("optimized_attention_override") is not None:
            raise RuntimeError("FastH3 VSA cannot stack with another optimized_attention override")
        existing = transformer.get("patches_replace", {}).get("dit", {})
        if existing:
            raise RuntimeError("FastH3 VSA requires an unmodified H3 DiT block chain")

        dm = getattr(getattr(patched, "model", None), "diffusion_model", None)
        if dm is None or type(dm).__name__ != "MiniMaxH3Model":
            raise ValueError("FastH3 VSA requires a native ComfyUI MiniMaxH3Model")
        blocks = list(getattr(dm, "blocks", ()))
        if len(blocks) != 50:
            raise RuntimeError(f"FastH3 VSA expected 50 H3 DiT blocks, found {len(blocks)}")
        object_patches = set(getattr(patched, "object_patches", {}))
        missing = [
            i
            for i, b in enumerate(blocks)
            if getattr(b.attn, "to_gate_compress", None) is None
            and f"diffusion_model.blocks.{i}.attn.to_gate_compress" not in object_patches
        ]
        if missing:
            raise RuntimeError(f"FastH3 VSA compression gate is absent from blocks {missing[:8]}")

        import comfy_kitchen as ck

        backends = ck.list_backends()
        if not backends.get("cuda", {}).get("available", False):
            raise RuntimeError("comfy-kitchen CUDA backend is unavailable")
        if "sol_attn" not in backends.get("cuda", {}).get("capabilities", []):
            raise RuntimeError("comfy-kitchen CUDA backend does not expose sol_attn")

        state = _VSAState(sparsity=sparsity)
        patched.add_wrapper_with_key(
            __import__("comfy.patcher_extension", fromlist=["WrappersMP"]).WrappersMP.DIFFUSION_MODEL,
            _WRAPPER_KEY,
            _make_diffusion_wrapper(state, dm),
        )
        for i, block in enumerate(blocks):
            patched.set_model_patch_replace(_make_block_replace(block, state), "dit", "double_block", i)

        options[_PATCH_MARKER] = {"version": 2, "sparsity": sparsity, "tile_size": 64}
        return (patched,)
