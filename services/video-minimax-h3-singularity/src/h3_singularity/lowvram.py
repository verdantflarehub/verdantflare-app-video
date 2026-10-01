"""H3-specific Low VRAM loading and packed attention.

The public workflow calls this contract ``CoffH3ModelLoader backend=auto``.
The runtime image does not ship that third-party node, so this module provides
the compatible boundary using ComfyUI's native ModelPatcher/DynamicVRAM and a
memory-bounded SDPA implementation for MiniMax H3's packed AV attention.
"""

from __future__ import annotations

import os
from typing import Any

import torch


Q_CHUNKS = (8192, 4096, 2048, 1024)


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def configure_comfy_low_vram(state_name: str | None = None) -> dict[str, Any]:
    """Select Comfy's model residency for the current worker.

    ``LOW_VRAM`` keeps a larger working set resident and is faster when it
    fits.  ``NO_VRAM`` is Comfy's more aggressive CPU offload mode: it keeps
    only a small execution window on the GPU, which is required when H3's
    token-refiner activation competes with the DiT weights on a 24 GiB card.
    The default remains ``low`` for backwards compatibility; deployments can
    opt into ``no`` with ``SINGULARITY_LOW_VRAM_STATE=no``.
    """

    import comfy.model_management as model_management

    previous = getattr(model_management, "vram_state", None)
    vram_states = getattr(model_management, "VRAMState", None)
    if vram_states is None:
        raise RuntimeError("ComfyUI low-vram state is unavailable")
    requested = (state_name or os.environ.get("SINGULARITY_LOW_VRAM_STATE", "low")).strip().lower()
    aliases = {"low": "LOW_VRAM", "low_vram": "LOW_VRAM", "no": "NO_VRAM", "no_vram": "NO_VRAM"}
    enum_name = aliases.get(requested)
    if enum_name is None:
        raise RuntimeError(f"invalid_low_vram_state:{requested}")
    if not hasattr(vram_states, enum_name):
        raise RuntimeError(f"ComfyUI VRAM state is unavailable:{enum_name}")
    model_management.vram_state = getattr(vram_states, enum_name)
    return {
        "previous_vram_state": getattr(previous, "name", str(previous)),
        "vram_state": enum_name,
        "requested_state": requested,
        "q_chunks": list(Q_CHUNKS),
    }


def restore_comfy_vram_state(previous_name: str | None) -> None:
    """Restore the state captured before a Low VRAM request."""

    if not previous_name:
        return
    import comfy.model_management as model_management

    state = getattr(model_management, "VRAMState", None)
    if state is not None and hasattr(state, previous_name):
        model_management.vram_state = getattr(state, previous_name)


class CoffH3ModelLoader:
    """Compatibility implementation of the workflow's H3 loader contract."""

    def __init__(self, nodes, diffusion_name: str, device: torch.device):
        self.nodes = nodes
        self.diffusion_name = diffusion_name
        self.device = device

    def load(self, backend: str = "auto"):
        if backend != "auto":
            raise ValueError(f"unsupported H3 Low VRAM backend: {backend}")
        model = self.nodes.UNETLoader().load_unet(self.diffusion_name, "default")[0]
        # Comfy's UNETLoader returns a ModelPatcher directly. A few older H3
        # node builds wrap it in ``.patcher``; accept both shapes.
        patcher = model if hasattr(model, "load_device") else getattr(model, "patcher", None)
        if patcher is None:
            raise RuntimeError("H3 model has no Comfy model patcher")
        patcher.load_device = self.device
        patcher.offload_device = torch.device("cpu")
        return model, {
            "loader": "CoffH3ModelLoader",
            "loader_implementation": "native-comfy-lowvram-compat",
            "backend": "auto",
            "load_device": str(self.device),
            "offload_device": "cpu",
        }


def make_h3_lowvram_attention(metrics: dict[str, Any]):
    """Build a packed H3 attention function with retryable Q chunking.

    H3 passes ``[1, heads, tokens, head_dim]`` through Comfy's
    ``AttentionTensorContainer`` with ``skip_reshape=True``.  Chunking only Q
    preserves the complete K/V context and therefore does not alter reference
    token visibility.
    """

    from comfy.ldm.modules.attention import wrap_attn

    initial = int(os.environ.get("SINGULARITY_LOW_VRAM_Q_CHUNK", str(Q_CHUNKS[0])))
    if initial not in Q_CHUNKS:
        raise ValueError(f"invalid Low VRAM Q chunk: {initial}")
    minimum = Q_CHUNKS[-1]

    @wrap_attn
    def h3_lowvram_attention(
        q,
        k,
        v,
        heads,
        mask=None,
        attn_precision=None,
        skip_reshape=False,
        skip_output_reshape=False,
        **kwargs,
    ):
        if not skip_reshape:
            # H3 uses skip_reshape. Keep a safe fallback for older Comfy H3
            # builds instead of silently producing a wrong layout.
            from comfy.ldm.modules.attention import attention_sub_quad

            return attention_sub_quad(
                q,
                k,
                v,
                heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )

        # Let Comfy/PyTorch SDPA apply the scale exactly once. Multiplying Q
        # here and then calling SDPA without its ``scale`` argument would
        # apply 1/sqrt(head_dim) twice and collapse the denoising trajectory.
        scale = kwargs.get("scale")
        # Delegate each query slice to ComfyUI's native PyTorch attention
        # implementation.  Calling torch SDPA directly here bypasses Comfy's
        # attention contract (mask normalization, GQA handling and backend
        # selection); on H3 this can produce valid-shaped but visibly corrupt
        # latents even when no CUDA error is raised.
        from comfy.ldm.modules.attention import attention_pytorch

        chunk = min(initial, q.shape[-2])
        while True:
            output = []
            try:
                for start in range(0, q.shape[-2], chunk):
                    stop = min(start + chunk, q.shape[-2])
                    query = q[:, :, start:stop, :]
                    output.append(
                        attention_pytorch(
                            query,
                            k,
                            v,
                            heads,
                            mask=mask,
                            attn_precision=attn_precision,
                            skip_reshape=True,
                            # Keep every slice in native [B,H,Q,D] form;
                            # flatten only after all Q slices are concatenated.
                            skip_output_reshape=True,
                            scale=scale,
                            enable_gqa=kwargs.get("enable_gqa", False),
                            _inside_attn_wrapper=True,
                        )
                    )
                result = torch.cat(output, dim=-2)
                # Comfy's attention contract returns [B, tokens, heads*D]
                # unless the caller explicitly requests the per-head layout.
                # SDPA itself returns [B, heads, tokens, D], so flattening the
                # head dimension here is required before H3's output
                # projection (and preserves the native attention semantics).
                if not skip_output_reshape:
                    result = result.transpose(1, 2).reshape(
                        result.shape[0], result.shape[2], heads * result.shape[3]
                    )
                metrics["last_q_chunk"] = chunk
                metrics["retry_count"] = metrics.get("retry_count", 0) + (1 if chunk < initial else 0)
                metrics["attention_calls"] = metrics.get("attention_calls", 0) + 1
                return result
            except Exception as exc:
                del output
                if not _is_oom(exc) or chunk <= minimum:
                    raise
                next_chunk = max(minimum, chunk // 2)
                metrics["oom_retries"] = metrics.get("oom_retries", 0) + 1
                metrics["retry_chunks"] = metrics.get("retry_chunks", []) + [next_chunk]
                chunk = next_chunk
                import comfy.model_management as model_management

                model_management.soft_empty_cache(force=True)

    return h3_lowvram_attention
