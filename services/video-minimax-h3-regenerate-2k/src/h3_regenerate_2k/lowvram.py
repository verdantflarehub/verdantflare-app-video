"""Memory-bounded packed H3 SDPA attention with retryable Q chunking for 2K IC tile refinement."""

from __future__ import annotations

import os
from typing import Any

import torch


Q_CHUNKS = (8192, 4096, 2048, 1024)


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def make_h3_lowvram_attention(metrics: dict[str, Any]):
    from comfy.ldm.modules.attention import wrap_attn

    initial = int(os.environ.get("REGENERATE_2K_LOW_VRAM_Q_CHUNK", str(Q_CHUNKS[0])))
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

        scale = kwargs.get("scale")
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
                            skip_output_reshape=True,
                            scale=scale,
                            enable_gqa=kwargs.get("enable_gqa", False),
                            _inside_attn_wrapper=True,
                        )
                    )
                result = torch.cat(output, dim=-2)
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
