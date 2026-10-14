# video-minimax-h3-api

Production MiniMax H3 Base Ref2VA inference for the VerdantFlare video
workflow. The service exposes SGLang's asynchronous OpenAI-compatible
`/v1/videos` API directly; it does not install ComfyUI or add an API proxy.

## Locked runtime

| Component | Version |
| --- | --- |
| Service image | `video-minimax-h3-api-v0.3.0` |
| SGLang | `bbbcbf9418f0d8fbea968d96f3b470f5b883bac3` |
| comfy-kitchen | `0.2.31` |
| FlashInfer Python/cubin | `0.6.17` |
| MiniMax H3 | `42ed227ee7df40d41602854ae760620d6eb651fe` |
| Full Ref2VA INT8 ConvRot | `4cc1d817b6184899b41293954329f576cb5ae86b` |

The production image promotes the exact runtime image that passed the Full
INT8 benchmark. Its only behavioral change is replacing the benchmark runner
with the long-running API entrypoint. Model weights and generated video remain
on persistent storage and do not enter the image or Git.

The Chengdu deployment is pinned to node `10.241.109.6` and two distinct whole
RTX 4090 GPUs allocated by `hami-scheduler`. It uses TP2, Ulysses1, exact
FlashAttention, eager execution, full DiT/text-encoder/VAE layerwise offload,
zero resident DiT layers, and the serialized Full INT8 transformer. The
checkpoint path is fixed to:

```text
/models/MiniMax-H3/serialized-int8/diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors
```

The verified file is `34038894550` bytes with SHA-256
`9eef934046a0671bc8a5daf87100705e1478419c574cfde70c50fbe6885f76a9`.

The entrypoint keeps single-GPU defaults for backward-compatible local smoke
tests. Production sets `H3_VISIBLE_GPU_COUNT=2`, `H3_NUM_GPUS=2`,
`H3_TP_SIZE=2`, `H3_ULYSSES_DEGREE=1`,
`H3_LAYERWISE_OFFLOAD_COMPONENTS=dit,text_encoder,vae`, and
`H3_DIT_LAYERWISE_RESIDENT_LAYERS=0`. Startup rejects mismatched counts,
invalid parallel degrees, or duplicate visible CUDA UUIDs.

## API

- `GET /health`
- `POST /v1/videos`
- `GET /v1/videos/{id}`
- `GET /v1/videos/{id}/content`

Run the technical smoke test after the server becomes healthy:

```bash
H3_BASE_URL=http://127.0.0.1:8000 \
  services/video-minimax-h3-api/smoke-test.sh
```

The smoke request sets `num_inference_steps` to `21`, which is 21 sigma points
and 20 denoising evaluations (Base 20 NFE). Passing verifies the runtime and
media contract only; creative motion and identity still require full-speed
human review and per-second contact-sheet review.

## License boundary

MiniMax H3 is not Apache or MIT licensed. Review `NOTICE.upstream.md` and the
license downloaded with the model before providing access. A public product
must implement the license's territory, user-terms, moderation, reporting,
disclosure, and UI-attribution requirements before third-party use.
