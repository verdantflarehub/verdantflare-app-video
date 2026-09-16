# Video Super Resolution API

Independent SeedVR2-3B runtime. The design repository owns the
[API and media contract](https://github.com/verdantflarehub/verdantflare-design/blob/dev/docs/design/app/video/video.sr.md).

Build from the application repository root:

```bash
docker build -f services/video-super-resolution-api/Dockerfile \
  -t video-super-resolution-api:0.1.0 services
```

The image pins SeedVR source, Torch and FlashAttention. It uses the reference Torch RMSNorm/LayerNorm configuration instead of Apex's fused implementations, exact request dimensions instead of upstream's area-derived crop, and one inference step. DiT parameters use BF16 on CUDA and return to the original mmap checkpoint storage on CPU; positional-frequency buffers remain FP32. Language RoPE computes only the requested coordinates (tested against the dense upstream formula), and the last two VAE upsampling blocks use upstream slicing. The model remains loaded between tasks, with DiT/VAE CPU offload between phases and VAE temporal caches stored on CPU and released between encoding and decoding. VAE convolution/normalization chunks use a 0.125 GiB input budget. CUDA uses `expandable_segments:True` by default; the allocator configuration is part of the capacity identity. No silent alternative backend is used.

Build from this service directory. Runtime and business logic are owned by `video_sr`. Provision external model bundles with `python -m video_sr.prepare_model --help`; weights are never included in the image.
Default model mount: `/models/seedvr2`; persistent queue: `/data/sr`.
Configure MCP with `VIDEO_SR_RUNTIME_URL` pointing to the internal HTTP service, and `VIDEO_SR_RUNTIME_VERSION=video-super-resolution-api-v0.1.0`.

Download the revision and SHA-256 values pinned in `backend-lock.json` into an external model directory:

```bash
python services/video-super-resolution-api/download_model.py \
  --destination "$VIDEO_MODEL_ROOT" --endpoint https://hf-mirror.com --workers 6
```

The downloader uses an explicit client User-Agent, resumes partial byte ranges, verifies every complete file and writes the manifest only after verification. Existing files with mismatched hashes are rejected without overwrite. The mirror returned HTTP 403 for the default Python User-Agent during local checks, while the explicit User-Agent supported API and weight downloads. Model weights stay outside Git and the image.

This checkout includes an implementation and CPU contract tests; Docker/CUDA/model-quality validation is recorded separately in the design repository.

## Measure the window before serving requests

After the exact image and verified model bundle are available, run inside that image with one visible GPU:

```bash
python /app/calibrate.py \
  --source "$CALIBRATION_INPUT_VIDEO" \
  --target-width 1152 --target-height 2016 \
  --max-frames 129 --repeats 3 --reserve-mib 1024 \
  --output "$NEW_CALIBRATION_OUTPUT_DIR"
```

Use a real input video with the intended input geometry and enough frames for the test budget. `--max-frames` bounds the measurement run, not production video length. Each trial loads the pinned model in a fresh process. Failures retain `run.json` and per-trial logs; only successful measurement creates `capacity.json`. The output directory must be new. A `lower_bound_only` result means a larger window may work; a `measured_budget_boundary` identifies a rejected next aligned window under the measured memory budget. Neither establishes a universal maximum across GPUs, versions or resolutions.

Mount the successful profile read-only and set `VIDEO_SR_CAPACITY_PROFILE=/config/sr-capacity.json`. One profile covers its exact input/output geometry and runtime identity. SR startup requires a profile; requests for unmeasured geometry fail explicitly. The selected window is internal and does not limit whole-video length. Source decoding, overlapping windows, hard-cut handling and final encoding operate as a stream. Validate temporal seams with real model outputs before accepting visual quality.
