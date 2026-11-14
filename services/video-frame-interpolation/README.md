# Video Frame Interpolation

Independent RIFE runtime. The design repository owns the
[API and media contract](https://github.com/verdantflarehub/verdantflare-design/blob/dev/docs/design/app/video/video.interpolate.md).

Build from this service directory:

```bash
docker build -f services/video-frame-interpolation/Dockerfile \
  -t video-frame-interpolation:0.2.6 services/video-frame-interpolation
```

The image pins ECCV2022-RIFE and Torch. It loads IFNet once, computes intermediate frames on CUDA and runs a fixed hard-cut heuristic before interpolation. It does not use the upstream demo's shell-based audio handling.

Build from this service directory. Runtime and business logic are owned by `video_interpolation`. Provision external model bundles with `python -m video_interpolation.prepare_model --help`; weights are never included in the image.
The backend lock identifies the archive linked by the upstream README, its SHA-256 content revision, and the extracted `train_log/flownet.pkl` hash. Extract only that weight file, verify its hash against the lock, then use the shared packaging command with the lock’s `weights_revision`. This avoids treating a Google Drive file ID as an immutable version.

Default model mount: `/models/rife`; persistent queue: `/data/interpolate`.
Configure MCP with `VIDEO_INTERPOLATION_RUNTIME_URL` pointing to the internal HTTP service, and `VIDEO_INTERPOLATION_RUNTIME_VERSION=video-frame-interpolation-v0.2.6`.

This checkout includes an implementation and CPU contract tests; Docker/CUDA/model-quality validation is recorded separately in the design repository.
