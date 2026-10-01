# Video MCP Server

Domain MCP boundary for video generation and processing. It imports immutable
project-scoped image, audio, and video Artifacts, persists idempotent domain
tasks, bridges them to approved runtimes or the server-side fal Queue adapter,
and validates completed video output before returning an Artifact.

## Tools

- `artifact.import`
- `video.generate`
- `video.status`
- `video.result`

The MCP never returns Runtime task IDs, internal service URLs, node names, GPU details, or storage paths. The internal `/runtime-artifacts/` route exists only for H3 condition downloads and must never be published by an ingress.

## Configuration

| Variable | Default |
| --- | --- |
| `VIDEO_ARTIFACT_ROOT` | `/data/video-mcp` |
| `VIDEO_ASSET_IMPORT_ORIGINS` | disabled |
| `VIDEO_MCP_PUBLIC_BASE_URL` | unset |
| `VIDEO_MCP_BEARER_TOKEN` | unset |
| `VIDEO_MCP_ALLOWED_HOSTS` | loopback only |
| `VIDEO_MCP_ALLOWED_ORIGINS` | loopback only |
| `VIDEO_MCP_URL` | client-facing unified Video MCP endpoint |
| `VIDEO_MCP_RUNTIME_BASE_URL` | `http://video-mcp-server:8000` |
| `H3_RUNTIME_URL` | `http://video-minimax-h3-api:8000` |
| `VIDEO_DEPTH_RUNTIME_URL` | `http://video-depth-anything-api:8000` |
| `H3_RUNTIME_VERSION` | `video-minimax-h3-api-v0.3.0` |
| `FAL_KEY` | unset; enables the server-side `fal` route |
| `FAL_QUEUE_BASE_URL` | `https://queue.fal.run` |
| `FAL_MODEL_ID` | `minimax/h3/reference-to-video` |
| `FAL_RUNTIME_VERSION` | `video-fal-adapter-v0.2.0` |
| `FAL_RESOLUTION` | `2K` |
| `FAL_RESULT_ORIGINS` | unset; additional exact HTTPS result origins |

Use `model=minimax-h3-ref2va` together with `route=fal`. The adapter maps up to
9 image, 3 video, and 3 audio Artifacts (12 total) to fal's reference lists,
submits to the asynchronous Queue API, polls the provider request, downloads
the result without forwarding the API key, validates it with `ffprobe`, and
stores it as a project Artifact. It never falls back to a self-hosted route.
`FAL_KEY` must be injected at runtime and must not be placed in client requests
or committed configuration.

`video.generate` uses 21 sigma points, which corresponds to the approved Base
20 NFE profile. This parameter is owned by the service and is not exposed as a
caller-controlled option.
