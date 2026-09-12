# Video MCP Server

Domain MCP boundary for MiniMax H3 Ref2VA. It imports immutable project-scoped image, audio, and video Artifacts, persists idempotent domain tasks, bridges them to the cluster-local H3 Runtime, and validates completed H.264 output before returning an Artifact.

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

`video.generate` uses 21 sigma points, which corresponds to the approved Base
20 NFE profile. This parameter is owned by the service and is not exposed as a
caller-controlled option.
