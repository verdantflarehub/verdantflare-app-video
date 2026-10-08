from __future__ import annotations

import asyncio
import contextlib
import hmac
from functools import wraps

import httpx
import json
import logging
import os

from mcp import types
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route

from .artifacts import ArtifactError, ArtifactNotFound, ArtifactStore, require_project_id
from .executor import ExecutionError, VideoExecutor
from .dashboard import Dashboard
from .registry import EtcdRegistry
from .tasks import TaskConflict, TaskNotFound, TaskStore

logging.getLogger("httpx").setLevel(logging.WARNING)

artifacts = ArtifactStore.from_environment()
tasks = TaskStore.from_environment()
executor = VideoExecutor(artifacts, tasks)
dashboard = Dashboard(executor)
mcp = MCPServer("VerdantFlare Video")

VIDEO_TOOLS_SCHEMA = [
    {
        "name": "video.create",
        "description": "Create an asynchronous video generation task using minimax-h3-ref2va or configured routes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Text prompt describing the desired video"},
                "model": {"type": "string", "default": "minimax-h3-ref2va", "description": "Business model to use"},
                "aspect_ratio": {"type": "string", "enum": ["adaptive", "21:9", "16:9", "9:16", "1:1", "4:3", "3:4"], "default": "16:9", "description": "Check video.capabilities for route-specific adapter limits, not model limits"},
                "duration_seconds": {"type": "integer", "minimum": 4, "maximum": 15, "default": 5},
                "route": {"type": "string", "description": "Execution channel; omitted uses the host default from video.capabilities"},
                "quality_profile": {"type": "string", "default": "du-0"},
                "references": {"type": "object", "description": "Reference assets mapping"},
                "project_id": {"type": "string", "description": "Project ID"},
                "idempotency_key": {"type": "string", "description": "Unique idempotency key"},
            },
            "required": ["prompt", "project_id", "idempotency_key"],
        },
    },
    {
        "name": "video.status",
        "description": "Query status, timing, and stage of a video task.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Video task ID"},
                "video_task_id": {"type": "string", "description": "Alias for task_id"},
            },
        },
    },
    {
        "name": "video.result",
        "description": "Retrieve final media artifacts and download URLs for a completed video task.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Video task ID"},
                "video_task_id": {"type": "string", "description": "Alias for task_id"},
            },
        },
    },
    {
        "name": "video.depth",
        "description": "Queue temporal RGB-to-depth conversion of a registered project video.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_artifact_id": {"type": "string", "description": "Source video artifact ID"},
                "model": {"type": "string", "default": "video-depth-anything"},
                "output_format": {"type": "string", "default": "mp4"},
                "project_id": {"type": "string"},
                "idempotency_key": {"type": "string"},
            },
            "required": ["source_artifact_id"],
        },
    },
    {
        "name": "video.sr",
        "description": "Restore a project video with SeedVR2.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_artifact_id": {"type": "string", "description": "Source video artifact ID"},
                "target_width": {"type": "integer"},
                "target_height": {"type": "integer"},
                "quality_mode": {"type": "string", "default": "standard"},
                "backend": {"type": "string", "default": "seedvr2"},
                "seed": {"type": "integer", "default": 666},
            },
            "required": ["source_artifact_id", "target_width", "target_height"],
        },
    },
    {
        "name": "video.interpolate",
        "description": "Double a project video CFR frame rate with RIFE.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_artifact_id": {"type": "string", "description": "Source video artifact ID"},
                "target_fps_num": {"type": "integer", "default": 48},
                "target_fps_den": {"type": "integer", "default": 1},
                "backend": {"type": "string", "default": "rife"},
            },
            "required": ["source_artifact_id"],
        },
    },
]

VIDEO_TOOLS_SCHEMA.extend([
    {"name": "video.capabilities", "description": "Read host defaults, adapter limits and live Singularity readiness; readiness does not prove GPU generation acceptance.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "video.import", "description": "Import an allowed HTTPS asset or up to 3 MiB of PNG/JPEG/WebP bytes into Video; does not resolve central ContentRefs.",
     "inputSchema": {"type": "object", "properties": {
         "project_id": {"type": "string"}, "filename": {"type": "string"},
         "expected_sha256": {"type": "string", "pattern": "^[0-9a-fA-F]{64}$"},
         "source_url": {"type": "string"}, "content_base64": {"type": "string", "maxLength": 4194304},
     }, "required": ["project_id", "filename", "expected_sha256"],
     "oneOf": [{"required": ["source_url"], "not": {"required": ["content_base64"]}},
               {"required": ["content_base64"], "not": {"required": ["source_url"]}}]}},
])
registry = EtcdRegistry(domain="video", tools=VIDEO_TOOLS_SCHEMA)


def _latent_errors(function):
    """Expose stable post-processing errors without forwarding internal messages."""
    @wraps(function)
    def guarded(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (TaskNotFound, TaskConflict, ValueError, ExecutionError, ArtifactError, httpx.HTTPError) as exc:
            retryable = False
            if isinstance(exc, TaskNotFound):
                code, message = 'task_not_found', 'The task is unavailable in the authorized scope.'
            elif isinstance(exc, TaskConflict):
                code, message = 'idempotency_conflict', 'The idempotency key belongs to different inputs.'
            elif isinstance(exc, ValueError):
                code, message = 'invalid_request', 'Provide a valid public task ID and supported parameters; MP4 input is not accepted.'
            elif isinstance(exc, ExecutionError):
                code = str(exc).split(':', 1)[0]
                messages = {
                    'source_not_found': 'The source is unavailable in the authorized scope.',
                    'archive_unavailable': 'Result archival is unavailable; no processing task was submitted.',
                    'runtime_not_ready': 'The post-processing runtime is not ready; no processing task was submitted.',
                    'source_not_ready': 'The source task has not completed successfully.',
                    'unsupported_source_route': 'The source route does not support latent post-processing.',
                    'missing_latent_bundle': 'The source has no retained latent bundle; MP4-only tasks cannot be processed.',
                    'source_node_mismatch': 'The source resources are not available on the processing node.',
                    'source_identity_mismatch': 'The source resource identity does not match the task.',
                    'invalid_source_descriptor': 'The source resource descriptor is invalid.',
                    'source_changed_after_registration': 'The registered source resources have changed.',
                }
                if code in messages:
                    message = messages[code]
                    retryable = code in {'archive_unavailable', 'runtime_not_ready'}
                else:
                    code, message = 'postprocessing_unavailable', 'Post-processing is unavailable; retain the task ID and retry the query.'
                    retryable = True
            else:
                code, message = 'postprocessing_unavailable', 'Resource access is unavailable; retain the task ID and retry the query.'
                retryable = True
            result = _result({'error': {'code': code, 'message': message, 'retryable': retryable}})
            result.is_error = True
            return result
    return guarded


@mcp.tool(name="video.h3.latent.upscale.generate")
@_latent_errors
def video_h3_latent_generate(source_video_task_id: str, project_id: str | None = None,
                            idempotency_key: str | None = None, profile_id: str | None = None,
                            target_width: int | None = None, target_height: int | None = None,
                            seed: int | None = None) -> types.CallToolResult:
    """Post-process a completed H3 task using retained same-node latent resources; never accept MP4 input."""
    adapter = executor.processing["h3-latent-upscale"]
    record = adapter.generate(source_video_task_id, project_id=project_id, idempotency_key=idempotency_key,
                              profile_id=profile_id, target_width=target_width, target_height=target_height, seed=seed)
    return _result(adapter.public_status(record))


@mcp.tool(name="video.h3.latent.upscale.status")
@_latent_errors
def video_h3_latent_status(video_task_id: str) -> types.CallToolResult:
    adapter = executor.processing["h3-latent-upscale"]
    return _result(adapter.public_status(adapter.status(video_task_id)))


@mcp.tool(name="video.h3.latent.upscale.result")
@_latent_errors
def video_h3_latent_result(video_task_id: str) -> types.CallToolResult:
    adapter = executor.processing["h3-latent-upscale"]
    return _result(adapter.public_result(adapter.result(video_task_id)))


@mcp.tool(name="video.h3.latent.upscale.preview")
@_latent_errors
def video_h3_latent_preview(video_task_id: str) -> types.CallToolResult:
    adapter = executor.processing["h3-latent-upscale"]
    return _result(adapter.public_result(adapter.result(video_task_id), preview=True))


@mcp.tool(name="video.depth.generate")
def video_depth_generate(project_id: str, idempotency_key: str, source_artifact_id: str,
                         model: str = "video-depth-anything", output_format: str = "mp4") -> types.CallToolResult:
    """Queue temporal RGB-to-depth conversion of a registered project video."""
    record = executor.depth.generate(project_id=project_id, idempotency_key=idempotency_key,
                                    source_artifact_id=source_artifact_id, model=model, output_format=output_format)
    return _result(executor.depth.public_status(record))


@mcp.tool(name="video.depth.status")
def video_depth_status(video_task_id: str) -> types.CallToolResult:
    return _result(executor.depth.public_status(executor.depth.status(video_task_id)))


@mcp.tool(name="video.depth.result")
def video_depth_result(video_task_id: str) -> types.CallToolResult:
    return _result(executor.depth.public_result(executor.depth.result(video_task_id)))


@mcp.tool(name="video.depth.preview")
def video_depth_preview(video_task_id: str) -> types.CallToolResult:
    """Get an immutable side-by-side source RGB / grayscale depth MP4."""
    return _result(executor.depth.public_result(executor.depth.result(video_task_id), preview=True))



@mcp.tool(name="video.sr.generate")
def video_sr_generate(project_id: str, idempotency_key: str, source_artifact_id: str,
                      target_width: int, target_height: int, quality_mode: str = "standard",
                      backend: str = "seedvr2", seed: int = 666) -> types.CallToolResult:
    """Restore a project video with SeedVR2; preserve CFR timing and aspect ratio."""
    adapter = executor.processing["sr"]
    record = adapter.generate(project_id=project_id, idempotency_key=idempotency_key,
        source_artifact_id=source_artifact_id, target_width=target_width, target_height=target_height,
        quality_mode=quality_mode, backend=backend, seed=seed)
    return _result(adapter.public_status(record))


@mcp.tool(name="video.interpolate.generate")
def video_interpolate_generate(project_id: str, idempotency_key: str, source_artifact_id: str,
                               target_fps_num: int = 48, target_fps_den: int = 1,
                               backend: str = "rife") -> types.CallToolResult:
    """Double a project video's CFR frame rate with RIFE, preserving duration and dimensions."""
    adapter = executor.processing["interpolate"]
    record = adapter.generate(project_id=project_id, idempotency_key=idempotency_key,
        source_artifact_id=source_artifact_id, target_fps_num=target_fps_num,
        target_fps_den=target_fps_den, backend=backend)
    return _result(adapter.public_status(record))


@mcp.tool(name="video.sr.status")
def video_sr_status(video_task_id: str) -> types.CallToolResult:
    adapter = executor.processing["sr"]
    return _result(adapter.public_status(adapter.status(video_task_id)))


@mcp.tool(name="video.sr.result")
def video_sr_result(video_task_id: str) -> types.CallToolResult:
    adapter = executor.processing["sr"]
    return _result(adapter.public_result(adapter.result(video_task_id)))


@mcp.tool(name="video.sr.preview")
def video_sr_preview(video_task_id: str) -> types.CallToolResult:
    adapter = executor.processing["sr"]
    return _result(adapter.public_result(adapter.result(video_task_id), preview=True))


@mcp.tool(name="video.interpolate.status")
def video_interpolate_status(video_task_id: str) -> types.CallToolResult:
    adapter = executor.processing["interpolate"]
    return _result(adapter.public_status(adapter.status(video_task_id)))


@mcp.tool(name="video.interpolate.result")
def video_interpolate_result(video_task_id: str) -> types.CallToolResult:
    adapter = executor.processing["interpolate"]
    return _result(adapter.public_result(adapter.result(video_task_id)))


@mcp.tool(name="video.interpolate.preview")
def video_interpolate_preview(video_task_id: str) -> types.CallToolResult:
    adapter = executor.processing["interpolate"]
    return _result(adapter.public_result(adapter.result(video_task_id), preview=True))


def _result(value: dict[str, object]) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(value, ensure_ascii=False))], structuredContent=value)


@mcp.tool(name="video.capabilities")
def video_capabilities() -> types.CallToolResult:
    """Read adapter limits and live Singularity readiness; readiness is not GPU acceptance."""
    return _result(executor.capabilities())


@mcp.tool(name="video.import")
def video_import(project_id: str, filename: str, expected_sha256: str,
                 source_url: str | None = None, content_base64: str | None = None) -> types.CallToolResult:
    """Import exactly one allowed HTTPS source or <=3 MiB image payload; never resolve central IDs."""
    if (source_url is None) == (content_base64 is None):
        raise ValueError("provide exactly one of source_url or content_base64")
    common = dict(project_id=project_id, filename=filename, expected_sha256=expected_sha256)
    record = (executor.import_image(**common, content_base64=content_base64) if content_base64 is not None
              else executor.import_asset(**common, source_url=source_url))
    return _result({"status": "completed", "project_id": record.project_id, "artifact": record.model_dump(),
                    "download_path": artifacts.download_path(record.artifact_id)})


@mcp.tool(name="video.import_prepare")
def video_import_prepare(project_id: str, idempotency_key: str, filename: str, size: int,
                         sha256: str, purpose: str, source_content_ref: dict[str, str] | None = None) -> types.CallToolResult:
    """Prepare resumable media import. A claimed ContentRef grants no central read permission."""
    return _result(executor.imports.prepare(project_id=project_id, idempotency_key=idempotency_key,
        filename=filename, size=size, sha256=sha256, purpose=purpose, source_content_ref=source_content_ref))


@mcp.tool(name="video.import_chunk")
def video_import_chunk(project_id: str, import_id: str, offset: int, content_base64: str, sha256: str) -> types.CallToolResult:
    """Append up to 512 KiB at the confirmed offset; identical repeated bytes are safe."""
    return _result(executor.imports.chunk(project_id=project_id, import_id=import_id, offset=offset,
        content_base64=content_base64, sha256=sha256))


@mcp.tool(name="video.import_status")
def video_import_status(project_id: str, import_id: str) -> types.CallToolResult:
    """Read the durable offset and final immutable identity after an interrupted upload."""
    return _result(executor.imports.status(project_id=project_id, import_id=import_id))


@mcp.tool(name="video.import_commit")
def video_import_commit(project_id: str, import_id: str) -> types.CallToolResult:
    """Verify all bytes and decode media before publishing a native Video artifact."""
    return _result(executor.imports.commit(project_id=project_id, import_id=import_id))


@mcp.tool(name="video.preflight")
def video_preflight(project_id: str, prompt: str, references: dict[str, list[dict[str, str]]],
                    duration_seconds: int = 5, aspect_ratio: str = "16:9",
                    model: str = "minimax-h3-ref2va", route: str | None = None,
                    quality_profile: str = "du-0") -> types.CallToolResult:
    """Validate Singularity media and reveal native output and reference labels without creating a task."""
    selected = route or executor.runtime_route
    if selected != "h3-singularity":
        raise ValueError("video.preflight currently supports h3-singularity")
    executor.runtime(selected)
    request, digest = executor._normalize(require_project_id(project_id), model, prompt,
        duration_seconds, aspect_ratio, references, selected, quality_profile)
    return _result({"status": "validated", "gpu_verified": False, "input_digest": digest,
                    "resolved_output": request["resolved_output"], "reference_mapping": request["reference_mapping"],
                    "reference_media": request["reference_media"], "prompt": request["prompt"]})


# Publish the registered parameter schemas instead of maintaining a second
# copy for Studio discovery. MCP is pinned; parity is checked in local tests.
for _name in ("video.import_prepare", "video.import_chunk", "video.import_status", "video.import_commit", "video.preflight"):
    _tool = mcp._tool_manager.get_tool(_name)
    VIDEO_TOOLS_SCHEMA.append({"name": _name, "description": _tool.description, "inputSchema": _tool.parameters})


@mcp.tool(name="artifact.import")
def artifact_import(project_id: str, source_url: str, filename: str, expected_sha256: str) -> types.CallToolResult:
    record = executor.import_asset(project_id=project_id, source_url=source_url, filename=filename, expected_sha256=expected_sha256)
    return _result({"status": "completed", "project_id": project_id, "artifact": record.model_dump(),
                    "download_path": artifacts.download_path(record.artifact_id)})


@mcp.tool(name="video.create")
def video_create(prompt: str,
                 project_id: str,
                 idempotency_key: str,
                 model: str = "minimax-h3-ref2va",
                 duration_seconds: int = 5,
                 aspect_ratio: str = "16:9",
                 references: dict[str, list[dict[str, str]]] | None = None,
                 route: str | None = None,
                 quality_profile: str = "du-0") -> types.CallToolResult:
    """Create an asynchronous video generation task."""
    record = executor.generate(project_id=project_id, idempotency_key=idempotency_key, model=model,
                               prompt=prompt, duration_seconds=duration_seconds,
                               aspect_ratio=aspect_ratio, references=references or {}, route=route,
                               quality_profile=quality_profile)
    return _result({"task_id": record.video_task_id, "video_task_id": record.video_task_id,
                    "status": record.status, "created_at": record.created_at, "error": record.error,
                    "resolved_output": record.request.get("resolved_output"),
                    "reference_mapping": record.request.get("reference_mapping")})


@mcp.tool(name="video.generate")
def video_generate(project_id: str, idempotency_key: str, model: str, prompt: str,
                   duration_seconds: int, aspect_ratio: str,
                   references: dict[str, list[dict[str, str]]], route: str,
                   quality_profile: str = "du-0") -> types.CallToolResult:
    """Generate with a business model and an explicitly selected runtime route.

    `route` selects a configured channel such as h3-sol, h3-vdn, or fal.
    The fal and self-hosted H3 channels use `model=minimax-h3-ref2va`.
    References must be registered project artifacts.
    Routes never fall back; retain the returned task id.
    """
    record = executor.generate(project_id=project_id, idempotency_key=idempotency_key, model=model,
                               prompt=prompt, duration_seconds=duration_seconds,
                               aspect_ratio=aspect_ratio, references=references, route=route,
                               quality_profile=quality_profile)
    return _result({"task_id": record.video_task_id, "video_task_id": record.video_task_id,
                    "status": record.status, "created_at": record.created_at, "error": record.error})


@mcp.tool(name="video.status")
def video_status(video_task_id: str = "", task_id: str = "") -> types.CallToolResult:
    tid = video_task_id or task_id
    if not tid:
        return _result({"error": "task_id is required"})
    record = executor.status(tid)
    return _result({"task_id": record.video_task_id, "video_task_id": record.video_task_id, "status": record.status,
                    "created_at": record.created_at, "updated_at": record.updated_at, "error": record.error,
                    "timing": record.timing, "runtime_metrics": record.runtime_metrics,
                    "service": record.service, "runtime_route": record.runtime_route,
                    "execution_instance_id": record.execution_instance_id, "stage": record.runtime_stage})


@mcp.tool(name="video.result")
def video_result(video_task_id: str = "", task_id: str = "") -> types.CallToolResult:
    tid = video_task_id or task_id
    if not tid:
        return _result({"error": "task_id is required"})
    record = executor.result(tid)
    if record.service in executor.processing:
        res = executor.processing[record.service].public_result(record)
        res["task_id"] = record.video_task_id
        return _result(res)
    if record.service == "depth":
        res = executor.depth.public_result(record)
        res["task_id"] = record.video_task_id
        return _result(res)
    artifact = artifacts.get(record.artifact_id, record.project_id)
    value = {"task_id": record.video_task_id, "video_task_id": record.video_task_id, "artifact_id": artifact.artifact_id,
             "model": record.request["model"], "runtime_version": record.runtime_version or executor.runtime_version,
             "runtime_route": record.runtime_route, "service": record.service,
             "input_digest": record.input_digest, "media": record.media,
             "download_path": artifacts.download_path(artifact.artifact_id)}
    return _result(value)


@mcp.tool(name="video.depth")
def video_depth(source_artifact_id: str, project_id: str, idempotency_key: str = "",
                model: str = "video-depth-anything", output_format: str = "mp4") -> types.CallToolResult:
    import uuid
    actual_idem = idempotency_key or f"idem_{uuid.uuid4().hex[:12]}"
    record = executor.depth.generate(project_id=project_id, idempotency_key=actual_idem,
                                     source_artifact_id=source_artifact_id, model=model, output_format=output_format)
    res = executor.depth.public_status(record)
    res["task_id"] = record.video_task_id
    return _result(res)


@mcp.tool(name="video.sr")
def video_sr(source_artifact_id: str, target_width: int, target_height: int,
             project_id: str, idempotency_key: str = "",
             quality_mode: str = "standard", backend: str = "seedvr2", seed: int = 666) -> types.CallToolResult:
    import uuid
    actual_idem = idempotency_key or f"idem_{uuid.uuid4().hex[:12]}"
    adapter = executor.processing["sr"]
    record = adapter.generate(project_id=project_id, idempotency_key=actual_idem,
        source_artifact_id=source_artifact_id, target_width=target_width, target_height=target_height,
        quality_mode=quality_mode, backend=backend, seed=seed)
    res = adapter.public_status(record)
    res["task_id"] = record.video_task_id
    return _result(res)


@mcp.tool(name="video.interpolate")
def video_interpolate(source_artifact_id: str, project_id: str, idempotency_key: str = "",
                      target_fps_num: int = 48, target_fps_den: int = 1,
                      backend: str = "rife") -> types.CallToolResult:
    import uuid
    actual_idem = idempotency_key or f"idem_{uuid.uuid4().hex[:12]}"
    adapter = executor.processing["interpolate"]
    record = adapter.generate(project_id=project_id, idempotency_key=actual_idem,
        source_artifact_id=source_artifact_id, target_fps_num=target_fps_num,
        target_fps_den=target_fps_den, backend=backend)
    res = adapter.public_status(record)
    res["task_id"] = record.video_task_id
    return _result(res)


def transport_security_from_environment() -> TransportSecuritySettings:
    hosts = [x.strip() for x in os.environ.get("VIDEO_MCP_ALLOWED_HOSTS", "").split(",") if x.strip()]
    origins = [x.strip() for x in os.environ.get("VIDEO_MCP_ALLOWED_ORIGINS", "").split(",") if x.strip()]
    return TransportSecuritySettings(enable_dns_rebinding_protection=True,
                                     allowed_hosts=hosts or ["127.0.0.1:*", "localhost:*", "[::1]:*"],
                                     allowed_origins=origins or ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"])


async def health(request: Request) -> JSONResponse:
    artifacts.ensure_ready(); tasks.ensure_ready()
    return JSONResponse({"status": "ok", "contract_version": "v1"})


async def artifact_content(request: Request) -> Response:
    try:
        record = artifacts.get(request.path_params["artifact_id"])
        return FileResponse(artifacts.content_path(record), media_type=record.media_type, filename=record.filename)
    except (ValueError, ArtifactNotFound):
        return JSONResponse({"error": "artifact_not_found"}, status_code=404)


class BearerAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/health" or request.url.path.startswith("/runtime-artifacts/"):
            return await call_next(request)
        if request.url.path in {"/dashboard", "/dashboard/"} or request.url.path.startswith(("/dashboard/frontend/", "/dashboard/static/", "/dashboard/tasks/")):
            response = await call_next(request)
            response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Referrer-Policy"] = "no-referrer"
            return response
        token = os.environ.get("VIDEO_MCP_BEARER_TOKEN", "").strip()
        if request.url.path.startswith("/api/") and not token:
            return JSONResponse({"error": "authentication_not_configured"}, status_code=503)
        if token and not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {token}"):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        response = await call_next(request)
        if request.url.path in {"/mcp", "/api/tasks", "/api/artifacts/import"} or request.url.path.endswith("/result"):
            dashboard.resources.record_request(response.status_code >= 400)
        return response


@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    artifacts.ensure_ready(); tasks.ensure_ready()
    await registry.start()
    async with mcp.session_manager.run():
        dashboard.recover_incomplete_submissions()
        pollers = [asyncio.create_task(coro) for coro in (dashboard.poll(), dashboard.resources.poll(), dashboard.resources.poll_protocol())]
        try:
            yield
        finally:
            await registry.stop()
            for poller in pollers:
                poller.cancel()
            for poller in pollers:
                with contextlib.suppress(asyncio.CancelledError):
                    await poller


app = Starlette(routes=[*dashboard.routes(), Route("/health", health),
                        Route("/artifacts/{artifact_id:str}/content", artifact_content),
                        Route("/runtime-artifacts/{artifact_id:str}/content", artifact_content),
                        Mount("/", app=mcp.streamable_http_app(json_response=True, stateless_http=True,
                                                               transport_security=transport_security_from_environment()))],
                lifespan=lifespan)
app.add_middleware(BearerAuthMiddleware)
