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

from .artifacts import ArtifactError, ArtifactNotFound, ArtifactStore
from .executor import ExecutionError, VideoExecutor
from .dashboard import Dashboard
from .tasks import TaskConflict, TaskNotFound, TaskStore

logging.getLogger("httpx").setLevel(logging.WARNING)

artifacts = ArtifactStore.from_environment()
tasks = TaskStore.from_environment()
executor = VideoExecutor(artifacts, tasks)
dashboard = Dashboard(executor)
mcp = MCPServer("VerdantFlare Video")


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
                    retryable = code == 'archive_unavailable'
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


@mcp.tool(name="artifact.import")
def artifact_import(project_id: str, source_url: str, filename: str, expected_sha256: str) -> types.CallToolResult:
    record = executor.import_asset(project_id=project_id, source_url=source_url, filename=filename, expected_sha256=expected_sha256)
    return _result({"status": "completed", "project_id": project_id, "artifact": record.model_dump(),
                    "download_path": artifacts.download_path(record.artifact_id)})


@mcp.tool(name="video.generate")
def video_generate(project_id: str, idempotency_key: str, model: str, prompt: str,
                   duration_seconds: int, aspect_ratio: str,
                   references: dict[str, list[dict[str, str]]], route: str) -> types.CallToolResult:
    """Generate with a business model and an explicitly selected runtime route.

    `model=minimax-h3-ref2va` remains the business contract. `route` selects a
    configured channel such as h3-sol or h3-vdn. References must be
    registered artifacts. Routes never fall back; retain the returned task id.
    """
    record = executor.generate(project_id=project_id, idempotency_key=idempotency_key, model=model,
                               prompt=prompt, duration_seconds=duration_seconds,
                               aspect_ratio=aspect_ratio, references=references, route=route)
    return _result({"video_task_id": record.video_task_id, "status": record.status,
                    "created_at": record.created_at, "error": record.error})


@mcp.tool(name="video.status")
def video_status(video_task_id: str) -> types.CallToolResult:
    record = executor.status(video_task_id)
    return _result({"video_task_id": record.video_task_id, "status": record.status,
                    "created_at": record.created_at, "updated_at": record.updated_at, "error": record.error,
                    "timing": record.timing, "runtime_metrics": record.runtime_metrics,
                    "service": record.service, "runtime_route": record.runtime_route,
                    "execution_instance_id": record.execution_instance_id, "stage": record.runtime_stage})


@mcp.tool(name="video.result")
def video_result(video_task_id: str) -> types.CallToolResult:
    record = executor.result(video_task_id)
    if record.service in executor.processing:
        return _result(executor.processing[record.service].public_result(record))
    if record.service == "depth":
        return _result(executor.depth.public_result(record))
    artifact = artifacts.get(record.artifact_id, record.project_id)
    value = {"video_task_id": record.video_task_id, "artifact_id": artifact.artifact_id,
             "model": record.request["model"], "runtime_version": record.runtime_version or executor.runtime_version,
             "runtime_route": record.runtime_route, "service": record.service,
             "input_digest": record.input_digest, "media": record.media,
             "download_path": artifacts.download_path(artifact.artifact_id)}
    return _result(value)


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
        if request.url.path in {"/dashboard", "/dashboard/"} or request.url.path.startswith(("/dashboard/static/", "/dashboard/tasks/")):
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
    async with mcp.session_manager.run():
        dashboard.recover_incomplete_submissions()
        pollers = [asyncio.create_task(coro) for coro in (dashboard.poll(), dashboard.resources.poll(), dashboard.resources.poll_protocol())]
        try:
            yield
        finally:
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
