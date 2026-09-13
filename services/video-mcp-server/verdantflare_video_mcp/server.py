from __future__ import annotations

import asyncio
import contextlib
import hmac
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


@mcp.tool(name="video.depth.generate")
def video_depth_generate(project_id: str, idempotency_key: str, source_artifact_id: str,
                         model: str = "video-depth-anything", output_format: str = "mp4") -> types.CallToolResult:
    """Submit RGB video depth conversion through the Video MCP runtime."""
    url = os.environ.get("VIDEO_DEPTH_RUNTIME_URL", "http://video-depth-anything-api:8000").rstrip("/")
    source = artifacts.get(source_artifact_id, project_id)
    try:
        response = executor.client.post(f"{url}/v1/depth", json={"project_id": project_id,
            "idempotency_key": idempotency_key, "source_artifact_id": str(artifacts.content_path(source)),
            "model": model, "output_format": output_format}, timeout=30)
        response.raise_for_status()
        return _result(response.json())
    except Exception as error:
        raise ExecutionError(f"depth runtime request failed: {error}") from error


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
                   references: dict[str, list[dict[str, str]]], service: str = "h3") -> types.CallToolResult:
    """Generate with explicit service: h3 (legacy default, 4-15s) or h3-sol (5/10/15s).

    Both use model=minimax-h3-ref2va, portrait 9:16 and seed=7. References must be
    registered artifacts. Routes never fall back; retain the returned task id.
    """
    record = executor.generate(project_id=project_id, idempotency_key=idempotency_key, model=model,
                               prompt=prompt, duration_seconds=duration_seconds,
                               aspect_ratio=aspect_ratio, references=references, service=service)
    return _result({"video_task_id": record.video_task_id, "status": record.status, "created_at": record.created_at})


@mcp.tool(name="video.status")
def video_status(video_task_id: str) -> types.CallToolResult:
    record = executor.status(video_task_id)
    return _result({"video_task_id": record.video_task_id, "status": record.status,
                    "created_at": record.created_at, "updated_at": record.updated_at, "error": record.error,
                    "service": record.service, "execution_instance_id": record.execution_instance_id, "stage": record.runtime_stage})


@mcp.tool(name="video.result")
def video_result(video_task_id: str) -> types.CallToolResult:
    record = executor.result(video_task_id)
    artifact = artifacts.get(record.artifact_id, record.project_id)
    value = {"video_task_id": record.video_task_id, "artifact_id": artifact.artifact_id,
             "model": record.request["model"], "runtime_version": record.runtime_version or executor.runtime_version, "service": record.service,
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
        if request.url.path in {"/dashboard", "/dashboard/"} or request.url.path.startswith("/dashboard/static/"):
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
