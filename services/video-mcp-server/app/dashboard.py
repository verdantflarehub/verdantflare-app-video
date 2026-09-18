from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.routing import Route
from starlette.concurrency import run_in_threadpool

from .artifacts import ArtifactError, ArtifactNotFound
from .executor import ExecutionError
from .resources import Resources
from .tasks import TaskConflict, TaskNotFound, TaskRecord, TaskStore

STATIC = Path(__file__).parent.parent / "frontend"
DIST = STATIC / "dist"


class Reference(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    artifact_id: str
    purpose: str = Field(min_length=1, max_length=256)


class References(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    images: list[Reference] = Field(default_factory=list, max_length=9)
    videos: list[Reference] = Field(default_factory=list, max_length=3)
    audios: list[Reference] = Field(default_factory=list, max_length=3)


class Submission(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    project_id: str = Field(min_length=1, max_length=64)
    idempotency_key: str = Field(min_length=1, max_length=128)
    model: str = "minimax-h3-ref2va"
    route: str = Field(min_length=1, max_length=64)
    prompt: str = Field(min_length=1, max_length=16000)
    duration_seconds: int = Field(ge=4, le=15)
    aspect_ratio: str = "9:16"
    references: References


class ImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    project_id: str
    source_url: str = Field(max_length=4096)
    filename: str = Field(max_length=128)
    expected_sha256: str = Field(min_length=64, max_length=64)


def public_task(record):
    record = TaskStore.with_timing(record)
    model = record.request.get("model")
    if not model:
        model = {
            "sr": "seedvr2",
            "interpolate": "rife",
            "depth": "video-depth-anything",
            "h3-latent-upscale": "h3-latent-upscaler",
        }.get(record.service)
    return {"video_task_id": record.video_task_id, "project_id": record.project_id,
            "idempotency_key": record.idempotency_key, "service": record.service, "route": record.runtime_route or record.request.get("route"),
            "model": model,
            "prompt": record.request.get("prompt", ""),
            "duration_seconds": record.request.get("duration_seconds"),
            "aspect_ratio": record.request.get("aspect_ratio"), "seed": record.request.get("seed", 7),
            "status": record.status, "created_at": record.created_at, "updated_at": record.updated_at,
            "completed_at": record.completed_at or (record.updated_at if record.status in {"succeeded", "failed", "cancelled"} else None),
            "artifact_id": record.artifact_id, "media": record.media, "error": record.error,
            "input_digest": record.input_digest, "execution_instance_id": record.execution_instance_id, "association_source": "runtime" if record.execution_instance_id else None, "runtime_stage": record.runtime_stage,
            "dispatched_at": record.dispatched_at, "timing": record.timing, "runtime_metrics": record.runtime_metrics}


class Dashboard:
    def __init__(self, executor):
        self.executor = executor
        self.resources = Resources()
        self.resources.task_provider = self.records
        self.result_lock = threading.Lock()
        self.services = {"h3": "unknown", "h3-sol": "not_connected", "mcp": "ready"}
        self.sync_errors = 0
        self.synced_at = None

    def records(self):
        self.executor.tasks.ensure_ready()
        records = []
        for path in self.executor.tasks.root.glob("video_task_*.json"):
            try:
                records.append(TaskRecord.model_validate_json(path.read_text()))
            except (OSError, ValueError):
                logging.getLogger(__name__).warning("Invalid task record omitted from dashboard")
        return sorted(records, key=lambda x: (x.created_at, x.video_task_id), reverse=True)

    def sync(self):
        try:
            response = self.executor.client.get(f"{self.executor.runtime_url}/health", timeout=5)
            self.services["h3"] = "ready" if response.is_success else "unavailable"
        except httpx.HTTPError:
            self.services["h3"] = "unavailable"
        if self.executor.sol_url and self.executor.sol_token:
            try:
                response = self.executor.client.get(f"{self.executor.sol_url}/health", timeout=5)
                self.services["h3-sol"] = "ready" if response.is_success else "unavailable"
            except httpx.HTTPError:
                self.services["h3-sol"] = "unavailable"
        errors = 0
        for record in self.records():
            if record.status in {"queued", "running"} and (record.runtime_task_id or record.service in {"depth", "sr", "interpolate"}):
                try:
                    self.executor.status(record.video_task_id)
                except (ExecutionError, OSError, ValueError):
                    errors += 1
        self.sync_errors = errors
        self.synced_at = datetime.now(UTC).isoformat()

    def recover_incomplete_submissions(self):
        for record in self.records():
            if record.status == "queued" and not record.runtime_task_id and record.service not in {"depth", "sr", "interpolate"}:
                self.executor.tasks.update(record, status="failed", error={
                    "code": "submission_unconfirmed",
                    "message": "Submission interrupted; do not resubmit automatically"})

    async def poll(self):
        while True:
            try:
                await run_in_threadpool(self.sync)
            except Exception:
                self.sync_errors += 1
                logging.getLogger(__name__).warning("Dashboard status synchronization failed")
            await asyncio.sleep(2)

    def snapshot(self, request):
        query = request.query_params
        page = max(1, int(query.get("page", "1")))
        size = max(1, min(100, int(query.get("page_size", "24"))))
        records = self.records()
        projects = sorted({r.project_id for r in records})
        rows = [public_task(r) for r in records]
        for key in ("project_id", "service"):
            value = query.get(key, "all")
            if value != "all":
                rows = [r for r in rows if r[key] == value]
        search = query.get("q", "").casefold()
        if search:
            rows = [r for r in rows if search in " ".join(str(r[k]) for k in (
                "prompt", "project_id", "idempotency_key", "video_task_id")).casefold()]
        counts = {s: sum(r["status"] == s for r in rows) for s in (
            "queued", "running", "succeeded", "failed", "cancelled")}
        status_filter = query.get("status", "all")
        if status_filter != "all":
            rows = [r for r in rows if r["status"] == status_filter]
        elapsed = [(datetime.fromisoformat(r["completed_at"]) - datetime.fromisoformat(r["created_at"])).total_seconds()
                   for r in rows if r["status"] == "succeeded"]
        return {"tasks": rows[(page-1)*size:page*size], "total": len(rows), "page": page,
                "page_size": size, "projects": projects, "counts": counts,
                "completion_times": [r["completed_at"] for r in rows if r["status"] == "succeeded"],
                "average_elapsed_seconds": round(sum(elapsed)/len(elapsed), 1) if elapsed else None,
                "services": dict(self.services), "synced_at": self.synced_at, "sync_errors": self.sync_errors}

    def detail(self, task_id):
        record = self.executor.tasks.get(task_id)
        value = public_task(record)
        value["runtime_version"] = record.runtime_version or self.executor.runtime_version
        value["references"] = []
        for kind, items in record.request.get("references", {}).items():
            for item in items:
                try:
                    artifact = self.executor.artifacts.get(item["artifact_id"], record.project_id)
                    value["references"].append({"kind": kind, "purpose": item["purpose"], **artifact.model_dump()})
                except (ArtifactError, ValueError):
                    value["references"].append({"kind": kind, "purpose": item["purpose"], "unavailable": True})
        if record.artifact_id:
            value["artifact"] = self.executor.artifacts.get(record.artifact_id, record.project_id).model_dump()
        return value

    def thumbnail(self, task_id):
        record = self.executor.tasks.get(task_id)
        reference = record.request.get("references", {}).get("images", [])
        artifact_id = record.artifact_id or (reference[0]["artifact_id"] if reference else None)
        if not artifact_id:
            return JSONResponse({"error": "preview_unavailable"}, status_code=404)
        artifact = self.executor.artifacts.get(artifact_id, record.project_id)
        directory = self.executor.artifacts.root / "previews"
        directory.mkdir(exist_ok=True)
        output = directory / f"{artifact.sha256}.jpg"
        with self.result_lock:
            if not output.exists():
                with tempfile.NamedTemporaryFile(suffix=".jpg", dir=directory) as temporary:
                    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i",
                                    str(self.executor.artifacts.content_path(artifact)),
                                    "-frames:v", "1", "-vf", "scale=480:320:force_original_aspect_ratio=decrease",
                                    temporary.name], check=True, capture_output=True, timeout=30)
                    output.write_bytes(Path(temporary.name).read_bytes())
        return FileResponse(output, media_type="image/jpeg", headers={"Cache-Control": "private, no-store"})

    async def endpoint(self, request: Request):
        try:
            action = request.url.path
            if action == "/api/dashboard":
                value = await run_in_threadpool(self.snapshot, request)
            elif action.endswith("/thumbnail"):
                return await run_in_threadpool(self.thumbnail, request.path_params["task_id"])
            elif request.method == "GET":
                value = await run_in_threadpool(self.detail, request.path_params["task_id"])
            elif action.endswith("/result"):
                def result():
                    with self.result_lock:
                        self.executor.result(request.path_params["task_id"])
                        return self.detail(request.path_params["task_id"])
                value = await run_in_threadpool(result)
            else:
                body = bytearray()
                async for chunk in request.stream():
                    body.extend(chunk)
                    if len(body) > 65536:
                        return JSONResponse({"error": "request_too_large"}, status_code=413)
                if action == "/api/artifacts/import":
                    inputs = ImportRequest.model_validate_json(body)
                    artifact = await run_in_threadpool(self.executor.import_asset, **inputs.model_dump())
                    value = artifact.model_dump()
                else:
                    inputs = Submission.model_validate_json(body)
                    service = "h3-sol" if inputs.route.startswith("h3-sol") else "h3"
                    if service not in {"h3", "h3-sol"}:
                        return JSONResponse({"error": "service_not_connected"}, status_code=409)
                    if service == "h3-sol":
                        selected_route = inputs.route
                        route_config = self.executor.runtime_routes.get(selected_route)
                        route_requires_token = route_config is not None and route_config["requires_token"]
                        if (route_config is None and not (self.executor.sol_url and self.executor.sol_token)) or (route_requires_token and not self.executor.sol_token):
                            return JSONResponse({"error": "service_not_connected"}, status_code=409)
                    kwargs = inputs.model_dump(exclude={"model"})
                    record = await run_in_threadpool(self.executor.generate, model=inputs.model, **kwargs)
                    value = public_task(record)
            return JSONResponse(value, headers={"Cache-Control": "no-store"})
        except (TaskNotFound, ArtifactNotFound):
            return JSONResponse({"error": "not_found"}, status_code=404)
        except TaskConflict:
            return JSONResponse({"error": "idempotency_conflict"}, status_code=409)
        except (ValidationError, ValueError, KeyError, TypeError):
            return JSONResponse({"error": "invalid_request"}, status_code=400)
        except (ExecutionError, ArtifactError, httpx.HTTPError):
            return JSONResponse({"error": "upstream_operation_failed"}, status_code=502)
        except Exception:
            logging.getLogger(__name__).warning("Dashboard operation failed")
            return JSONResponse({"error": "operation_failed"}, status_code=500)

    def routes(self):
        async def shell(request):
            if request.url.path.endswith("/"):
                return RedirectResponse("../dashboard")
            entry = DIST / "index.html" if (DIST / "index.html").is_file() else STATIC / "dashboard.html"
            return FileResponse(entry, headers={"Cache-Control": "no-store"})

        async def task_shell(request):
            return FileResponse(STATIC / "task-detail.html", headers={"Cache-Control": "no-store"})

        async def asset(request):
            name = request.path_params["name"]
            candidate = (DIST / name).resolve() if DIST.is_dir() else (STATIC / name).resolve()
            root = DIST.resolve() if DIST.is_dir() else STATIC.resolve()
            if candidate != root and root not in candidate.parents or not candidate.is_file():
                return JSONResponse({"error": "not_found"}, status_code=404)
            return FileResponse(candidate, headers={"Cache-Control": "no-cache"})

        return [*self.resources.routes(), Route("/dashboard", shell), Route("/dashboard/", shell),
                Route("/dashboard/frontend/{name:path}", asset),
                Route("/dashboard/tasks/{task_id:str}", task_shell),
                Route("/api/dashboard", self.endpoint), Route("/api/tasks", self.endpoint, methods=["POST"]),
                Route("/api/tasks/{task_id:str}", self.endpoint),
                Route("/api/tasks/{task_id:str}/thumbnail", self.endpoint),
                Route("/api/tasks/{task_id:str}/result", self.endpoint, methods=["POST"]),
                Route("/api/artifacts/import", self.endpoint, methods=["POST"])]
