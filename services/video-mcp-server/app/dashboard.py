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
from .collection import public_collection
from . import titles
from .project_labels import project_options

_PACKAGE_FRONTEND = Path(__file__).parent.parent / "frontend"
_WORKSPACE_FRONTEND = Path.cwd() / "services" / "video-mcp-server" / "frontend"
STATIC = _WORKSPACE_FRONTEND if (_WORKSPACE_FRONTEND / "dist").is_dir() else _PACKAGE_FRONTEND
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
    title: object = None
    model: str | None = None
    route: str = Field(min_length=1, max_length=64)
    prompt: str = Field(min_length=1, max_length=16000)
    duration_seconds: int = Field(ge=1, le=15)
    aspect_ratio: str | None = None
    mode: str | None = None
    resolution: str | None = None
    channel_account_id: str | None = Field(default=None, max_length=36)
    budget_limit: str | None = Field(default=None, max_length=32)
    references: References


class ImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    project_id: str
    source_url: str = Field(max_length=4096)
    filename: str = Field(max_length=128)
    expected_sha256: str = Field(min_length=64, max_length=64)


class TitleUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    title: str
    expected_title_revision: int = Field(ge=0)


def task_channel(record):
    route = record.request.get("route") or record.runtime_route or record.service
    local = {"h3", "h3-sol", "h3-vdn", "h3-singularity", "minimax-h3-sol-ref2va",
             "h3-latent-upscale", "sr", "interpolate", "depth"}
    return "local" if route in local else route or "unknown"


def channel_label(channel):
    return {"local": "本地", "higgsfield": "Higgsfield", "fal": "fal.ai",
            "grok": "xAI", "unknown": "未记录渠道"}.get(channel, channel)


def task_presentation(record):
    """UI labels only: keep provider identifiers and stored requests unchanged."""
    local = {"h3", "h3-sol", "h3-vdn", "h3-singularity"}
    generating = record.service in local | {"fal", "grok", "higgsfield"}
    route = record.request.get("route") or record.runtime_route or record.service
    channel = task_channel(record)
    local_channel = channel == "local"
    model = record.request.get("model")
    if not model and record.service in local:
        model = "minimax-h3-ref2va"  # Historical H3 contract, never a provider fallback.
    models = {"minimax-h3-ref2va": "MiniMax H3", "minimax-h3": "MiniMax H3",
              "wan3_0": "Wan 3.0", "grok_video_v15_lite": "Grok Imagine 1.5 Lite",
              "grok-imagine-video": "Grok Imagine", "seedance_2_5": "Seedance 2.5",
              "seedance_2_5_draft": "Seedance 2.5 Draft"}
    modes = {"text_to_video": ("T2V", "文生视频"), "image_to_video": ("I2V", "图生视频"),
             "first_last_frame_to_video": ("F2V", "首尾帧视频"),
             "reference_to_video": ("R2V", "参考生视频"), "ref2va": ("R2V", "参考生视频")}
    mode = record.request.get("mode")
    if not mode and (record.service in local or model == "minimax-h3-ref2va"):
        mode = "ref2va"
    code, explanation = modes.get(mode, (None, None)) if generating else (None, None)
    return {"is_generation": generating, "channel_id": channel,
            "model_label": models.get(model, model or "未记录模型") if generating else None,
            "channel_label": channel_label(channel) if generating else None,
            "channel_icon": ("local" if local_channel else "higgsfield" if route == "higgsfield" else None) if generating else None,
            "generation_code": code, "generation_label": explanation}


def public_task(record, store=None):
    record = TaskStore.with_timing(record)
    model = record.request.get("model")
    if not model:
        model = {
            "sr": "seedvr2",
            "interpolate": "rife",
            "depth": "video-depth-anything",
            "h3-latent-upscale": "h3-latent-upscaler",
        }.get(record.service)
    naming = titles.project(store, record) if store is not None else {'title': titles.fallback(record), 'title_source': 'fallback', 'title_revision': 0, 'title_updated_at': None}
    return {**naming, "video_task_id": record.video_task_id, "project_id": record.project_id,
            "idempotency_key": record.idempotency_key, "service": record.service, "route": record.runtime_route or record.request.get("route"),
            "model": model, "presentation": task_presentation(record),
            "mode": record.request.get("mode"), "resolution": record.request.get("resolution"),
            "needs_attention": record.runtime_stage == "submission_unknown" or bool(record.error),
            "progress": (record.provider_state or {}).get("progress"),
            "prompt": record.request.get("prompt", ""),
            "duration_seconds": record.request.get("duration_seconds"),
            "aspect_ratio": record.request.get("aspect_ratio"),
            "seed": record.request.get("seed", None if record.service in {"fal", "grok"} else 7),
            "status": record.status, "created_at": record.created_at, "updated_at": record.updated_at,
            "completed_at": record.completed_at or (record.updated_at if record.status in {"succeeded", "failed", "cancelled"} else None),
            "artifact_id": (record.delivery or {}).get("final_artifact_id") or record.artifact_id, "media": record.media, "error": record.error, "collection": public_collection(record),
            "input_digest": record.input_digest, "execution_instance_id": record.execution_instance_id, "association_source": "runtime" if record.execution_instance_id else None, "runtime_stage": record.runtime_stage,
            "dispatched_at": record.dispatched_at, "timing": record.timing, "runtime_metrics": record.runtime_metrics}


class Dashboard:
    def __init__(self, executor):
        self.executor = executor
        self.resources = Resources()
        self.result_lock = threading.Lock()
        self.services = {"h3": "unknown", "h3-sol": "not_connected", "fal": "not_connected", "mcp": "ready"}
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
        self.services["grok"] = "configured" if self.executor.grok.configured() and self.executor.grok.enabled else "not_connected"
        self.services["fal"] = "configured" if self.executor.fal.connected() else "not_connected"
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
            channel_pending = record.service == 'fal' and self.executor.fal._channel_record(record)
            if record.status in {"queued", "running"} and (record.runtime_task_id or channel_pending or record.service in {"depth", "sr", "interpolate", "h3-singularity", "grok", "higgsfield"}):
                try:
                    self.executor.status(record.video_task_id)
                except (ExecutionError, OSError, ValueError):
                    errors += 1
        self.sync_errors = errors
        self.synced_at = datetime.now(UTC).isoformat()

    def recover_incomplete_submissions(self):
        for record in self.records():
            if record.service == 'fal' and self.executor.fal._channel_record(record):
                continue  # The channel ledger resolves unknown acceptance by the original operation key.
            if record.status == "queued" and not record.runtime_task_id and record.service not in {"depth", "sr", "interpolate", "grok", "higgsfield"}:
                self.executor.tasks.update(record, status="queued" if record.service == "h3-singularity" else "failed", error={
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
        visible = []
        for record in records:
            if record.service == "higgsfield":
                try:
                    self.executor.higgsfield.authorize_record(record)
                except (PermissionError, ValueError, httpx.HTTPError):
                    continue
            if record.service == "fal":
                try:
                    self.executor.fal.authorize_record(record)
                except (PermissionError, ValueError, httpx.HTTPError):
                    continue
            visible.append(record)
        records = visible
        projects = sorted({r.project_id for r in records})
        options = project_options(self.executor.client, projects)
        project_labels = {item["id"]: item["label"] for item in options}
        channels = [{"id": c, "label": channel_label(c)} for c in sorted({task_channel(r) for r in records})]
        rows = [public_task(r, self.executor.tasks) for r in records]
        channel = query.get("channel", "all")
        if channel != "all":
            rows = [r for r in rows if r["presentation"]["channel_id"] == channel]
        for key in ("project_id", "service"):
            value = query.get(key, "all")
            if value != "all":
                rows = [r for r in rows if r[key] == value]
        search = query.get("q", "").casefold()
        if search:
            rows = [r for r in rows if search in " ".join(str(r[k]) for k in (
                "title", "prompt", "project_id", "idempotency_key", "video_task_id")).casefold()
                    or search in project_labels.get(r["project_id"], "").casefold()]
        counts = {s: sum(r["status"] == s for r in rows) for s in (
            "queued", "running", "succeeded", "failed", "cancelled")}
        status_filter = query.get("status", "all")
        if status_filter != "all":
            rows = [r for r in rows if r["status"] == status_filter]
        elapsed = [r["timing"]["processing_seconds"] if r["service"] == "higgsfield" else r["timing"]["total_seconds"]
                   for r in rows if r["status"] == "succeeded"]
        elapsed = [value for value in elapsed if value is not None]
        return {"tasks": rows[(page-1)*size:page*size], "total": len(rows), "page": page,
                "page_size": size, "projects": projects, "project_options": options, "channels": channels, "counts": counts,
                "completion_times": [r["completed_at"] for r in rows if r["status"] == "succeeded"],
                "average_elapsed_seconds": round(sum(elapsed)/len(elapsed), 1) if elapsed else None,
                "services": dict(self.services), "grok_capability": self.executor.grok.capability(),
                "synced_at": self.synced_at, "sync_errors": self.sync_errors}

    def detail(self, task_id):
        record = self.executor.tasks.get(task_id)
        if record.service == "higgsfield":
            self.executor.higgsfield.authorize_record(record)
        if record.service == "fal":
            self.executor.fal.authorize_record(record)
        value = public_task(record, self.executor.tasks)
        value["runtime_version"] = record.runtime_version or self.executor.runtime_version
        value["references"] = []
        for kind, items in record.request.get("references", {}).items():
            for item in items:
                try:
                    artifact = self.executor.artifacts.get(item["artifact_id"], record.project_id)
                    value["references"].append({"kind": kind, "purpose": item["purpose"], **artifact.model_dump()})
                except (ArtifactError, ValueError):
                    value["references"].append({"kind": kind, "purpose": item["purpose"], "unavailable": True})
        artifact_id = (record.delivery or {}).get("final_artifact_id") or record.artifact_id
        if artifact_id:
            value["artifact"] = self.executor.artifacts.get(artifact_id, record.project_id).model_dump()
        if record.service == "higgsfield" and record.status == "succeeded":
            value.update(self.executor.higgsfield.public_result(record))
        return value

    def thumbnail(self, task_id):
        record = self.executor.tasks.get(task_id)
        if record.service == "higgsfield":
            self.executor.higgsfield.authorize_record(record)
        if record.service == "fal":
            self.executor.fal.authorize_record(record)
        if record.status != "succeeded":
            return JSONResponse({"error": "preview_unavailable"}, status_code=404)
        native = record.service == "higgsfield"
        artifact_id = (record.delivery or {}).get("final_artifact_id") or record.artifact_id
        if native:
            digest = (record.media or {}).get("sha256")
            if not digest:
                return JSONResponse({"error": "preview_unavailable"}, status_code=404)
            try:
                source_path = self.executor.higgsfield.content_path(record)
            except (ArtifactNotFound, FileNotFoundError):
                return JSONResponse({"error": "preview_unavailable"}, status_code=404)
            except (ExecutionError, ArtifactError, OSError, ValueError):
                return JSONResponse({"error": "preview_extraction_failed"}, status_code=422)
        else:
            if not artifact_id:
                return JSONResponse({"error": "preview_unavailable"}, status_code=404)
            try:
                artifact = self.executor.artifacts.get(artifact_id, record.project_id)
                if not (artifact.media_type or "").startswith("video/"):
                    return JSONResponse({"error": "preview_unavailable"}, status_code=404)
                source_path = self.executor.artifacts.content_path(artifact)
                digest = artifact.sha256
            except (ArtifactNotFound, FileNotFoundError):
                return JSONResponse({"error": "preview_unavailable"}, status_code=404)
            except (ArtifactError, OSError, ValueError):
                return JSONResponse({"error": "preview_extraction_failed"}, status_code=422)
        if not Path(source_path).is_file():
            return JSONResponse({"error": "preview_unavailable"}, status_code=404)
        directory = self.executor.artifacts.root / "previews"
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / f"{digest}.jpg"
        with self.result_lock:
            valid_cached = False
            if output.is_file():
                try:
                    cached = output.read_bytes()
                    valid_cached = len(cached) > 4 and cached.startswith(b"\xff\xd8")
                except OSError:
                    valid_cached = False
            if not valid_cached:
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".jpg", dir=directory, delete=False) as temporary:
                        temp_path = Path(temporary.name)
                    subprocess.run(
                        [
                            "ffmpeg", "-v", "error", "-nostdin", "-y", "-i",
                            str(source_path),
                            "-frames:v", "1", "-vf", "scale=480:320:force_original_aspect_ratio=decrease",
                            str(temp_path),
                        ],
                        check=True,
                        capture_output=True,
                        timeout=30,
                    )
                    extracted = temp_path.read_bytes()
                    if len(extracted) <= 4 or not extracted.startswith(b"\xff\xd8"):
                        raise ValueError("Invalid JPEG preview")
                    temp_path.replace(output)
                    temp_path = None
                except (subprocess.SubprocessError, OSError, ValueError):
                    output.unlink(missing_ok=True)
                    return JSONResponse({"error": "preview_extraction_failed"}, status_code=422)
                finally:
                    if temp_path is not None:
                        temp_path.unlink(missing_ok=True)
        return FileResponse(output, media_type="image/jpeg", headers={"Cache-Control": "private, no-store"})

    async def endpoint(self, request: Request):
        try:
            action = request.url.path
            if action == "/api/model-catalog":
                value = self.executor.catalog_view()
            elif action == "/api/capabilities":
                value = self.executor.capabilities()
            elif action == "/api/dashboard":
                value = await run_in_threadpool(self.snapshot, request)
            elif action.endswith("/thumbnail"):
                return await run_in_threadpool(self.thumbnail, request.path_params["task_id"])
            elif action.endswith("/content"):
                record = self.executor.tasks.get(request.path_params["task_id"])
                if record.service != "higgsfield":
                    return JSONResponse({"error": "not_found"}, status_code=404)
                path = await run_in_threadpool(self.executor.higgsfield.content_path, record)
                return FileResponse(path, media_type="video/mp4", filename=record.video_task_id + ".mp4",
                                    headers={"Cache-Control": "private, no-store"})
            elif request.method == "GET":
                value = await run_in_threadpool(self.detail, request.path_params["task_id"])
            elif action.endswith("/result"):
                def result():
                    record = self.executor.tasks.get(request.path_params["task_id"])
                    if record.service == "higgsfield":
                        self.executor.higgsfield.authorize_record(record)
                    if record.service == "fal":
                        self.executor.fal.authorize_record(record)
                    self.executor.result(request.path_params["task_id"])
                    return self.detail(request.path_params["task_id"])
                value = await run_in_threadpool(result)
            else:
                body = bytearray()
                async for chunk in request.stream():
                    body.extend(chunk)
                    if len(body) > 65536:
                        return JSONResponse({"error": "request_too_large"}, status_code=413)
                if action.endswith('/title'):
                    inputs = TitleUpdate.model_validate_json(body)
                    naming = await run_in_threadpool(titles.rename, self.executor, request.path_params['task_id'], **inputs.model_dump())
                    value = {'video_task_id': request.path_params['task_id'], **naming}
                elif action == "/api/artifacts/import":
                    inputs = ImportRequest.model_validate_json(body)
                    artifact = await run_in_threadpool(self.executor.import_asset, **inputs.model_dump())
                    value = artifact.model_dump()
                else:
                    inputs = Submission.model_validate_json(body)
                    service = ("higgsfield" if inputs.route == "higgsfield" else "grok" if inputs.route == "grok" else "fal" if inputs.route == "fal" else
                               "h3-sol" if inputs.route.startswith("h3-sol") else
                               "h3-singularity" if inputs.route == "h3-singularity" else "h3")
                    if service not in {"h3", "h3-sol", "h3-singularity", "fal", "grok", "higgsfield"}:
                        return JSONResponse({"error": "service_not_connected"}, status_code=409)
                    if service == "fal" and not self.executor.fal.connected():
                        return JSONResponse({"error": "service_not_connected"}, status_code=409)
                    if service == "h3-sol":
                        selected_route = inputs.route
                        route_config = self.executor.runtime_routes.get(selected_route)
                        route_requires_token = route_config is not None and route_config["requires_token"]
                        if (route_config is None and not (self.executor.sol_url and self.executor.sol_token)) or (route_requires_token and not self.executor.sol_token):
                            return JSONResponse({"error": "service_not_connected"}, status_code=409)
                    if service == "h3-singularity" and not self.executor.singularity_token:
                        return JSONResponse({"error": "service_not_connected"}, status_code=409)
                    kwargs = inputs.model_dump(exclude={"model"})
                    record, naming = await run_in_threadpool(titles.create, self.executor, model=inputs.model, **kwargs)
                    value = {**public_task(record, self.executor.tasks), **naming}
            return JSONResponse(value, headers={"Cache-Control": "no-store"})
        except (TaskNotFound, ArtifactNotFound):
            return JSONResponse({"error": "not_found"}, status_code=404)
        except PermissionError:
            return JSONResponse({"error": "permission_denied"}, status_code=403)
        except titles.TitleConflict:
            return JSONResponse({"error": "title_conflict"}, status_code=409)
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
        from .member_dashboard import MemberDashboard
        members = MemberDashboard(self)
        async def shell(request):
            if request.url.path.endswith("/"):
                return RedirectResponse("../dashboard")
            if request.query_params.get("embed") == "1":
                return FileResponse(STATIC / "dashboard.html", headers={"Cache-Control": "no-store"})
            entry = DIST / "index.html" if (DIST / "index.html").is_file() else STATIC / "dashboard.html"
            return FileResponse(entry, headers={"Cache-Control": "no-store"})

        async def task_shell(request):
            return FileResponse(STATIC / "task-detail.html", headers={"Cache-Control": "no-store"})

        async def asset(request):
            name = request.path_params["name"]
            for directory in (DIST, STATIC):
                if not directory.is_dir():
                    continue
                root = directory.resolve()
                candidate = (root / name).resolve()
                if (candidate == root or root in candidate.parents) and candidate.is_file():
                    return FileResponse(candidate, headers={"Cache-Control": "no-cache"})
            return JSONResponse({"error": "not_found"}, status_code=404)

        return [*self.resources.routes(), *members.routes(), Route("/dashboard", shell), Route("/dashboard/", shell),
                Route("/dashboard/frontend/{name:path}", asset),
                Route("/dashboard/static/{name:path}", asset),
                Route("/dashboard/tasks/{task_id:str}", task_shell),
                Route("/api/dashboard", self.endpoint), Route("/api/capabilities", self.endpoint),
                Route("/api/model-catalog", self.endpoint), Route("/api/tasks", self.endpoint, methods=["POST"]),
                Route("/api/tasks/{task_id:str}", self.endpoint),
                Route("/api/tasks/{task_id:str}/thumbnail", self.endpoint),
                Route("/api/tasks/{task_id:str}/content", self.endpoint),
                Route("/api/tasks/{task_id:str}/result", self.endpoint, methods=["POST"]),
                Route("/api/tasks/{task_id:str}/title", self.endpoint, methods=["POST"]),
                Route("/api/artifacts/import", self.endpoint, methods=["POST"])]
