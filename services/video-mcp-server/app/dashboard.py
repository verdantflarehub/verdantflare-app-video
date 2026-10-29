from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route
from starlette.concurrency import run_in_threadpool

from .artifacts import ArtifactError, ArtifactNotFound
from .channel_errors import ChannelError
from .content_rating import ContentRatingError, ContentRatingStore, include_nsfw_from_headers
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
    expected_title_revision: int | None = Field(default=None, ge=0)


class ContentRatingUpdate(BaseModel):
    model_config = ConfigDict(extra="ignore")
    action: str
    reason: str | None = None
    note: str | None = None


def task_channel(record):
    route = record.request.get("route") or record.runtime_route or record.service
    local = {"h3", "h3-sol", "h3-vdn", "h3-singularity", "minimax-h3-sol-ref2va",
             "h3-latent-upscale", "sr", "interpolate", "depth"}
    return "local" if route in local else route or "unknown"


def channel_label(channel):
    return {"local": "本地", "higgsfield": "Higgsfield", "fal": "fal.ai",
            "grok": "xAI", "google": "Google", "unknown": "未记录渠道"}.get(channel, channel)


def task_presentation(record, catalog=None):
    """UI labels only: keep provider identifiers and stored requests unchanged."""
    local = {"h3", "h3-sol", "h3-vdn", "h3-singularity"}
    generating = record.service in local | {"fal", "grok", "google", "higgsfield"}
    route = record.request.get("route") or record.runtime_route or record.service
    channel = task_channel(record)
    local_channel = channel == "local"
    model = record.request.get("model")
    if not model and record.service in local:
        model = "minimax-h3-ref2va"  # Historical H3 contract, never a provider fallback.
    label = None
    for entry_channel in (catalog or {}).get("channels", []):
        if entry_channel["id"] != channel:
            continue
        for entry in entry_channel["models"]:
            execution = (catalog or {}).get("execution", {}).get(channel, {}).get(entry["id"], {})
            if model and model in {entry.get("model_id"), entry.get("upstream_model_id"), execution.get("model")}:
                label = entry.get("display_name")
                break
    label = label or "未记录模型"
    modes = {"text_to_video": ("T2V", "文生视频"), "image_to_video": ("I2V", "图生视频"),
             "lip_sync": ("A2V", "音频驱动口型"),
             "first_last_frame_to_video": ("F2V", "首尾帧视频"),
             "reference_to_video": ("R2V", "参考生视频"), "ref2va": ("R2V", "参考生视频")}
    mode = record.request.get("mode")
    if not mode and (record.service in local or model == "minimax-h3-ref2va"):
        mode = "ref2va"
    code, explanation = modes.get(mode, (None, None)) if generating else (None, None)
    return {"is_generation": generating, "channel_id": channel,
            "model_label": label if generating else None,
            "channel_label": channel_label(channel) if generating else None,
            "channel_icon": ("local" if local_channel else "higgsfield" if route == "higgsfield" else None) if generating else None,
            "generation_code": code, "generation_label": explanation}


def public_task(record, store=None, ratings_store=None, catalog=None):
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
    ui_fields = store.record_ui_fields(record, ratings_store, can_write=True) if store is not None else {"content_rating": "unrated", "can_clear_rating": False, "can_mark_rating": True}
    return {**naming, "video_task_id": record.video_task_id, "project_id": record.project_id,
            "idempotency_key": record.idempotency_key, "service": record.service, "route": record.runtime_route or record.request.get("route"),
            "model": model, "presentation": task_presentation(record, catalog),
            "mode": record.request.get("mode"), "resolution": record.request.get("resolution"),
            "needs_attention": record.runtime_stage == "submission_unknown" or bool(record.error),
            "progress": (record.provider_state or {}).get("progress"),
            "prompt": record.request.get("prompt", ""),
            "duration_seconds": record.request.get("duration_seconds"),
            "aspect_ratio": record.request.get("aspect_ratio"),
            "seed": record.request.get("seed", None if record.service in {"fal", "grok", "google"} else 7),
            "status": record.status, "created_at": record.created_at, "updated_at": record.updated_at,
            "completed_at": record.completed_at or (record.updated_at if record.status in {"succeeded", "failed", "cancelled"} else None),
            "artifact_id": (record.delivery or {}).get("final_artifact_id") or record.artifact_id, "media": record.media, "error": record.error, "collection": public_collection(record),
            "input_digest": record.input_digest, "execution_instance_id": record.execution_instance_id, "association_source": "runtime" if record.execution_instance_id else None, "runtime_stage": record.runtime_stage,
            "dispatched_at": record.dispatched_at, "timing": record.timing, "runtime_metrics": record.runtime_metrics,
            **ui_fields}


def _is_valid_jpeg(path: Path) -> bool:
    try:
        st = path.stat()
        if st.st_size <= 4:
            return False
        with path.open("rb") as f:
            head = f.read(4)
        return len(head) == 4 and head.startswith(b"\xff\xd8")
    except OSError:
        return False


def _etag_matches(if_none_match: str | None, etag: str) -> bool:
    if not if_none_match:
        return False
    target = etag.strip()
    if target.startswith("W/"):
        target = target[2:].strip()
    for raw in if_none_match.split(","):
        candidate = raw.strip()
        if candidate == "*":
            return True
        if candidate.startswith("W/"):
            candidate = candidate[2:].strip()
        if candidate == target:
            return True
    return False


class Dashboard:
    def __init__(self, executor):
        self.executor = executor
        self.resources = Resources()
        self.result_lock = threading.Lock()
        self._thumb_locks_guard = threading.Lock()
        self._thumb_locks: dict[str, threading.Lock] = {}
        self._ffmpeg_sem = threading.Semaphore(2)
        self._records_cache_lock = threading.RLock()
        self._records_cache: dict[Path, tuple[int, int, TaskRecord]] = {}
        self.services = {"h3": "unknown", "h3-sol": "not_connected", "fal": "not_connected", "mcp": "ready"}
        self.sync_errors = 0
        self.synced_at = None

    def _get_thumb_lock(self, digest: str) -> threading.Lock:
        with self._thumb_locks_guard:
            lock = self._thumb_locks.get(digest)
            if lock is None:
                lock = threading.Lock()
                self._thumb_locks[digest] = lock
            return lock

    def _public_task(self, record, store=None, ratings_store=None):
        # Labels may use a short-lived display cache; submission never uses it.
        now = time.monotonic()
        if now >= getattr(self, "_label_catalog_until", 0):
            self._label_catalog = getattr(self.executor, "catalog_view", lambda: {})()
            self._label_catalog_until = now + 30
        return public_task(record, store, ratings_store, self._label_catalog)

    def _stores(self, request: Request | None = None):
        tasks_store = getattr(request.app.state, "tasks", None) if request is not None else None
        if tasks_store is None:
            tasks_store = self.executor.tasks
        artifacts_store = getattr(request.app.state, "artifacts", None) if request is not None else None
        if artifacts_store is None:
            artifacts_store = self.executor.artifacts
        ratings_store = getattr(request.app.state, "ratings", None) if request is not None else None
        if ratings_store is None or ratings_store.db_path != tasks_store.db_path:
            ratings_store = ContentRatingStore(tasks_store.db_path, domain="video")
            if request is not None:
                request.app.state.ratings = ratings_store
        return tasks_store, artifacts_store, ratings_store

    def records(self, tasks_store=None):
        store = tasks_store or self.executor.tasks
        store.ensure_ready()
        records = []
        seen_paths = set()
        for path in store.root.glob("video_task_*.json"):
            seen_paths.add(path)
            try:
                st = path.stat()
                key = (st.st_mtime_ns, st.st_size)
                with self._records_cache_lock:
                    cached = self._records_cache.get(path)
                if cached is not None and (cached[0], cached[1]) == key:
                    records.append(cached[2])
                    continue
                parsed = TaskRecord.model_validate_json(path.read_text())
                with self._records_cache_lock:
                    self._records_cache[path] = (st.st_mtime_ns, st.st_size, parsed)
                records.append(parsed)
            except (OSError, ValueError):
                logging.getLogger(__name__).warning("Invalid task record omitted from dashboard")
        with self._records_cache_lock:
            stale = [p for p in self._records_cache if p not in seen_paths and p.parent == store.root]
            for p in stale:
                self._records_cache.pop(p, None)
        return sorted(records, key=lambda x: (x.created_at, x.video_task_id), reverse=True)

    def sync(self):
        self.services["grok"] = "configured" if self.executor.grok.configured() and self.executor.grok.enabled else "not_connected"
        self.services["google"] = "configured" if self.executor.google.configured() and self.executor.google.enabled else "not_connected"
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
            if record.status in {"queued", "running"} and (record.runtime_task_id or channel_pending or record.service in {"depth", "sr", "interpolate", "h3-singularity", "grok", "google", "higgsfield"}):
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
            if record.status == "queued" and not record.runtime_task_id and record.service not in {"depth", "sr", "interpolate", "grok", "google", "higgsfield"}:
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
        tasks_store, _, ratings_store = self._stores(request)
        include_nsfw = include_nsfw_from_headers(request.headers)
        query = request.query_params
        page = max(1, int(query.get("page", "1")))
        size = max(1, min(100, int(query.get("page_size", "24"))))
        records = self.records(tasks_store)
        visible = []
        for record in records:
            if not include_nsfw and tasks_store.is_record_nsfw(record, ratings_store):
                continue
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
        rows = [
            {
                **self._public_task(r, tasks_store),
                "task_id": r.video_task_id,
                **tasks_store.record_ui_fields(r, ratings_store, can_write=True),
            }
            for r in records
        ]
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

    def list_tasks_api(self, request):
        tasks_store, _, ratings_store = self._stores(request)
        include_nsfw = include_nsfw_from_headers(request.headers)
        query = request.query_params
        project_id = query.get("project_id")
        service = query.get("service")
        route = query.get("route")
        status = query.get("status")
        limit = max(1, min(200, int(query.get("limit", "50"))))
        offset = max(0, int(query.get("offset", "0")))
        records, total = tasks_store.list_tasks(
            project_id=project_id,
            service=service,
            route=route,
            status=status,
            limit=limit,
            offset=offset,
            include_nsfw=include_nsfw,
            ratings_store=ratings_store,
        )
        items = [
            {
                **self._public_task(r, tasks_store),
                "task_id": r.video_task_id,
                **tasks_store.record_ui_fields(r, ratings_store, can_write=True),
            }
            for r in records
        ]
        return {"tasks": items, "total": total, "limit": limit, "offset": offset}

    def task_stats_api(self, request):
        tasks_store, _, ratings_store = self._stores(request)
        include_nsfw = include_nsfw_from_headers(request.headers)
        project_id = request.query_params.get("project_id")
        return tasks_store.get_stats(
            project_id=project_id,
            include_nsfw=include_nsfw,
            ratings_store=ratings_store,
        )

    def detail(self, task_id, request=None):
        tasks_store, artifacts_store, ratings_store = self._stores(request)
        include_nsfw = include_nsfw_from_headers(request.headers) if request is not None else False
        record = tasks_store.get(task_id)
        if not include_nsfw and tasks_store.is_record_nsfw(record, ratings_store):
            raise TaskNotFound(f"NOT_FOUND: {task_id}")
        if record.service == "higgsfield":
            self.executor.higgsfield.authorize_record(record)
        if record.service == "fal":
            self.executor.fal.authorize_record(record)
        value = self._public_task(record, tasks_store)
        value["task_id"] = record.video_task_id
        value.update(tasks_store.record_ui_fields(record, ratings_store, can_write=True))
        value["runtime_version"] = record.runtime_version or self.executor.runtime_version
        value["references"] = []
        for kind, items in record.request.get("references", {}).items():
            for item in items:
                try:
                    artifact = artifacts_store.get(item["artifact_id"], record.project_id)
                    value["references"].append({"kind": kind, "purpose": item["purpose"], **artifact.model_dump()})
                except (ArtifactError, ValueError):
                    value["references"].append({"kind": kind, "purpose": item["purpose"], "unavailable": True})
        artifact_id = (record.delivery or {}).get("final_artifact_id") or record.artifact_id
        if artifact_id:
            value["artifact"] = artifacts_store.get(artifact_id, record.project_id).model_dump()
        if record.service == "higgsfield" and record.status == "succeeded":
            value.update(self.executor.higgsfield.public_result(record))
        return value

    def thumbnail(self, task_id, request=None):
        tasks_store, artifacts_store, ratings_store = self._stores(request)
        include_nsfw = include_nsfw_from_headers(request.headers) if request is not None else False
        record = tasks_store.get(task_id)
        if not include_nsfw and tasks_store.is_record_nsfw(record, ratings_store):
            raise TaskNotFound(f"NOT_FOUND: {task_id}")
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
                artifact = artifacts_store.get(artifact_id, record.project_id)
                if not (artifact.media_type or "").startswith("video/"):
                    return JSONResponse({"error": "preview_unavailable"}, status_code=404)
                source_path = artifacts_store.content_path(artifact)
                digest = artifact.sha256
            except (ArtifactNotFound, FileNotFoundError):
                return JSONResponse({"error": "preview_unavailable"}, status_code=404)
            except (ArtifactError, OSError, ValueError):
                return JSONResponse({"error": "preview_extraction_failed"}, status_code=422)
        if not Path(source_path).is_file():
            return JSONResponse({"error": "preview_unavailable"}, status_code=404)
        directory = artifacts_store.root / "previews"
        output = directory / f"{digest}.jpg"
        etag = f'"{digest}-v1"'
        cache_headers = {
            "Cache-Control": "private, max-age=2592000, immutable",
            "ETag": etag,
        }
        # Lock-free fast path on disk cache hit.
        if _is_valid_jpeg(output):
            if request is not None and _etag_matches(request.headers.get("if-none-match"), etag):
                return Response(status_code=304, headers=cache_headers)
            return FileResponse(output, media_type="image/jpeg", headers=cache_headers)

        directory.mkdir(parents=True, exist_ok=True)
        with self._get_thumb_lock(digest):
            if not _is_valid_jpeg(output):
                temp_path = None
                try:
                    with self._ffmpeg_sem:
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
                        if not _is_valid_jpeg(temp_path):
                            raise ValueError("Invalid JPEG preview")
                        temp_path.replace(output)
                        temp_path = None
                except (subprocess.SubprocessError, OSError, ValueError):
                    output.unlink(missing_ok=True)
                    return JSONResponse({"error": "preview_extraction_failed"}, status_code=422)
                finally:
                    if temp_path is not None:
                        temp_path.unlink(missing_ok=True)
        if request is not None and _etag_matches(request.headers.get("if-none-match"), etag):
            return Response(status_code=304, headers=cache_headers)
        return FileResponse(output, media_type="image/jpeg", headers=cache_headers)

    def prewarm_thumbnail(self, task_id: str, request: Request | None = None) -> None:
        try:
            self.thumbnail(task_id, request)
        except Exception:
            pass

    async def endpoint(self, request: Request):
        try:
            action = request.url.path
            if action.startswith("/video/"):
                action = action[len("/video"):]
            if action == "/api/model-catalog":
                value = self.executor.catalog_view()
            elif action == "/api/capabilities":
                value = self.executor.capabilities()
            elif action == "/api/dashboard":
                value = await run_in_threadpool(self.snapshot, request)
            elif action == "/api/tasks/stats":
                value = await run_in_threadpool(self.task_stats_api, request)
            elif action == "/api/tasks" and request.method == "GET":
                value = await run_in_threadpool(self.list_tasks_api, request)
            elif action.endswith("/thumbnail"):
                return await run_in_threadpool(self.thumbnail, request.path_params["task_id"], request)
            elif action.endswith("/content"):
                tasks_store, _, ratings_store = self._stores(request)
                include_nsfw = include_nsfw_from_headers(request.headers)
                record = tasks_store.get(request.path_params["task_id"])
                if not include_nsfw and tasks_store.is_record_nsfw(record, ratings_store):
                    return JSONResponse({"error": "not_found"}, status_code=404)
                if record.service != "higgsfield":
                    return JSONResponse({"error": "not_found"}, status_code=404)
                path = await run_in_threadpool(self.executor.higgsfield.content_path, record)
                digest = (record.media or {}).get("sha256") or record.video_task_id
                etag = f'"{digest}"'
                cache_headers = {
                    "Cache-Control": "private, max-age=2592000, immutable",
                    "ETag": etag,
                }
                if _etag_matches(request.headers.get("if-none-match"), etag):
                    return Response(status_code=304, headers=cache_headers)
                return FileResponse(path, media_type="video/mp4", filename=record.video_task_id + ".mp4",
                                    headers=cache_headers)
            elif request.method == "GET":
                value = await run_in_threadpool(self.detail, request.path_params["task_id"], request)
            elif request.method == "DELETE" or action.endswith("/delete"):
                task_id = request.path_params["task_id"]
                def _delete():
                    tasks_store, _, _ = self._stores(request)
                    return tasks_store.delete_task(task_id)
                value = await run_in_threadpool(_delete)
            elif action.endswith("/result"):
                def result():
                    tasks_store, _, ratings_store = self._stores(request)
                    include_nsfw = include_nsfw_from_headers(request.headers)
                    record = tasks_store.get(request.path_params["task_id"])
                    if not include_nsfw and tasks_store.is_record_nsfw(record, ratings_store):
                        raise TaskNotFound(f"NOT_FOUND: {request.path_params['task_id']}")
                    if record.service == "higgsfield":
                        self.executor.higgsfield.authorize_record(record)
                    if record.service == "fal":
                        self.executor.fal.authorize_record(record)
                    self.executor.result(request.path_params["task_id"])
                    self.prewarm_thumbnail(request.path_params["task_id"], request)
                    return self.detail(request.path_params["task_id"], request)
                value = await run_in_threadpool(result)
            else:
                body = bytearray()
                async for chunk in request.stream():
                    body.extend(chunk)
                    if len(body) > 65536:
                        return JSONResponse({"error": "request_too_large"}, status_code=413)
                if action.endswith('/title'):
                    inputs = TitleUpdate.model_validate_json(body)
                    task_id = request.path_params['task_id']
                    if inputs.expected_title_revision is not None:
                        naming = await run_in_threadpool(
                            titles.rename,
                            self.executor,
                            task_id,
                            title=inputs.title,
                            expected_title_revision=inputs.expected_title_revision,
                        )
                    else:
                        def _rename_simple():
                            tasks_store, _, _ = self._stores(request)
                            record = tasks_store.get(task_id)
                            current = titles.project(tasks_store, record)
                            return titles.update(tasks_store, record, inputs.title, current["title_revision"])
                        naming = await run_in_threadpool(_rename_simple)
                    value = {'video_task_id': task_id, **naming}
                elif action.endswith('/content-rating'):
                    inputs = ContentRatingUpdate.model_validate_json(body or b"{}")
                    act = inputs.action.strip().lower()
                    if act not in {"mark", "clear"}:
                        raise ValueError("action must be 'mark' or 'clear'")
                    task_id = request.path_params["task_id"]
                    actor_user_id = (
                        request.headers.get("X-User-Id")
                        or request.headers.get("X-VF-User-Id")
                        or "dashboard_user"
                    ).strip() or "dashboard_user"
                    reason_text = (inputs.reason or inputs.note or ("manual_mark_nsfw" if act == "mark" else "manual_clear")).strip()
                    def _update_rating():
                        tasks_store, _, ratings_store = self._stores(request)
                        record = tasks_store.get(task_id)
                        vids = tasks_store.task_version_ids(record)
                        target_vids = vids if vids else [record.video_task_id]
                        for vid in target_vids:
                            if act == "mark":
                                ratings_store.mark_rating(
                                    version_id=vid,
                                    project_id=record.project_id,
                                    task_id=record.video_task_id,
                                    actor_user_id=actor_user_id,
                                    reason=reason_text,
                                )
                            else:
                                ratings_store.clear_rating(
                                    version_id=vid,
                                    project_id=record.project_id,
                                    task_id=record.video_task_id,
                                    actor_user_id=actor_user_id,
                                    note=reason_text,
                                )
                        new_rating = "nsfw" if act == "mark" else "general"
                        return {
                            "status": "ok",
                            "video_task_id": record.video_task_id,
                            "content_rating": new_rating,
                        }
                    value = await run_in_threadpool(_update_rating)
                elif action == "/api/artifacts/import":
                    inputs = ImportRequest.model_validate_json(body)
                    artifact = await run_in_threadpool(self.executor.import_asset, **inputs.model_dump())
                    value = artifact.model_dump()
                else:
                    inputs = Submission.model_validate_json(body)
                    service = ("higgsfield" if inputs.route == "higgsfield" else "google" if inputs.route == "google" else "grok" if inputs.route == "grok" else "fal" if inputs.route == "fal" else
                               "h3-sol" if inputs.route.startswith("h3-sol") else
                               "h3-singularity" if inputs.route == "h3-singularity" else "h3")
                    if service not in {"h3", "h3-sol", "h3-singularity", "fal", "grok", "google", "higgsfield"}:
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
                    value = {**self._public_task(record, self.executor.tasks), **naming}
            return JSONResponse(value, headers={"Cache-Control": "no-store"})
        except ChannelError as error:
            status = error.http_status if type(error.http_status) is int and 400 <= error.http_status <= 599 else 400
            return JSONResponse({"error": error.public()}, status_code=status, headers={"Cache-Control": "no-store"})
        except (TaskNotFound, ArtifactNotFound):
            return JSONResponse({"error": "not_found"}, status_code=404)
        except PermissionError:
            return JSONResponse({"error": "permission_denied"}, status_code=403)
        except titles.TitleConflict:
            return JSONResponse({"error": "title_conflict"}, status_code=409)
        except TaskConflict:
            return JSONResponse({"error": "idempotency_conflict"}, status_code=409)
        except (ValidationError, ValueError, KeyError, TypeError, ContentRatingError):
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
                Route("/api/dashboard", self.endpoint), Route("/video/api/dashboard", self.endpoint),
                Route("/api/capabilities", self.endpoint), Route("/video/api/capabilities", self.endpoint),
                Route("/api/model-catalog", self.endpoint), Route("/video/api/model-catalog", self.endpoint),
                Route("/api/tasks", self.endpoint, methods=["GET", "POST"]),
                Route("/video/api/tasks", self.endpoint, methods=["GET", "POST"]),
                Route("/api/tasks/stats", self.endpoint, methods=["GET"]),
                Route("/video/api/tasks/stats", self.endpoint, methods=["GET"]),
                Route("/api/tasks/{task_id:str}", self.endpoint, methods=["GET", "DELETE"]),
                Route("/video/api/tasks/{task_id:str}", self.endpoint, methods=["GET", "DELETE"]),
                Route("/api/tasks/{task_id:str}/delete", self.endpoint, methods=["POST"]),
                Route("/video/api/tasks/{task_id:str}/delete", self.endpoint, methods=["POST"]),
                Route("/api/tasks/{task_id:str}/content-rating", self.endpoint, methods=["POST"]),
                Route("/video/api/tasks/{task_id:str}/content-rating", self.endpoint, methods=["POST"]),
                Route("/api/tasks/{task_id:str}/thumbnail", self.endpoint),
                Route("/video/api/tasks/{task_id:str}/thumbnail", self.endpoint),
                Route("/api/tasks/{task_id:str}/content", self.endpoint),
                Route("/video/api/tasks/{task_id:str}/content", self.endpoint),
                Route("/api/tasks/{task_id:str}/result", self.endpoint, methods=["POST"]),
                Route("/video/api/tasks/{task_id:str}/result", self.endpoint, methods=["POST"]),
                Route("/api/tasks/{task_id:str}/title", self.endpoint, methods=["POST"]),
                Route("/video/api/tasks/{task_id:str}/title", self.endpoint, methods=["POST"]),
                Route("/api/artifacts/import", self.endpoint, methods=["POST"])]
