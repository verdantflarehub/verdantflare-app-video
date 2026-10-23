from __future__ import annotations

import json
try:
    import fcntl
except ImportError:  # pragma: no cover - Windows local test fallback
    fcntl = None  # type: ignore[assignment]
import os
import re
import tempfile
import uuid
import sqlite3
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

TASK_ID_PATTERN = re.compile(r"video_task_[0-9a-f]{32}")


class TaskConflict(RuntimeError):
    pass


class TaskNotFound(RuntimeError):
    pass


class TaskRecord(BaseModel):
    schema_version: int = 1
    service: str = "h3"
    runtime_version: str | None = None
    runtime_route: str | None = None
    execution_instance_id: str | None = None
    runtime_stage: str | None = None
    timing: dict[str, object] | None = None
    runtime_metrics: dict[str, object] | None = None
    video_task_id: str
    project_id: str
    idempotency_key: str
    input_digest: str
    request: dict[str, object]
    runtime_task_id: str
    status: str
    created_at: str
    updated_at: str
    completed_at: str | None = None
    dispatched_at: str | None = None
    artifact_id: str | None = None
    collection: dict[str, object] | None = None
    media: dict[str, object] | None = None
    error: dict[str, str] | None = None
    provider_state: dict[str, object] | None = None
    delivery: dict[str, object] | None = None
    authority: dict[str, str] | None = None
    runtime_endpoint: str | None = None


class TaskStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve() / "tasks"
        self._mutex = threading.RLock()
        self._local = threading.local()

    @property
    def db_path(self) -> Path:
        return self.root / "coordination.sqlite3"

    @contextmanager
    def exclusive(self):
        # File records remain backwards compatible. SQLite provides a shared
        # process lock and a unique identity index; nested executor calls reuse it.
        with self._mutex:
            if getattr(self._local, 'connection', None) is not None:
                yield
                return
            self.ensure_ready()
            db = sqlite3.connect(self.root / 'coordination.sqlite3', timeout=120)
            try:
                db.execute('CREATE TABLE IF NOT EXISTS identities (project TEXT, key TEXT, task TEXT, PRIMARY KEY(project,key))')
                db.execute('BEGIN IMMEDIATE')
                self._local.connection = db
                yield
                db.commit()
            finally:
                self._local.connection = None
                db.close()

    @contextmanager
    def task_lock(self, task_id: str, *, blocking: bool = True, purpose: str = "execution"):
        """Reentrant per-task, cross-process lock; never holds the identity DB lock."""
        if not TASK_ID_PATTERN.fullmatch(task_id):
            raise ValueError("video_task_id is invalid")
        self.ensure_ready()
        key = (task_id, purpose)
        held = getattr(self._local, "task_locks", None)
        if held is None:
            held = self._local.task_locks = set()
        if key in held:
            yield True
            return
        with (self.root / f".{task_id}.{purpose}.lock").open("a") as stream:
            if fcntl is not None:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
                except BlockingIOError:
                    yield False
                    return
            held.add(key)
            try:
                yield True
            finally:
                held.remove(key)
                if fcntl is not None:
                    fcntl.flock(stream, fcntl.LOCK_UN)

    @classmethod
    def from_environment(cls) -> "TaskStore":
        return cls(Path(os.environ.get("VIDEO_ARTIFACT_ROOT", "/data/video-mcp")))

    def ensure_ready(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o750)

    def _write(self, record: TaskRecord) -> None:
        self.ensure_ready()
        fd, temporary = tempfile.mkstemp(prefix=".task-", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                output.write(record.model_dump_json(indent=2) + "\n")
            os.replace(temporary, self.root / f"{record.video_task_id}.json")
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    def find_idempotency(self, project_id: str, key: str) -> TaskRecord | None:
        self.ensure_ready()
        for path in self.root.glob("video_task_*.json"):
            record = TaskRecord.model_validate_json(path.read_text(encoding="utf-8"))
            if record.project_id == project_id and record.idempotency_key == key:
                return record
        return None

    def create(self, *, project_id: str, idempotency_key: str, input_digest: str,
               request: dict[str, object], runtime_task_id: str | None, status: str,
               error: dict[str, str] | None = None, authority: dict[str, str] | None = None,
               runtime_endpoint: str | None = None, runtime_version: str | None = None,
               runtime_route: str | None = None) -> TaskRecord:
        existing = self.find_idempotency(project_id, idempotency_key)
        if existing:
            if existing.input_digest != input_digest:
                raise TaskConflict("idempotency key already exists with different input")
            return existing
        now = datetime.now(UTC).isoformat()
        if request.get("service") in {"depth", "sr", "interpolate", "h3-latent-upscale"}:
            service = str(request["service"])
        elif request.get("route") in {"fal", "grok", "higgsfield"}:
            service = str(request["route"])
        elif request.get("service") == "h3-sol":
            service = "h3-sol"
        else:
            route = str(request.get("route", "h3"))
            service = ("h3-sol" if route.startswith("h3-sol") else
                       "h3-singularity" if route == "h3-singularity" else "h3")
        task_id = f"video_task_{uuid.uuid4().hex}"
        record = TaskRecord(video_task_id=task_id, service=service, project_id=project_id,
                            idempotency_key=idempotency_key, input_digest=input_digest,
                            request=request, runtime_task_id=task_id if runtime_task_id is None else runtime_task_id,
                            status=status, error=error,
                            authority=authority, runtime_endpoint=runtime_endpoint,
                            runtime_version=runtime_version, runtime_route=runtime_route,
                            created_at=now, updated_at=now)
        self._write(record)
        from .titles import initialize
        initialize(self, record)
        db = getattr(self._local, 'connection', None)
        if db is not None:
            db.execute('INSERT INTO identities(project,key,task) VALUES(?,?,?)',
                       (project_id, idempotency_key, task_id))
        return record

    def get(self, video_task_id: str) -> TaskRecord:
        if not TASK_ID_PATTERN.fullmatch(video_task_id):
            raise ValueError("video_task_id is invalid")
        path = self.root / f"{video_task_id}.json"
        if not path.is_file():
            raise TaskNotFound("video task does not exist")
        record = TaskRecord.model_validate_json(path.read_text(encoding="utf-8"))
        return self.with_timing(record)

    @staticmethod
    def with_timing(record: TaskRecord) -> TaskRecord:
        def seconds(a, b):
            if not a or not b: return None
            return max(0.0, (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds())
        dispatched = record.dispatched_at
        completed = record.completed_at
        now = datetime.now(UTC).isoformat()
        generation_end = None
        if record.service == "higgsfield":
            state = record.provider_state or {}
            candidate = state.get("generation_completed_at")
            try:
                stamp = datetime.fromisoformat(candidate)
                start = datetime.fromisoformat(dispatched or record.created_at)
                end = datetime.fromisoformat(completed or now)
                if stamp.tzinfo and start <= stamp <= end:
                    generation_end = candidate
            except (ValueError, TypeError):
                pass
        processing_end = generation_end if record.service == "higgsfield" else completed
        timing = {"queue_seconds": seconds(record.created_at, dispatched),
                  "queue_elapsed_seconds": seconds(record.created_at, now) if not dispatched and not completed and record.status == "queued" else None,
                  "processing_seconds": seconds(dispatched, processing_end),
                  "processing_elapsed_seconds": seconds(dispatched, now) if dispatched and not completed and not processing_end else None,
                  "total_seconds": seconds(record.created_at, completed)}
        if record.service == "higgsfield":
            timing.update(generation_completed_at=generation_end,
                          generation_time_source=(record.provider_state or {}).get("generation_time_source") if generation_end else None)
        return record.model_copy(update={"timing": timing})

    def update(self, record: TaskRecord, **values: object) -> TaskRecord:
        with self.task_lock(record.video_task_id, purpose="write"):
            return self._update_current(self.get(record.video_task_id), **values)

    def _update_current(self, record: TaskRecord, **values: object) -> TaskRecord:
        now = datetime.now(UTC).isoformat()
        if not record.completed_at:
            if record.status in {"succeeded", "failed", "cancelled"}:
                values["completed_at"] = record.updated_at
            elif values.get("status") in {"succeeded", "failed", "cancelled"}:
                values["completed_at"] = now
        if record.dispatched_at:
            values["dispatched_at"] = record.dispatched_at
        updated = self.with_timing(record.model_copy(update={**values, "updated_at": now}))
        self._write(updated)
        return updated

    @staticmethod
    def task_version_ids(record: TaskRecord) -> list[str]:
        candidates: list[str] = []
        delivery = record.delivery or {}
        media = record.media or {}
        provider_state = record.provider_state or {}
        content_ref = provider_state.get("content_ref") if isinstance(provider_state.get("content_ref"), dict) else {}
        collection = record.collection or {}
        col_artifacts = collection.get("artifacts") if isinstance(collection.get("artifacts"), dict) else {}

        for raw in (
            delivery.get("final_artifact_id"),
            record.artifact_id,
            content_ref.get("version_id") if isinstance(content_ref, dict) else None,
            content_ref.get("artifact_id") if isinstance(content_ref, dict) else None,
            delivery.get("preview_artifact_id"),
            media.get("preview_artifact_id"),
            *(col_artifacts.values() if isinstance(col_artifacts, dict) else ()),
            record.video_task_id,
        ):
            if isinstance(raw, str) and raw.strip() and raw.strip() not in candidates:
                candidates.append(raw.strip())
        return candidates

    def is_record_nsfw(self, record: TaskRecord, ratings_store: Any | None = None) -> bool:
        store = ratings_store
        if store is None:
            from .content_rating import ContentRatingStore
            store = ContentRatingStore(self.db_path, domain="video")
        for vid in self.task_version_ids(record):
            if store.is_nsfw(vid, task_id=record.video_task_id):
                return True
        return store.is_nsfw(None, task_id=record.video_task_id)

    def record_ui_fields(
        self,
        record: TaskRecord,
        ratings_store: Any | None = None,
        *,
        can_write: bool = True,
    ) -> dict[str, Any]:
        store = ratings_store
        if store is None:
            from .content_rating import ContentRatingStore
            store = ContentRatingStore(self.db_path, domain="video")
        vids = self.task_version_ids(record)
        chosen_vid = vids[0] if vids else record.video_task_id
        for vid in vids:
            rec = store.get(vid, task_id=record.video_task_id)
            if rec.get("rating") == "nsfw":
                chosen_vid = vid
                break
            if rec.get("rating") != "unrated" and chosen_vid == vids[0]:
                chosen_vid = vid
        return store.ui_fields(chosen_vid, task_id=record.video_task_id, can_write=can_write)

    def find_by_artifact_or_task(self, version_id: str) -> TaskRecord | None:
        vid = (version_id or "").strip()
        if not vid:
            return None
        self.ensure_ready()
        if TASK_ID_PATTERN.fullmatch(vid):
            try:
                return self.get(vid)
            except TaskNotFound:
                pass
        records: list[TaskRecord] = []
        for path in self.root.glob("video_task_*.json"):
            try:
                rec = TaskRecord.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if vid in self.task_version_ids(rec):
                records.append(self.with_timing(rec))
        if not records:
            return None
        records.sort(key=lambda r: (r.created_at, r.video_task_id), reverse=True)
        return records[0]

    def list_tasks(
        self,
        project_id: str | None = None,
        service: str | None = None,
        route: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
        include_nsfw: bool = False,
        ratings_store: Any | None = None,
    ) -> tuple[list[TaskRecord], int]:
        self.ensure_ready()
        all_records: list[TaskRecord] = []
        for path in self.root.glob("video_task_*.json"):
            try:
                rec = TaskRecord.model_validate_json(path.read_text(encoding="utf-8"))
                all_records.append(self.with_timing(rec))
            except (OSError, ValueError):
                continue
        all_records.sort(key=lambda r: (r.created_at, r.video_task_id), reverse=True)

        filtered: list[TaskRecord] = []
        for rec in all_records:
            if project_id and rec.project_id != project_id:
                continue
            if service and rec.service != service:
                continue
            if route:
                rec_route = rec.runtime_route or rec.request.get("route") or rec.service
                if rec_route != route and rec.service != route:
                    continue
            if status:
                if status == "completed":
                    if rec.status not in {"succeeded", "completed"}:
                        continue
                elif rec.status != status:
                    continue
            if not include_nsfw and self.is_record_nsfw(rec, ratings_store):
                continue
            filtered.append(rec)

        total = len(filtered)
        return filtered[offset : offset + limit], total

    def get_stats(
        self,
        *,
        project_id: str | None = None,
        include_nsfw: bool = False,
        ratings_store: Any | None = None,
    ) -> dict[str, Any]:
        records, total = self.list_tasks(
            project_id=project_id,
            limit=100000,
            offset=0,
            include_nsfw=include_nsfw,
            ratings_store=ratings_store,
        )
        queued = sum(1 for r in records if r.status == "queued")
        running = sum(1 for r in records if r.status == "running")
        succeeded = sum(1 for r in records if r.status in {"succeeded", "completed"})
        failed = sum(1 for r in records if r.status == "failed")
        cancelled = sum(1 for r in records if r.status in {"cancelled", "canceled"})
        durations: list[float] = []
        for r in records:
            if r.status in {"succeeded", "completed"} and r.timing:
                dur = r.timing.get("processing_seconds") if r.service == "higgsfield" else r.timing.get("total_seconds")
                if isinstance(dur, (int, float)):
                    durations.append(float(dur))
        avg_duration = round(sum(durations) / len(durations), 2) if durations else 0.0
        return {
            "total": total,
            "queued": queued,
            "running": running,
            "succeeded": succeeded,
            "completed": succeeded,
            "failed": failed,
            "cancelled": cancelled,
            "canceled": cancelled,
            "avg_duration": avg_duration,
        }
