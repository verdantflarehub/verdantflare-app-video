from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

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
    media: dict[str, object] | None = None
    error: dict[str, str] | None = None


class TaskStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve() / "tasks"

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
               request: dict[str, object], runtime_task_id: str, status: str) -> TaskRecord:
        existing = self.find_idempotency(project_id, idempotency_key)
        if existing:
            if existing.input_digest != input_digest:
                raise TaskConflict("idempotency key already exists with different input")
            return existing
        now = datetime.now(UTC).isoformat()
        if request.get("service") in {"depth", "sr", "interpolate"}:
            service = str(request["service"])
        elif request.get("service") == "h3-sol":
            service = "h3-sol"
        else:
            route = str(request.get("route", "h3"))
            service = "h3-sol" if route.startswith("h3-sol") else "h3"
        record = TaskRecord(video_task_id=f"video_task_{uuid.uuid4().hex}", service=service, project_id=project_id,
                            idempotency_key=idempotency_key, input_digest=input_digest,
                            request=request, runtime_task_id=runtime_task_id, status=status,
                            created_at=now, updated_at=now)
        self._write(record)
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
        timing = {"queue_seconds": seconds(record.created_at, dispatched),
                  "processing_seconds": seconds(dispatched, completed),
                  "processing_elapsed_seconds": seconds(dispatched, now) if dispatched and not completed else None,
                  "total_seconds": seconds(record.created_at, completed)}
        return record.model_copy(update={"timing": timing})

    def update(self, record: TaskRecord, **values: object) -> TaskRecord:
        now = datetime.now(UTC).isoformat()
        if not record.completed_at:
            if record.status in {"succeeded", "failed", "cancelled"}:
                values["completed_at"] = record.updated_at
            elif values.get("status") in {"succeeded", "failed", "cancelled"}:
                values["completed_at"] = now
        updated = record.model_copy(update={**values, "updated_at": now})
        self._write(updated)
        return updated
