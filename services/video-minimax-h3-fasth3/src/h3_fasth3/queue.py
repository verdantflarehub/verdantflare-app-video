"""Durable single-worker SQLite queue for FastH3 video generation tasks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
import uuid


TASK_ID = re.compile(r"fasth3_[0-9a-f]{32}")
IDEMPOTENCY_KEY_RE = re.compile(r"video_task_[0-9a-f]{32}")
MAX_ACTIVE_TASKS = 2


class QueueError(ValueError):
    pass


def _lock_owner_file(handle) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write("0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise BlockingIOError("owner lock busy") from None
    else:
        import fcntl

        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_owner_file(handle) -> None:
    if os.name == "nt":
        import msvcrt

        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
    else:
        import fcntl

        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        except OSError:
            pass


class Queue:
    def __init__(self, root: str | Path, *, acquire_lock: bool = True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.owner = None
        if acquire_lock:
            self.owner = (self.root / "owner.lock").open("a+")
            try:
                _lock_owner_file(self.owner)
            except BlockingIOError:
                self.owner.close()
                self.owner = None
                raise QueueError("worker_already_running") from None
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.root / "queue.sqlite", timeout=30.0, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA busy_timeout=30000")
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                idempotency_key TEXT UNIQUE NOT NULL,
                digest TEXT NOT NULL,
                request TEXT NOT NULL,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                created REAL NOT NULL,
                updated REAL NOT NULL,
                execution_instance_id TEXT,
                runtime_metrics TEXT,
                result TEXT,
                error TEXT,
                started REAL,
                finished REAL,
                worker_settled INTEGER NOT NULL DEFAULT 0,
                cancel_requested INTEGER NOT NULL DEFAULT 0
            )"""
        )
        if acquire_lock:
            now = time.time()
            with self.db:
                self.db.execute(
                    """UPDATE tasks
                       SET status='failed', stage='interrupted', error='interrupted',
                           finished=COALESCE(finished, ?), worker_settled=1, updated=?
                       WHERE status='in_progress'""",
                    (now, now),
                )
                self.db.execute(
                    """UPDATE tasks
                       SET worker_settled=1
                       WHERE status IN ('completed', 'failed', 'cancelled')
                         AND worker_settled=0"""
                )

    def close(self):
        self.db.close()
        if self.owner is not None:
            _unlock_owner_file(self.owner)
            self.owner.close()
            self.owner = None

    def _read_row(self, task_id: str) -> sqlite3.Row:
        if not isinstance(task_id, str) or not TASK_ID.fullmatch(task_id):
            raise QueueError("invalid_task_id")
        row = self.db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        return row

    @staticmethod
    def _format_video(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "object": "video",
            "status": row["status"],
            "stage": row["stage"],
            "request": json.loads(row["request"]),
            "created_at": row["created"],
            "started_at": row["started"],
            "finished_at": row["finished"],
            "updated_at": row["updated"],
            "worker_settled": bool(row["worker_settled"]),
            "cancel_requested": bool(row["cancel_requested"]),
            "execution_instance_id": row["execution_instance_id"],
            "runtime_metrics": json.loads(row["runtime_metrics"]) if row["runtime_metrics"] else None,
            "result": json.loads(row["result"]) if row["result"] else None,
            "error": {"code": row["error"]} if row["error"] else None,
        }

    def get(self, task_id: str) -> dict:
        with self.lock:
            row = self._read_row(task_id)
        return self._format_video(row)

    def _active_count(self) -> int:
        return int(
            self.db.execute(
                "SELECT COUNT(*) FROM tasks WHERE status IN ('queued','in_progress')"
            ).fetchone()[0]
        )

    def submit(self, request: dict) -> dict:
        key = request.get("idempotency_key")
        if not isinstance(key, str) or not IDEMPOTENCY_KEY_RE.fullmatch(key):
            raise QueueError("invalid_idempotency_key")
        encoded = json.dumps(request, sort_keys=True, ensure_ascii=False, allow_nan=False)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        with self.lock, self.db:
            previous = self.db.execute(
                "SELECT * FROM tasks WHERE idempotency_key=?", (key,)
            ).fetchone()
            if previous:
                if previous["digest"] != digest:
                    raise QueueError("idempotency_conflict")
                return self._format_video(previous)
            if self._active_count() >= MAX_ACTIVE_TASKS:
                raise QueueError("queue_full")
            task_id = "fasth3_" + uuid.uuid4().hex
            now = time.time()
            self.db.execute(
                """INSERT INTO tasks (
                    id, idempotency_key, digest, request, status, stage,
                    created, updated, execution_instance_id, runtime_metrics,
                    result, error, started, finished, worker_settled
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    task_id,
                    key,
                    digest,
                    encoded,
                    "queued",
                    "queued",
                    now,
                    now,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    0,
                ),
            )
            row = self._read_row(task_id)
        return self._format_video(row)

    def find_idempotency(self, key: str) -> dict | None:
        if not isinstance(key, str) or not IDEMPOTENCY_KEY_RE.fullmatch(key):
            raise QueueError("invalid_idempotency_key")
        with self.lock:
            row = self.db.execute(
                "SELECT id FROM tasks WHERE idempotency_key=?",
                (key,),
            ).fetchone()
            return self.get(row["id"]) if row else None

    def take(self) -> dict | None:
        with self.lock, self.db:
            row = self.db.execute(
                "SELECT id FROM tasks WHERE status='queued' ORDER BY created, id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            now = time.time()
            self.db.execute(
                """UPDATE tasks
                   SET status='in_progress', stage='downloading',
                       started=COALESCE(started, ?), worker_settled=0, updated=?
                   WHERE id=?""",
                (now, now, row["id"]),
            )
        return self.get(row["id"])

    def cancel_queued_or_check(self, task_id: str) -> tuple[str, dict]:
        with self.lock, self.db:
            row = self._read_row(task_id)
            status = row["status"]
            if status == "cancelled":
                return ("already_cancelled", self._format_video(row))
            if status in {"completed", "failed"}:
                raise QueueError("task_already_terminal")
            if status == "queued":
                now = time.time()
                self.db.execute(
                    """UPDATE tasks
                       SET status='cancelled', stage='cancelled', cancel_requested=1,
                           finished=?, worker_settled=1, updated=?
                       WHERE id=?""",
                    (now, now, task_id),
                )
                return ("cancelled_queued", self._format_video(self._read_row(task_id)))
            now = time.time()
            self.db.execute(
                "UPDATE tasks SET cancel_requested=1, updated=? WHERE id=?",
                (now, task_id),
            )
            return ("active", self._format_video(self._read_row(task_id)))

    def is_cancel_requested(self, task_id: str) -> bool:
        with self.lock:
            row = self._read_row(task_id)
        return bool(row["cancel_requested"])

    def mark_settled(
        self,
        task_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        error: str | None = None,
    ) -> dict:
        now = time.time()
        with self.lock, self.db:
            row = self._read_row(task_id)
            next_status = status if status is not None else row["status"]
            next_stage = stage if stage is not None else row["stage"]
            next_error = error if error is not None else row["error"]
            self.db.execute(
                """UPDATE tasks
                   SET status=?, stage=?, error=?, finished=COALESCE(finished, ?),
                       worker_settled=1, updated=?
                   WHERE id=?""",
                (next_status, next_stage, next_error, now, now, task_id),
            )
            refreshed = self._read_row(task_id)
        return self._format_video(refreshed)

    def update(self, task_id: str, **values):
        allowed = {
            "status",
            "stage",
            "execution_instance_id",
            "runtime_metrics",
            "result",
            "error",
            "started",
            "finished",
            "worker_settled",
        }
        if set(values) - allowed:
            raise QueueError("invalid_task_update")
        now = time.time()
        assignments = ["updated=?"]
        args: list[object] = [now]
        if values.get("status") in {"completed", "failed", "cancelled"} and "finished" not in values:
            assignments.append("finished=COALESCE(finished, ?)")
            args.append(now)
        for name, value in values.items():
            assignments.append(f"{name}=?")
            if name in {"runtime_metrics", "result"} and value is not None:
                args.append(json.dumps(value, allow_nan=False))
            elif name == "worker_settled" and isinstance(value, bool):
                args.append(int(value))
            else:
                args.append(value)
        args.append(task_id)
        with self.lock, self.db:
            self._read_row(task_id)
            self.db.execute(f"UPDATE tasks SET {','.join(assignments)} WHERE id=?", args)
