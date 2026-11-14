"""Durable single-worker SQLite queue isolating P1 (video) and P2 (regenerate_2k) tasks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
import time


TASK_ID = re.compile(r"singularity_[0-9a-f]{32}")
REGENERATE_TASK_ID = re.compile(r"video_task_[0-9a-f]{32}")
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
                kind TEXT NOT NULL DEFAULT 'video',
                started REAL,
                finished REAL,
                worker_settled INTEGER NOT NULL DEFAULT 0,
                cancel_requested INTEGER NOT NULL DEFAULT 0
            )"""
        )
        self._migrate_columns()
        if acquire_lock:
            now = time.time()
            with self.db:
                self.db.execute(
                    """UPDATE tasks
                       SET status='failed', stage='interrupted', error='interrupted',
                           finished=COALESCE(finished, ?), worker_settled=1, updated=?
                       WHERE status IN ('in_progress', 'running')""",
                    (now, now),
                )
                self.db.execute(
                    """UPDATE tasks
                       SET worker_settled=1
                       WHERE status IN ('completed', 'succeeded', 'failed', 'cancelled')
                         AND worker_settled=0"""
                )

    def _migrate_columns(self) -> None:
        existing = {row["name"] for row in self.db.execute("PRAGMA table_info(tasks)").fetchall()}
        migrations = (
            ("kind", "ALTER TABLE tasks ADD COLUMN kind TEXT NOT NULL DEFAULT 'video'"),
            ("started", "ALTER TABLE tasks ADD COLUMN started REAL"),
            ("finished", "ALTER TABLE tasks ADD COLUMN finished REAL"),
            ("worker_settled", "ALTER TABLE tasks ADD COLUMN worker_settled INTEGER NOT NULL DEFAULT 0"),
            ("cancel_requested", "ALTER TABLE tasks ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0"),
        )
        with self.db:
            for col_name, ddl in migrations:
                if col_name not in existing:
                    self.db.execute(ddl)

    def close(self):
        self.db.close()
        if self.owner is not None:
            _unlock_owner_file(self.owner)
            self.owner.close()
            self.owner = None

    def _read_video(self, task_id: str) -> sqlite3.Row:
        if not isinstance(task_id, str) or not TASK_ID.fullmatch(task_id):
            raise QueueError("invalid_task_id")
        row = self.db.execute("SELECT * FROM tasks WHERE id=? AND kind='video'", (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        return row

    def _read_regenerate(self, video_task_id: str) -> sqlite3.Row:
        if not isinstance(video_task_id, str) or not REGENERATE_TASK_ID.fullmatch(video_task_id):
            raise QueueError("invalid_task_id")
        row = self.db.execute(
            "SELECT * FROM tasks WHERE id=? AND kind='regenerate_2k'",
            (video_task_id,),
        ).fetchone()
        if row is None:
            raise KeyError(video_task_id)
        return row

    def _read_any(self, task_id: str) -> sqlite3.Row:
        if not isinstance(task_id, str) or not (
            TASK_ID.fullmatch(task_id) or REGENERATE_TASK_ID.fullmatch(task_id)
        ):
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

    @staticmethod
    def _format_regenerate(row: sqlite3.Row) -> dict:
        return {
            "video_task_id": row["id"],
            "status": row["status"],
            "stage": row["stage"],
            "created_at": row["created"],
            "started_at": row["started"],
            "finished_at": row["finished"],
            "updated_at": row["updated"],
            "worker_settled": bool(row["worker_settled"]),
            "cancel_requested": bool(row["cancel_requested"]),
            "runtime_metrics": json.loads(row["runtime_metrics"]) if row["runtime_metrics"] else None,
            "result": json.loads(row["result"]) if row["result"] else None,
            "error": (
                {"code": row["error"], "message": "Video 2K regeneration failed"}
                if row["error"]
                else None
            ),
        }

    def get(self, task_id: str) -> dict:
        with self.lock:
            row = self._read_video(task_id)
        return self._format_video(row)

    def get_regenerate(self, video_task_id: str) -> dict:
        with self.lock:
            row = self._read_regenerate(video_task_id)
        return self._format_regenerate(row)

    def get_any(self, task_id: str) -> dict:
        with self.lock:
            row = self._read_any(task_id)
        return {
            "id": row["id"],
            "kind": row["kind"],
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
            "error": row["error"],
        }

    def _active_count(self) -> int:
        return int(
            self.db.execute(
                "SELECT COUNT(*) FROM tasks WHERE status IN ('queued','in_progress','running')"
            ).fetchone()[0]
        )

    def submit(self, request: dict) -> dict:
        key = request.get("idempotency_key")
        if not isinstance(key, str) or not REGENERATE_TASK_ID.fullmatch(key):
            raise QueueError("invalid_idempotency_key")
        encoded = json.dumps(request, sort_keys=True, ensure_ascii=False, allow_nan=False)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        with self.lock, self.db:
            previous = self.db.execute(
                "SELECT * FROM tasks WHERE idempotency_key=?", (key,)
            ).fetchone()
            if previous:
                if previous["kind"] != "video" or previous["digest"] != digest:
                    raise QueueError("idempotency_conflict")
                return self._format_video(previous)
            if self._active_count() >= MAX_ACTIVE_TASKS:
                raise QueueError("queue_full")
            task_id = "singularity_" + __import__("uuid").uuid4().hex
            now = time.time()
            self.db.execute(
                """INSERT INTO tasks (
                    id, idempotency_key, digest, request, status, stage,
                    created, updated, execution_instance_id, runtime_metrics,
                    result, error, kind, started, finished, worker_settled
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                    "video",
                    None,
                    None,
                    0,
                ),
            )
            row = self._read_video(task_id)
        return self._format_video(row)

    def find_regenerate_idempotency(self, key: str, digest: str) -> dict | None:
        if not isinstance(key, str) or not REGENERATE_TASK_ID.fullmatch(key):
            raise QueueError("invalid_idempotency_key")
        with self.lock:
            previous = self.db.execute(
                "SELECT * FROM tasks WHERE id=? OR idempotency_key=?",
                (key, key),
            ).fetchone()
            if previous is None:
                return None
            if previous["kind"] != "regenerate_2k" or previous["digest"] != digest:
                raise QueueError("idempotency_conflict")
            return self._format_regenerate(previous)

    def submit_regenerate(self, request: dict) -> dict:
        key = request.get("idempotency_key")
        if not isinstance(key, str) or not REGENERATE_TASK_ID.fullmatch(key):
            raise QueueError("invalid_idempotency_key")
        encoded = json.dumps(request, sort_keys=True, ensure_ascii=False, allow_nan=False)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        with self.lock, self.db:
            previous = self.db.execute(
                "SELECT * FROM tasks WHERE id=? OR idempotency_key=?",
                (key, key),
            ).fetchone()
            if previous:
                if previous["kind"] != "regenerate_2k" or previous["digest"] != digest:
                    raise QueueError("idempotency_conflict")
                return self._format_regenerate(previous)
            if self._active_count() >= MAX_ACTIVE_TASKS:
                raise QueueError("queue_full")
            now = time.time()
            self.db.execute(
                """INSERT INTO tasks (
                    id, idempotency_key, digest, request, status, stage,
                    created, updated, execution_instance_id, runtime_metrics,
                    result, error, kind, started, finished, worker_settled
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    key,
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
                    "regenerate_2k",
                    None,
                    None,
                    0,
                ),
            )
            row = self._read_regenerate(key)
        return self._format_regenerate(row)

    def find_idempotency(self, key: str) -> dict | None:
        if not isinstance(key, str) or not REGENERATE_TASK_ID.fullmatch(key):
            raise QueueError("invalid_idempotency_key")
        with self.lock:
            row = self.db.execute(
                "SELECT id FROM tasks WHERE idempotency_key=? AND kind='video'",
                (key,),
            ).fetchone()
            return self.get(row["id"]) if row else None

    def take(self) -> dict | None:
        with self.lock, self.db:
            row = self.db.execute(
                "SELECT id, kind FROM tasks WHERE status='queued' ORDER BY created, id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            now = time.time()
            active_status = "in_progress" if row["kind"] == "video" else "running"
            active_stage = "downloading" if row["kind"] == "video" else "validating"
            self.db.execute(
                """UPDATE tasks
                   SET status=?, stage=?, started=COALESCE(started, ?), worker_settled=0, updated=?
                   WHERE id=?""",
                (active_status, active_stage, now, now, row["id"]),
            )
        return self.get_any(row["id"])

    def cancel_queued_or_check(self, task_id: str, kind: str) -> tuple[str, dict]:
        """Cancel a queued task immediately, return 'already_cancelled', or signal 'active' for Supervisor."""
        with self.lock, self.db:
            row = (
                self._read_video(task_id)
                if kind == "video"
                else self._read_regenerate(task_id)
            )
            formatter = self._format_video if kind == "video" else self._format_regenerate
            status = row["status"]
            if status == "cancelled":
                return ("already_cancelled", formatter(row))
            terminal = {"completed", "failed"} if kind == "video" else {"succeeded", "failed"}
            if status in terminal:
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
                updated_row = (
                    self._read_video(task_id)
                    if kind == "video"
                    else self._read_regenerate(task_id)
                )
                return ("cancelled_queued", formatter(updated_row))
            now = time.time()
            self.db.execute(
                "UPDATE tasks SET cancel_requested=1, updated=? WHERE id=?",
                (now, task_id),
            )
            updated_row = (
                self._read_video(task_id)
                if kind == "video"
                else self._read_regenerate(task_id)
            )
            return ("active", formatter(updated_row))

    def is_cancel_requested(self, task_id: str) -> bool:
        with self.lock:
            row = self._read_any(task_id)
        return bool(row["cancel_requested"])

    def mark_settled(
        self,
        task_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        error: str | None = None,
    ) -> dict:
        """Mark worker_settled=1 ONLY after Supervisor reaps the worker subprocess group."""
        now = time.time()
        with self.lock, self.db:
            row = self._read_any(task_id)
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
            refreshed = self._read_any(task_id)
        if refreshed["kind"] == "video":
            return self.get(task_id)
        return self.get_regenerate(task_id)

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
        if values.get("status") in {"completed", "succeeded", "failed", "cancelled"} and "finished" not in values:
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
            self._read_any(task_id)
            self.db.execute(f"UPDATE tasks SET {','.join(assignments)} WHERE id=?", args)
