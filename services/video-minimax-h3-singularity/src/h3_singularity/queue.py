"""Durable, single-worker queue for the GPU-bound Singularity runtime."""

from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import threading
import time


TASK_ID = re.compile(r"singularity_[0-9a-f]{32}")


class QueueError(ValueError):
    pass


class Queue:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.owner = (self.root / "owner.lock").open("a")
        try:
            fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.owner.close()
            raise QueueError("worker_already_running") from None
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.root / "queue.sqlite", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE NOT NULL,
                digest TEXT NOT NULL, request TEXT NOT NULL,
                status TEXT NOT NULL, stage TEXT NOT NULL,
                created REAL NOT NULL, updated REAL NOT NULL,
                execution_instance_id TEXT, runtime_metrics TEXT,
                result TEXT, error TEXT)"""
        )
        with self.db:
            self.db.execute(
                "UPDATE tasks SET status='failed', stage='interrupted', error='interrupted', updated=? WHERE status='in_progress'",
                (time.time(),),
            )

    def close(self):
        self.db.close()
        self.owner.close()

    def _read(self, task_id: str):
        if not isinstance(task_id, str) or not TASK_ID.fullmatch(task_id):
            raise QueueError("invalid_task_id")
        row = self.db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        return row

    def get(self, task_id: str) -> dict:
        with self.lock:
            row = self._read(task_id)
        return {
            "id": row["id"],
            "object": "video",
            "status": row["status"],
            "stage": row["stage"],
            "request": json.loads(row["request"]),
            "created_at": row["created"],
            "updated_at": row["updated"],
            "execution_instance_id": row["execution_instance_id"],
            "runtime_metrics": json.loads(row["runtime_metrics"]) if row["runtime_metrics"] else None,
            "result": json.loads(row["result"]) if row["result"] else None,
            "error": {"code": row["error"]} if row["error"] else None,
        }

    def submit(self, request: dict) -> dict:
        key = request.get("idempotency_key")
        if not isinstance(key, str) or not re.fullmatch(r"video_task_[0-9a-f]{32}", key):
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
                return self.get(previous["id"])
            active = self.db.execute(
                "SELECT COUNT(*) FROM tasks WHERE status IN ('queued','in_progress')"
            ).fetchone()[0]
            if active >= 2:
                raise QueueError("queue_full")
            task_id = "singularity_" + __import__("uuid").uuid4().hex
            now = time.time()
            self.db.execute(
                "INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (task_id, key, digest, encoded, "queued", "queued", now, now, None, None, None, None),
            )
        return self.get(task_id)

    def take(self) -> dict | None:
        with self.lock, self.db:
            row = self.db.execute(
                "SELECT id FROM tasks WHERE status='queued' ORDER BY created,id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            self.db.execute(
                "UPDATE tasks SET status='in_progress',stage='downloading',updated=? WHERE id=?",
                (time.time(), row["id"]),
            )
        return self.get(row["id"])

    def update(self, task_id: str, **values):
        allowed = {"status", "stage", "execution_instance_id", "runtime_metrics", "result", "error"}
        if set(values) - allowed:
            raise QueueError("invalid_task_update")
        assignments = ["updated=?"]
        args = [time.time()]
        for name, value in values.items():
            assignments.append(f"{name}=?")
            args.append(json.dumps(value, allow_nan=False) if name in {"runtime_metrics", "result"} and value is not None else value)
        args.append(task_id)
        with self.lock, self.db:
            self._read(task_id)
            self.db.execute(f"UPDATE tasks SET {','.join(assignments)} WHERE id=?", args)
