"""Bounded, restart-safe collection of completed native Video MCP results."""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from .tasks import TaskRecord


def public_collection(record):
    state = record.collection or {}
    if state.get("state") not in {"downloading", "retry_wait", "ready"}:
        return None
    return {key: state.get(key) for key in (
        "state", "attempts", "started_at", "completed_at", "next_attempt_at", "last_error")}


class ResultCollector:
    def __init__(self, executor, workers=3):
        self.executor = executor
        self.tasks = executor.tasks
        self.workers = workers
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="video-result")
        self.active = {}
        self.lock = threading.Lock()
        self.closed = False

    @staticmethod
    def eligible(record):
        # Grok/Google/Higgsfield already collect as part of their own authorized state
        # machine; never duplicate their central Artifact or delivery workflow.
        needs_preview = record.service in {"depth", "sr", "interpolate"} and not (record.media or {}).get("preview_artifact_id")
        return (record.status == "succeeded" and (not record.artifact_id or needs_preview)
                and record.service not in {"higgsfield", "grok", "google", "h3-latent-upscale"})

    def sync(self):
        with self.lock:
            if self.closed:
                return
            for task_id, future in list(self.active.items()):
                if future.done():
                    try:
                        future.result()
                    except Exception:
                        logging.getLogger(__name__).warning("Result reconciliation deferred")
                    del self.active[task_id]
            available = self.workers - len(self.active)
            if not available:
                return
            candidates = []
            for path in self.tasks.root.glob("video_task_*.json"):
                try:
                    record = TaskRecord.model_validate_json(path.read_text())
                    due = float((record.collection or {}).get("next_attempt_at", 0))
                    if self.eligible(record) and due <= time.time() and record.video_task_id not in self.active:
                        candidates.append((due, record.created_at, record.video_task_id))
                except (OSError, ValueError, TypeError):
                    continue
            for _, _, task_id in sorted(candidates)[:available]:
                self.active[task_id] = self.pool.submit(self.collect, task_id)

    def collect(self, task_id):
        # Another MCP caller/worker may already be collecting the same result.
        with self.tasks.task_lock(task_id, blocking=False) as acquired:
            if not acquired:
                return
            record = self.tasks.get(task_id)
            if not self.eligible(record):
                return
            state = dict(record.collection or {})
            if float(state.get("next_attempt_at", 0)) > time.time():
                return
            attempts = int(state.get("attempts", 0)) + 1
            state.update(state="downloading", attempts=attempts,
                         started_at=datetime.now(UTC).isoformat(), next_attempt_at=0)
            self.tasks.update(record, collection=state)
            try:
                result = self.executor.result(task_id)
                if not result.artifact_id:
                    raise ValueError("result_not_ready")
            except Exception:
                # Never persist exception text: provider URLs may contain credentials.
                delay = min(300, 5 * 2 ** min(attempts - 1, 6))
                state.update(state="retry_wait", next_attempt_at=time.time() + delay,
                             last_error="result_collection_failed")
                self.tasks.update(self.tasks.get(task_id), collection=state)
                logging.getLogger(__name__).warning("Video result collection deferred: %s", task_id)
            else:
                state.update(state="ready", next_attempt_at=0, last_error=None,
                             completed_at=datetime.now(UTC).isoformat())
                self.tasks.update(result, collection=state)

    def close(self):
        with self.lock:
            self.closed = True
            self.pool.shutdown(wait=False, cancel_futures=True)
