"""Bounded, restart-safe collection of completed native Video MCP results."""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from .tasks import TaskRecord
from .artifacts import ArtifactError

MANAGED_COLLECTORS = {"higgsfield", "grok", "google", "h3-latent-upscale"}


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
        self.scan_samples = deque(maxlen=30)
        self.last_scan_log = 0.0

    @staticmethod
    def eligible(record):
        # Grok/Google/Higgsfield already collect as part of their own authorized state
        # machine; never duplicate their central Artifact or delivery workflow.
        needs_preview = record.service in {"depth", "sr", "interpolate", "regenerate_2k"} and not (record.media or {}).get("preview_artifact_id")
        return (record.status == "succeeded" and (not record.artifact_id or needs_preview)
                and record.service not in MANAGED_COLLECTORS)

    @staticmethod
    def stale_cached(record):
        return (record.status == "succeeded" and bool(record.artifact_id)
                and record.service not in MANAGED_COLLECTORS
                and (record.collection or {}).get("state") in {"downloading", "retry_wait"})

    def reconcile_cached(self, record):
        """Called under the existing task lock; never fetch or generate media."""
        if not self.stale_cached(record) or self.eligible(record):
            return record
        ids = [record.artifact_id]
        if record.service in {"depth", "sr", "interpolate", "regenerate_2k"}:
            ids.append(record.media["preview_artifact_id"])
        completed = []
        try:
            for artifact_id in ids:
                artifact = self.executor.artifacts.get(artifact_id, record.project_id)
                path = self.executor.artifacts.content_path(artifact)
                if path.stat().st_size != artifact.size:
                    return record
                with path.open("rb") as content:
                    if hashlib.file_digest(content, "sha256").hexdigest() != artifact.sha256:
                        return record
                try:
                    stamp = datetime.fromisoformat(artifact.created_at)
                    completed.append(stamp.astimezone(UTC) if stamp.tzinfo else None)
                except (ValueError, TypeError):
                    completed.append(None)
        except (ArtifactError, OSError, ValueError):
            return record
        # File creation is evidence of reception, unlike generation completion.
        # Missing historical time remains unknown; do not invent an old duration.
        completed_at = max(completed).isoformat() if all(completed) else None
        state = dict(record.collection)
        state.update(state="ready", next_attempt_at=0, last_error=None, completed_at=completed_at)
        return self.tasks.update(record, collection=state)

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
            scan_started = time.perf_counter()
            scanned = 0
            candidates = []
            for path in self.tasks.root.glob("video_task_*.json"):
                scanned += 1
                try:
                    record = TaskRecord.model_validate_json(path.read_text())
                    due = float((record.collection or {}).get("next_attempt_at", 0))
                    if (self.eligible(record) or self.stale_cached(record)) and due <= time.time() and record.video_task_id not in self.active:
                        candidates.append((due, record.created_at, record.video_task_id))
                except (OSError, ValueError, TypeError):
                    continue
            due_candidates = sorted(candidates)[:available]
            self.observe_scan(scanned, time.perf_counter() - scan_started)
            for _, _, task_id in due_candidates:
                self.active[task_id] = self.pool.submit(self.collect, task_id)

    def observe_scan(self, files, elapsed):
        self.scan_samples.append(elapsed)
        now = time.monotonic()
        if now - self.last_scan_log < 60:
            return
        self.last_scan_log = now
        samples = sorted(self.scan_samples)
        p95_ms = samples[(len(samples) * 95 + 99) // 100 - 1] * 1000
        # The service runs under Uvicorn, whose error logger emits INFO by
        # default; the module logger would hide healthy samples.
        logger = logging.getLogger("uvicorn.error")
        message = "Video collection scan: files=%d p95_ms=%.1f samples=%d"
        if files >= 5000 or p95_ms >= 200:
            logger.warning(message, files, p95_ms, len(samples))
        else:
            logger.info(message, files, p95_ms, len(samples))

    def collect(self, task_id):
        # Another MCP caller/worker may already be collecting the same result.
        with self.tasks.task_lock(task_id, blocking=False) as acquired:
            if not acquired:
                return
            record = self.tasks.get(task_id)
            if self.stale_cached(record) and not self.eligible(record):
                reconciled = self.reconcile_cached(record)
                if (reconciled.collection or {}).get("state") != "ready":
                    # Broken cached bytes need investigation, not a tight hash
                    # loop or an automatic replacement of the original output.
                    self.tasks.update(record, collection={**record.collection,
                        "next_attempt_at": time.time() + 300})
                return
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
                result = self.reconcile_cached(self.tasks.get(task_id))
                if not result.artifact_id or (result.collection or {}).get("state") != "ready":
                    raise ValueError("result_not_ready")
            except Exception:
                current = self.reconcile_cached(self.tasks.get(task_id))
                if (current.collection or {}).get("state") == "ready":
                    return
                # Never persist exception text: provider URLs may contain credentials.
                delay = min(300, 5 * 2 ** min(attempts - 1, 6))
                state.update(state="retry_wait", next_attempt_at=time.time() + delay,
                             last_error="result_collection_failed")
                self.tasks.update(self.tasks.get(task_id), collection=state)
                logging.getLogger(__name__).warning("Video result collection deferred: %s", task_id)

    def close(self):
        with self.lock:
            self.closed = True
            self.pool.shutdown(wait=False, cancel_futures=True)
