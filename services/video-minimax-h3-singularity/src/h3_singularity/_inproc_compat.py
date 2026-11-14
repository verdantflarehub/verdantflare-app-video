"""Test-only in-process engine harness when `create_app(..., engine_factory=...)` is explicitly supplied."""

from __future__ import annotations

import json
import time

from .errors import RuntimeErrorCode
from .queue import Queue, QueueError


def run_inproc_engine_worker(
    *,
    queue: Queue,
    engine_factory,
    state: dict[str, object],
    lock,
    stop,
    download_references,
) -> None:
    try:
        state["engine"] = engine_factory()
        state["ready"] = True
        while not stop.wait(0.05):
            task = queue.take()
            if task is None:
                continue
            task_id = task["id"]
            if queue.is_cancel_requested(task_id):
                queue.mark_settled(task_id, status="cancelled", stage="cancelled")
                continue
            with lock:
                state["active_task_id"] = task_id
            directory = queue.root / task_id
            download_started = time.perf_counter()
            download_seconds = None
            try:
                directory.mkdir(exist_ok=False)
                request = (
                    task["request"]
                    if isinstance(task.get("request"), dict)
                    else json.loads(task["request"])
                )
                queue.update(task_id, stage="downloading", worker_settled=0)
                files = download_references(directory, request.get("conditions", []))
                download_seconds = time.perf_counter() - download_started
                queue.update(task_id, stage="warming", worker_settled=0)
                result = state["engine"].generate(
                    request,
                    files,
                    directory / "output.mp4",
                    lambda stage: queue.update(task_id, stage=stage, worker_settled=0),
                )
                metrics = dict(result.pop("runtime_metrics", {}) or {})
                metrics["reference_download_seconds"] = download_seconds
                queue.update(
                    task_id,
                    status="completed",
                    stage="completed",
                    execution_instance_id=result.pop("execution_instance_id"),
                    runtime_metrics=metrics,
                    result=result,
                    worker_settled=1,
                )
            except Exception as exc:
                code = getattr(exc, "code", type(exc).__name__)
                metrics = dict(getattr(exc, "runtime_metrics", None) or {})
                if download_seconds is None:
                    metrics["reference_download_seconds"] = time.perf_counter() - download_started
                queue.update(
                    task_id,
                    status="failed",
                    stage="engine_failed",
                    runtime_metrics=metrics,
                    error=code,
                    worker_settled=1,
                )
                fatal_capacity = isinstance(code, str) and code.endswith("_out_of_memory")
                if fatal_capacity or not isinstance(
                    exc, (RuntimeErrorCode, ValueError, QueueError, OSError)
                ):
                    state["ready"] = False
                    state["error"] = code
                    return
            finally:
                with lock:
                    state["active_task_id"] = None
    except Exception as exc:
        state["error"] = getattr(exc, "code", type(exc).__name__)
    finally:
        state["ready"] = False
        if not stop.is_set() and state["error"] is None:
            state["error"] = "worker_stopped"
