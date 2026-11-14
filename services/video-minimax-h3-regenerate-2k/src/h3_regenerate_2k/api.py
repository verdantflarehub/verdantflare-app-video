"""Authenticated internal HTTP contract for Open-H3-Regenerate-2K-IC (Zero-CUDA Supervisor)."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hashlib
import hmac
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from .contract import check_weights_health, validate_regenerate_request
from .errors import RuntimeErrorCode
from .queue import Queue, QueueError
from .regenerate import probe_and_validate_source_video, resolve_source_artifact_path


MAX_REQUEST_BYTES = 128 * 1024


def _spawn_worker_process(cmd: list[str], env: dict[str, str]) -> subprocess.Popen:
    if os.name == "nt":
        return subprocess.Popen(
            cmd,
            env=env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    return subprocess.Popen(
        cmd,
        env=env,
        start_new_session=True,
    )


def _process_group_alive(pgid: int) -> bool:
    if os.name == "nt" or pgid <= 0:
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _terminate_and_reap_process_group(
    proc: subprocess.Popen,
    grace_seconds: float,
    *,
    kill_wait_seconds: float = 2.0,
) -> bool:
    if os.name == "nt":
        if proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass
            try:
                proc.wait(timeout=max(0.05, grace_seconds))
                return True
            except subprocess.TimeoutExpired:
                pass
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.wait(timeout=max(0.05, kill_wait_seconds))
            return True
        except subprocess.TimeoutExpired:
            return False

    pgid = proc.pid
    reaped_leader = proc.poll() is not None
    if reaped_leader:
        proc.wait()
    if reaped_leader and not _process_group_alive(pgid):
        return True

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + max(0.05, grace_seconds)
    while time.monotonic() < deadline:
        if not reaped_leader and proc.poll() is not None:
            proc.wait()
            reaped_leader = True
        if reaped_leader and not _process_group_alive(pgid):
            return True
        time.sleep(0.05)

    if not reaped_leader or _process_group_alive(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    kill_deadline = time.monotonic() + max(0.05, kill_wait_seconds)
    while (not reaped_leader or _process_group_alive(pgid)) and time.monotonic() < kill_deadline:
        if not reaped_leader and proc.poll() is not None:
            proc.wait()
            reaped_leader = True
        if reaped_leader and not _process_group_alive(pgid):
            return True
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        time.sleep(0.02)

    if not reaped_leader and proc.poll() is not None:
        proc.wait()
        reaped_leader = True
    return bool(reaped_leader and not _process_group_alive(pgid))


async def _read_bounded_json(request: Request) -> dict:
    try:
        length = int(request.headers.get("content-length", "0"))
    except ValueError:
        raise HTTPException(400, "invalid_content_length") from None
    if length > MAX_REQUEST_BYTES:
        raise HTTPException(413, "body_too_large")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_REQUEST_BYTES:
            raise HTTPException(413, "body_too_large")
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("json_object_required")
    return payload


def create_app(
    queue: Queue,
    token: str = "",
    *,
    worker_cmd_builder=None,
    health_checker=None,
    kill_grace_seconds: float = 5.0,
    kill_wait_seconds: float = 2.0,
    cancel_endpoint_wait_seconds: float | None = None,
):
    if not token:
        raise ValueError("REGENERATE_2K_RUNTIME_TOKEN is required")

    lock = threading.RLock()
    stop = threading.Event()
    cancel_requested = threading.Event()
    cancel_completed = threading.Event()
    thread: threading.Thread | None = None

    state: dict[str, object] = {
        "error": None,
        "active_task_id": None,
        "active_proc": None,
        "cancelling_task_id": None,
        "reap_stalled_task_id": None,
    }

    def current_health() -> dict:
        with lock:
            active_id = state["active_task_id"]
            active_ids = [active_id] if isinstance(active_id, str) else []
            gpu_active = bool(active_ids)
            reap_stalled = state["reap_stalled_task_id"] is not None
        checker = health_checker or check_weights_health
        info = dict(
            checker(
                gpu_worker_active=gpu_active,
                active_task_ids=active_ids,
            )
        )
        if reap_stalled:
            info["ready"] = False
            info["reap_stalled"] = True
        return info

    def is_ready_for_new_submission() -> bool:
        try:
            info = current_health()
            return bool(info.get("ready"))
        except Exception:
            return False

    def supervisor_loop():
        try:
            while not stop.wait(0.1):
                task = queue.take()
                if task is None:
                    continue
                task_id = task["id"]
                with lock:
                    if stop.is_set() or queue.is_cancel_requested(task_id):
                        proc = None
                    else:
                        cmd = (
                            worker_cmd_builder(task_id, queue.root)
                            if worker_cmd_builder is not None
                            else [sys.executable, "-m", "h3_regenerate_2k.worker", task_id]
                        )
                        env = os.environ.copy()
                        env["REGENERATE_2K_TASK_ROOT"] = str(queue.root)
                        cancel_requested.clear()
                        cancel_completed.clear()
                        proc = _spawn_worker_process(cmd, env)
                        state["active_task_id"] = task_id
                        state["active_proc"] = proc
                        state["reap_stalled_task_id"] = None

                if proc is None:
                    if queue.is_cancel_requested(task_id):
                        queue.mark_settled(task_id, status="cancelled", stage="cancelled")
                    else:
                        queue.mark_settled(
                            task_id,
                            status="failed",
                            stage="interrupted",
                            error="interrupted",
                        )
                    cancel_completed.set()
                    continue

                was_cancelled = False
                group_reaped = False
                while proc.poll() is None:
                    cancelled_now = queue.is_cancel_requested(task_id) or (
                        cancel_requested.is_set() and state.get("cancelling_task_id") == task_id
                    )
                    if stop.is_set() or cancelled_now:
                        was_cancelled = cancelled_now
                        if was_cancelled:
                            queue.update(
                                task_id,
                                status="cancelled",
                                stage="cancelled",
                                worker_settled=False,
                            )
                        group_reaped = _terminate_and_reap_process_group(
                            proc,
                            kill_grace_seconds,
                            kill_wait_seconds=kill_wait_seconds,
                        )
                        break
                    time.sleep(0.05)
                else:
                    proc.wait()
                    if os.name == "nt" or not _process_group_alive(proc.pid):
                        group_reaped = True
                    else:
                        group_reaped = _terminate_and_reap_process_group(
                            proc,
                            kill_grace_seconds,
                            kill_wait_seconds=kill_wait_seconds,
                        )

                if not was_cancelled and queue.is_cancel_requested(task_id):
                    was_cancelled = True
                    if not group_reaped:
                        queue.update(
                            task_id,
                            status="cancelled",
                            stage="cancelled",
                            worker_settled=False,
                        )

                if not group_reaped:
                    with lock:
                        state["reap_stalled_task_id"] = task_id
                    retry_wait = min(0.25, max(0.05, kill_wait_seconds))
                    while not group_reaped and not stop.is_set():
                        if not was_cancelled and queue.is_cancel_requested(task_id):
                            was_cancelled = True
                            queue.update(
                                task_id,
                                status="cancelled",
                                stage="cancelled",
                                worker_settled=False,
                            )
                        group_reaped = _terminate_and_reap_process_group(
                            proc,
                            0.05,
                            kill_wait_seconds=retry_wait,
                        )
                    if not group_reaped and stop.is_set():
                        group_reaped = _terminate_and_reap_process_group(
                            proc,
                            0.05,
                            kill_wait_seconds=retry_wait,
                        )

                if not group_reaped:
                    if was_cancelled:
                        queue.update(
                            task_id,
                            status="cancelled",
                            stage="cancelled",
                            worker_settled=False,
                        )
                    break

                rc = proc.wait()
                with lock:
                    state["active_proc"] = None
                    state["active_task_id"] = None
                    state["cancelling_task_id"] = None
                    state["reap_stalled_task_id"] = None

                if was_cancelled:
                    queue.mark_settled(task_id, status="cancelled", stage="cancelled")
                elif stop.is_set():
                    current = queue.get(task_id)
                    if current["status"] == "running":
                        queue.mark_settled(
                            task_id,
                            status="failed",
                            stage="interrupted",
                            error="interrupted",
                        )
                    else:
                        queue.mark_settled(task_id)
                else:
                    current = queue.get(task_id)
                    if current["status"] == "running":
                        err_code = f"worker_exit_{rc}" if rc != 0 else "worker_incomplete"
                        queue.mark_settled(
                            task_id,
                            status="failed",
                            stage="engine_failed",
                            error=err_code,
                        )
                    else:
                        queue.mark_settled(task_id)
                cancel_completed.set()
        except Exception as exc:
            state["error"] = getattr(exc, "code", type(exc).__name__)
        finally:
            if not stop.is_set() and state["error"] is None:
                state["error"] = "worker_stopped"

    @asynccontextmanager
    async def lifespan(app):
        nonlocal thread
        thread = threading.Thread(target=supervisor_loop, name="regenerate-2k-supervisor", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            if thread is not None:
                thread.join(timeout=max(1.0, kill_grace_seconds + kill_wait_seconds + 1.0))

    app = FastAPI(lifespan=lifespan)

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if request.url.path not in {"/health", "/live"}:
            supplied = request.headers.get("Authorization", "")
            if not hmac.compare_digest(supplied.encode(), ("Bearer " + token).encode()):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)

    @app.get("/live")
    def live():
        alive = thread is not None and thread.is_alive() and state["error"] is None
        ready_flag = is_ready_for_new_submission() if alive else False
        return JSONResponse(
            {"live": alive, "ready": ready_flag, "error": state["error"]},
            status_code=200 if alive else 503,
        )

    @app.get("/health")
    def health():
        if thread is None or not thread.is_alive() or state["error"] is not None:
            raise HTTPException(503, detail={"code": state["error"] or "not_ready"})
        try:
            return current_health()
        except RuntimeErrorCode as exc:
            raise HTTPException(503, detail={"code": exc.code}) from None

    @app.post("/v1/regenerate-2k", status_code=202)
    async def submit_regenerate_2k(request: Request):
        try:
            raw_payload = await _read_bounded_json(request)
            normalized = validate_regenerate_request(raw_payload)
            encoded = json.dumps(normalized, sort_keys=True, ensure_ascii=False, allow_nan=False)
            digest = hashlib.sha256(encoded.encode()).hexdigest()
            existing = queue.find_idempotency(normalized["idempotency_key"], digest)
            if existing is not None:
                return existing
            if not is_ready_for_new_submission():
                raise HTTPException(503, {"code": "not_ready"})
            artifact_root = Path(os.environ.get("REGENERATE_2K_ARTIFACT_ROOT", "/source/video"))
            source_mp4 = resolve_source_artifact_path(normalized, artifact_root)
            await asyncio.to_thread(probe_and_validate_source_video, source_mp4, normalized["grid_mode"])
            return queue.submit(normalized)
        except HTTPException:
            raise
        except QueueError as exc:
            code = str(exc)
            status_code = 409 if "conflict" in code else 429 if code == "queue_full" else 422
            raise HTTPException(status_code, {"code": code}) from None
        except RuntimeErrorCode as exc:
            status_code = 404 if exc.code == "source_not_found" else 422
            raise HTTPException(status_code, {"code": exc.code}) from None
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(422, {"code": str(exc)}) from None

    @app.get("/v1/regenerate-2k/{video_task_id}")
    def regenerate_status(video_task_id: str):
        try:
            return queue.get(video_task_id)
        except (KeyError, QueueError):
            raise HTTPException(404, {"code": "task_not_found"}) from None

    @app.get("/v1/regenerate-2k/{video_task_id}/content")
    def regenerate_content(video_task_id: str):
        try:
            record = queue.get(video_task_id)
        except (KeyError, QueueError):
            raise HTTPException(404, {"code": "task_not_found"}) from None
        if record["status"] != "succeeded":
            raise HTTPException(409, {"code": "result_not_ready"})
        path = queue.root / video_task_id / "output.mp4"
        if not path.is_file():
            raise HTTPException(404, {"code": "content_missing"})
        return FileResponse(path, media_type="video/mp4", filename=f"{video_task_id}-content.mp4")

    @app.get("/v1/regenerate-2k/{video_task_id}/preview")
    def regenerate_preview(video_task_id: str):
        try:
            record = queue.get(video_task_id)
        except (KeyError, QueueError):
            raise HTTPException(404, {"code": "task_not_found"}) from None
        if record["status"] != "succeeded":
            raise HTTPException(409, {"code": "result_not_ready"})
        path = queue.root / video_task_id / "preview.mp4"
        if not path.is_file():
            raise HTTPException(404, {"code": "preview_missing"})
        return FileResponse(path, media_type="video/mp4", filename=f"{video_task_id}-preview.mp4")

    @app.delete("/v1/regenerate-2k/{video_task_id}")
    def cancel_regenerate(video_task_id: str):
        try:
            outcome, snapshot = queue.cancel_queued_or_check(video_task_id)
        except QueueError as exc:
            code = str(exc)
            if code == "task_already_terminal":
                raise HTTPException(409, {"code": "task_already_terminal"}) from None
            raise HTTPException(404, {"code": "task_not_found"}) from None
        except KeyError:
            raise HTTPException(404, {"code": "task_not_found"}) from None

        if outcome == "cancelled_queued" or (
            outcome == "already_cancelled" and snapshot["worker_settled"]
        ):
            return snapshot

        with lock:
            if state["active_task_id"] == video_task_id and outcome != "already_cancelled":
                state["cancelling_task_id"] = video_task_id
                cancel_completed.clear()
                cancel_requested.set()

        endpoint_wait = (
            cancel_endpoint_wait_seconds
            if cancel_endpoint_wait_seconds is not None
            else max(2.0, kill_grace_seconds + kill_wait_seconds + 3.0)
        )
        wait_deadline = time.monotonic() + max(0.05, endpoint_wait)
        while time.monotonic() < wait_deadline:
            snap = queue.get(video_task_id)
            if snap["worker_settled"]:
                break
            cancel_completed.wait(timeout=0.05)

        return queue.get(video_task_id)

    return app


def main():
    import uvicorn

    queue = Queue(os.environ.get("REGENERATE_2K_TASK_ROOT", "/data/projects/regenerate-2k/tasks"))
    app = create_app(queue, os.environ.get("REGENERATE_2K_RUNTIME_TOKEN", ""))
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), workers=1)


if __name__ == "__main__":
    main()
