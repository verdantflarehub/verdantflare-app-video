"""Authenticated internal HTTP contract for Video LipSync (Zero-CUDA Supervisor)."""

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

from .contract import check_weights_health, validate_lipsync_request
from .errors import RuntimeErrorCode
from .fence import (
    EpochLeaseFenceGuard,
    FencePersistenceError,
    StaleFenceError,
    cleanup_task_shm,
    enable_child_subreaper,
    read_runtime_identity,
    reap_adopted_children,
)
from .lipsync import probe_and_validate_lipsync_inputs
from .queue import Queue, QueueError


MAX_REQUEST_BYTES = 128 * 1024


def _spawn_worker_process(cmd: list[str], env: dict[str, str]) -> subprocess.Popen:
    if os.name == "nt":
        return subprocess.Popen(cmd, env=env, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    return subprocess.Popen(cmd, env=env, start_new_session=True)


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
    reap_adopted_children()
    return bool(reaped_leader and not _process_group_alive(pgid))


async def _read_bounded_json(request: Request) -> dict:
    try:
        length = int(request.headers.get("content-length", "0"))
    except ValueError:
        raise HTTPException(400, {"code": "invalid_content_length"}) from None
    if length > MAX_REQUEST_BYTES:
        raise HTTPException(413, {"code": "body_too_large"})
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_REQUEST_BYTES:
            raise HTTPException(413, {"code": "body_too_large"})
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeErrorCode("invalid_json") from None
    if not isinstance(payload, dict):
        raise ValueError("json_object_required")
    return payload


def create_app(
    queue: Queue,
    token: str = "",
    *,
    worker_cmd_builder=None,
    health_checker=None,
    input_validator=None,
    kill_grace_seconds: float = 5.0,
    kill_wait_seconds: float = 2.0,
    cancel_endpoint_wait_seconds: float | None = None,
):
    if not token:
        raise ValueError("LIPSYNC_RUNTIME_TOKEN is required")

    fence_guard = EpochLeaseFenceGuard(state_path=queue.root / "gpu_guard_state.json")
    identity = read_runtime_identity()
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
        info = dict(checker(gpu_worker_active=gpu_active, active_task_ids=active_ids))
        if reap_stalled:
            info["ready"] = False
            info["reap_stalled"] = True
        info["gpu_epoch"] = fence_guard.current_epoch
        info["runtime_identity"] = identity
        return info

    def is_ready_for_new_submission() -> bool:
        try:
            return bool(current_health().get("ready"))
        except Exception:
            return False

    def supervisor_loop():
        try:
            while not stop.wait(0.1):
                task = queue.take()
                if task is None:
                    continue
                task_id = task["id"]
                stale_cancelled = False
                fence_corrupted = False
                with lock:
                    if stop.is_set() or queue.is_cancel_requested(task_id):
                        proc = None
                    else:
                        try:
                            epoch = fence_guard.allocate_worker_epoch(task_id)
                        except StaleFenceError:
                            proc = None
                            stale_cancelled = True
                        except FencePersistenceError:
                            proc = None
                            fence_corrupted = True
                        else:
                            cmd = (
                                worker_cmd_builder(task_id, queue.root)
                                if worker_cmd_builder is not None
                                else [sys.executable, "-m", "video_lipsync.worker", task_id]
                            )
                            env = os.environ.copy()
                            env.pop("CUDA_VISIBLE_DEVICES", None)
                            env["LIPSYNC_TASK_ROOT"] = str(queue.root)
                            env["VF_GPU_EPOCH"] = str(epoch)
                            env["VF_SHM_PREFIX"] = f"/dev/shm/vf_{task_id}_"
                            cancel_requested.clear()
                            cancel_completed.clear()
                            proc = _spawn_worker_process(cmd, env)
                            state["active_task_id"] = task_id
                            state["active_proc"] = proc
                            state["reap_stalled_task_id"] = None

                if proc is None:
                    try:
                        fence_guard.seal_task_epoch(task_id)
                    except FencePersistenceError:
                        pass
                    cleanup_task_shm(task_id)
                    if stale_cancelled or queue.is_cancel_requested(task_id):
                        queue.mark_settled(task_id, status="cancelled", stage="cancelled")
                    elif fence_corrupted:
                        queue.mark_settled(
                            task_id, status="failed", stage="fence_state_corrupted", error="fence_state_corrupted"
                        )
                    else:
                        queue.mark_settled(task_id, status="failed", stage="interrupted", error="interrupted")
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
                            queue.update(task_id, status="cancelled", stage="cancelled", worker_settled=False)
                        group_reaped = _terminate_and_reap_process_group(
                            proc, kill_grace_seconds, kill_wait_seconds=kill_wait_seconds
                        )
                        break
                    time.sleep(0.05)
                else:
                    proc.wait()
                    group_reaped = (
                        True
                        if (os.name == "nt" or not _process_group_alive(proc.pid))
                        else _terminate_and_reap_process_group(
                            proc, kill_grace_seconds, kill_wait_seconds=kill_wait_seconds
                        )
                    )

                if not was_cancelled and queue.is_cancel_requested(task_id):
                    was_cancelled = True

                if not group_reaped:
                    with lock:
                        state["reap_stalled_task_id"] = task_id
                    break

                rc = proc.wait()
                reap_adopted_children()
                cleanup_task_shm(task_id)
                with lock:
                    state["active_proc"] = None
                    state["active_task_id"] = None
                    state["cancelling_task_id"] = None
                    state["reap_stalled_task_id"] = None
                    try:
                        fence_guard.seal_task_epoch(task_id)
                    except FencePersistenceError:
                        pass

                if was_cancelled:
                    queue.mark_settled(task_id, status="cancelled", stage="cancelled")
                elif stop.is_set():
                    current = queue.get(task_id)
                    if current["status"] == "running":
                        queue.mark_settled(task_id, status="failed", stage="interrupted", error="interrupted")
                    else:
                        queue.mark_settled(task_id)
                else:
                    current = queue.get(task_id)
                    if current["status"] == "running":
                        err_code = f"worker_exit_{rc}" if rc != 0 else "worker_incomplete"
                        queue.mark_settled(task_id, status="failed", stage="engine_failed", error=err_code)
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
        enable_child_subreaper()
        thread = threading.Thread(target=supervisor_loop, name="video-lipsync-supervisor", daemon=True)
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
            allowed = [f"Bearer {token}"]
            if request.url.path in {"/internal/v1/gpu-residency/revoke-lease", "/internal/v1/gpu-guard/reclaim"}:
                sched_tok = os.environ.get("STATION_GPU_SCHEDULER_TOKEN", "").strip()
                if sched_tok:
                    allowed.append(f"Bearer {sched_tok}")
            if not any(hmac.compare_digest(supplied.encode(), expected.encode()) for expected in allowed):
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
            info = current_health()
            return JSONResponse(info, status_code=200 if info["ready"] else 503)
        except RuntimeErrorCode as exc:
            raise HTTPException(503, detail={"code": exc.code}) from None

    def _enrich_task(record: dict) -> dict:
        enriched = dict(record)
        tid = str(enriched.get("video_task_id") or enriched.get("id") or "")
        try:
            ep = fence_guard.task_epoch(tid)
            if ep is not None:
                enriched["gpu_epoch"] = ep
            with lock:
                active_count = 1 if (state["active_task_id"] is not None or state["reap_stalled_task_id"] is not None) else 0
                is_terminal = str(enriched.get("status") or "") in {"completed", "succeeded", "failed", "cancelled"}
                proof = fence_guard.build_epoch_reclaim_proof(
                    task_id=tid,
                    active_tasks=active_count,
                    settled=bool(is_terminal and enriched.get("worker_settled", False)),
                    identity=identity,
                )
        except FencePersistenceError as exc:
            raise HTTPException(503, {"code": str(exc)}) from None
        if proof is not None:
            enriched["reclaim_proof"] = proof
        return enriched

    @app.post("/v1/lipsync", status_code=202)
    async def submit_lipsync(request: Request):
        try:
            raw_payload = await _read_bounded_json(request)
            normalized = validate_lipsync_request(raw_payload)
            encoded = json.dumps(normalized, sort_keys=True, ensure_ascii=False, allow_nan=False)
            digest = hashlib.sha256(encoded.encode()).hexdigest()
            try:
                _, replay_key = fence_guard.extract_and_validate(
                    request.headers,
                    key_id=normalized["idempotency_key"],
                    request_digest=digest,
                    identity=identity,
                )
            except FencePersistenceError as exc:
                raise HTTPException(503, {"code": str(exc)}) from None
            except PermissionError as exc:
                raise HTTPException(403, {"code": str(exc)}) from None
            except (StaleFenceError, ValueError) as exc:
                raise HTTPException(409, {"code": str(exc)}) from None

            if replay_key is not None:
                replayed = queue.find_idempotency(replay_key, digest)
                if replayed is not None and replay_key != normalized["idempotency_key"]:
                    return _enrich_task(replayed)
                if replayed is None and fence_guard.is_key_revoked_or_sealed(replay_key):
                    raise HTTPException(409, {"code": "stale_or_revoked_fencing_token"})

            try:
                existing = queue.find_idempotency(normalized["idempotency_key"], digest)
                if existing is not None:
                    return _enrich_task(existing)
                if not is_ready_for_new_submission():
                    raise HTTPException(503, {"code": "not_ready"})
                artifact_root = Path(os.environ.get("LIPSYNC_ARTIFACT_ROOT", "/data/projects/video"))
                validator = input_validator or probe_and_validate_lipsync_inputs
                await asyncio.to_thread(validator, normalized, artifact_root)
                accepted = queue.submit(normalized)
                fence_guard.bind_task_id(
                    normalized["idempotency_key"],
                    accepted.get("video_task_id") or normalized["idempotency_key"],
                )
            except BaseException:
                if replay_key is None:
                    try:
                        fence_guard.seal_task_epoch(normalized["idempotency_key"])
                    except FencePersistenceError:
                        pass
                raise
            return _enrich_task(accepted)
        except FencePersistenceError as exc:
            raise HTTPException(503, {"code": str(exc)}) from None
        except HTTPException:
            raise
        except QueueError as exc:
            code = str(exc)
            status_code = 409 if "conflict" in code else 429 if code == "queue_full" else 422
            raise HTTPException(status_code, {"code": code}) from None
        except RuntimeErrorCode as exc:
            status_code = 404 if exc.code in {"source_not_found", "audio_not_found"} else 422
            raise HTTPException(status_code, {"code": exc.code}) from None
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(422, {"code": str(exc)}) from None

    @app.get("/v1/lipsync/{video_task_id}")
    def lipsync_status(video_task_id: str):
        try:
            return _enrich_task(queue.get(video_task_id))
        except (KeyError, QueueError):
            raise HTTPException(404, {"code": "task_not_found"}) from None

    @app.get("/v1/lipsync/{video_task_id}/content")
    def lipsync_content(video_task_id: str):
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

    @app.get("/v1/lipsync/{video_task_id}/preview")
    def lipsync_preview(video_task_id: str):
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

    @app.delete("/v1/lipsync/{video_task_id}")
    def cancel_lipsync(video_task_id: str):
        try:
            outcome, snapshot = queue.cancel_queued_or_check(video_task_id)
        except QueueError as exc:
            code = str(exc)
            if code == "task_already_terminal":
                raise HTTPException(409, {"code": "task_already_terminal"}) from None
            raise HTTPException(404, {"code": "task_not_found"}) from None
        except KeyError:
            raise HTTPException(404, {"code": "task_not_found"}) from None

        if outcome == "cancelled_queued":
            fence_guard.seal_task_epoch(video_task_id)
            return _enrich_task(snapshot)
        if outcome == "already_cancelled" and snapshot["worker_settled"]:
            return _enrich_task(snapshot)

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

        return _enrich_task(queue.get(video_task_id))

    @app.post("/internal/v1/gpu-residency/revoke-lease")
    async def revoke_lease(request: Request):
        payload = await _read_bounded_json(request)
        lease_id = str(payload.get("lease_id", "")).strip()
        try:
            fencing_token = int(payload.get("fencing_token", 0))
        except (TypeError, ValueError):
            raise HTTPException(422, {"code": "invalid_fencing_token"}) from None
        try:
            affected = fence_guard.revoke_lease(lease_id, fencing_token)
        except FencePersistenceError as exc:
            raise HTTPException(503, {"code": str(exc)}) from None
        for tid in affected:
            if tid.startswith("video_task_"):
                try:
                    cancel_lipsync(tid)
                except HTTPException:
                    pass
        try:
            with lock:
                active_count = 1 if (state["active_task_id"] is not None or state["reap_stalled_task_id"] is not None) else 0
                proof = fence_guard.reclaim_and_seal(
                    lease_id=lease_id,
                    fencing_token=fencing_token,
                    active_tasks=active_count,
                    settled=(active_count == 0),
                    identity=identity,
                ) or {}
        except FencePersistenceError as exc:
            raise HTTPException(503, {"code": str(exc)}) from None
        is_settled = bool(proof.get("reclaimed"))
        return {**proof, "revoked": True, "settled": is_settled}

    @app.post("/internal/v1/gpu-guard/reclaim")
    async def reclaim_lease(request: Request):
        payload = await _read_bounded_json(request)
        lease_id = str(payload.get("lease_id", "")).strip()
        try:
            fencing_token = int(payload.get("fencing_token", 0))
        except (TypeError, ValueError):
            raise HTTPException(422, {"code": "invalid_fencing_token"}) from None
        try:
            with lock:
                active_count = 1 if (state["active_task_id"] is not None or state["reap_stalled_task_id"] is not None) else 0
                proof = fence_guard.reclaim_and_seal(
                    lease_id=lease_id,
                    fencing_token=fencing_token,
                    active_tasks=active_count,
                    settled=True,
                    identity=identity,
                )
        except FencePersistenceError as exc:
            raise HTTPException(503, {"code": str(exc)}) from None
        if proof is None:
            raise HTTPException(400, {"code": "missing_lease_or_token"})
        if not proof.get("reclaimed"):
            raise HTTPException(
                409,
                {
                    "code": "worker_not_settled",
                    "settled": False,
                    "active_tasks": int(proof.get("active_tasks", 1)),
                    "reclaim_proof": proof,
                },
            )
        return {
            "ok": True,
            "settled": True,
            "active_tasks": 0,
            "gpu_epoch": int(proof.get("sealed_gpu_epoch", 0)),
            "reclaim_proof": proof,
        }

    return app


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import uvicorn

    queue = Queue(os.environ.get("LIPSYNC_TASK_ROOT", "/data/projects/lipsync/tasks"))
    app = create_app(queue, os.environ.get("LIPSYNC_RUNTIME_TOKEN", ""))
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), workers=1)


if __name__ == "__main__":
    main()
