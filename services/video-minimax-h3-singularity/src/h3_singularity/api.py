"""Authenticated internal HTTP contract consumed by Video MCP."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import threading
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from .engine import Engine, RuntimeErrorCode
from .queue import Queue, QueueError


MAX_REQUEST_BYTES = 128 * 1024
MAX_ARTIFACT_BYTES = 2 * 1024**3


def _download_references(task_root: Path, conditions: list[dict]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {"images": [], "videos": [], "audios": []}
    client = httpx.Client(timeout=httpx.Timeout(connect=10, read=600, write=30, pool=10), follow_redirects=False)
    try:
        for index, item in enumerate(conditions):
            if not isinstance(item, dict) or item.get("role") != "reference":
                raise RuntimeErrorCode("invalid_reference")
            kind = {"image": "images", "video": "videos", "audio": "audios"}.get(item.get("type"))
            uri = item.get("uri")
            if kind is None or not isinstance(uri, str) or not uri.startswith("http://"):
                raise RuntimeErrorCode("invalid_reference")
            target = task_root / f"reference_{index:02d}_{kind[:-1]}.bin"
            digest = hashlib.sha256()
            size = 0
            with client.stream("GET", uri, headers={"Accept": "image/*,video/*,audio/*,application/octet-stream"}) as response:
                response.raise_for_status()
                with target.open("wb") as stream:
                    for block in response.iter_bytes(1024 * 1024):
                        size += len(block)
                        if size > MAX_ARTIFACT_BYTES:
                            raise RuntimeErrorCode("reference_too_large")
                        digest.update(block)
                        stream.write(block)
            if item.get("size") is not None and int(item["size"]) != size:
                raise RuntimeErrorCode("reference_size_mismatch")
            if item.get("sha256") is not None and str(item["sha256"]).lower() != digest.hexdigest():
                raise RuntimeErrorCode("reference_hash_mismatch")
            result[kind].append(target)
    except httpx.HTTPError as exc:
        raise RuntimeErrorCode("reference_download_failed") from exc
    finally:
        client.close()
    return result


def create_app(queue: Queue, engine_factory, token: str):
    if not token:
        raise ValueError("SINGULARITY_RUNTIME_TOKEN is required")
    state: dict[str, object] = {"ready": False, "engine": None, "error": None}
    stop = threading.Event()

    def worker():
        try:
            state["engine"] = engine_factory()
            state["ready"] = True
        except Exception as exc:
            state["error"] = getattr(exc, "code", type(exc).__name__)
            return
        while not stop.wait(0.2):
            task = queue.take()
            if task is None:
                continue
            task_id = task["id"]
            directory = queue.root / task_id
            directory.mkdir(exist_ok=False)
            download_started = time.perf_counter()
            download_seconds = None
            try:
                request = task["request"] if isinstance(task.get("request"), dict) else json.loads(task["request"])
                queue.update(task_id, stage="downloading")
                files = _download_references(directory, request.get("conditions", []))
                download_seconds = time.perf_counter() - download_started
                queue.update(task_id, stage="warming")
                result = state["engine"].generate(request, files, directory / "output.mp4", lambda stage: queue.update(task_id, stage=stage))
                metrics = result.pop("runtime_metrics", {})
                metrics["reference_download_seconds"] = download_seconds
                queue.update(task_id, status="completed", stage="completed", execution_instance_id=result.pop("execution_instance_id"), runtime_metrics=metrics, result=result)
            except Exception as exc:
                code = getattr(exc, "code", type(exc).__name__)
                metrics = dict(getattr(exc, "runtime_metrics", None) or {})
                if download_seconds is None:
                    metrics["reference_download_seconds"] = time.perf_counter() - download_started
                queue.update(task_id, status="failed", stage="engine_failed", runtime_metrics=metrics, error=code)
                # A CUDA exception can leave the context unusable. The pod is
                # restarted by Kubernetes rather than attempting a second GPU job.
                fatal_capacity = isinstance(code, str) and code.endswith("_out_of_memory")
                if fatal_capacity or not isinstance(exc, (RuntimeErrorCode, ValueError, QueueError)):
                    state["ready"] = False
                    state["error"] = code
                    return

    @asynccontextmanager
    async def lifespan(app):
        thread = threading.Thread(target=worker, name="singularity-worker", daemon=True)
        thread.start()
        yield
        stop.set()

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
        return {"live": True, "ready": bool(state["ready"]), "error": state["error"]}

    @app.get("/health")
    def health():
        if not state["ready"]:
            raise HTTPException(503, detail={"code": state["error"] or "not_ready"})
        return state["engine"].health()

    @app.post("/v1/videos")
    async def submit(request: Request):
        try:
            length = int(request.headers.get("content-length", "0"))
        except ValueError:
            raise HTTPException(400, "invalid_content_length") from None
        if length > MAX_REQUEST_BYTES:
            raise HTTPException(413, "body_too_large")
        try:
            payload = json.loads(await request.body())
            required = {"idempotency_key", "model", "task", "prompt", "seconds", "conditions", "target"}
            if not isinstance(payload, dict) or not required <= payload.keys():
                raise ValueError("invalid_fields")
            if payload["model"] != "MiniMaxAI/MiniMax-H3" or payload["task"] != "ref2va":
                raise ValueError("unsupported_model_or_task")
            if not isinstance(payload["conditions"], list) or not payload["conditions"]:
                raise ValueError("references_required")
            if int(payload.get("num_inference_steps", 4)) != 4:
                raise ValueError("singularity_requires_4_nfe")
            if not state["ready"]:
                raise HTTPException(503, "not_ready")
            accepted = queue.submit(payload)
            return {"id": accepted["id"], "object": "video", "status": accepted["status"]}
        except QueueError as exc:
            code = str(exc)
            raise HTTPException(409 if "conflict" in code else 429 if code == "queue_full" else 422, code) from None
        except RuntimeErrorCode as exc:
            raise HTTPException(422, exc.code) from None
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/v1/videos/{task_id}")
    def status(task_id: str):
        try:
            return queue.get(task_id)
        except (KeyError, QueueError):
            raise HTTPException(404, "not_found") from None

    @app.get("/v1/videos/{task_id}/content")
    def content(task_id: str):
        try:
            result = queue.get(task_id)
        except (KeyError, QueueError):
            raise HTTPException(404, "not_found") from None
        if result["status"] != "completed":
            raise HTTPException(409, "not_completed")
        path = queue.root / task_id / "output.mp4"
        if not path.is_file():
            raise HTTPException(404, "content_missing")
        return FileResponse(path, media_type="video/mp4", filename=f"{task_id}.mp4")

    return app


def main():
    import uvicorn
    queue = Queue(os.environ.get("SINGULARITY_TASK_ROOT", "/data/projects/singularity/tasks"))
    app = create_app(queue, Engine, os.environ.get("SINGULARITY_RUNTIME_TOKEN", ""))
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), workers=1)


if __name__ == "__main__":
    main()
