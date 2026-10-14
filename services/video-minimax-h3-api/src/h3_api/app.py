"""Bounded, authenticated proxy for the existing Singularity queue on 5090."""

from __future__ import annotations

from contextlib import asynccontextmanager
import hmac
import json
import os
import re
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

MAX_REQUEST_BYTES = 128 * 1024
TASK_ID = re.compile(r"singularity_[0-9a-f]{32}\Z")
IDEMPOTENCY_KEY = re.compile(r"video_task_[0-9a-f]{32}\Z")


def _upstream_url(value: str) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password
            or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
        raise ValueError("H3_API_UPSTREAM_URL must be an internal HTTP origin")
    return value.rstrip("/")


def create_app(upstream_url: str, inbound_token: str, runtime_token: str,
               client: httpx.AsyncClient | None = None) -> FastAPI:
    if not inbound_token or not runtime_token:
        raise ValueError("H3_API_TOKEN and SINGULARITY_RUNTIME_TOKEN are required")
    upstream = _upstream_url(upstream_url)
    owns_client = client is None
    backend = client or httpx.AsyncClient(timeout=httpx.Timeout(connect=5, read=30, write=30, pool=5),
                                          follow_redirects=False)
    upstream_headers = {"Authorization": f"Bearer {runtime_token}"}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            if owns_client:
                await backend.aclose()

    app = FastAPI(title="VerdantFlare MiniMax H3 API", version="0.5.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if request.url.path not in {"/health", "/live"}:
            supplied = request.headers.get("Authorization", "")
            if not hmac.compare_digest(supplied.encode(), f"Bearer {inbound_token}".encode()):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)

    async def proxy_json(method: str, path: str, body: bytes | None = None):
        try:
            response = await backend.request(method, upstream + path, content=body,
                                             headers={**upstream_headers, "Content-Type": "application/json"}
                                             if body is not None else upstream_headers)
        except httpx.RequestError:
            raise HTTPException(503, "runtime_unavailable") from None
        try:
            payload = response.json()
        except (ValueError, UnicodeDecodeError):
            raise HTTPException(502, "invalid_runtime_response") from None
        if not isinstance(payload, dict):
            raise HTTPException(502, "invalid_runtime_response")
        return JSONResponse(payload, status_code=response.status_code)

    @app.get("/live")
    async def live():
        return {"live": True}

    @app.get("/health")
    async def health():
        try:
            response = await backend.get(upstream + "/health", headers=upstream_headers, timeout=5)
            healthy = response.status_code == 200 and response.json().get("ready") is True
        except (httpx.RequestError, ValueError, AttributeError):
            healthy = False
        return JSONResponse({"ready": healthy, "runtime": "h3-singularity"},
                            status_code=200 if healthy else 503)

    @app.get("/capabilities")
    async def capabilities():
        return {"runtime": "h3-singularity", "model": "MiniMaxAI/MiniMax-H3", "task": "ref2va",
                "reference_count": {"minimum": 1, "maximum": 12},
                "duration_seconds": {"minimum": 4, "maximum": 15},
                "short_edge": 768, "num_inference_steps": 4,
                "unsupported": ["text_to_video", "first_last_frame", "2K", "external_media_url", "v2_video_generation"]}

    @app.post("/v1/videos")
    async def submit(request: Request):
        try:
            length = int(request.headers.get("content-length", "0"))
        except ValueError:
            raise HTTPException(400, "invalid_content_length") from None
        if length > MAX_REQUEST_BYTES:
            raise HTTPException(413, "body_too_large")
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > MAX_REQUEST_BYTES:
                raise HTTPException(413, "body_too_large")
        try:
            payload = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            raise HTTPException(422, "invalid_json") from None
        if not isinstance(payload, dict):
            raise HTTPException(422, "invalid_request")
        return await proxy_json("POST", "/v1/videos", bytes(data))

    @app.get("/v1/videos/by-idempotency/{key}")
    async def by_idempotency(key: str):
        if not IDEMPOTENCY_KEY.fullmatch(key):
            raise HTTPException(422, "invalid_idempotency_key")
        return await proxy_json("GET", f"/v1/videos/by-idempotency/{key}")

    @app.get("/v1/videos/{task_id}")
    async def status(task_id: str):
        if not TASK_ID.fullmatch(task_id):
            raise HTTPException(422, "invalid_task_id")
        return await proxy_json("GET", f"/v1/videos/{task_id}")

    @app.get("/v1/videos/{task_id}/content")
    async def content(task_id: str):
        if not TASK_ID.fullmatch(task_id):
            raise HTTPException(422, "invalid_task_id")
        try:
            outbound = backend.build_request("GET", upstream + f"/v1/videos/{task_id}/content",
                                             headers=upstream_headers)
            response = await backend.send(outbound, stream=True)
        except httpx.RequestError:
            raise HTTPException(503, "runtime_unavailable") from None
        if response.status_code != 200:
            await response.aclose()
            raise HTTPException(response.status_code if 400 <= response.status_code < 600 else 502,
                                "content_unavailable")
        headers = {"Content-Disposition": f'attachment; filename="{task_id}.mp4"'}
        if "content-length" in response.headers:
            headers["Content-Length"] = response.headers["content-length"]
        return StreamingResponse(response.aiter_bytes(), media_type="video/mp4", headers=headers,
                                 background=BackgroundTask(response.aclose))

    return app


def main():
    import uvicorn

    app = create_app(os.environ.get("H3_API_UPSTREAM_URL", "http://video-minimax-h3-singularity:8000"),
                     os.environ.get("H3_API_TOKEN", ""), os.environ.get("SINGULARITY_RUNTIME_TOKEN", ""))
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), workers=1)


if __name__ == "__main__":
    main()
