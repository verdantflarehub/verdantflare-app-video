"""MCP status and explicit tombstones for the retired runtime resource APIs."""
from __future__ import annotations

import asyncio
import os
import threading
import time
from datetime import UTC, datetime

import httpx
from starlette.responses import JSONResponse
from starlette.routing import Route

from . import __version__

VERSION = __version__

def stamp(epoch):
    return datetime.fromtimestamp(epoch, UTC).isoformat() if epoch is not None else None

class Resources:
    def __init__(self, clock=time.time):
        self.clock = clock
        self.lock = threading.RLock()
        self.started_at = clock()
        self.protocol = {"state": "unavailable", "sampled_at": None, "status": "unknown"}
        self.requests = {"count": 0, "errors": 0, "last_at": None, "last_error_at": None}

    def record_request(self, failed):
        with self.lock:
            self.requests["count"] += 1
            self.requests["errors"] += int(failed)
            self.requests["last_at"] = stamp(self.clock())
            if failed:
                self.requests["last_error_at"] = self.requests["last_at"]

    async def poll_protocol(self):
        while True:
            status = "unavailable"
            token = os.environ.get("VIDEO_MCP_BEARER_TOKEN", "").strip()
            if token:
                try:
                    async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
                        response = await client.post("http://127.0.0.1:8000/mcp",
                            headers={"Authorization": f"Bearer {token}", "Accept": "application/json, text/event-stream"},
                            json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                                "protocolVersion": "2025-03-26", "capabilities": {},
                                "clientInfo": {"name": "dashboard-health", "version": VERSION}}})
                        data = response.json()
                        if response.is_success and data.get("result", {}).get("protocolVersion"):
                            status = "ready"
                except (httpx.HTTPError, ValueError, TypeError):
                    pass
            with self.lock:
                self.protocol = {"status": status, "state": "fresh", "sampled_at": stamp(self.clock())}
            await asyncio.sleep(10)

    def mcp_status(self):
        with self.lock:
            protocol = dict(self.protocol)
            if protocol["sampled_at"] and self.clock() - datetime.fromisoformat(protocol["sampled_at"]).timestamp() > 30:
                protocol["state"] = "stale"
            return {"state": "fresh", "sampled_at": stamp(self.clock()), "service": "MCP", "version": VERSION,
                    "started_at": stamp(self.started_at), "protocol": protocol, "requests": dict(self.requests)}

    async def endpoint(self, request):
        if request.url.path == "/api/mcp/status":
            return JSONResponse(self.mcp_status(), headers={"Cache-Control": "no-store"})
        return JSONResponse({"error": "runtime_resources_retired"}, status_code=410,
                            headers={"Cache-Control": "no-store"})

    def routes(self):
        return [Route(path, self.endpoint) for path in ("/api/mcp/status", "/api/models", "/api/models/{model:str}/instances",
                "/api/models/{model:str}/instances/{instance_id:str}",
                "/api/models/{model:str}/instances/{instance_id:str}/gpus/{gpu_id:str}")]
