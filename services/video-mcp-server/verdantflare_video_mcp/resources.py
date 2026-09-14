"""Read-only, namespace-scoped model inventory and UUID-scoped GPU telemetry."""
from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import re
import ssl
import threading
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

import httpx
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from starlette.routing import Route

VERSION = "0.6.2"
MODEL_TYPE = "minimax-h3-ref2va"
MODELS = {"h3": ("h3", "video-minimax-h3-api"), "h3-sol": ("h3-sol", "video-minimax-h3-sol-api"),
          "h3-vdn": ("h3-vdn", "video-minimax-h3-vdn")}
FIELDS = {"DCGM_FI_DEV_GPU_UTIL": ("utilization_percent", 100, 1),
          "DCGM_FI_DEV_FB_USED": ("memory_used_gib", 1048576, 1024),
          "DCGM_FI_DEV_FB_FREE": ("memory_free_gib", 1048576, 1024),
          "DCGM_FI_DEV_GPU_TEMP": ("temperature_celsius", 200, 1),
          "DCGM_FI_DEV_POWER_USAGE": ("power_watts", 5000, 1)}
UUID = re.compile(r"GPU-[0-9a-fA-F-]{36}$")
SAMPLE = re.compile(r'^(DCGM_\w+)\{(.*)\}\s+(\S+)(?:\s+(\S+))?$')
LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def stamp(epoch):
    return datetime.fromtimestamp(epoch, UTC).isoformat() if epoch is not None else None


def owner(obj, kind, uid):
    return any(r.get("kind") == kind and r.get("uid") == uid and r.get("controller") is True
               for r in obj.get("metadata", {}).get("ownerReferences", []))


def parse_metrics(text, allowed, now):
    """Reject missing, ambiguous, invalid and timestamp-stale samples, retaining real zero."""
    values, seen, conflicts = {}, set(), set()
    for line in text.splitlines():
        match = SAMPLE.match(line)
        if not match or match[1] not in FIELDS:
            continue
        labels = {k: json.loads('"' + v + '"') for k, v in LABEL.findall(match[2])}
        uid = labels.get("UUID")
        if uid not in allowed:
            continue
        field, maximum, divisor = FIELDS[match[1]]
        key = (uid, field)
        if key in seen:
            conflicts.add(key)
        seen.add(key)
        try:
            value = float(match[3])
            if not math.isfinite(value) or not 0 <= value <= maximum:
                continue
            if match[4] and not 0 <= now - float(match[4]) / 1000 <= 30:
                continue
        except ValueError:
            continue
        row = values.setdefault(uid, {"name": labels.get("modelName", "GPU")})
        row[field] = round(value / divisor, 3)
    for uid, field in conflicts:
        values.get(uid, {}).pop(field, None)
    for row in values.values():
        if "memory_used_gib" in row and "memory_free_gib" in row:
            row["memory_total_gib"] = round(row["memory_used_gib"] + row["memory_free_gib"], 3)
        row.pop("memory_free_gib", None)
    return values


class ClusterSource:
    def __init__(self):
        self.namespace = "verdantflare-video"
        self.account = Path("/var/run/secrets/kubernetes.io/serviceaccount")
        self.metrics_urls = [v.strip() for v in os.environ.get("VIDEO_DCGM_URLS", "").split(",") if v.strip()][:16]

    def inventory(self):
        # Read the projected token afresh so Kubernetes rotation is respected.
        token = (self.account / "token").read_text().strip()
        with httpx.Client(base_url="https://kubernetes.default.svc", verify=ssl.create_default_context(cafile=str(self.account / "ca.crt")),
                          timeout=5, trust_env=False, headers={"Authorization": f"Bearer {token}"}) as client:
            result = []
            for group, resource in (("apis/apps/v1", "deployments"), ("apis/apps/v1", "replicasets"), ("api/v1", "pods")):
                rows, cursor = [], ""
                for _ in range(20):
                    response = client.get(f"/{group}/namespaces/{self.namespace}/{resource}", params={"limit": 500, "continue": cursor})
                    response.raise_for_status()
                    data = response.json()
                    rows.extend(data["items"])
                    cursor = data.get("metadata", {}).get("continue", "")
                    if not cursor:
                        break
                if cursor:
                    raise ValueError("Inventory limit exceeded")
                result.append(rows)
            return result

    def metrics(self):
        texts = []
        with httpx.Client(timeout=4, trust_env=False, follow_redirects=False) as client:
            for url in self.metrics_urls:
                try:
                    with client.stream("GET", url) as response:
                        response.raise_for_status()
                        data = bytearray()
                        for chunk in response.iter_bytes():
                            data.extend(chunk)
                            if len(data) > 4 * 1024 * 1024:
                                raise ValueError("Metrics limit exceeded")
                        texts.append(data.decode())
                except (httpx.HTTPError, ValueError):
                    continue
        return "\n".join(texts)


class Resources:
    def __init__(self, source=None, clock=time.time):
        self.source = source or ClusterSource()
        self.clock = clock
        self.lock = threading.RLock()
        self.models = {}
        self.inventory_at = None
        self.inventory_failed = False
        self.history = {}
        self.latest = {}
        self.task_provider = lambda: []
        self.started_at = clock()
        self.protocol = {"state": "unavailable", "sampled_at": None, "status": "unknown"}
        self.requests = {"count": 0, "errors": 0, "last_at": None, "last_error_at": None}

    @staticmethod
    def route_connected(route):
        if route == "h3":
            return True
        try:
            routes = json.loads(os.environ.get("H3_RUNTIME_ROUTES", "{}"))
        except json.JSONDecodeError:
            return False
        config = routes.get(route)
        return bool(config and config.get("url") and ((not config.get("requires_token") and route != "h3-vdn") or os.environ.get("H3_VDN_RUNTIME_TOKEN" if route == "h3-vdn" else "H3_SOL_RUNTIME_TOKEN")))

    def record_request(self, failed):
        with self.lock:
            self.requests["count"] += 1
            self.requests["errors"] += int(failed)
            self.requests["last_at"] = stamp(self.clock())
            if failed:
                self.requests["last_error_at"] = self.requests["last_at"]

    def sync_inventory(self):
        now = self.clock()
        try:
            deployments, replicasets, pods = self.source.inventory()
            models = {}
            for model, (name, workload) in MODELS.items():
                deployment = next((d for d in deployments if d["metadata"]["name"] == workload), None)
                instances, desired = [], 0
                if deployment:
                    desired = deployment.get("spec", {}).get("replicas", 1)
                    rs_ids = {r["metadata"]["uid"] for r in replicasets if owner(r, "Deployment", deployment["metadata"]["uid"])}
                    for pod in pods:
                        meta, spec, status = pod["metadata"], pod.get("spec", {}), pod.get("status", {})
                        if meta.get("deletionTimestamp") or status.get("phase") in {"Succeeded", "Failed"}:
                            continue
                        if not any(owner(pod, "ReplicaSet", uid) for uid in rs_ids):
                            continue
                        allocated = meta.get("annotations", {}).get("hami.io/vgpu-devices-allocated", "")
                        gpu_ids = sorted({part.split(",")[0] for part in re.split(r"[:;]", allocated) if UUID.fullmatch(part.split(",")[0])})
                        ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in status.get("conditions", []))
                        containers = status.get("containerStatuses", [])
                        versions = [c.get("image", "").rsplit("/", 1)[-1] for c in spec.get("containers", [])]
                        instances.append({"id": meta["uid"], "name": meta["name"], "model": model,
                                          "node": spec.get("nodeName"), "ready": ready, "phase": status.get("phase", "Unknown"),
                                          "model_phase": None, "started_at": status.get("startTime"),
                                          "restart_count": sum(c.get("restartCount", 0) for c in containers),
                                          "versions": versions, "gpu_ids": gpu_ids, "sampled_at": stamp(now)})
                ready = sum(i["ready"] for i in instances)
                state = ("not_deployed" if not deployment else "scaled_zero" if not desired and not instances else
                         "not_ready" if not ready else "partial" if ready < desired or ready < len(instances) else "online")
                models[model] = {"id": model, "name": name, "model_type": MODEL_TYPE, "route": model, "deployment_status": state,
                                 "ready": ready, "current": len(instances), "desired": desired,
                                 "route_status": "connected" if self.route_connected(model) else "not_connected", "instances": instances}
            with self.lock:
                self.models, self.inventory_at, self.inventory_failed = models, now, False
                allowed = {g for m in models.values() for i in m["instances"] for g in i["gpu_ids"]}
                self.latest = {g: v for g, v in self.latest.items() if g in allowed}
                self.history = {g: v for g, v in self.history.items() if g in allowed}
        except (OSError, ValueError, KeyError, TypeError, httpx.HTTPError):
            with self.lock:
                self.inventory_failed = True

    def sync_metrics(self):
        now = self.clock()
        with self.lock:
            allowed = {g for m in self.models.values() for i in m["instances"] for g in i["gpu_ids"]} if self._state() == "fresh" else set()
        if not allowed:
            return
        try:
            values = parse_metrics(self.source.metrics(), allowed, now)
        except (OSError, ValueError, httpx.HTTPError):
            values = {}
        with self.lock:
            for uid in allowed:
                row = values.get(uid, {})
                point = {"sampled_at": stamp(now), "epoch": now,
                         **{k: row.get(k) for k in ("utilization_percent", "memory_used_gib", "memory_total_gib", "temperature_celsius", "power_watts")}}
                self.history.setdefault(uid, deque(maxlen=360)).append(point)
                # Missing values remain null for this sample; they are never carried forward as live.
                if any(v is not None for k, v in point.items() if k not in {"sampled_at", "epoch"}):
                    self.latest[uid] = {**point, "name": row.get("name", "GPU")}

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

    def sync_runtime_phase(self):
        url = os.environ.get("H3_SOL_RUNTIME_URL", "").rstrip("/")
        if not url:
            return
        try:
            with httpx.Client(timeout=5, trust_env=False) as client:
                response = client.get(url + "/health")
                data = response.json()
            if data.get("stage") not in {"gpu_check", "model_integrity", "loading", "ready", "downloading", "warming", "generating", "saving", "failed", "stopping"}:
                return
            with self.lock:
                for instance in self.models.get("h3-sol", {}).get("instances", []):
                    if instance["id"] == data.get("execution_instance_id"):
                        instance["model_phase"] = data["stage"]
        except (httpx.HTTPError, ValueError, TypeError):
            pass

    async def poll(self):
        while True:
            await run_in_threadpool(self.sync_inventory)
            await run_in_threadpool(self.sync_metrics)
            await run_in_threadpool(self.sync_runtime_phase)
            await asyncio.sleep(10)

    def _state(self):
        if self.inventory_at is None:
            return "unavailable"
        return "stale" if self.inventory_failed or self.clock() - self.inventory_at > 30 else "fresh"

    def snapshot(self):
        with self.lock:
            state = self._state()
            rows = []
            for model, (name, _) in MODELS.items():
                row = copy.deepcopy(self.models.get(model, {"id": model, "name": name, "model_type": MODEL_TYPE, "route": model, "route_status": "connected" if self.route_connected(model) else "not_connected"}))
                row.pop("instances", None)
                if state != "fresh":
                    row.update(deployment_status="unknown", ready=None, current=None, desired=None)
                rows.append(row)
            return {"state": state, "sampled_at": stamp(self.inventory_at), "models": rows}

    def get_instances(self, model):
        if model not in MODELS:
            raise KeyError(model)
        with self.lock:
            state = self._state()
            return {"state": state, "sampled_at": stamp(self.inventory_at),
                    "instances": copy.deepcopy(self.models.get(model, {}).get("instances", [])) if state == "fresh" else []}

    def instance(self, model, instance_id):
        data = self.get_instances(model)
        if data["state"] != "fresh":
            return data
        row = next((r for r in data["instances"] if r["id"] == instance_id), None)
        if row is None:
            raise KeyError(instance_id)
        return {"state": "fresh", "sampled_at": data["sampled_at"], "instance": row,
                "gpus": [self.gpu(model, instance_id, uid, 15) for uid in row["gpu_ids"]],
                "tasks": [{"video_task_id": t.video_task_id, "status": t.status} for t in self.task_provider()
                          if t.execution_instance_id == instance_id and t.service == model]}

    def gpu(self, model, instance_id, uid, minutes):
        with self.lock:
            data = self.get_instances(model)
            known = next((i for i in self.models.get(model, {}).get("instances", []) if i["id"] == instance_id), None)
            if known is None or uid not in known["gpu_ids"]:
                raise KeyError(uid)
            if data["state"] != "fresh":
                return {"state": data["state"], "sampled_at": None, "id": uid, "metrics": None, "history": []}
            instance = next((i for i in data["instances"] if i["id"] == instance_id), None)
            if instance is None or uid not in instance["gpu_ids"]:
                raise KeyError(uid)
            point = copy.deepcopy(self.latest.get(uid))
            state = "unavailable" if not point else "stale" if self.clock() - point["epoch"] > 30 else "fresh"
            history = [{k: v for k, v in p.items() if k != "epoch"} for p in self.history.get(uid, []) if p["epoch"] >= self.clock() - minutes * 60]
            if point:
                point.pop("epoch")
            return {"id": uid, "state": state, "sampled_at": point.get("sampled_at") if point else None,
                    "metrics": point if state == "fresh" else None, "last_sample": point,
                    "history": history, "window_minutes": minutes}

    def mcp_status(self):
        with self.lock:
            protocol = dict(self.protocol)
            if protocol["sampled_at"] and self.clock() - datetime.fromisoformat(protocol["sampled_at"]).timestamp() > 30:
                protocol["state"] = "stale"
            return {"state": "fresh", "sampled_at": stamp(self.clock()), "service": "MCP", "version": VERSION,
                    "started_at": stamp(self.started_at), "protocol": protocol, "requests": dict(self.requests)}

    async def endpoint(self, request):
        try:
            params = request.path_params
            if request.url.path == "/api/mcp/status":
                value = self.mcp_status()
            elif request.url.path == "/api/models":
                value = self.snapshot()
            elif "gpu_id" in params:
                window = request.query_params.get("window", "15m")
                if window not in {"15m", "60m"}:
                    return JSONResponse({"error": "invalid_window"}, status_code=400)
                value = self.gpu(params["model"], params["instance_id"], params["gpu_id"], int(window[:-1]))
            elif "instance_id" in params:
                value = self.instance(params["model"], params["instance_id"])
            else:
                value = self.get_instances(params["model"])
            return JSONResponse(value, headers={"Cache-Control": "no-store"})
        except KeyError:
            return JSONResponse({"error": "not_found"}, status_code=404)

    def routes(self):
        return [Route(path, self.endpoint) for path in ("/api/mcp/status", "/api/models", "/api/models/{model:str}/instances",
                "/api/models/{model:str}/instances/{instance_id:str}",
                "/api/models/{model:str}/instances/{instance_id:str}/gpus/{gpu_id:str}")]
