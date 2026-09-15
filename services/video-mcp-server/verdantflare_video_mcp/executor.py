from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import subprocess
import tempfile
import threading
import uuid
from functools import wraps
from pathlib import Path
from urllib.parse import unquote, urlsplit

import httpx

from .artifacts import MAX_ARTIFACT_BYTES, ArtifactRecord, ArtifactStore, require_filename, require_project_id
from .tasks import TaskConflict, TaskRecord, TaskStore

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
MEDIA_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
}
H3_DURATION_TOLERANCE_MS = 1000


class ExecutionError(RuntimeError):
    pass


def serialized(method):
    @wraps(method)
    def invoke(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return invoke


class VideoExecutor:
    def __init__(self, artifacts: ArtifactStore, tasks: TaskStore, client: httpx.Client | None = None) -> None:
        self._lock = threading.RLock()
        self.artifacts = artifacts
        self.tasks = tasks
        self.runtime_url = os.environ.get("H3_RUNTIME_URL", "http://video-minimax-h3-api:8000").rstrip("/")
        self.runtime_route = os.environ.get("H3_RUNTIME_ROUTE", "h3")
        self.runtime_routes = self._load_routes()
        self.runtime_artifact_url = os.environ.get("VIDEO_MCP_RUNTIME_BASE_URL", "http://video-mcp-server:8000").rstrip("/")
        self.sol_url = os.environ.get("H3_SOL_RUNTIME_URL", "").rstrip("/")
        self.sol_version = os.environ.get("H3_SOL_RUNTIME_VERSION", "video-minimax-h3-sol-v0.2.1")
        self.sol_route = os.environ.get("H3_SOL_RUNTIME_ROUTE", "minimax-h3-sol-ref2va")
        self.sol_token = os.environ.get("H3_SOL_RUNTIME_TOKEN", "")
        self.vdn_token = os.environ.get("H3_VDN_RUNTIME_TOKEN", "")
        self.runtime_version = os.environ.get("H3_RUNTIME_VERSION", "video-minimax-h3-api-v0.3.0")
        self.allowed_origins = frozenset(x.strip() for x in os.environ.get("VIDEO_ASSET_IMPORT_ORIGINS", "").split(",") if x.strip())
        self.import_rewrites = json.loads(os.environ.get("VIDEO_ASSET_IMPORT_REWRITES", "{}"))
        if not isinstance(self.import_rewrites, dict):
            raise ValueError("VIDEO_ASSET_IMPORT_REWRITES must be an object")
        for source, target in self.import_rewrites.items():
            if not isinstance(source, str) or not isinstance(target, str) or not source.endswith("/") or not target.endswith("/"):
                raise ValueError("import rewrite prefixes must end with a slash")
            for value in (source, target):
                parsed = urlsplit(value)
                if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                    raise ValueError("invalid import rewrite prefix")
        self.client = client or httpx.Client(timeout=httpx.Timeout(connect=10, read=3600, write=600, pool=10), follow_redirects=False)
        from .depth import DepthExecutor
        self.depth = DepthExecutor(self)
        from .processing import ProcessingExecutor
        self.processing = {name: ProcessingExecutor(self, name) for name in ("sr", "interpolate")}

    def _load_routes(self) -> dict[str, dict[str, object]]:
        raw = os.environ.get("H3_RUNTIME_ROUTES", "")
        if not raw:
            return {}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("H3_RUNTIME_ROUTES must be valid JSON") from error
        if not isinstance(value, dict):
            raise ValueError("H3_RUNTIME_ROUTES must be an object")
        routes: dict[str, dict[str, object]] = {}
        for route, config in value.items():
            if not isinstance(route, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", route):
                raise ValueError("runtime route names must be lowercase slugs")
            if not isinstance(config, dict) or not isinstance(config.get("url"), str) or not config["url"].startswith("http"):
                raise ValueError("each runtime route requires an HTTP URL")
            routes[route] = {"url": config["url"].rstrip("/"),
                             "version": str(config.get("version", "unknown")),
                             "requires_token": bool(config.get("requires_token", False))}
        return routes

    @staticmethod
    def _service_for_route(route: str) -> str:
        return "h3-vdn" if route == "h3-vdn" else "h3-sol" if route.startswith("h3-sol") else "h3"

    def runtime(self, route=None):
        selected = route or self.runtime_route
        service = self._service_for_route(selected)
        if selected in self.runtime_routes:
            config = self.runtime_routes[selected]
            if config["requires_token"] or service == "h3-vdn":
                token = self.vdn_token if service == "h3-vdn" else self.sol_token
                if not token:
                    raise ExecutionError("Selected runtime route is not connected")
                return config["url"], {"Authorization": f"Bearer {token}"}, config["version"], selected
            return config["url"], {}, config["version"], selected
        if service == "h3" and selected in {"h3", self.runtime_route}:
            return self.runtime_url, {}, self.runtime_version, selected
        if service == "h3-sol" and selected in {"h3-sol", self.sol_route} and self.sol_url and self.sol_token:
            return self.sol_url, {"Authorization": f"Bearer {self.sol_token}"}, self.sol_version, self.sol_route
        raise ExecutionError("Selected runtime is not connected")

    def _record_route(self, record: TaskRecord) -> str:
        route = record.request.get("route")
        if isinstance(route, str) and route:
            return route
        # Read old records created before route became the public selector.
        return self.sol_route if record.service == "h3-sol" else self.runtime_route

    def import_asset(self, *, project_id: str, source_url: str, filename: str, expected_sha256: str) -> ArtifactRecord:
        project_id = require_project_id(project_id)
        filename = require_filename(filename)
        media_type = MEDIA_TYPES.get(Path(filename).suffix.lower())
        if media_type is None:
            raise ValueError("filename uses an unsupported media extension")
        digest = expected_sha256.strip().lower()
        if not SHA256_PATTERN.fullmatch(digest):
            raise ValueError("expected_sha256 must contain exactly 64 hexadecimal characters")
        parsed = urlsplit(source_url)
        origin = f"{parsed.scheme}://{parsed.hostname}" if parsed.port in {None, 443} else f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
        if parsed.scheme != "https" or parsed.username or parsed.password or parsed.fragment or origin not in self.allowed_origins:
            raise ValueError("source_url must be an allowed absolute HTTPS object URL")
        decoded_path = parsed.path
        for _ in range(5):
            if "\\" in decoded_path or any(part in {".", ".."} for part in decoded_path.split("/")):
                raise ValueError("source_url must not contain path traversal")
            decoded = unquote(decoded_path)
            if decoded == decoded_path:
                break
            decoded_path = decoded
        else:
            raise ValueError("source_url path has excessive encoding")
        download_url = source_url
        for source, target in self.import_rewrites.items():
            if source_url.startswith(source):
                download_url = target + source_url[len(source):]
                break
        try:
            with self.client.stream("GET", download_url, headers={"Accept": "image/*, audio/*, video/*, application/octet-stream"}) as response:
                if not 200 <= response.status_code < 300:
                    raise ExecutionError(f"asset import returned HTTP {response.status_code}")
                length = response.headers.get("content-length")
                if length and not 0 < int(length) <= MAX_ARTIFACT_BYTES:
                    raise ExecutionError("asset import size must be between 1 byte and 1 GiB")
                return self.artifacts.create_from_chunks(project_id=project_id, operation="artifact.import",
                                                         filename=filename, media_type=media_type,
                                                         chunks=response.iter_bytes(), expected_sha256=digest)
        except httpx.HTTPError as error:
            raise ExecutionError("asset import request failed") from error

    def _normalize(self, project_id: str, model: str, prompt: str, duration_seconds: int,
                   aspect_ratio: str, references: dict[str, list[dict[str, str]]],
                   route: str = "h3") -> tuple[dict[str, object], str]:
        if not isinstance(route, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", route):
            raise ValueError("route must be a lowercase slug")
        service = self._service_for_route(route)
        if service in {"h3-sol", "h3-vdn"} and duration_seconds not in {5, 10, 15}:
            raise ValueError("Selected channel duration must be 5, 10, or 15 seconds")
        if model != "minimax-h3-ref2va":
            raise ValueError("model must be minimax-h3-ref2va")
        if not 4 <= duration_seconds <= 15 or aspect_ratio != "9:16" or not prompt.strip():
            raise ValueError("duration_seconds, aspect_ratio, or prompt is invalid")
        limits = {"images": 9, "videos": 3, "audios": 3}
        if not references.get("images") and not references.get("videos"):
            raise ValueError("at least one image or video reference is required")
        if service == "h3-vdn" and sum(len(references.get(k, [])) for k in limits) > 12:
            raise ValueError("VDN supports at most 12 references")
        reference_bytes = 0
        normalized: dict[str, list[dict[str, str]]] = {}
        for kind, limit in limits.items():
            items = references.get(kind, [])
            if len(items) > limit:
                raise ValueError(f"too many {kind} references")
            normalized[kind] = []
            for item in items:
                artifact = self.artifacts.get(item["artifact_id"], project_id)
                reference_bytes += artifact.size
                expected_prefix = {"images": "image/", "videos": "video/", "audios": "audio/"}[kind]
                if not artifact.media_type.startswith(expected_prefix) or not item.get("purpose", "").strip():
                    raise ValueError(f"invalid {kind} reference")
                normalized[kind].append({"artifact_id": artifact.artifact_id, "purpose": item["purpose"].strip()})
        if service == "h3-vdn" and reference_bytes > 2 * 1024**3:
            raise ValueError("VDN references exceed 2 GiB")
        request = {"project_id": project_id, "model": model, "prompt": prompt.strip(),
                   "duration_seconds": duration_seconds, "aspect_ratio": aspect_ratio, "references": normalized}
        request["route"] = route
        canonical = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return request, "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()

    @serialized
    def generate(self, *, project_id: str, idempotency_key: str, model: str, prompt: str,
                 duration_seconds: int, aspect_ratio: str,
                 references: dict[str, list[dict[str, str]]], route: str = "h3") -> TaskRecord:
        project_id = require_project_id(project_id)
        if not idempotency_key.strip() or len(idempotency_key) > 128:
            raise ValueError("idempotency_key is invalid")
        request, digest = self._normalize(project_id, model, prompt, duration_seconds, aspect_ratio, references, route)
        service = self._service_for_route(route)
        existing = self.tasks.find_idempotency(project_id, idempotency_key)
        if existing:
            if existing.input_digest != digest:
                raise TaskConflict("idempotency key already exists with different input")
            return existing
        runtime_url, runtime_headers, runtime_version, runtime_route = self.runtime(route)
        conditions = []
        material_tags = []
        singular = {"images": "image", "videos": "video", "audios": "audio"}
        tag_name = {"images": "Picture", "videos": "Video", "audios": "Audio"}
        # Diffusers numbers video soundtracks as audio references too. Put explicit
        # audio first so <Audio 1> continues to name the first audio Artifact.
        kinds = ("images", "audios", "videos") if service == "h3-vdn" else ("images", "videos", "audios")
        for kind in kinds:
            for index, item in enumerate(request["references"][kind], start=1):
                conditions.append({"type": singular[kind],
                                   "uri": f"{self.runtime_artifact_url}/runtime-artifacts/{item['artifact_id']}/content",
                                   "role": "reference"})
                if service in {"h3-sol", "h3-vdn"}:
                    asset = self.artifacts.get(item["artifact_id"], project_id)
                    conditions[-1].update(sha256=asset.sha256, size=asset.size)
                material_tags.append(f"<{tag_name[kind]} {index}> is the approved {item['purpose']} reference")
        compiled_prompt = "; ".join(material_tags) + ". " + request["prompt"]
        payload = {"model": "MiniMaxAI/MiniMax-H3", "task": "ref2va", "prompt": compiled_prompt,
                   "seconds": duration_seconds, "conditions": conditions,
                   "target": {"short_edge": 768, "aspect_ratio": aspect_ratio, "duration_seconds": float(duration_seconds)},
                   "num_outputs_per_prompt": 1, "num_inference_steps": 8 if service == "h3-vdn" else 4 if service == "h3-sol" else 21, "flow_shift": 12.0,
                   "audio_flow_shift": 3.0, "seed": 7}
        # Persist the attempt before calling the runtime. An ambiguous network
        # failure must not permit a duplicate GPU request under the same key.
        reserved = self.tasks.create(project_id=project_id, idempotency_key=idempotency_key,
                                     input_digest=digest, request=request, runtime_task_id="", status="queued")
        reserved = self.tasks.update(reserved, service=service, runtime_version=runtime_version, runtime_route=runtime_route)
        if service in {"h3-sol", "h3-vdn"}:
            payload["idempotency_key"] = reserved.video_task_id
        try:
            response = self.client.post(f"{runtime_url}/v1/videos", json=payload, headers=runtime_headers)
            response.raise_for_status()
            runtime_task_id = response.json()["id"]
        except (httpx.HTTPError, KeyError, ValueError) as error:
            self.tasks.update(reserved, status="failed", error={"code": "submission_unconfirmed",
                              "message": "Runtime submission could not be confirmed; do not resubmit automatically"})
            raise ExecutionError("H3 runtime submission failed") from error
        return self.tasks.update(reserved, runtime_task_id=runtime_task_id)

    @serialized
    def status(self, video_task_id: str) -> TaskRecord:
        record = self.tasks.get(video_task_id)
        if record.service in self.processing:
            return self.processing[record.service].status(video_task_id)
        if record.service == "depth":
            return self.depth.status(video_task_id)
        if record.status in {"succeeded", "failed", "cancelled"}:
            return record
        runtime_url, runtime_headers, _, _ = self.runtime(self._record_route(record))
        try:
            response = self.client.get(f"{runtime_url}/v1/videos/{record.runtime_task_id}", timeout=10, headers=runtime_headers)
            if response.status_code == 404:
                return self.tasks.update(record, status="failed", error={
                    "code": "runtime_task_lost",
                    "message": "H3 runtime task no longer exists",
                })
            response.raise_for_status()
            runtime_data = response.json()
            runtime_status = runtime_data["status"]
        except (httpx.HTTPError, KeyError, ValueError) as error:
            raise ExecutionError("H3 runtime status request failed") from error
        mapped = {"queued": "queued", "in_progress": "running", "completed": "succeeded",
                  "failed": "failed", "failure": "failed", "cancelled": "cancelled"}.get(runtime_status)
        if mapped is None:
            raise ExecutionError("H3 runtime returned an unknown status")
        error = {"code": "runtime_failed", "message": "H3 generation failed"} if mapped == "failed" else None
        identity = runtime_data.get("execution_instance_id")
        if identity is not None:
            try:
                identity = str(uuid.UUID(identity))
            except (ValueError, TypeError, AttributeError):
                raise ExecutionError("Runtime returned invalid instance identity")
        stage = runtime_data.get("stage")
        if stage not in {"queued", "downloading", "warming", "generating", "saving", "completed", "interrupted", "download_failed", "engine_failed"}:
            stage = None
        return self.tasks.update(record, status=mapped, error=error, execution_instance_id=identity,
                                 runtime_stage=stage, timing=runtime_data.get("timing"))

    @serialized
    def result(self, video_task_id: str) -> TaskRecord:
        record = self.status(video_task_id)
        if record.service in self.processing:
            return self.processing[record.service].result(video_task_id)
        if record.service == "depth":
            return self.depth.result(video_task_id)
        if record.status != "succeeded":
            raise ExecutionError("video task has not succeeded")
        if record.artifact_id:
            return record
        runtime_url, runtime_headers, _, _ = self.runtime(self._record_route(record))
        try:
            response = self.client.get(f"{runtime_url}/v1/videos/{record.runtime_task_id}/content", headers=runtime_headers)
            response.raise_for_status()
            payload = response.content
        except httpx.HTTPError as error:
            raise ExecutionError("H3 result download failed") from error
        with tempfile.NamedTemporaryFile(suffix=".mp4") as temp:
            temp.write(payload); temp.flush()
            probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                    "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels",
                                    "-of", "json", temp.name], check=True, capture_output=True, text=True)
        data = json.loads(probe.stdout)
        video = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
        audio = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
        if not video or video.get("codec_name") != "h264" or video.get("r_frame_rate") != "24/1":
            raise ExecutionError("H3 result failed the H.264/24 FPS media contract")
        duration_ms = round(float(data["format"]["duration"]) * 1000)
        expected_ms = int(record.request["duration_seconds"]) * 1000
        # H3 rounds generation to its internal temporal frame bucket, which can
        # leave less than one second of edit handle beyond the requested length.
        if abs(duration_ms - expected_ms) > H3_DURATION_TOLERANCE_MS:
            raise ExecutionError("H3 result duration is outside the approved tolerance")
        artifact = self.artifacts.create_from_chunks(project_id=record.project_id, operation="video.result",
                                                     filename=f"{record.video_task_id}.mp4", media_type="video/mp4",
                                                     chunks=(payload,))
        media = {"duration_ms": duration_ms, "width": video["width"], "height": video["height"],
                 "frame_rate": 24, "video_codec": "h264", "audio_codec": audio.get("codec_name") if audio else None,
                 "audio_sample_rate": int(audio["sample_rate"]) if audio and audio.get("sample_rate") else None,
                 "audio_channels": audio.get("channels") if audio else None}
        return self.tasks.update(record, artifact_id=artifact.artifact_id, media=media)
