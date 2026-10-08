from __future__ import annotations

import base64
import binascii
import hashlib
import json
import mimetypes
import os
import re
import subprocess
import tempfile
import threading
import uuid
from datetime import UTC, datetime
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
MAX_INLINE_IMAGE_BYTES = 3 * 1024 * 1024


class ExecutionError(RuntimeError):
    pass


def _check_media(command, **kwargs):
    try:
        return subprocess.run(command, check=True, capture_output=True, **kwargs)
    except (subprocess.SubprocessError, OSError) as error:
        raise ExecutionError("Generated video could not be probed or fully decoded") from error


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
        self.singularity_token = os.environ.get("H3_SINGULARITY_RUNTIME_TOKEN", "")
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
        from .h3_latent import H3LatentExecutor
        self.processing["h3-latent-upscale"] = H3LatentExecutor(self)
        from .providers.fal import FalAdapter
        self.fal = FalAdapter(self)

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
        return ("fal" if route == "fal" else
                "h3-vdn" if route == "h3-vdn" else
                "h3-sol" if route.startswith("h3-sol") else
                "h3-singularity" if route == "h3-singularity" else "h3")

    def runtime(self, route=None):
        selected = route or self.runtime_route
        service = self._service_for_route(selected)
        if selected in self.runtime_routes:
            config = self.runtime_routes[selected]
            if config["requires_token"] or service in {"h3-vdn", "h3-singularity"}:
                token = (self.vdn_token if service == "h3-vdn" else
                         self.singularity_token if service == "h3-singularity" else self.sol_token)
                if not token:
                    raise ExecutionError("Selected runtime route is not connected")
                return config["url"], {"Authorization": f"Bearer {token}"}, config["version"], selected
            return config["url"], {}, config["version"], selected
        if service == "h3" and selected in {"h3", self.runtime_route}:
            return self.runtime_url, {}, self.runtime_version, selected
        if service == "h3-singularity" and selected == self.runtime_route and self.singularity_token:
            return self.runtime_url, {"Authorization": f"Bearer {self.singularity_token}"}, self.runtime_version, selected
        if service == "h3-sol" and selected in {"h3-sol", self.sol_route} and self.sol_url and self.sol_token:
            return self.sol_url, {"Authorization": f"Bearer {self.sol_token}"}, self.sol_version, self.sol_route
        raise ExecutionError("Selected runtime is not connected")

    def _record_route(self, record: TaskRecord) -> str:
        route = record.request.get("route")
        if isinstance(route, str) and route:
            return route
        # Read old records created before route became the public selector.
        return self.sol_route if record.service == "h3-sol" else self.runtime_route

    def capabilities(self) -> dict[str, object]:
        """Separate configured adapters, live readiness and unverified GPU combinations."""
        from .providers.fal import FAL_ASPECT_RATIOS
        from .singularity import ASPECT_RATIOS
        routes = []
        names = {self.runtime_route, "fal", *self.runtime_routes}
        if self.sol_url:
            names.add("h3-sol")
        for route in sorted(names):
            service = self._service_for_route(route)
            try:
                if route == "fal":
                    configured = self.fal.connected()
                else:
                    self.runtime(route)
                    configured = True
            except ExecutionError:
                configured = False
            routes.append({
                "route": route, "model": "minimax-h3-ref2va", "configured": configured,
                "readiness": "not_checked",
                "limit_scope": "adapter",  # These are implementation limits, not model limits.
                "duration_seconds": {"minimum": 5 if service in {"fal", "h3-sol", "h3-vdn"} else 4, "maximum": 15},
                "aspect_ratios": sorted(FAL_ASPECT_RATIOS) if route == "fal" else list(ASPECT_RATIOS) if service == "h3-singularity" else ["9:16"],
            })
            if service == "h3-singularity":
                routes[-1].update(quality_profiles=["du-0"], gpu_verified_combinations=[],
                                  verification_status="not_recorded", frame_grid="17k+5", fps=24)
                if configured:
                    url, headers, version, _ = self.runtime(route)
                    routes[-1]["configured_runtime_version"] = version
                    try:
                        response = self.client.get(f"{url}/health", headers=headers, timeout=10)
                        health = response.json()
                        routes[-1]["readiness"] = "ready" if response.status_code == 200 and health.get("ready") is True else "not_ready"
                        if response.status_code == 200:
                            routes[-1]["runtime_version"] = health.get("runtime_version", "unknown")
                            routes[-1]["runtime_capabilities"] = health.get("capabilities", {})
                    except (httpx.HTTPError, ValueError, AttributeError):
                        routes[-1]["readiness"] = "unreachable"
        return {"default_model": "minimax-h3-ref2va", "default_route": self.runtime_route,
                "model_specification": {"aspect_ratios": ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
                                        "omni_adaptive": True},
                "routes": routes, "inline_image_max_bytes": MAX_INLINE_IMAGE_BYTES,
                "chunk_import": {"max_chunk_bytes": 512 * 1024, "max_file_bytes": MAX_ARTIFACT_BYTES,
                                 "formats": ["PNG", "JPEG", "WebP", "MP4/H.264", "MP4/H.265", "WAV/PCM", "MP3"],
                                 "reference_duration_seconds": {"minimum": 2, "maximum": 15},
                                 "max_image_pixels": 16 * 1024 * 1024, "max_video_pixels": 4096 * 2160,
                                 "max_video_decoded_pixels": 1_000_000_000, "max_video_fps": 120,
                                 "audio_channels": [1, 2]}}

    @property
    def imports(self):
        # Lazy initialization keeps unrelated read-only operations independent
        # of upload storage and avoids creating files during tool discovery.
        with self._lock:
            if not hasattr(self, "_imports"):
                from .imports import ImportStore
                from .media_validation import validate_reference_media
                self._imports = ImportStore(self.artifacts, validate_reference_media)
            return self._imports

    def import_image(self, *, project_id: str, content_base64: str,
                     filename: str, expected_sha256: str) -> ArtifactRecord:
        """Import caller-owned bytes, without fetching URLs or accepting access claims."""
        project_id = require_project_id(project_id)
        filename = require_filename(filename)
        media_type = MEDIA_TYPES.get(Path(filename).suffix.lower())
        digest = expected_sha256.strip().lower()
        if media_type not in {"image/png", "image/jpeg", "image/webp"}:
            raise ValueError("inline import supports PNG, JPEG, and WebP images only")
        if not SHA256_PATTERN.fullmatch(digest):
            raise ValueError("expected_sha256 must contain exactly 64 hexadecimal characters")
        if not content_base64 or len(content_base64) > 4 * ((MAX_INLINE_IMAGE_BYTES + 2) // 3):
            raise ValueError("inline image must contain 1 byte to 3 MiB")
        try:
            content = base64.b64decode(content_base64, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("content_base64 is invalid") from error
        magic = (content.startswith(b"\x89PNG\r\n\x1a\n") if media_type == "image/png" else
                 content.startswith(b"\xff\xd8\xff") if media_type == "image/jpeg" else
                 content.startswith(b"RIFF") and content[8:12] == b"WEBP")
        if not magic or len(content) > MAX_INLINE_IMAGE_BYTES:
            raise ValueError("inline image content does not match filename or size limit")
        return self.artifacts.create_from_chunks(project_id=project_id, operation="video.import",
            filename=filename, media_type=media_type, chunks=(content,), expected_sha256=digest)

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
                   route: str = "h3", quality_profile: str = "du-0") -> tuple[dict[str, object], str]:
        if not isinstance(route, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", route):
            raise ValueError("route must be a lowercase slug")
        service = self._service_for_route(route)
        if quality_profile not in {"du-0", "du-1", "du-2", "du-3", "hr-refine-tile-v1", "hr-refine-global-v2"}:
            raise ValueError("quality_profile must be du-0, du-1, du-2, du-3, hr-refine-tile-v1, or hr-refine-global-v2")
        if service != "h3-singularity" and quality_profile != "du-0":
            raise ValueError("quality_profile du-1, du-2, du-3, hr-refine-tile-v1, and hr-refine-global-v2 require h3-singularity")
        if service in {"h3-sol", "h3-vdn"} and not 5 <= duration_seconds <= 15:
            raise ValueError("Selected channel duration must be an integer from 5 to 15 seconds")
        if model != "minimax-h3-ref2va":
            raise ValueError("model must be minimax-h3-ref2va")
        if type(duration_seconds) is not int or not 4 <= duration_seconds <= 15 or not prompt.strip():
            raise ValueError("duration_seconds must be an integer from 4 to 15 and prompt must not be empty")
        from .singularity import ASPECT_RATIOS, output_spec
        if service == "h3-singularity" and quality_profile != "du-0":
            raise ValueError("Selected Singularity production route supports only du-0")
        if service == "h3-singularity" and aspect_ratio not in ASPECT_RATIOS:
            raise ValueError("Unsupported Singularity aspect_ratio")
        if service != "h3-singularity" and aspect_ratio != "9:16":
            raise ValueError("Selected local H3 channel supports only 9:16; query video.capabilities before submission")
        limits = {"images": 9, "videos": 3, "audios": 3}
        if not isinstance(references, dict) or set(references) - limits.keys() or any(not isinstance(v, list) for v in references.values()):
            raise ValueError("references must contain only images, videos and audios arrays")
        if service == "h3-singularity" and not any(references.values()):
            raise ValueError("at least one reference is required")
        if service != "h3-singularity" and not references.get("images") and not references.get("videos"):
            raise ValueError("at least one image or video reference is required")
        if service in {"h3-vdn", "h3-singularity"} and sum(len(references.get(k, [])) for k in limits) > 12:
            raise ValueError("Selected route supports at most 12 references")
        reference_bytes = 0
        media_by_kind = {kind: [] for kind in limits}
        normalized: dict[str, list[dict[str, str]]] = {}
        for kind, limit in limits.items():
            items = references.get(kind, [])
            if len(items) > limit:
                raise ValueError(f"too many {kind} references")
            normalized[kind] = []
            for item in items:
                if not isinstance(item, dict) or set(item) != {"artifact_id", "purpose"} or not isinstance(item["purpose"], str):
                    raise ValueError("reference requires only artifact_id and purpose")
                artifact = self.artifacts.get(item["artifact_id"], project_id)
                reference_bytes += artifact.size
                expected_prefix = {"images": "image/", "videos": "video/", "audios": "audio/"}[kind]
                if not artifact.media_type.startswith(expected_prefix) or not item.get("purpose", "").strip():
                    raise ValueError(f"invalid {kind} reference")
                normalized[kind].append({"artifact_id": artifact.artifact_id, "purpose": item["purpose"].strip()})
                if service == "h3-singularity":
                    from .media_validation import validate_reference_media
                    media_by_kind[kind].append(validate_reference_media(self.artifacts.content_path(artifact), artifact.media_type))
        if service == "h3-vdn" and reference_bytes > 2 * 1024**3:
            raise ValueError("VDN references exceed 2 GiB")
        request = {"project_id": project_id, "model": model, "prompt": prompt.strip(),
                   "duration_seconds": duration_seconds, "aspect_ratio": aspect_ratio,
                   "references": normalized, "quality_profile": quality_profile}
        request["route"] = route
        if service == "h3-singularity":
            for kind in ("videos", "audios"):
                if sum(m["duration_seconds"] for m in media_by_kind[kind]) > 15.001:
                    raise ValueError(f"{kind} references exceed 15 seconds in total")
            visuals = media_by_kind["images"] or media_by_kind["videos"]
            request["resolved_output"] = output_spec(aspect_ratio, duration_seconds, visuals[0] if visuals else None)
            for media in media_by_kind["videos"]:
                count = min(media["resampled_frames"], request["resolved_output"]["frames"])
                count -= (count - 5) % 17
                media.update(conditioning_frames=count, conditioning_duration_seconds=count / 24,
                             soundtrack_alignment="same_interval_as_video")
            request["reference_media"] = media_by_kind
            mapping = []
            audio_index = 0
            for kind, label in (("images", "Picture"), ("videos", "Video"), ("audios", "Audio")):
                for index, (item, media) in enumerate(zip(normalized[kind], media_by_kind[kind]), 1):
                    if kind == "videos" and media["has_audio"]:
                        audio_index += 1
                        mapping.append({"tag": f"<Audio {audio_index}>", "artifact_id": item["artifact_id"], "role": "video_soundtrack"})
                    if kind == "audios":
                        audio_index += 1
                    mapping.append({"tag": f"<{label} {audio_index if kind == 'audios' else index}>",
                                    "artifact_id": item["artifact_id"], "purpose": item["purpose"], "role": kind})
            request["reference_mapping"] = mapping
        canonical = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return request, "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()

    @serialized
    def generate(self, *, project_id: str, idempotency_key: str, model: str, prompt: str,
                 duration_seconds: int, aspect_ratio: str,
                 references: dict[str, list[dict[str, str]]], route: str | None = None,
                 quality_profile: str = "du-0") -> TaskRecord:
        project_id = require_project_id(project_id)
        if not idempotency_key.strip() or len(idempotency_key) > 128:
            raise ValueError("idempotency_key is invalid")
        route = route or self.runtime_route
        if route == "fal":
            return self.fal.generate(project_id=project_id, idempotency_key=idempotency_key,
                                     model=model, prompt=prompt, duration_seconds=duration_seconds,
                                     aspect_ratio=aspect_ratio, references=references)
        request, digest = self._normalize(project_id, model, prompt, duration_seconds, aspect_ratio,
                                          references, route, quality_profile)
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
                if service in {"h3-sol", "h3-vdn", "h3-singularity"}:
                    asset = self.artifacts.get(item["artifact_id"], project_id)
                    conditions[-1].update(sha256=asset.sha256, size=asset.size)
                material_tags.append(f"<{tag_name[kind]} {index}> is the approved {item['purpose']} reference")
        compiled_prompt = request["prompt"] if service == "h3-singularity" else "; ".join(material_tags) + ". " + request["prompt"]
        payload = {"model": "MiniMaxAI/MiniMax-H3", "task": "ref2va", "prompt": compiled_prompt,
                   "seconds": duration_seconds, "conditions": conditions,
                   "target": {"short_edge": 768, "aspect_ratio": aspect_ratio, "duration_seconds": float(duration_seconds)},
                   "num_outputs_per_prompt": 1, "num_inference_steps": 4 if service in {"h3-vdn", "h3-sol", "h3-singularity"} else 21, "flow_shift": 12.0,
                   "audio_flow_shift": 3.0, "seed": 7,
                   "quality_profile": quality_profile}
        # Persist the attempt before calling the runtime. An ambiguous network
        # failure must not permit a duplicate GPU request under the same key.
        reserved = self.tasks.create(project_id=project_id, idempotency_key=idempotency_key,
                                     input_digest=digest, request=request, runtime_task_id="", status="queued")
        reserved = self.tasks.update(reserved, service=service, runtime_version=runtime_version, runtime_route=runtime_route)
        if service in {"h3-vdn", "h3-singularity"}:
            try:
                health = self.client.get(f"{runtime_url}/health", headers=runtime_headers, timeout=10)
                if health.status_code == 503:
                    return self.tasks.update(reserved, status="failed", error={
                        "code": "runtime_not_ready", "message": "H3 runtime is not ready; generation was not submitted"})
                health.raise_for_status()
                health_data = health.json()
                if not isinstance(health_data, dict) or not isinstance(health_data.get("ready"), bool):
                    raise ValueError("Invalid readiness response")
                if not health_data["ready"]:
                    return self.tasks.update(reserved, status="failed", error={
                        "code": "runtime_not_ready", "message": "H3 runtime is not ready; generation was not submitted"})
            except (httpx.ConnectError, httpx.ConnectTimeout):
                return self.tasks.update(reserved, status="failed", error={
                    "code": "runtime_unavailable", "message": "H3 runtime is unreachable; generation was not submitted"})
            except (httpx.HTTPError, ValueError):
                return self.tasks.update(reserved, status="failed", error={
                    "code": "runtime_health_check_failed", "message": "H3 readiness check failed; generation was not submitted"})
        if service == "h3-vdn":
            payload["project_id"] = project_id
        if service in {"h3-sol", "h3-vdn", "h3-singularity"}:
            payload["idempotency_key"] = reserved.video_task_id
        try:
            response = self.client.post(f"{runtime_url}/v1/videos", json=payload, headers=runtime_headers)
            response.raise_for_status()
            runtime_task_id = response.json()["id"]
            if not isinstance(runtime_task_id, str) or not runtime_task_id.strip():
                raise ValueError("Invalid runtime task id")
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return self.tasks.update(reserved, status="failed", error={
                "code": "runtime_unavailable", "message": "H3 runtime is unreachable; generation was not submitted"})
        except httpx.HTTPStatusError as error:
            if 400 <= error.response.status_code < 500:
                return self.tasks.update(reserved, status="failed", error={"code": "runtime_rejected",
                    "message": f"Runtime rejected submission (HTTP {error.response.status_code})"})
            return self.tasks.update(reserved, status="queued" if service == "h3-singularity" else "failed",
                error={"code": "submission_unconfirmed", "message": "Runtime submission could not be confirmed; do not resubmit automatically"})
        except (httpx.HTTPError, KeyError, ValueError, TypeError):
            return self.tasks.update(reserved, status="queued" if service == "h3-singularity" else "failed", error={"code": "submission_unconfirmed",
                              "message": "Runtime submission could not be confirmed; do not resubmit automatically"})
        return self.tasks.update(reserved, runtime_task_id=runtime_task_id,
                                 dispatched_at=datetime.now(UTC).isoformat())

    @serialized
    def status(self, video_task_id: str) -> TaskRecord:
        record = self.tasks.get(video_task_id)
        if (record.service == "h3-singularity" and not record.runtime_task_id
            and (record.error or {}).get("code") == "submission_unconfirmed"):
            url, headers, _, _ = self.runtime(self._record_route(record))
            try:
                response = self.client.get(f"{url}/v1/videos/by-idempotency/{record.video_task_id}", headers=headers, timeout=10)
                if response.status_code == 404:
                    # A delayed submission can still arrive; absence is not
                    # permission to change keys or dispatch another GPU job.
                    return self.tasks.update(record, status="queued")
                response.raise_for_status()
                recovered = response.json()
                identity = recovered.get("id")
                if not isinstance(identity, str) or not re.fullmatch(r"singularity_[0-9a-f]{32}", identity):
                    raise ValueError("invalid recovered identity")
            except (httpx.HTTPError, ValueError, AttributeError) as error:
                raise ExecutionError("Singularity submission remains unconfirmed; retain the original task and key") from error
            record = self.tasks.update(record, runtime_task_id=identity, status="queued", error=None)
        if record.service == "fal":
            from .providers.fal import FalError
            try:
                return self.fal.status(record)
            except FalError as error:
                raise ExecutionError(str(error)) from error
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
        metrics = runtime_data.get("runtime_metrics") or ({"runtime_total_seconds": runtime_data["result"]["generate_seconds"]} if isinstance(runtime_data.get("result"), dict) and isinstance(runtime_data["result"].get("generate_seconds"), (int, float)) else {})
        if record.service == "h3-singularity" and isinstance(runtime_data.get("result"), dict):
            metrics = {**metrics, "result_sha256": runtime_data["result"].get("content_sha256")}
        return self.tasks.update(record, status=mapped, error=error, execution_instance_id=identity,
                                 runtime_stage=stage, runtime_metrics=metrics or None)

    @serialized
    def result(self, video_task_id: str) -> TaskRecord:
        record = self.tasks.get(video_task_id)
        if record.service == "fal":
            from .providers.fal import FalError
            try:
                return self.fal.result(record)
            except FalError as error:
                raise ExecutionError(str(error)) from error
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
        expected = record.request.get("resolved_output") if record.service == "h3-singularity" else None
        digest = hashlib.sha256(payload).hexdigest()
        if expected and digest != (record.runtime_metrics or {}).get("result_sha256"):
            raise ExecutionError("Singularity result SHA-256 mismatch or missing runtime digest")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.mp4"
            path.write_bytes(payload)
            probe = _check_media(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
                                    "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,nb_read_frames,duration,sample_rate,channels",
                                    "-of", "json", str(path)], text=True, timeout=120)
            if expected:
                _check_media(["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-map", "0:v:0",
                              "-map", "0:a:0", "-f", "null", "-"], timeout=180)
        data = json.loads(probe.stdout)
        video = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
        audio = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
        if not video or video.get("codec_name") != "h264" or video.get("r_frame_rate") != "24/1":
            raise ExecutionError("H3 result failed the H.264/24 FPS media contract")
        duration_ms = round(float(data["format"]["duration"]) * 1000)
        expected_ms = int(record.request["duration_seconds"]) * 1000
        # H3 rounds generation to its internal temporal frame bucket, which can
        # leave less than one second of edit handle beyond the requested length.
        if expected:
            if (video.get("width"), video.get("height"), int(video.get("nb_read_frames", 0))) != (expected["width"], expected["height"], expected["frames"]):
                raise ExecutionError("Singularity output geometry or decoded frame count differs from resolved request")
            if video.get("avg_frame_rate") != "24/1":
                raise ExecutionError("Singularity output must have constant 24 FPS")
            if not audio or audio.get("codec_name") != "aac" or int(audio.get("channels", 0)) != 2:
                raise ExecutionError("Singularity result requires stereo AAC audio")
            if abs(float(video.get("duration", 0)) - expected["video_duration_seconds"]) > 1 / 24:
                raise ExecutionError("Singularity video stream duration mismatch")
            if abs(float(audio.get("duration", 0)) - expected["video_duration_seconds"]) > 0.1:
                raise ExecutionError("Singularity audio stream duration mismatch")
        elif abs(duration_ms - expected_ms) > H3_DURATION_TOLERANCE_MS:
            raise ExecutionError("H3 result duration is outside the approved tolerance")
        artifact = self.artifacts.create_from_chunks(project_id=record.project_id, operation="video.result",
                                                     filename=f"{record.video_task_id}.mp4", media_type="video/mp4",
                                                     chunks=(payload,))
        media = {"duration_ms": duration_ms, "width": video["width"], "height": video["height"],
                 "frame_rate": 24, "video_codec": "h264", "audio_codec": audio.get("codec_name") if audio else None,
                 "audio_sample_rate": int(audio["sample_rate"]) if audio and audio.get("sample_rate") else None,
                 "audio_channels": audio.get("channels") if audio else None}
        if expected:
            media.update(frames=int(video["nb_read_frames"]), video_duration_seconds=float(video["duration"]),
                         audio_duration_seconds=float(audio["duration"]), sha256=digest, resolved_output=expected,
                         complete_decode_verified=True)
        return self.tasks.update(record, artifact_id=artifact.artifact_id, media=media)
