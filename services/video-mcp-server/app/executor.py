from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import mimetypes
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid
import ssl
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from urllib.parse import unquote, urlsplit

import httpx

from .artifacts import MAX_ARTIFACT_BYTES, ArtifactRecord, ArtifactStore, require_filename, require_project_id
from .channel_errors import MESSAGES
from .gpu_lease import GpuCoordinator, LeaseBusy, LocalGpuStageSlot
from .regenerate import (
    SINGULARITY_PROFILE_NFE,
    SINGULARITY_QUALITY_PROFILES,
    default_singularity_quality_profile,
)
from .tasks import TaskConflict, TaskRecord, TaskStore

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
MEDIA_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
}
H3_DURATION_TOLERANCE_MS = 1000
MAX_INLINE_IMAGE_BYTES = 3 * 1024 * 1024
LOCAL_H3_SERVICES = {"h3", "h3-vdn", "h3-sol", "h3-singularity"}
LOCAL_GPU_SERVICES = LOCAL_H3_SERVICES | {"regenerate_2k", "interpolate"}
TERMINAL_TASK_STATUS = {"succeeded", "failed", "cancelled", "canceled"}


def _cancel_backoff_ready(provider_state: dict | None, now_epoch: float) -> bool:
    if not provider_state:
        return True
    next_at = provider_state.get("_next_cancel_retry_at")
    if not isinstance(next_at, (int, float)):
        return True
    return now_epoch >= float(next_at)


def _next_cancel_backoff_epoch(attempts: int, now_epoch: float) -> float:
    if attempts <= 1:
        return 0.0
    delay = min(10.0, 0.5 * (2 ** min(attempts - 2, 5)))
    return now_epoch + delay


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
        with self._lock, self.tasks.exclusive():
            return method(self, *args, **kwargs)
    return invoke


def task_serialized(method):
    @wraps(method)
    def invoke(self, video_task_id, *args, **kwargs):
        # Submission still owns the global identity lock. A newly written task
        # must not be polled before its dispatch receipt has been persisted,
        # while already-running peers remain pollable so stage slots can settle.
        if method.__name__ == "status":
            root_executor = getattr(self, "executor", self)
            defer_fn = getattr(root_executor, "_should_defer_status", None)
            if defer_fn is not None:
                if defer_fn(video_task_id):
                    return self.tasks.get(video_task_id)
            else:
                if not self._lock.acquire(blocking=False):
                    return self.tasks.get(video_task_id)
                self._lock.release()
        # Status reads remain responsive while this task's result is downloading.
        with self.tasks.task_lock(video_task_id, blocking=method.__name__ != "status") as acquired:
            if not acquired:
                return self.tasks.get(video_task_id)
            return method(self, video_task_id, *args, **kwargs)
    return invoke


class VideoExecutor:
    def __init__(self, artifacts: ArtifactStore, tasks: TaskStore, client: httpx.Client | None = None) -> None:
        from .catalog import ModelCatalog
        self.catalog = ModelCatalog()
        self._lock = threading.RLock()
        self._state_guard = threading.Lock()
        self._reserving_count = 0
        self._submitting_tasks: set[str] = set()
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
        trust = ssl.create_default_context()
        if os.environ.get('VIDEO_PROJECT_CA_FILE'):
            trust.load_verify_locations(os.environ['VIDEO_PROJECT_CA_FILE'])
        self.client = client or httpx.Client(timeout=httpx.Timeout(connect=10, read=3600, write=600, pool=10), follow_redirects=False, verify=trust)
        self.gpu = GpuCoordinator.from_environment()
        self.stage_slot = LocalGpuStageSlot.from_environment(self.tasks)
        from .depth import DepthExecutor
        self.depth = DepthExecutor(self)
        from .processing import ProcessingExecutor
        self.processing = {name: ProcessingExecutor(self, name) for name in ("sr", "interpolate", "regenerate_2k")}
        from .h3_latent import H3LatentExecutor
        self.processing["h3-latent-upscale"] = H3LatentExecutor(self)
        from .providers.fal import FalAdapter
        self.fal = FalAdapter(self)
        from .providers.grok import GrokVideoAdapter
        self.grok = GrokVideoAdapter(self)
        from .providers.google import GoogleVideoAdapter
        self.google = GoogleVideoAdapter(self)
        from .providers.higgsfield import HiggsfieldAdapter
        self.higgsfield = HiggsfieldAdapter(self)
        from .delivery import Delivery
        self.delivery = Delivery(self)
        from .collection import ResultCollector
        self.collector = ResultCollector(self)

    def _begin_reservation(self) -> None:
        with self._state_guard:
            self._reserving_count += 1

    def _end_reservation(self) -> None:
        with self._state_guard:
            self._reserving_count = max(0, self._reserving_count - 1)

    def _should_defer_status(self, video_task_id: str) -> bool:
        with self._state_guard:
            if self._reserving_count > 0 or video_task_id in self._submitting_tasks:
                return True
        if self._lock.acquire(blocking=False):
            self._lock.release()
            return False
        with self._state_guard:
            if self._submitting_tasks and self._reserving_count == 0:
                return False
        return True

    def _reconcile_slot_holder(self, holder_task_id: str, holder_service: str) -> None:
        try:
            if holder_service in self.processing:
                self.processing[holder_service].status(holder_task_id)
            else:
                self.status(holder_task_id)
        except Exception:
            pass

    def acquire_local_gpu(self, task_id: str, service: str, *, record: TaskRecord | None = None) -> bool:
        """Non-blocking local GPU slot + K8s lease admission.

        Returns True when immediately admitted (or when local GPU gating is disabled),
        False when busy and still within GPU_QUEUE_SECONDS (caller keeps task queued asynchronously),
        or raises LeaseBusy once the task's durable queue_started_at exceeds GPU_QUEUE_SECONDS.
        """
        if service not in LOCAL_GPU_SERVICES:
            return True
        if not self.gpu.enabled and not self.stage_slot.enabled:
            return True
        from .delivery import _to_epoch

        rec = record or self.tasks.get(task_id)
        pstate = dict(rec.provider_state or {})
        monotonic_fn = self.gpu._monotonic if self.gpu.enabled else self.stage_slot._monotonic
        queue_seconds = self.gpu.queue_seconds if self.gpu.enabled else self.stage_slot.queue_seconds
        now_mono = monotonic_fn()
        now_wall = time.time()
        q_started_iso = pstate.get("queue_started_at") or rec.created_at
        q_wall = _to_epoch(q_started_iso)
        elapsed = max(0.0, now_wall - q_wall) if q_wall is not None else 0.0
        q_mono = pstate.get("_queue_started_mono")
        if isinstance(q_mono, (int, float)):
            elapsed = max(elapsed, max(0.0, now_mono - float(q_mono)))

        if self.stage_slot.enabled:
            if not self.stage_slot.try_acquire(task_id, service):
                current = self.stage_slot.current_holder()
                if current and current[0] != task_id:
                    self._reconcile_slot_holder(current[0], current[1])
                if not self.stage_slot.try_acquire(task_id, service):
                    if queue_seconds <= 0 or elapsed >= queue_seconds:
                        holder_name = current[0] if current else "local_gpu_stage_slot"
                        raise LeaseBusy(holder_name)
                    return False

        if self.gpu.enabled:
            if not self.gpu.try_acquire():
                if self.stage_slot.enabled:
                    self.stage_slot.release(task_id, force=True)
                if queue_seconds <= 0 or elapsed >= queue_seconds:
                    raise LeaseBusy(self.gpu.lease_name)
                return False
        return True

    def defer_local_gpu_queue(self, record: TaskRecord, *, waiting_reason: str | None = None) -> TaskRecord:
        pstate = dict(record.provider_state or {})
        now_iso = datetime.now(UTC).isoformat()
        monotonic_fn = self.gpu._monotonic if self.gpu.enabled else self.stage_slot._monotonic
        if waiting_reason is None:
            if self.stage_slot.enabled and self.stage_slot.current_holder() is not None:
                waiting_reason = "gpu_stage_slot"
            else:
                waiting_reason = "gpu_lease"
        pstate.setdefault("queue_started_at", record.created_at or now_iso)
        pstate.setdefault("_queue_started_mono", monotonic_fn())
        pstate["waiting_reason"] = waiting_reason
        pstate["gpu_dispatched"] = False
        return self.tasks.update(record, status="queued", error=None, provider_state=pstate)

    def dispatch_queued_local_tasks(self) -> None:
        root = getattr(self.tasks, "root", None)
        if root is None or not root.is_dir():
            return
        queued_candidates: list[tuple[str, str]] = []
        try:
            with os.scandir(root) as entries:
                for entry in entries:
                    if not entry.name.startswith("video_task_") or not entry.name.endswith(".json"):
                        continue
                    try:
                        rec = TaskRecord.model_validate_json(Path(entry.path).read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    if (
                        rec.service in LOCAL_GPU_SERVICES
                        and rec.status == "queued"
                        and (rec.provider_state or {}).get("gpu_dispatched") is False
                    ):
                        queued_candidates.append((rec.created_at or "", rec.video_task_id))
        except OSError:
            return
        queued_candidates.sort()
        for _, tid in queued_candidates:
            try:
                self.status(tid)
            except Exception:
                continue

    def release_or_reconcile_local_gpu(self, task_id: str) -> None:
        try:
            record = self.tasks.get(task_id)
            if self.stage_slot.is_task_settled(record):
                self.stage_slot.release(task_id)
        except Exception:
            pass
        if self.gpu.enabled:
            try:
                self.reconcile_gpu()
            except Exception:
                logging.getLogger(__name__).warning("GPU lease reconciliation deferred after dispatch")

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
        return ("google" if route == "google" else "grok" if route == "grok" else "fal" if route == "fal" else
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

    def catalog_view(self) -> dict[str, object]:
        from .channel_catalog import project
        return project(self)

    def require_channel(self, route: str, model: str) -> None:
        """New submissions only; callers check their stored idempotency record first."""
        from .channel_errors import ChannelError, safe_code
        catalog = self.catalog_view()
        entries = [entry for entry in catalog.get("execution", {}).get(route, {}).values()
                   if entry.get("model") == model]
        if any(entry.get("can_submit") is True for entry in entries):
            return
        for entry in entries:
            reason = entry.get('reason_code')
            if entry.get('state') in {'disabled', 'catalog_only'} and safe_code(reason, route):
                raise ChannelError(reason, source=route, accepted=False)
        raise ChannelError("channel_unavailable", source=route, accepted=False)

    def capabilities(self) -> dict[str, object]:
        """Separate configured adapters, live readiness and unverified GPU combinations."""
        from .providers.fal import FAL_ASPECT_RATIOS, FAL_LIP_PUBLIC_MODEL
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
            if route == "fal":
                routes[-1]["modes"] = ["reference_to_video"]
            if service == "h3-singularity":
                default_qp = default_singularity_quality_profile()
                routes[-1].update(quality_profiles=list(SINGULARITY_QUALITY_PROFILES), gpu_verified_combinations=[],
                                  verification_status="not_recorded", frame_grid="17k+5", fps=24,
                                  modes=["reference_to_video"],
                                  image_guidance={"modes": ["reference"], "default": "reference",
                                      "experimental": True, "gpu_verified": False, "exact_pixel_preservation": False},
                                  audio_guidance={"modes": ["reference"], "default": "reference",
                                      "timeline_experimental": True, "gpu_verified": False,
                                      "timeline_audio_format": "pcm_wav", "timeline_maximum_requested_seconds": 14,
                                      "exact_audio_preservation": False, "lip_sync_verified": False},
                                  defaults={"mode": "reference_to_video", "duration_seconds": 5,
                                            "aspect_ratio": "16:9", "quality_profile": default_qp},
                                  reference_limits={"images": 9, "videos": 3, "audios": 3,
                                      "minimum_total": 1, "total": 12, "max_file_bytes": MAX_ARTIFACT_BYTES,
                                      "total_duration_seconds": {"videos": 15, "audios": 15}})
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
                            from .singularity import runtime_supports_timeline, runtime_supports_first_frame
                            if runtime_supports_first_frame(health):
                                routes[-1]["image_guidance"]["modes"].append("first_frame")
                            if runtime_supports_timeline(health):
                                routes[-1]["audio_guidance"]["modes"].append("timeline")
                    except (httpx.HTTPError, ValueError, AttributeError):
                        routes[-1]["readiness"] = "unreachable"
        routes.append(self.grok.capability())
        routes.append(self.google.capability())
        routes.append({"route": "fal", "model": "bytedance-seedance-2.0-reference-to-video",
                       "configured": self.fal.connected(), "readiness": "not_checked",
                       "limit_scope": "adapter", "duration_seconds": {"minimum": 4, "maximum": 15},
                       "aspect_ratios": ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
                       "modes": ["reference_to_video"],
                       "resolutions": ["480p", "720p"]})
        routes.extend(self.higgsfield.capabilities())
        routes.append({"route": "fal", "model": FAL_LIP_PUBLIC_MODEL,
                       "configured": self.fal.connected() and self.fal.lip_sync_declared(),
                       "readiness": "not_checked", "limit_scope": "adapter",
                       "modes": ["lip_sync"], "duration_seconds": {"minimum": 5, "maximum": 15},
                       "duration_basis": "source_audio", "aspect_ratios": ["adaptive"],
                       "resolutions": ["480p", "768p"], "prompt_usage": "task_description_only",
                       "reference_limits": {"images": 1, "audios": 1, "videos": 0}})
        catalog = self.catalog_view()
        dynamic_models = []
        for channel, entries in catalog.get("execution", {}).items():
            for entry in entries.values():
                if entry.get("model"):
                    dynamic_models.append({"model": entry["model"], "routes": [channel],
                        "execution_state": entry["state"], "can_submit": entry.get("can_submit", False)})
        external = {"fal", "grok", "google", "higgsfield"}
        from .channel_specs import external_routes
        route_templates = [item for item in routes if item["route"] in external]
        routes = [item for item in routes if item["route"] not in external]
        routes.extend(external_routes(route_templates, catalog))
        return {"catalog": catalog, "channel_capabilities": catalog.get("channel_capabilities", {}),
                "default_model": "minimax-h3-ref2va", "default_route": self.runtime_route,
                "models": [{"model": "minimax-h3-ref2va"}, *dynamic_models],
                "model_specification": {"aspect_ratios": ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
                                        "omni_adaptive": True},
                "postprocess": {
                    "interpolate": self.delivery.capability(),
                    "regenerate_2k": self.delivery.regenerate_capability(),
                },
                "routes": routes, "inline_image_max_bytes": MAX_INLINE_IMAGE_BYTES,
                "chunk_import": {"max_chunk_bytes": 512 * 1024, "max_file_bytes": MAX_ARTIFACT_BYTES,
                                 "formats": ["PNG", "JPEG", "WebP", "MP4/H.264", "MP4/H.265", "WAV/PCM", "MP3"],
                                 "reference_duration_seconds": {"minimum": 2, "maximum": 15},
                                 "max_image_pixels": 16 * 1024 * 1024, "max_video_pixels": 4096 * 2160,
                                 "max_video_decoded_pixels": 1_000_000_000, "max_video_fps": 120,
                                 "audio_channels": [1, 2]},
                "content_rating": {
                    "supported": True,
                    "tool": "video.rate_content",
                    "rubric_version": "rubric v1.0",
                    "add_only": True,
                }}

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
                   route: str = "h3", quality_profile: str = "du-0", audio_guidance: str | None = None, image_guidance: str | None = None) -> tuple[dict[str, object], str]:
        if not isinstance(route, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", route):
            raise ValueError("route must be a lowercase slug")
        service = self._service_for_route(route)
        from .singularity import check_audio_guidance, check_image_guidance
        check_audio_guidance(audio_guidance, service)
        check_image_guidance(image_guidance, service)
        if quality_profile not in {"fasth3-nvfp4", "du-0", "du-1", "du-2", "du-3", "hr-refine-tile-v1", "hr-refine-global-v2"}:
            raise ValueError("quality_profile_not_available")
        if service != "h3-singularity" and quality_profile != "du-0":
            raise ValueError("quality_profile fasth3-nvfp4, du-1, du-2, du-3, hr-refine-tile-v1, and hr-refine-global-v2 require h3-singularity")
        if service in {"h3-sol", "h3-vdn"} and not 5 <= duration_seconds <= 15:
            raise ValueError("Selected channel duration must be an integer from 5 to 15 seconds")
        if model != "minimax-h3-ref2va":
            raise ValueError("model must be minimax-h3-ref2va")
        if type(duration_seconds) is not int or not 4 <= duration_seconds <= 15 or not prompt.strip():
            raise ValueError("duration_seconds must be an integer from 4 to 15 and prompt must not be empty")
        from .singularity import ASPECT_RATIOS, output_spec
        if service == "h3-singularity" and quality_profile not in SINGULARITY_QUALITY_PROFILES:
            raise ValueError("Selected Singularity production route supports only du-0 or fasth3-nvfp4")
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
        if audio_guidance == "timeline" and (len(references.get("audios", [])) != 1 or references.get("videos")):
            raise ValueError("timeline_requires_one_audio_and_no_video")
        if audio_guidance == "timeline" and duration_seconds > 14:
            raise ValueError("timeline_audio_requested_seconds_must_be_4_to_14")
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
        if image_guidance is not None:
            request["image_guidance"] = image_guidance
        if audio_guidance is not None:
            request["audio_guidance"] = audio_guidance
        if service == "h3-singularity":
            for kind in ("videos", "audios"):
                if sum(m["duration_seconds"] for m in media_by_kind[kind]) > 15.001:
                    raise ValueError(f"{kind} references exceed 15 seconds in total")
            visuals = media_by_kind["images"] or media_by_kind["videos"]
            request["resolved_output"] = output_spec(aspect_ratio, duration_seconds, visuals[0] if visuals else None)
            if image_guidance == "first_frame":
                if not media_by_kind["images"]:
                    raise ValueError("first_frame_requires_image")
                image = media_by_kind["images"][0]
                if any(image[k] != request["resolved_output"][k] for k in ("width", "height")):
                    raise ValueError("first_frame_canvas_mismatch")
            if audio_guidance == "timeline":
                from .singularity import validate_timeline_wav
                audio = self.artifacts.get(normalized["audios"][0]["artifact_id"], project_id)
                if audio.media_type != "audio/wav":
                    raise ValueError("timeline_audio_requires_pcm_wav")
                request["timeline_audio"] = validate_timeline_wav(self.artifacts.content_path(audio), request["resolved_output"]["frames"])
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

    def generate(self, *, project_id: str, idempotency_key: str, model: str, prompt: str,
                 duration_seconds: int, aspect_ratio: str,
                 references: dict[str, list[dict[str, str]]], route: str | None = None,
                 quality_profile: str | None = None, mode: str | None = None,
                 resolution: str | None = None, postprocess: dict | None = None,
                 channel_account_id: str | None = None, budget_limit: str | None = None,
                 price_ceiling: float | None = None, audio_guidance: str | None = None, image_guidance: str | None = None) -> TaskRecord:
        with self._lock:
            project_id = require_project_id(project_id)
            if not idempotency_key.strip() or len(idempotency_key) > 128:
                raise ValueError("idempotency_key is invalid")
            route = route or self.runtime_route
            service = self._service_for_route(route)
            from .singularity import check_audio_guidance, check_image_guidance
            check_audio_guidance(audio_guidance, service)
            check_image_guidance(image_guidance, service)
            if route != "higgsfield" and price_ceiling is not None:
                raise ValueError("price_ceiling_not_supported")
            if route not in {'fal', 'grok', 'google'} and (channel_account_id is not None or budget_limit is not None):
                raise ValueError('channel_budget_not_supported')
            from .delivery import normalize, authorize
            normalize(postprocess, route)
            if route == "higgsfield":
                with self.tasks.exclusive():
                    return self.higgsfield.generate(project_id=project_id, idempotency_key=idempotency_key,
                        model=model, prompt=prompt, duration_seconds=duration_seconds, aspect_ratio=aspect_ratio,
                        references=references, quality_profile=quality_profile, mode=mode, resolution=resolution,
                        price_ceiling=price_ceiling)
            if route == "grok":
                with self.tasks.exclusive():
                    return self.grok.generate(project_id=project_id, idempotency_key=idempotency_key,
                        model=model, prompt=prompt, duration_seconds=duration_seconds, aspect_ratio=aspect_ratio,
                        references=references, quality_profile=quality_profile, mode=mode, resolution=resolution,
                        channel_account_id=channel_account_id, budget_limit=budget_limit)
            if route == "google":
                with self.tasks.exclusive():
                    return self.google.generate(project_id=project_id, idempotency_key=idempotency_key,
                        model=model, prompt=prompt, duration_seconds=duration_seconds, aspect_ratio=aspect_ratio,
                        references=references, quality_profile=quality_profile, mode=mode, resolution=resolution,
                        channel_account_id=channel_account_id, budget_limit=budget_limit)
            model = model or "minimax-h3-ref2va"
            aspect_ratio = aspect_ratio or "16:9"
            if route == "fal":
                resolved_qp = quality_profile or "du-0"
                from .providers.fal import FAL_LIP_PUBLIC_MODEL
                allowed_modes = {None, "lip_sync"} if model == FAL_LIP_PUBLIC_MODEL else {None, "reference_to_video"}
                if mode not in allowed_modes or resolved_qp != "du-0":
                    raise ValueError("unsupported_parameter")
                with self.tasks.exclusive():
                    return self.fal.generate(project_id=project_id, idempotency_key=idempotency_key,
                                             model=model, prompt=prompt, duration_seconds=duration_seconds,
                                             aspect_ratio=aspect_ratio, references=references,
                                             channel_account_id=channel_account_id, budget_limit=budget_limit,
                                             resolution=resolution)
            allowed_modes = {None, "reference_to_video"} if service == "h3-singularity" else {None}
            if mode not in allowed_modes or resolution is not None:
                raise ValueError("unsupported_parameter")

            self._begin_reservation()
            try:
                with self.tasks.exclusive():
                    existing = self.tasks.find_idempotency(project_id, idempotency_key)
                    if quality_profile is None:
                        if existing is not None and isinstance(existing.request.get("quality_profile"), str):
                            resolved_qp = existing.request["quality_profile"]
                        elif service == "h3-singularity":
                            resolved_qp = default_singularity_quality_profile()
                        else:
                            resolved_qp = "du-0"
                    else:
                        resolved_qp = quality_profile

                    request, digest = self._normalize(
                        project_id, model, prompt, duration_seconds, aspect_ratio,
                        references, route, resolved_qp, audio_guidance, image_guidance,
                    )
                    request, digest = self.delivery.prepare(request, digest, postprocess, check=False)
                    authority = authorize(self.client, project_id) if postprocess is not None else None
                    if existing:
                        if existing.input_digest != digest:
                            raise TaskConflict("idempotency key already exists with different input")
                        return existing
                    if postprocess is not None:
                        self.delivery.prepare(request, digest, postprocess)
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
                    steps = (
                        SINGULARITY_PROFILE_NFE[resolved_qp]
                        if service == "h3-singularity"
                        else (4 if service in {"h3-vdn", "h3-sol"} else 21)
                    )
                    payload = {"model": "MiniMaxAI/MiniMax-H3", "task": "ref2va", "prompt": compiled_prompt,
                               "seconds": duration_seconds, "conditions": conditions,
                               "target": {"short_edge": 768, "aspect_ratio": aspect_ratio, "duration_seconds": float(duration_seconds)},
                               "num_outputs_per_prompt": 1, "num_inference_steps": steps, "flow_shift": 12.0,
                               "audio_flow_shift": 3.0, "seed": 7,
                               "quality_profile": resolved_qp}
                    if "image_guidance" in request:
                        payload["image_guidance"] = request["image_guidance"]
                    if "audio_guidance" in request:
                        payload["audio_guidance"] = request["audio_guidance"]
                    # Persist the attempt before calling the runtime. An ambiguous network
                    # failure must not permit a duplicate GPU request under the same key.
                    reserved = self.tasks.create(project_id=project_id, idempotency_key=idempotency_key,
                                                 input_digest=digest, request=request, runtime_task_id="", status="queued",
                                                 authority=authority, runtime_endpoint=runtime_url,
                                                 runtime_version=runtime_version, runtime_route=runtime_route)
                    reserved = self.tasks.update(reserved, service=service, runtime_version=runtime_version, runtime_route=runtime_route,
                                                 runtime_endpoint=runtime_url, authority=authority,
                                                 delivery=self.delivery.initial(reserved) if postprocess is not None else None)
                    with self._state_guard:
                        self._submitting_tasks.add(reserved.video_task_id)
            finally:
                self._end_reservation()

        if service == "h3-vdn":
            payload["project_id"] = project_id
        if service in {"h3-sol", "h3-vdn", "h3-singularity"}:
            payload["idempotency_key"] = reserved.video_task_id
        return self._dispatch_local_generation(reserved, payload=payload, already_marked_submitting=True)

    def _build_local_generation_payload(self, record: TaskRecord) -> dict[str, object]:
        service = record.service
        project_id = record.project_id
        request = record.request
        resolved_qp = request["quality_profile"]
        duration_seconds = request["duration_seconds"]
        aspect_ratio = request["aspect_ratio"]
        conditions = []
        material_tags = []
        singular = {"images": "image", "videos": "video", "audios": "audio"}
        tag_name = {"images": "Picture", "videos": "Video", "audios": "Audio"}
        kinds = ("images", "audios", "videos") if service == "h3-vdn" else ("images", "videos", "audios")
        for kind in kinds:
            for index, item in enumerate(request["references"][kind], start=1):
                conditions.append({
                    "type": singular[kind],
                    "uri": f"{self.runtime_artifact_url}/runtime-artifacts/{item['artifact_id']}/content",
                    "role": "reference",
                })
                if service in {"h3-sol", "h3-vdn", "h3-singularity"}:
                    asset = self.artifacts.get(item["artifact_id"], project_id)
                    conditions[-1].update(sha256=asset.sha256, size=asset.size)
                material_tags.append(f"<{tag_name[kind]} {index}> is the approved {item['purpose']} reference")
        compiled_prompt = (
            request["prompt"]
            if service == "h3-singularity"
            else "; ".join(material_tags) + ". " + request["prompt"]
        )
        steps = (
            SINGULARITY_PROFILE_NFE[resolved_qp]
            if service == "h3-singularity"
            else (4 if service in {"h3-vdn", "h3-sol"} else 21)
        )
        payload: dict[str, object] = {
            "model": "MiniMaxAI/MiniMax-H3",
            "task": "ref2va",
            "prompt": compiled_prompt,
            "seconds": duration_seconds,
            "conditions": conditions,
            "target": {
                "short_edge": 768,
                "aspect_ratio": aspect_ratio,
                "duration_seconds": float(duration_seconds),
            },
            "num_outputs_per_prompt": 1,
            "num_inference_steps": steps,
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
            "seed": 7,
            "quality_profile": resolved_qp,
        }
        if "image_guidance" in request:
            payload["image_guidance"] = request["image_guidance"]
        if "audio_guidance" in request:
            payload["audio_guidance"] = request["audio_guidance"]
        if service == "h3-vdn":
            payload["project_id"] = project_id
        if service in {"h3-sol", "h3-vdn", "h3-singularity"}:
            payload["idempotency_key"] = record.video_task_id
        return payload

    def _dispatch_local_generation(
        self,
        record: TaskRecord,
        *,
        payload: dict[str, object] | None = None,
        already_marked_submitting: bool = False,
    ) -> TaskRecord:
        if not already_marked_submitting:
            with self._state_guard:
                self._submitting_tasks.add(record.video_task_id)

        service = record.service
        route = self._record_route(record)
        runtime_url, runtime_headers, _, _ = self.runtime(route)
        runtime_url = record.runtime_endpoint or runtime_url
        image_guidance = record.request.get("image_guidance")
        audio_guidance = record.request.get("audio_guidance")

        def _fail_pre_submit(code: str, message: str, *, retryable: bool = False) -> TaskRecord:
            err: dict[str, object] = {"code": code, "message": message}
            if retryable:
                err["retryable"] = True
            cleaned_pstate = {
                k: v
                for k, v in (record.provider_state or {}).items()
                if k not in {"waiting_reason", "gpu_dispatched"}
            }
            cleaned_pstate["worker_settled"] = True
            updates: dict[str, object] = {
                "status": "failed",
                "error": err,
                "provider_state": cleaned_pstate,
            }
            if record.delivery is not None:
                updates["delivery"] = {
                    **record.delivery,
                    "status": "failed",
                    "stage": "generation",
                    "error": err,
                    "completed_at": datetime.now(UTC).isoformat(),
                }
            return self.tasks.update(record, **updates)

        acquired = False
        try:
            if service in LOCAL_H3_SERVICES and (self.gpu.enabled or self.stage_slot.enabled):
                try:
                    acquired = self.acquire_local_gpu(record.video_task_id, service, record=record)
                except LeaseBusy:
                    return _fail_pre_submit("gpu_busy", MESSAGES["gpu_busy"], retryable=True)
                if not acquired:
                    return self.defer_local_gpu_queue(record)

            if service in {"h3-vdn", "h3-singularity"}:
                try:
                    health = self.client.get(f"{runtime_url}/health", headers=runtime_headers, timeout=10)
                    if health.status_code == 503:
                        return _fail_pre_submit(
                            "runtime_not_ready",
                            "H3 runtime is not ready; generation was not submitted",
                            retryable=True,
                        )
                    health.raise_for_status()
                    health_data = health.json()
                    if not isinstance(health_data, dict) or not isinstance(health_data.get("ready"), bool):
                        raise ValueError("Invalid readiness response")
                    if image_guidance == "first_frame":
                        from .singularity import runtime_supports_first_frame
                        if not runtime_supports_first_frame(health_data):
                            return _fail_pre_submit(
                                "first_frame_not_supported",
                                "Selected runtime does not support first-frame guidance; generation was not submitted",
                            )
                    if audio_guidance == "timeline":
                        from .singularity import runtime_supports_timeline
                        if not runtime_supports_timeline(health_data):
                            return _fail_pre_submit(
                                "timeline_audio_not_supported",
                                "Selected runtime does not support timeline audio; generation was not submitted",
                            )
                    if not health_data["ready"]:
                        return _fail_pre_submit(
                            "runtime_not_ready",
                            "H3 runtime is not ready; generation was not submitted",
                            retryable=True,
                        )
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    return _fail_pre_submit(
                        "runtime_unavailable",
                        "H3 runtime is unreachable; generation was not submitted",
                    )
                except (httpx.HTTPError, ValueError):
                    return _fail_pre_submit(
                        "runtime_health_check_failed",
                        "H3 readiness check failed; generation was not submitted",
                    )

            if payload is None:
                payload = self._build_local_generation_payload(record)

            dispatched_pstate = {
                k: v for k, v in (record.provider_state or {}).items() if k != "waiting_reason"
            }
            dispatched_pstate["gpu_dispatched"] = True
            record = self.tasks.update(record, provider_state=dispatched_pstate)

            try:
                response = self.client.post(f"{runtime_url}/v1/videos", json=payload, headers=runtime_headers)
                response.raise_for_status()
                runtime_task_id = response.json()["id"]
                if not isinstance(runtime_task_id, str) or not runtime_task_id.strip():
                    raise ValueError("Invalid runtime task id")
                return self.tasks.update(
                    record,
                    runtime_task_id=runtime_task_id,
                    status="queued",
                    error=None,
                    dispatched_at=datetime.now(UTC).isoformat(),
                )
            except (httpx.ConnectError, httpx.ConnectTimeout):
                return _fail_pre_submit(
                    "runtime_unavailable",
                    "H3 runtime is unreachable; generation was not submitted",
                )
            except httpx.HTTPStatusError as error:
                if 400 <= error.response.status_code < 500:
                    return _fail_pre_submit(
                        "runtime_rejected",
                        f"Runtime rejected submission (HTTP {error.response.status_code})",
                    )
                return self.tasks.update(
                    record,
                    status="queued" if service == "h3-singularity" else "failed",
                    error={
                        "code": "submission_unconfirmed",
                        "message": "Runtime submission could not be confirmed; do not resubmit automatically",
                    },
                )
            except (httpx.HTTPError, KeyError, ValueError, TypeError):
                return self.tasks.update(
                    record,
                    status="queued" if service == "h3-singularity" else "failed",
                    error={
                        "code": "submission_unconfirmed",
                        "message": "Runtime submission could not be confirmed; do not resubmit automatically",
                    },
                )
        finally:
            with self._state_guard:
                self._submitting_tasks.discard(record.video_task_id)
            if acquired:
                # A failed HTTP observation is not evidence of stopped GPU
                # work. Reconcile the durable task set, including unknowns.
                self.release_or_reconcile_local_gpu(record.video_task_id)

    def reconcile_gpu(self) -> None:
        if not self.gpu.enabled:
            return
        with self._state_guard:
            in_flight = self._reserving_count > 0 or bool(self._submitting_tasks)
        if in_flight:
            self.gpu.renew_existing()
            return
        if not self._lock.acquire(blocking=False):
            self.gpu.renew_existing()
            return
        try:
            try:
                active = self._local_h3_active()
            except Exception:
                self.gpu.renew_existing()
                raise
            if active:
                if not self.gpu.try_acquire():
                    logging.getLogger(__name__).warning("GPU lease recovery waiting for existing holder")
            else:
                self.gpu.release()
        finally:
            self._lock.release()
        self.dispatch_queued_local_tasks()

    def _local_h3_active(self) -> bool:
        root = getattr(self.tasks, "root", None)
        if root is None or not root.is_dir():
            raise ExecutionError("GPU task reconciliation storage is unavailable")
        saw_interpolate = False
        unsettled_records: list[TaskRecord] = []
        # scandir propagates unreadable-directory errors; glob can silently
        # produce an empty collection and incorrectly imply an idle backend.
        with os.scandir(root) as entries:
            for entry in entries:
                if not entry.name.startswith("video_task_") or not entry.name.endswith(".json"):
                    continue
                record = TaskRecord.model_validate_json(Path(entry.path).read_text(encoding="utf-8"))
                if (record.provider_state or {}).get("gpu_dispatched") is False and not record.runtime_task_id:
                    continue
                if record.service == "interpolate":
                    saw_interpolate = True
                if record.service in LOCAL_GPU_SERVICES and (
                    record.status not in TERMINAL_TASK_STATUS or
                    (record.status == "failed" and (record.error or {}).get("code") in
                     {"submission_unconfirmed", "runtime_task_lost"})
                ):
                    return True
                if (record.delivery or {}).get("status") in {"queued", "running"} and (
                    record.service in LOCAL_GPU_SERVICES or record.request.get("route") == "h3-singularity"
                ):
                    return True
                if record.service in LOCAL_GPU_SERVICES and (record.provider_state or {}).get("worker_settled") is False:
                    unsettled_records.append(record)

        for record in unsettled_records:
            try:
                if record.service in self.processing:
                    self.processing[record.service].status(record.video_task_id)
                else:
                    self.status(record.video_task_id)
            except Exception:
                return True
            refreshed = self.tasks.get(record.video_task_id)
            if (refreshed.provider_state or {}).get("worker_settled") is False:
                return True

        health_targets: list[tuple[str, dict[str, str]]] = []
        try:
            s_url, s_headers, _, _ = self.runtime("h3-singularity")
            health_targets.append((f"{s_url}/health", s_headers))
        except ExecutionError:
            pass
        if (
            saw_interpolate
            or "VIDEO_INTERPOLATION_RUNTIME_URL" in os.environ
            or "VIDEO_PROJECT_SERVICE_URL" in os.environ
        ):
            health_targets.append((f"{self.processing['interpolate'].url}/health", {}))

        for h_url, h_headers in health_targets:
            try:
                resp = self.client.get(h_url, headers=h_headers, timeout=5)
                if resp.status_code == 200:
                    body = resp.json()
                    if isinstance(body, dict) and body.get("gpu_worker_active") is True:
                        return True
            except Exception:
                continue
        return False

    def cancel_generation(self, record: TaskRecord, *, timeout_error: dict[str, object] | None = None) -> TaskRecord:
        record = self.tasks.get(record.video_task_id)
        prev_state = dict(record.provider_state or {})
        now_epoch = time.time()
        attempts = int(prev_state.get("_cancel_attempts", 0)) + 1
        worker_settled = bool(prev_state.get("worker_settled", False))
        cancel_delivered = bool(prev_state.get("cancel_delivered", False))
        started_at = prev_state.get("started_at")
        finished_at = prev_state.get("finished_at")
        force_backoff = False

        if prev_state.get("gpu_dispatched") is False or not record.runtime_task_id:
            worker_settled = True
            cancel_delivered = True
        else:
            runtime_url, runtime_headers, _, _ = self.runtime(self._record_route(record))
            runtime_url = record.runtime_endpoint or runtime_url
            try:
                resp = self.client.delete(
                    f"{runtime_url}/v1/videos/{record.runtime_task_id}",
                    headers=runtime_headers,
                    timeout=15,
                )
                if resp.status_code == 200:
                    body = resp.json()
                    worker_settled = bool(body.get("worker_settled", False))
                    started_at = body.get("started_at", started_at)
                    finished_at = body.get("finished_at", finished_at)
                    if (
                        worker_settled
                        or body.get("cancel_requested") is True
                        or body.get("status") in {"cancelled", "canceled", "failed", "succeeded", "completed"}
                    ):
                        cancel_delivered = True
                    else:
                        force_backoff = True
                elif resp.status_code == 409:
                    cancel_delivered = True
                    poll = self.client.get(
                        f"{runtime_url}/v1/videos/{record.runtime_task_id}",
                        headers=runtime_headers,
                        timeout=10,
                    )
                    if poll.status_code == 200:
                        body = poll.json()
                        worker_settled = bool(body.get("worker_settled", True))
                        started_at = body.get("started_at", started_at)
                        finished_at = body.get("finished_at", finished_at)
                elif resp.status_code == 404:
                    worker_settled = True
                    cancel_delivered = True
                else:
                    worker_settled = False
            except (httpx.HTTPError, ValueError, TypeError):
                worker_settled = False

        next_retry_at = (
            0.0
            if cancel_delivered
            else _next_cancel_backoff_epoch(max(attempts, 2) if force_backoff else attempts, now_epoch)
        )
        provider_state = {
            **prev_state,
            "started_at": started_at,
            "finished_at": finished_at,
            "worker_settled": worker_settled,
            "cancel_delivered": cancel_delivered,
            "_cancel_requested": True,
            "_cancel_attempts": attempts,
            "_next_cancel_retry_at": next_retry_at,
        }
        provider_state.pop("waiting_reason", None)
        if prev_state.get("gpu_dispatched") is False:
            provider_state.pop("gpu_dispatched", None)
        if timeout_error is not None and not worker_settled:
            provider_state["_pending_timeout_error"] = timeout_error
            target_status = "running"
            target_error = None
        else:
            provider_state.pop("_pending_timeout_error", None)
            target_status = "failed" if timeout_error is not None else "cancelled"
            target_error = timeout_error or {"code": "cancelled", "message": "Video generation was cancelled"}
        updated = self.tasks.update(
            record,
            status=target_status,
            error=target_error,
            provider_state=provider_state,
        )
        if worker_settled:
            self.stage_slot.release(updated.video_task_id)
        return updated

    @task_serialized
    def status(self, video_task_id: str) -> TaskRecord:
        record = self.tasks.get(video_task_id)
        if record.service == "higgsfield":
            return self.higgsfield.status(record)
        if record.service == "grok":
            return self.grok.status(record)
        if record.service == "google":
            return self.google.status(record)
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

        if (
            record.service in LOCAL_H3_SERVICES
            and record.status == "queued"
            and (record.provider_state or {}).get("gpu_dispatched") is False
        ):
            return self._dispatch_local_generation(record)

        pstate_init = record.provider_state or {}
        if (
            record.service in LOCAL_H3_SERVICES
            and record.runtime_task_id
            and pstate_init.get("worker_settled") is False
            and (pstate_init.get("_pending_timeout_error") is not None or pstate_init.get("_cancel_requested") is True)
            and pstate_init.get("cancel_delivered") is not True
            and _cancel_backoff_ready(pstate_init, time.time())
        ):
            record = self.cancel_generation(record, timeout_error=pstate_init.get("_pending_timeout_error"))
            if record.status in {"succeeded", "failed", "cancelled"} and (record.provider_state or {}).get("worker_settled") is not False:
                return record
            if (record.provider_state or {}).get("cancel_delivered") is not True:
                return record

        if (record.service == "h3-singularity" and not record.runtime_task_id
            and (record.status in {"queued", "running"} or (record.error or {}).get("code") == "submission_unconfirmed")):
            url, headers, _, _ = self.runtime(self._record_route(record))
            url = record.runtime_endpoint or url
            try:
                response = self.client.get(f"{url}/v1/videos/by-idempotency/{record.video_task_id}", headers=headers, timeout=10)
                if response.status_code == 404:
                    if record.request.get('postprocess'):
                        # The opted-in workflow owns a durable dispatch intent.
                        # Reusing the exact runtime key closes a crash before POST
                        # as well as a lost response, without duplicate inference.
                        from .delivery import authorize
                        authorize(self.client, record.project_id, record.authority)
                        req = record.request
                        payload = self._build_local_generation_payload(record)
                        if req.get('image_guidance') == 'first_frame':
                            from .singularity import runtime_supports_first_frame
                            health = self.client.get(f'{url}/health', headers=headers, timeout=10)
                            health.raise_for_status()
                            data = health.json()
                            if not runtime_supports_first_frame(data) or data.get('ready') is not True:
                                raise ValueError('first_frame_not_supported_or_not_ready')
                        acquired = False
                        try:
                            acquired = self.acquire_local_gpu(record.video_task_id, record.service, record=record)
                        except LeaseBusy:
                            return self.tasks.update(
                                record,
                                status="failed",
                                provider_state={**(record.provider_state or {}), "worker_settled": True},
                                error={"code": "gpu_busy", "message": MESSAGES["gpu_busy"], "retryable": True},
                            )
                        if not acquired:
                            return self.defer_local_gpu_queue(record)
                        try:
                            response = self.client.post(f'{url}/v1/videos', headers=headers, json=payload, timeout=30)
                        finally:
                            if acquired:
                                self.release_or_reconcile_local_gpu(record.video_task_id)
                    else:
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
            record = self.tasks.update(
                record,
                runtime_task_id=identity,
                status="queued",
                error=None,
                dispatched_at=record.dispatched_at or datetime.now(UTC).isoformat(),
            )
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
            if record.service == "h3-singularity" and record.runtime_task_id and (record.provider_state or {}).get("worker_settled") is False:
                runtime_url, runtime_headers, _, _ = self.runtime(self._record_route(record))
                runtime_url = record.runtime_endpoint or runtime_url
                try:
                    poll = self.client.get(f"{runtime_url}/v1/videos/{record.runtime_task_id}", timeout=10, headers=runtime_headers)
                    if poll.status_code == 200:
                        body = poll.json()
                        pstate = {
                            **(record.provider_state or {}),
                            "worker_settled": bool(body.get("worker_settled", False)),
                        }
                        for key in ("started_at", "finished_at"):
                            if key in body:
                                pstate[key] = body[key]
                        record = self.tasks.update(record, provider_state=pstate)
                        if pstate.get("worker_settled") is True:
                            self.stage_slot.release(record.video_task_id)
                    elif poll.status_code == 404:
                        pstate = {**(record.provider_state or {}), "worker_settled": True}
                        record = self.tasks.update(record, provider_state=pstate)
                        self.stage_slot.release(record.video_task_id)
                except (httpx.HTTPError, ValueError, TypeError):
                    pass
            return record
        runtime_url, runtime_headers, _, _ = self.runtime(self._record_route(record))
        runtime_url = record.runtime_endpoint or runtime_url
        try:
            response = self.client.get(f"{runtime_url}/v1/videos/{record.runtime_task_id}", timeout=10, headers=runtime_headers)
            if response.status_code == 404:
                return self.tasks.update(record, error={
                    "code": "runtime_task_lost",
                    "message": "H3 runtime task could not be found; execution outcome remains unconfirmed",
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
        error = None
        if mapped == "failed":
            rt_code = (runtime_data.get("error") or {}).get("code") if isinstance(runtime_data.get("error"), dict) else None
            if rt_code in {"gpu_memory_occupancy_changed", "interrupted", "runtime_not_ready"}:
                error = {"code": rt_code, "message": MESSAGES.get(rt_code, "H3 generation failed")}
            else:
                error = {"code": "runtime_failed", "message": "H3 generation failed"}
        elif mapped == "cancelled":
            if record.error and str(record.error.get("code", "")).startswith(("stage_timeout_", "chain_active_timeout")):
                error = record.error
            else:
                error = {"code": "cancelled", "message": "H3 generation was cancelled"}
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
        provider_state = dict(record.provider_state or {})
        for key in ("started_at", "finished_at", "worker_settled"):
            if key in runtime_data:
                provider_state[key] = runtime_data[key]
        if mapped in {"succeeded", "failed", "cancelled"} and "worker_settled" not in runtime_data:
            provider_state.setdefault("worker_settled", True)
        pending_timeout = provider_state.get("_pending_timeout_error")
        if pending_timeout is not None:
            if provider_state.get("worker_settled") is True:
                provider_state.pop("_pending_timeout_error", None)
                mapped = "failed"
                error = pending_timeout
            else:
                mapped = "running"
                error = None
        updated = self.tasks.update(
            record,
            status=mapped,
            error=error,
            execution_instance_id=identity,
            runtime_stage=stage,
            runtime_metrics=metrics or None,
            provider_state=provider_state or None,
        )
        if self.stage_slot.is_task_settled(updated):
            self.stage_slot.release(updated.video_task_id)
        if (
            updated.service == "h3-singularity"
            and updated.status == "running"
            and not updated.request.get("postprocess")
            and (updated.provider_state or {}).get("_pending_timeout_error") is None
        ):
            from .delivery import P1_STAGE_TIMEOUT_SECONDS, _to_epoch
            import time as _time
            started_epoch = _to_epoch((updated.provider_state or {}).get("started_at"))
            if started_epoch is not None and (_time.time() - started_epoch) > P1_STAGE_TIMEOUT_SECONDS:
                return self.cancel_generation(
                    updated,
                    timeout_error={
                        "code": "stage_timeout_generation",
                        "message": MESSAGES["stage_timeout_generation"],
                        "retryable": False,
                    },
                )
        return updated

    @task_serialized
    def result(self, video_task_id: str) -> TaskRecord:
        return self.collector.reconcile_cached(self._result(video_task_id))

    def _result(self, video_task_id: str) -> TaskRecord:
        record = self.tasks.get(video_task_id)
        if record.service == "higgsfield":
            return self.higgsfield.result(record)
        if record.service == "grok":
            return self.grok.result(record)
        if record.service == "google":
            return self.google.result(record)
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
        runtime_url = record.runtime_endpoint or runtime_url
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
