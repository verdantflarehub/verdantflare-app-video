from __future__ import annotations

import asyncio
import contextlib
import hmac
from functools import wraps

import httpx
import json
import logging
import os
from typing import Annotated

from pydantic import Field, StrictInt
from mcp import types
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route

from .artifacts import ArtifactError, ArtifactNotFound, ArtifactStore, require_project_id
from . import __version__, titles
from .content_rating import ContentRatingError, ContentRatingStore, include_nsfw_from_headers
from .executor import ExecutionError, VideoExecutor
from .channel_errors import MESSAGES, tool_errors
from .dashboard import Dashboard, _etag_matches, public_admission_reconciliation, public_gpu_queue, public_task
from .registry import EtcdRegistry
from .collection import public_collection
from .tasks import TaskConflict, TaskNotFound, TaskRecord, TaskStore
from .delivery import CALLER, normalize as normalize_postprocess, normalize_regenerate_postprocess, authorize
from .regenerate import ROUTE_QUALITY_PROFILE, is_valid_h3_frame_bucket, resolve_2k_target_geometry, route_quality_profile

logging.getLogger("httpx").setLevel(logging.WARNING)

artifacts = ArtifactStore.from_environment()
tasks = TaskStore.from_environment()
ratings = ContentRatingStore(tasks.db_path, domain="video")
executor = VideoExecutor(artifacts, tasks)
dashboard = Dashboard(executor)
mcp = MCPServer("VerdantFlare Video")


def _active_tasks() -> TaskStore:
    state = getattr(globals().get("app"), "state", None)
    return getattr(state, "tasks", None) or executor.tasks


def _active_artifacts() -> ArtifactStore:
    state = getattr(globals().get("app"), "state", None)
    return getattr(state, "artifacts", None) or executor.artifacts


def _active_ratings() -> ContentRatingStore:
    state = getattr(globals().get("app"), "state", None)
    return getattr(state, "ratings", None) or ratings


def _is_task_nsfw(record) -> bool:
    return _active_tasks().is_record_nsfw(record, _active_ratings())


def _resolve_rating_target(version_id: str, project_id: str | None = None) -> tuple[str, str, str | None]:
    vid = (version_id or "").strip()
    if not vid:
        raise TaskNotFound("NOT_FOUND: empty version_id")
    t_store = _active_tasks()
    a_store = _active_artifacts()
    task_rec = t_store.find_by_artifact_or_task(vid)
    if task_rec is not None:
        if project_id and task_rec.project_id != project_id:
            raise TaskNotFound(f"NOT_FOUND: {vid}")
        primary_vid = (task_rec.delivery or {}).get("final_artifact_id") or task_rec.artifact_id or vid
        resolved_vid = vid if vid != task_rec.video_task_id else primary_vid
        return resolved_vid, task_rec.project_id, task_rec.video_task_id
    try:
        art = a_store.get(vid, project_id)
    except (ArtifactError, ValueError) as e:
        raise TaskNotFound(f"NOT_FOUND: {vid}") from e
    return art.artifact_id, art.project_id, None

IMAGE_GUIDANCE_DESCRIPTION = (
    "Local H3 (h3-fasth3 / h3-singularity) only; experimental frame-0 conditioning. "
    "Exact output canvas required; not pixel preservation. Single-image frame-0 consumption is verified, "
    "but multi-reference cross-style conversion acts as soft conditioning without hard geometry lock "
    "(render a matching photoreal first-frame keyframe before motion generation)."
)
from .image_guides import DESCRIPTION as IMAGE_GUIDES_DESCRIPTION, SCHEMA as IMAGE_GUIDES_SCHEMA

VIDEO_TOOLS_SCHEMA = [
    {
        "name": "video.create",
        "description": "Create an asynchronous video generation task using minimax-h3-ref2va or configured routes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Text prompt describing the desired video"},
                "model": {"type": "string", "description": "Business model; omitted resolves for the selected route"},
                "aspect_ratio": {"type": "string", "enum": ["adaptive", "21:9", "16:9", "9:16", "1:1", "4:3", "3:4"], "description": "Check video.capabilities for route-specific adapter limits, not model limits"},
                "duration_seconds": {"type": "integer", "minimum": 1, "maximum": 15, "default": 5},
                "route": {"type": "string", "description": "Execution channel; omitted uses the host default from video.capabilities"},
                "quality_profile": {"type": "string", "description": "H3 only; omit for Grok"},
                "image_guidance": {
                    "type": "string",
                    "enum": ["reference", "first_frame"],
                    "description": IMAGE_GUIDANCE_DESCRIPTION,
                },
                "audio_guidance": {"type": "string", "enum": ["reference", "timeline"], "description": "H3 Singularity only; timeline is experimental full-duration PCM audio conditioning, not guaranteed lip sync"},
                "mode": {"type": "string", "enum": ["text_to_video", "image_to_video", "reference_to_video"]},
                "resolution": {"type": "string", "enum": ["480p", "720p"]},
                "references": {"type": "object", "description": "Reference assets mapping"},
                "project_id": {"type": "string", "description": "Project ID"},
                "idempotency_key": {"type": "string", "description": "Unique idempotency key"},
            },
            "required": ["prompt", "project_id", "idempotency_key"],
        },
    },
    {
        "name": "video.status",
        "description": "Query status, timing, and stage of a video task.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Video task ID"},
                "video_task_id": {"type": "string", "description": "Alias for task_id"},
            },
        },
    },
    {
        "name": "video.result",
        "description": "Retrieve final media artifacts and download URLs for a completed video task.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "execution_proof_only": {"type": "boolean", "default": False},
                "task_id": {"type": "string", "description": "Video task ID"},
                "video_task_id": {"type": "string", "description": "Alias for task_id"},
            },
        },
    },
    {
        "name": "video.depth",
        "description": "Queue temporal RGB-to-depth conversion of a registered project video.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_artifact_id": {"type": "string", "description": "Source video artifact ID"},
                "model": {"type": "string", "default": "video-depth-anything"},
                "output_format": {"type": "string", "default": "mp4"},
                "project_id": {"type": "string"},
                "idempotency_key": {"type": "string"},
            },
            "required": ["source_artifact_id"],
        },
    },
    {
        "name": "video.sr",
        "description": "Restore a project video with SeedVR2.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_artifact_id": {"type": "string", "description": "Source video artifact ID"},
                "target_width": {"type": "integer"},
                "target_height": {"type": "integer"},
                "quality_mode": {"type": "string", "default": "standard"},
                "backend": {"type": "string", "default": "seedvr2"},
                "seed": {"type": "integer", "default": 666},
            },
            "required": ["source_artifact_id", "target_width", "target_height"],
        },
    },
    {
        "name": "video.interpolate",
        "description": "Double a project video CFR frame rate with RIFE.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_artifact_id": {"type": "string", "description": "Source video artifact ID"},
                "project_id": {"type": "string", "description": "Owning Studio project ID"},
                "idempotency_key": {"type": "string", "description": "Stable key for submission and recovery"},
                "target_fps_num": {"type": "integer", "default": 48},
                "target_fps_den": {"type": "integer", "default": 1},
                "backend": {"type": "string", "default": "rife"},
            },
            "required": ["source_artifact_id", "project_id", "idempotency_key"],
        },
    },
    {
        "name": "video.character_swap",
        "description": "Replace the specified person in a source video with one character image.",
        "inputSchema": {
            "type": "object", "additionalProperties": False,
            "required": ["project_id", "idempotency_key", "source_artifact_id", "character_image_artifact_id", "prompt"],
            "properties": {
                "project_id": {"type": "string"}, "idempotency_key": {"type": "string"},
                "source_artifact_id": {"type": "string"}, "character_image_artifact_id": {"type": "string"},
                "prompt": {"type": "string", "minLength": 1, "maxLength": 16384},
                "seed": {"type": "integer", "minimum": 0, "maximum": 4294967295, "default": 7},
                "seed_mode": {"enum": ["fixed"]}, "character_lora_version": {"enum": ["v1.5", "v1"]},
                "turbo_variant": {"enum": ["fl2v_turbo_8step", "pika_h3_flex_turbo_8step_hd_ref2va"]},
                "resolution_preset": {"enum": ["768p", "480p"]},
                "lora_strength": {"type": "number", "minimum": 0.5, "maximum": 1.2},
            },
        },
    },
    {
        "name": "video.regenerate_2k",
        "description": "Regenerate a project 24 FPS H3 video into strict aspect-preserving 2K with Open-H3-Regenerate-2K-IC, preserving lossless AAC audio.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "source_artifact_id": {"type": "string", "description": "Source 24 FPS H3 video artifact ID"},
                "project_id": {"type": "string", "description": "Owning Studio project ID"},
                "idempotency_key": {"type": "string", "description": "Stable key for submission and recovery"},
                "profile": {"type": "string", "enum": ["open_h3_2k_ic"], "default": "open_h3_2k_ic"},
                "grid_mode": {"type": "string", "enum": ["budget_1440p", "exact_2x"], "default": "budget_1440p"},
                "prompt": {"type": "string", "description": "Optional conditioning prompt; auto-resolved from same-project H3 parent task when omitted"},
                "seed": {"type": "integer", "minimum": 0, "maximum": 4294967295, "default": 7},
                "title": {"type": "string", "description": "Optional readable task title"},
                "postprocess": {"type": "object", "description": "Optional postprocess, e.g. {'interpolate': {'profile': 'rife_2x'}}"},
            },
            "required": ["source_artifact_id", "project_id", "idempotency_key"],
        },
    },
]

VIDEO_TOOLS_SCHEMA.extend([
    {"name": "video.capabilities", "description": "Read host defaults, adapter limits and live Singularity readiness; readiness does not prove GPU generation acceptance.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "video.import", "description": "Import an allowed HTTPS asset or up to 3 MiB of PNG/JPEG/WebP bytes into Video; does not resolve central ContentRefs.",
     "inputSchema": {"type": "object", "properties": {
         "project_id": {"type": "string"}, "filename": {"type": "string"},
         "expected_sha256": {"type": "string", "pattern": "^[0-9a-fA-F]{64}$"},
         "source_url": {"type": "string"}, "content_base64": {"type": "string", "maxLength": 4194304},
     }, "required": ["project_id", "filename", "expected_sha256"],
     "oneOf": [{"required": ["source_url"], "not": {"required": ["content_base64"]}},
               {"required": ["content_base64"], "not": {"required": ["source_url"]}}]}},
])
registry = EtcdRegistry(domain="video", tools=VIDEO_TOOLS_SCHEMA, version=__version__)


def _latent_errors(function):
    """Expose stable post-processing errors without forwarding internal messages."""
    @wraps(function)
    def guarded(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (TaskNotFound, TaskConflict, ValueError, ExecutionError, ArtifactError, httpx.HTTPError) as exc:
            retryable = False
            if isinstance(exc, TaskNotFound):
                code, message = 'task_not_found', 'The task is unavailable in the authorized scope.'
            elif isinstance(exc, TaskConflict):
                code, message = 'idempotency_conflict', 'The idempotency key belongs to different inputs.'
            elif isinstance(exc, ValueError):
                code, message = 'invalid_request', 'Provide a valid public task ID and supported parameters; MP4 input is not accepted.'
            elif isinstance(exc, ExecutionError):
                code = str(exc).split(':', 1)[0]
                messages = {
                    'source_not_found': 'The source is unavailable in the authorized scope.',
                    'archive_unavailable': 'Result archival is unavailable; no processing task was submitted.',
                    'runtime_not_ready': 'The post-processing runtime is not ready; no processing task was submitted.',
                    'source_not_ready': 'The source task has not completed successfully.',
                    'unsupported_source_route': 'The source route does not support latent post-processing.',
                    'missing_latent_bundle': 'The source has no retained latent bundle; MP4-only tasks cannot be processed.',
                    'source_node_mismatch': 'The source resources are not available on the processing node.',
                    'source_identity_mismatch': 'The source resource identity does not match the task.',
                    'invalid_source_descriptor': 'The source resource descriptor is invalid.',
                    'source_changed_after_registration': 'The registered source resources have changed.',
                }
                if code in messages:
                    message = messages[code]
                    retryable = code in {'archive_unavailable', 'runtime_not_ready'}
                else:
                    code, message = 'postprocessing_unavailable', 'Post-processing is unavailable; retain the task ID and retry the query.'
                    retryable = True
            else:
                code, message = 'postprocessing_unavailable', 'Resource access is unavailable; retain the task ID and retry the query.'
                retryable = True
            result = _result({'error': {'code': code, 'message': message, 'retryable': retryable}})
            result.is_error = True
            return result
    return guarded


@mcp.tool(name="video.h3.latent.upscale.generate")
@_latent_errors
def video_h3_latent_generate(source_video_task_id: str, project_id: str | None = None,
                            idempotency_key: str | None = None, profile_id: str | None = None,
                            target_width: int | None = None, target_height: int | None = None,
                            seed: int | None = None) -> types.CallToolResult:
    """Post-process a completed H3 task using retained same-node latent resources; never accept MP4 input."""
    adapter = executor.processing["h3-latent-upscale"]
    record = adapter.generate(source_video_task_id, project_id=project_id, idempotency_key=idempotency_key,
                              profile_id=profile_id, target_width=target_width, target_height=target_height, seed=seed)
    return _result({**adapter.public_status(record), **titles.project(executor.tasks, record)})


@mcp.tool(name="video.h3.latent.upscale.status")
@_latent_errors
def video_h3_latent_status(video_task_id: str) -> types.CallToolResult:
    if _is_task_nsfw(_active_tasks().get(video_task_id)):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    adapter = executor.processing["h3-latent-upscale"]
    record = adapter.status(video_task_id)
    if _is_task_nsfw(record):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    return _result({**adapter.public_status(record), **titles.project(executor.tasks, record)})


@mcp.tool(name="video.h3.latent.upscale.result")
@_latent_errors
def video_h3_latent_result(video_task_id: str) -> types.CallToolResult:
    if _is_task_nsfw(_active_tasks().get(video_task_id)):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    adapter = executor.processing["h3-latent-upscale"]
    record = adapter.result(video_task_id)
    if _is_task_nsfw(record):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    return _result({**adapter.public_result(record), **titles.project(executor.tasks, record)})


@mcp.tool(name="video.h3.latent.upscale.preview")
@_latent_errors
def video_h3_latent_preview(video_task_id: str) -> types.CallToolResult:
    if _is_task_nsfw(_active_tasks().get(video_task_id)):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    adapter = executor.processing["h3-latent-upscale"]
    record = adapter.result(video_task_id)
    if _is_task_nsfw(record):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    return _result({**adapter.public_result(record, preview=True), **titles.project(executor.tasks, record)})


@mcp.tool(name="video.depth.generate")
def video_depth_generate(project_id: str, idempotency_key: str, source_artifact_id: str,
                         model: str = "video-depth-anything", output_format: str = "mp4") -> types.CallToolResult:
    """Queue temporal RGB-to-depth conversion of a registered project video."""
    record = executor.depth.generate(project_id=project_id, idempotency_key=idempotency_key,
                                    source_artifact_id=source_artifact_id, model=model, output_format=output_format)
    return _result({**executor.depth.public_status(record), **titles.project(executor.tasks, record)})


@mcp.tool(name="video.depth.status")
def video_depth_status(video_task_id: str) -> types.CallToolResult:
    if _is_task_nsfw(_active_tasks().get(video_task_id)):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    record = executor.depth.status(video_task_id)
    if _is_task_nsfw(record):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    return _result({**executor.depth.public_status(record), **titles.project(executor.tasks, record)})


@mcp.tool(name="video.depth.result")
def video_depth_result(video_task_id: str) -> types.CallToolResult:
    if _is_task_nsfw(_active_tasks().get(video_task_id)):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    record = executor.depth.result(video_task_id)
    if _is_task_nsfw(record):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    return _result({**executor.depth.public_result(record), **titles.project(executor.tasks, record)})


@mcp.tool(name="video.depth.preview")
def video_depth_preview(video_task_id: str) -> types.CallToolResult:
    """Get an immutable side-by-side source RGB / grayscale depth MP4."""
    if _is_task_nsfw(_active_tasks().get(video_task_id)):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    record = executor.depth.result(video_task_id)
    if _is_task_nsfw(record):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    return _result({**executor.depth.public_result(record, preview=True), **titles.project(executor.tasks, record)})



@mcp.tool(name="video.sr.generate")
def video_sr_generate(project_id: str, idempotency_key: str, source_artifact_id: str,
                      target_width: int, target_height: int, quality_mode: str = "standard",
                      backend: str = "seedvr2", seed: int = 666) -> types.CallToolResult:
    """Restore a project video with SeedVR2; preserve CFR timing and aspect ratio."""
    adapter = executor.processing["sr"]
    record = adapter.generate(project_id=project_id, idempotency_key=idempotency_key,
        source_artifact_id=source_artifact_id, target_width=target_width, target_height=target_height,
        quality_mode=quality_mode, backend=backend, seed=seed)
    return _result({**adapter.public_status(record), **titles.project(executor.tasks, record)})


@mcp.tool(name="video.interpolate.generate")
def video_interpolate_generate(project_id: str, idempotency_key: str, source_artifact_id: str,
                               target_fps_num: int = 48, target_fps_den: int = 1,
                               backend: str = "rife") -> types.CallToolResult:
    """Double a project video's CFR frame rate with RIFE, preserving duration and dimensions."""
    authorize(executor.client, project_id)
    adapter = executor.processing["interpolate"]
    record = adapter.generate(project_id=project_id, idempotency_key=idempotency_key,
        source_artifact_id=source_artifact_id, target_fps_num=target_fps_num,
        target_fps_den=target_fps_den, backend=backend)
    return _result({**adapter.public_status(record), **titles.project(executor.tasks, record)})


@mcp.tool(name="video.sr.status")
def video_sr_status(video_task_id: str) -> types.CallToolResult:
    if _is_task_nsfw(_active_tasks().get(video_task_id)):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    adapter = executor.processing["sr"]
    record = adapter.status(video_task_id)
    if _is_task_nsfw(record):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    return _result({**adapter.public_status(record), **titles.project(executor.tasks, record)})


@mcp.tool(name="video.sr.result")
def video_sr_result(video_task_id: str) -> types.CallToolResult:
    if _is_task_nsfw(_active_tasks().get(video_task_id)):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    adapter = executor.processing["sr"]
    record = adapter.result(video_task_id)
    if _is_task_nsfw(record):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    return _result({**adapter.public_result(record), **titles.project(executor.tasks, record)})


@mcp.tool(name="video.sr.preview")
def video_sr_preview(video_task_id: str) -> types.CallToolResult:
    if _is_task_nsfw(_active_tasks().get(video_task_id)):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    adapter = executor.processing["sr"]
    record = adapter.result(video_task_id)
    if _is_task_nsfw(record):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    return _result({**adapter.public_result(record, preview=True), **titles.project(executor.tasks, record)})


@mcp.tool(name="video.interpolate.status")
def video_interpolate_status(video_task_id: str) -> types.CallToolResult:
    existing = _active_tasks().get(video_task_id)
    if _is_task_nsfw(existing):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    authorize(executor.client, existing.project_id)
    adapter = executor.processing["interpolate"]
    record = adapter.status(video_task_id)
    if _is_task_nsfw(record):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    return _result({**adapter.public_status(record), **titles.project(executor.tasks, record)})


@mcp.tool(name="video.interpolate.result")
def video_interpolate_result(video_task_id: str) -> types.CallToolResult:
    existing = _active_tasks().get(video_task_id)
    if _is_task_nsfw(existing):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    authorize(executor.client, existing.project_id)
    adapter = executor.processing["interpolate"]
    record = adapter.result(video_task_id)
    if _is_task_nsfw(record):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    return _result({**adapter.public_result(record), **titles.project(executor.tasks, record)})


@mcp.tool(name="video.interpolate.preview")
def video_interpolate_preview(video_task_id: str) -> types.CallToolResult:
    existing = _active_tasks().get(video_task_id)
    if _is_task_nsfw(existing):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    authorize(executor.client, existing.project_id)
    adapter = executor.processing["interpolate"]
    record = adapter.result(video_task_id)
    if _is_task_nsfw(record):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    return _result({**adapter.public_result(record, preview=True), **titles.project(executor.tasks, record)})


def _submit_regenerate_2k(
    *,
    source_artifact_id: str,
    project_id: str,
    idempotency_key: str,
    profile: str = "open_h3_2k_ic",
    grid_mode: str = "budget_1440p",
    prompt: str | None = None,
    seed: int = 7,
    title: object = None,
    postprocess: dict | None = None,
) -> dict[str, object]:
    authorize(executor.client, project_id)
    adapter = executor.processing["regenerate_2k"]
    warning = None
    cleaned_title = None
    if title is not None:
        try:
            cleaned_title = titles.validate_title(title)
        except ValueError:
            cleaned_title, warning = None, "invalid_optional_title; readable fallback used"
    token = titles.PROVIDED_TITLE.set(cleaned_title)
    try:
        record = adapter.generate(
            project_id=project_id,
            idempotency_key=idempotency_key,
            source_artifact_id=source_artifact_id,
            profile=profile,
            grid_mode=grid_mode,
            prompt=prompt,
            seed=seed,
            postprocess=postprocess,
        )
    finally:
        titles.PROVIDED_TITLE.reset(token)
    naming = titles.project(executor.tasks, record)
    if warning:
        naming["title_warning"] = warning
    res = adapter.public_status(record)
    res["task_id"] = record.video_task_id
    return {**res, **naming}


@mcp.tool(name="video.character_swap")
@mcp.tool(name="video.character_swap.generate")
@tool_errors
def video_character_swap(project_id: str, idempotency_key: str, source_artifact_id: str,
                         character_image_artifact_id: str, prompt: str, seed: int = 7,
                         character_lora_version: str = "v1.5", lora_strength: float = 1.0,
                         turbo_variant: str = "fl2v_turbo_8step", resolution_preset: str = "768p",
                         seed_mode: str = "fixed") -> types.CallToolResult:
    """Replace one person in a project video using its target character image."""
    authorize(executor.client, project_id)
    adapter = executor.processing["character_swap"]
    record = adapter.generate(project_id=project_id, idempotency_key=idempotency_key,
        source_artifact_id=source_artifact_id, character_image_artifact_id=character_image_artifact_id,
        prompt=prompt, seed=seed, character_lora_version=character_lora_version, lora_strength=lora_strength,
        turbo_variant=turbo_variant, resolution_preset=resolution_preset, seed_mode=seed_mode)
    return _result({**adapter.public_status(record), "task_id": record.video_task_id,
                    **titles.project(executor.tasks, record)})


@mcp.tool(name="video.regenerate_2k.generate")
@tool_errors
def video_regenerate_2k_generate(
    project_id: str,
    idempotency_key: str,
    source_artifact_id: str,
    profile: str = "open_h3_2k_ic",
    grid_mode: str = "budget_1440p",
    prompt: str | None = None,
    seed: int = 7,
    title: str | None = None,
    postprocess: dict | None = None,
) -> types.CallToolResult:
    """Regenerate an existing 24 FPS H3 video into strict aspect-preserving 2K."""
    return _result(
        _submit_regenerate_2k(
            source_artifact_id=source_artifact_id,
            project_id=project_id,
            idempotency_key=idempotency_key,
            profile=profile,
            grid_mode=grid_mode,
            prompt=prompt,
            seed=seed,
            title=title,
            postprocess=postprocess,
        )
    )


@mcp.tool(name="video.regenerate_2k.status")
@tool_errors
def video_regenerate_2k_status(video_task_id: str) -> types.CallToolResult:
    existing = _active_tasks().get(video_task_id)
    if _is_task_nsfw(existing):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    authorize(executor.client, existing.project_id)
    adapter = executor.processing["regenerate_2k"]
    record = adapter.status(video_task_id)
    if _is_task_nsfw(record):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    return _result({**adapter.public_status(record), **titles.project(executor.tasks, record)})


@mcp.tool(name="video.regenerate_2k.result")
@tool_errors
def video_regenerate_2k_result(video_task_id: str) -> types.CallToolResult:
    existing = _active_tasks().get(video_task_id)
    if _is_task_nsfw(existing):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    authorize(executor.client, existing.project_id)
    adapter = executor.processing["regenerate_2k"]
    record = adapter.result(video_task_id)
    if _is_task_nsfw(record):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    return _result({**adapter.public_result(record), **titles.project(executor.tasks, record)})


@mcp.tool(name="video.regenerate_2k.preview")
@tool_errors
def video_regenerate_2k_preview(video_task_id: str) -> types.CallToolResult:
    existing = _active_tasks().get(video_task_id)
    if _is_task_nsfw(existing):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    authorize(executor.client, existing.project_id)
    adapter = executor.processing["regenerate_2k"]
    record = adapter.result(video_task_id)
    if _is_task_nsfw(record):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    return _result({**adapter.public_result(record, preview=True), **titles.project(executor.tasks, record)})


@mcp.tool(name="video.character_swap.status")
@tool_errors
def video_character_swap_status(video_task_id: str) -> types.CallToolResult:
    existing = _active_tasks().get(video_task_id)
    if existing.service != "character_swap" or _is_task_nsfw(existing):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    authorize(executor.client, existing.project_id)
    adapter = executor.processing["character_swap"]
    record = adapter.status(video_task_id)
    if _is_task_nsfw(record):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    return _result({**adapter.public_status(record), **titles.project(executor.tasks, record)})


@mcp.tool(name="video.character_swap.result")
@tool_errors
def video_character_swap_result(video_task_id: str) -> types.CallToolResult:
    existing = _active_tasks().get(video_task_id)
    if existing.service != "character_swap" or _is_task_nsfw(existing):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    authorize(executor.client, existing.project_id)
    adapter = executor.processing["character_swap"]
    record = adapter.result(video_task_id)
    if _is_task_nsfw(record):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    return _result({**adapter.public_result(record), **titles.project(executor.tasks, record)})


@mcp.tool(name="video.character_swap.preview")
@tool_errors
def video_character_swap_preview(video_task_id: str) -> types.CallToolResult:
    existing = _active_tasks().get(video_task_id)
    if existing.service != "character_swap" or _is_task_nsfw(existing):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    authorize(executor.client, existing.project_id)
    adapter = executor.processing["character_swap"]
    record = adapter.result(video_task_id)
    if _is_task_nsfw(record):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    return _result({**adapter.public_result(record, preview=True), **titles.project(executor.tasks, record)})


@mcp.tool(name="video.lipsync")
@mcp.tool(name="video.lipsync.generate")
@tool_errors
def video_lipsync(
    project_id: str,
    idempotency_key: str,
    source_video_artifact_id: str,
    audio_artifact_id: str,
    inference_steps: int = 20,
    guidance_scale: float = 1.5,
    seed: int = 1247,
    audio_mix_mode: str = "driver_wav",
    title: str | None = None,
) -> types.CallToolResult:
    """Synchronize lip movement on a project video using a dry vocal PCM WAV via LatentSync 1.6."""
    authorize(executor.client, project_id)
    # Enforce NSFW source video isolation (S5)
    t_store = _active_tasks()
    r_store = _active_ratings()
    if getattr(t_store, "root", None) is not None and t_store.root.is_dir():
        for path in t_store.root.glob("video_task_*.json"):
            try:
                src_rec = TaskRecord.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if src_rec.project_id == project_id and (
                src_rec.artifact_id == source_video_artifact_id
                or (src_rec.delivery or {}).get("final_artifact_id") == source_video_artifact_id
            ):
                if t_store.is_record_nsfw(src_rec, r_store):
                    raise TaskNotFound(f"NOT_FOUND: {source_video_artifact_id}")
    tok = titles.PROVIDED_TITLE.set(titles.validate_title(title) if title is not None else None)
    try:
        adapter = executor.processing["lipsync"]
        record = adapter.generate(
            project_id=project_id,
            idempotency_key=idempotency_key,
            source_artifact_id=source_video_artifact_id,
            audio_artifact_id=audio_artifact_id,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            audio_mix_mode=audio_mix_mode,
        )
    finally:
        titles.PROVIDED_TITLE.reset(tok)
    return _result({**adapter.public_status(record), "task_id": record.video_task_id,
                    **titles.project(executor.tasks, record)})


@mcp.tool(name="video.lipsync.status")
@tool_errors
def video_lipsync_status(video_task_id: str) -> types.CallToolResult:
    existing = _active_tasks().get(video_task_id)
    if existing.service != "lipsync" or _is_task_nsfw(existing):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    authorize(executor.client, existing.project_id)
    adapter = executor.processing["lipsync"]
    record = adapter.status(video_task_id)
    if _is_task_nsfw(record):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    return _result({**adapter.public_status(record), **titles.project(executor.tasks, record)})


@mcp.tool(name="video.lipsync.result")
@tool_errors
def video_lipsync_result(video_task_id: str) -> types.CallToolResult:
    existing = _active_tasks().get(video_task_id)
    if existing.service != "lipsync" or _is_task_nsfw(existing):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    authorize(executor.client, existing.project_id)
    adapter = executor.processing["lipsync"]
    record = adapter.result(video_task_id)
    if _is_task_nsfw(record):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    return _result({**adapter.public_result(record), **titles.project(executor.tasks, record)})


@mcp.tool(name="video.lipsync.preview")
@tool_errors
def video_lipsync_preview(video_task_id: str) -> types.CallToolResult:
    existing = _active_tasks().get(video_task_id)
    if existing.service != "lipsync" or _is_task_nsfw(existing):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    authorize(executor.client, existing.project_id)
    adapter = executor.processing["lipsync"]
    record = adapter.result(video_task_id)
    if _is_task_nsfw(record):
        raise TaskNotFound(f"NOT_FOUND: {video_task_id}")
    return _result({**adapter.public_result(record, preview=True), **titles.project(executor.tasks, record)})


def _result(value: dict[str, object]) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(value, ensure_ascii=False))], structuredContent=value)


@mcp.tool(name="video.capabilities")
def video_capabilities() -> types.CallToolResult:
    """Read adapter limits and live Singularity readiness; readiness is not GPU acceptance."""
    return _result(executor.capabilities())


@mcp.tool(name="video.import")
@tool_errors
def video_import(project_id: str, filename: str, expected_sha256: str,
                 source_url: str | None = None, content_base64: str | None = None) -> types.CallToolResult:
    """Import exactly one allowed HTTPS source or <=3 MiB image payload; never resolve central IDs."""
    if (source_url is None) == (content_base64 is None):
        raise ValueError("provide exactly one of source_url or content_base64")
    common = dict(project_id=project_id, filename=filename, expected_sha256=expected_sha256)
    record = (executor.import_image(**common, content_base64=content_base64) if content_base64 is not None
              else executor.import_asset(**common, source_url=source_url))
    return _result({"status": "completed", "project_id": record.project_id, "artifact": record.model_dump(),
                    "download_path": artifacts.download_path(record.artifact_id)})


@mcp.tool(name="video.import_prepare")
@tool_errors
def video_import_prepare(project_id: str, idempotency_key: str, filename: str, size: int,
                         sha256: str, purpose: str, source_content_ref: dict[str, str] | None = None) -> types.CallToolResult:
    """Prepare resumable media import. A claimed ContentRef grants no central read permission."""
    return _result(executor.imports.prepare(project_id=project_id, idempotency_key=idempotency_key,
        filename=filename, size=size, sha256=sha256, purpose=purpose, source_content_ref=source_content_ref))


@mcp.tool(name="video.import_chunk")
@tool_errors
def video_import_chunk(project_id: str, import_id: str, offset: int, content_base64: str, sha256: str) -> types.CallToolResult:
    """Append up to 512 KiB at the confirmed offset; identical repeated bytes are safe."""
    return _result(executor.imports.chunk(project_id=project_id, import_id=import_id, offset=offset,
        content_base64=content_base64, sha256=sha256))


@mcp.tool(name="video.import_status")
@tool_errors
def video_import_status(project_id: str, import_id: str) -> types.CallToolResult:
    """Read the durable offset and final immutable identity after an interrupted upload."""
    return _result(executor.imports.status(project_id=project_id, import_id=import_id))


@mcp.tool(name="video.import_commit")
def video_import_commit(project_id: str, import_id: str) -> types.CallToolResult:
    """Verify all bytes and decode media before publishing a native Video artifact."""
    try:
        committed = executor.imports.commit(project_id=project_id, import_id=import_id)
    except ValueError as exc:
        # Validators carry stable codes, but arbitrary exception text may contain
        # private paths. Never echo it or infer retryability from its wording.
        messages = {code: message for code, message in MESSAGES.items()
                    if code.startswith(("reference_", "import_", "invalid_import_"))}
        code = str(exc)
        if code not in messages:
            code = "import_validation_failed"
        result = _result({"error": {
            "code": code,
            "message": messages.get(code, "The import could not be validated. Check the media and import parameters."),
            "retryable": code == "import_incomplete",
            "source": "video",
        }})
        result.is_error = True
        return result
    return _result(committed)


@mcp.tool(name="video.preflight")
@tool_errors
def video_preflight(project_id: str, prompt: str = "",
                    references: dict[str, list[dict[str, str]]] | None = None,
                    duration_seconds: int = 5, aspect_ratio: str | None = None,
                    model: str | None = None, route: str | None = None,
                    quality_profile: str | None = None, mode: str | None = None,
                    resolution: str | None = None, postprocess: dict | None = None,
                    channel_account_id: str | None = None, budget_limit: str | None = None,
                    price_ceiling: float | None = None, audio_guidance: str | None = None,
                    image_guidance: Annotated[str | None, Field(default=None, description=IMAGE_GUIDANCE_DESCRIPTION)] = None,
                    image_guides: Annotated[list[dict[str, object]] | None, Field(default=None, description=IMAGE_GUIDES_DESCRIPTION, json_schema_extra=IMAGE_GUIDES_SCHEMA)] = None,
                    source_artifact_id: str | None = None,
                    character_image_artifact_id: str | None = None,
                    character_lora_version: str = "v1.5", turbo_variant: str = "fl2v_turbo_8step",
                    lora_strength: float = 1.0, resolution_preset: str = "768p",
                    profile: str | None = None,
                    grid_mode: str | None = None,
                    seed: int | None = None,
                    workflow_params: dict[str, object] | None = None,
                    upscale: str | None = None,
                    audio_artifact_id: str | None = None,
                    inference_steps: int = 20,
                    guidance_scale: float = 1.5,
                    audio_mix_mode: str = "driver_wav") -> types.CallToolResult:
    """Validate exact media and parameters and read the selected channel's real pricing and admission."""
    if image_guides is not None and (
        source_artifact_id is not None or character_image_artifact_id is not None
        or route in {"character_swap", "regenerate_2k", "lipsync"}
        or mode in {"character_swap", "lipsync"}
        or model == "latentsync-1.6" or audio_artifact_id is not None
    ):
        raise ValueError("image_guides_require_h3_fasth3")
    if route == "lipsync" or mode == "lipsync" or model == "latentsync-1.6" or audio_artifact_id is not None:
        from .lipsync import prepare_request as prepare_lipsync_request
        authorize(executor.client, project_id)
        if not source_artifact_id or not audio_artifact_id:
            raise ValueError("source video and audio artifact are required")
        request = prepare_lipsync_request(executor.artifacts, project_id, source_artifact_id, {
            "audio_artifact_id": audio_artifact_id,
            "inference_steps": inference_steps,
            "guidance_scale": guidance_scale,
            "seed": 1247 if seed is None else seed,
            "audio_mix_mode": audio_mix_mode,
        })
        try:
            readiness = "ready" if executor.gpu.engine_replicas("lipsync") else "scaled_to_zero"
        except Exception:
            readiness = "scaled_to_zero"
        return _result({
            **request,
            "business_type": "video_lipsync",
            "model_id": "latentsync-1.6",
            "backend": "latentsync_1_6",
            "readiness": readiness,
            "status": "validated",
            "gpu_verified": False,
        })
    if route == "character_swap" or mode == "character_swap" or character_image_artifact_id is not None:
        from .character_swap import prepare_request
        authorize(executor.client, project_id)
        if not source_artifact_id or not character_image_artifact_id:
            raise ValueError("source video and character image are required")
        request = prepare_request(executor.artifacts, project_id, source_artifact_id, {
            "character_image_artifact_id": character_image_artifact_id, "prompt": prompt,
            "character_lora_version": character_lora_version, "turbo_variant": turbo_variant,
            "lora_strength": lora_strength, "resolution_preset": resolution_preset,
            "seed": 7 if seed is None else seed})
        capability = executor.delivery.character_swap_capability(
            character_lora_version=request["character_lora_version"], turbo_variant=request["turbo_variant"])
        if capability["readiness"] not in {"ready", "scaled_to_zero"}:
            raise ValueError(capability["reason"])
        return _result({**request, **capability, "status": "validated", "gpu_verified": False})
    if source_artifact_id is not None or route == "regenerate_2k":
        if source_artifact_id is None:
            raise ValueError("source_artifact_id is required")
        return _result(executor.processing["regenerate_2k"].preflight(
            project_id=project_id,
            source_artifact_id=source_artifact_id,
            profile=profile,
            grid_mode=grid_mode,
            prompt=prompt if prompt != "" else None,
            seed=seed,
            postprocess=postprocess,
        ))
    references = {} if references is None else references
    selected = (
        "comfyui"
        if (route in {None, "local"} and isinstance(model, str) and model.strip().startswith("comfyui:"))
        else (route or executor.runtime_route)
    )
    from .singularity import check_audio_guidance, runtime_supports_timeline, check_image_guidance, runtime_supports_first_frame
    check_audio_guidance(audio_guidance, executor._service_for_route(selected))
    check_image_guidance(image_guidance, executor._service_for_route(selected))
    from .image_guides import check_route as check_image_guides, runtime_supports as supports_image_guides
    check_image_guides(image_guides, executor._service_for_route(selected), image_guidance)
    norm_post = normalize_postprocess(postprocess, selected)
    if selected != "comfyui" and workflow_params is not None:
        raise ValueError("unsupported_parameter")
    if selected != "google-flow" and upscale is not None:
        raise ValueError("unsupported_upscale")
    if selected not in {'fal', 'grok', 'google', 'google-flow'} and (channel_account_id is not None or budget_limit is not None):
        raise ValueError('channel_budget_not_supported')
    if selected != 'higgsfield' and price_ceiling is not None:
        raise ValueError('channel_budget_not_supported')
    if selected == "comfyui":
        return _result(executor.comfyui.preflight(
            project_id=require_project_id(project_id),
            prompt=prompt,
            model=model,
            duration_seconds=duration_seconds,
            aspect_ratio=aspect_ratio,
            references=references,
            mode=mode,
            resolution=resolution,
            quality_profile=quality_profile,
            workflow_params=workflow_params,
        ))
    if selected == "google-flow":
        return _result(executor.google_flow.preflight(project_id=require_project_id(project_id),
            prompt=prompt, model=model, duration_seconds=duration_seconds, aspect_ratio=aspect_ratio,
            references=references, mode=mode, resolution=resolution, quality_profile=quality_profile,
            budget_limit=budget_limit, channel_account_id=channel_account_id, upscale=upscale))
    if selected == "higgsfield":
        executor.higgsfield.authorize(project_id)
        request, digest = executor.higgsfield.normalize(project_id=require_project_id(project_id),
            prompt=prompt, model=model, duration_seconds=duration_seconds, aspect_ratio=aspect_ratio,
            references=references, mode=mode, resolution=resolution, quality_profile=quality_profile)
        executor.require_channel('higgsfield', request['model'])
        quote = executor.higgsfield.quote(request)
        if price_ceiling is not None:
            import math
            if not math.isfinite(price_ceiling) or price_ceiling < 0:
                raise ValueError('invalid_price_ceiling')
            if quote['credits'] > price_ceiling:
                quote = {**quote, 'can_submit': False, 'blocked_reason': 'quote_changed'}
        return _result({"status": "validated", "generation_verified": False, "input_digest": digest,
            "resolved_output": {k: request[k] for k in ("model", "mode", "resolution", "duration_seconds", "aspect_ratio")},
            "reference_mapping": request["references"],
            **quote, 'quoted_credits': quote['credits'], 'quote_source': 'provider_cost_api', 'unit': 'Credits'})
    if selected == "grok":
        request, digest = executor.grok.normalize(project_id=require_project_id(project_id),
            prompt=prompt, model=model, duration_seconds=duration_seconds, aspect_ratio=aspect_ratio,
            references=references, mode=mode, resolution=resolution, quality_profile=quality_profile)
        return _result({"status": "validated", "generation_verified": False, "input_digest": digest,
            "resolved_output": {k: request[k] for k in ("mode", "resolution", "duration_seconds", "aspect_ratio")},
            "reference_mapping": request["references"],
            **executor.grok.preflight(request, channel_account_id=channel_account_id, budget_limit=budget_limit)})
    if selected == "google":
        request, digest = executor.google.normalize(project_id=require_project_id(project_id),
            prompt=prompt, model=model, duration_seconds=duration_seconds, aspect_ratio=aspect_ratio,
            references=references, mode=mode, resolution=resolution, quality_profile=quality_profile)
        return _result({"status": "validated", "generation_verified": False, "input_digest": digest,
            "resolved_output": {k: request[k] for k in ("model", "mode", "resolution", "duration_seconds", "aspect_ratio")},
            "reference_mapping": request["references"],
            **executor.google.preflight(request, channel_account_id=channel_account_id, budget_limit=budget_limit)})
    if selected == 'fal':
        return _result(executor.fal.preflight(project_id=require_project_id(project_id), prompt=prompt,
            model=model, duration_seconds=duration_seconds, aspect_ratio=aspect_ratio, references=references,
            mode=mode, resolution=resolution, quality_profile=quality_profile,
            channel_account_id=channel_account_id, budget_limit=budget_limit))
    if selected not in {"h3-fasth3", "h3-singularity"}:
        raise ValueError("video.preflight currently supports h3-fasth3 and h3-singularity")
    if mode not in {None, "reference_to_video"} or resolution is not None:
        raise ValueError("unsupported_parameter")
    url, headers, _, _ = executor.runtime(selected)
    expected_profile = route_quality_profile(selected)
    effective_profile = quality_profile if quality_profile is not None else expected_profile
    request, digest = executor._normalize(require_project_id(project_id), model or "minimax-h3-ref2va", prompt,
        duration_seconds, aspect_ratio or "16:9", references, selected, effective_profile, audio_guidance, image_guidance,
        image_guides=image_guides)
    try:
        response = executor.client.get(f"{url}/health", headers=headers, timeout=10)
        if response.status_code != 200:
            raise ValueError("runtime_not_ready")
        health = response.json()
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        raise ValueError("runtime_not_ready") from exc
    if not isinstance(health, dict) or health.get("ready") is not True:
        raise ValueError("runtime_not_ready")
    if "default_quality_profile" in health and health.get("default_quality_profile") != expected_profile:
        raise ValueError("runtime_not_ready")
    caps = health.get("capabilities") or {}
    if isinstance(caps.get("quality_profiles"), list) and effective_profile not in caps["quality_profiles"]:
        raise ValueError("quality_profile_not_available")
    if audio_guidance == "timeline" and not runtime_supports_timeline(health):
        raise ValueError("timeline_audio_not_supported_or_not_ready")
    if image_guidance == "first_frame" and not runtime_supports_first_frame(health):
        raise ValueError("first_frame_not_supported_or_not_ready")
    if image_guides is not None and not supports_image_guides(health):
        raise ValueError("image_guides_not_supported_or_not_ready")
    if norm_post is not None:
        authorize(executor.client, project_id)
    request, digest = executor.delivery.prepare(request, digest, postprocess)
    delivery_output = None
    if norm_post is not None:
        delivery_output = dict(request["resolved_output"])
        if "regenerate_2k" in norm_post:
            g_mode = norm_post["regenerate_2k"]["grid_mode"]
            geo = resolve_2k_target_geometry(
                int(delivery_output["width"]),
                int(delivery_output["height"]),
                g_mode,
            )
            delivery_output.update(
                {
                    "width": geo["target_w"],
                    "height": geo["target_h"],
                    "grid_mode": g_mode,
                    "levels": geo["levels"],
                }
            )
        if "interpolate" in norm_post:
            delivery_output.update({"fps": 48, "frames": int(delivery_output["frames"]) * 2})
    return _result({"status": "validated", "gpu_verified": False, "input_digest": digest,
                    "delivery_output": delivery_output, "postprocess": norm_post,
                    "resolved_output": request["resolved_output"], "reference_mapping": request["reference_mapping"],
                    "reference_media": request["reference_media"], "prompt": request["prompt"],
                    **({"image_guidance": request["image_guidance"]} if "image_guidance" in request else {}),
                    **({"image_guides": request["image_guides"]} if "image_guides" in request else {}),
                    **({"audio_guidance": request["audio_guidance"], "timeline_audio": request.get("timeline_audio")}
                       if "audio_guidance" in request else {})})


# Publish the registered parameter schemas instead of maintaining a second
# copy for Studio discovery. MCP is pinned; parity is checked in local tests.
for _name in ("video.import_prepare", "video.import_chunk", "video.import_status", "video.import_commit", "video.preflight"):
    _tool = mcp._tool_manager.get_tool(_name)
    VIDEO_TOOLS_SCHEMA.append({"name": _name, "description": _tool.description, "inputSchema": _tool.parameters})


@mcp.tool(name="artifact.import")
def artifact_import(project_id: str, source_url: str, filename: str, expected_sha256: str) -> types.CallToolResult:
    record = executor.import_asset(project_id=project_id, source_url=source_url, filename=filename, expected_sha256=expected_sha256)
    return _result({"status": "completed", "project_id": project_id, "artifact": record.model_dump(),
                    "download_path": artifacts.download_path(record.artifact_id)})


@mcp.tool(name="video.create")
@tool_errors
def video_create(prompt: str,
                 project_id: str,
                 idempotency_key: str,
                 model: str | None = None,
                 duration_seconds: int = 5,
                 aspect_ratio: str | None = None,
                 references: dict[str, list[dict[str, str]]] | None = None,
                 route: str | None = None,
                 quality_profile: str | None = None, mode: str | None = None,
                  resolution: str | None = None, postprocess: dict | None = None, title: object = None,
                  channel_account_id: str | None = None, budget_limit: str | None = None,
                  price_ceiling: float | None = None, audio_guidance: str | None = None,
                  image_guidance: Annotated[str | None, Field(default=None, description=IMAGE_GUIDANCE_DESCRIPTION)] = None,
                  image_guides: Annotated[list[dict[str, object]] | None, Field(default=None, description=IMAGE_GUIDES_DESCRIPTION, json_schema_extra=IMAGE_GUIDES_SCHEMA)] = None,
                  workflow_params: dict[str, object] | None = None,
                  upscale: str | None = None) -> types.CallToolResult:
    """Create an asynchronous video generation task."""
    record, naming = titles.create(executor, title=title, project_id=project_id, idempotency_key=idempotency_key, model=model,
                               prompt=prompt, duration_seconds=duration_seconds,
                               aspect_ratio=aspect_ratio, references=references or {}, route=route,
                               quality_profile=quality_profile, mode=mode, resolution=resolution, postprocess=postprocess,
                               channel_account_id=channel_account_id, budget_limit=budget_limit, price_ceiling=price_ceiling,
                               audio_guidance=audio_guidance, image_guidance=image_guidance,
                               image_guides=image_guides,
                               workflow_params=workflow_params, upscale=upscale)
    return _result({**naming, "task_id": record.video_task_id, "video_task_id": record.video_task_id,
                    "status": record.status, "created_at": record.created_at, "error": record.error, "delivery": record.delivery,
                    "resolved_output": record.request.get("resolved_output"),
                    "reference_mapping": record.request.get("reference_mapping")})


@mcp.tool(name="video.generate")
@tool_errors
def video_generate(project_id: str, idempotency_key: str, model: str, prompt: str,
                   duration_seconds: int, aspect_ratio: str,
                   references: dict[str, list[dict[str, str]]], route: str,
                   quality_profile: str | None = None, mode: str | None = None,
                  resolution: str | None = None, postprocess: dict | None = None, title: object = None,
                  channel_account_id: str | None = None, budget_limit: str | None = None,
                  price_ceiling: float | None = None, audio_guidance: str | None = None,
                  image_guidance: Annotated[str | None, Field(default=None, description=IMAGE_GUIDANCE_DESCRIPTION)] = None,
                  image_guides: Annotated[list[dict[str, object]] | None, Field(default=None, description=IMAGE_GUIDES_DESCRIPTION, json_schema_extra=IMAGE_GUIDES_SCHEMA)] = None,
                  workflow_params: dict[str, object] | None = None,
                  upscale: str | None = None) -> types.CallToolResult:
    """Generate with a business model and an explicitly selected runtime route.

    `route` selects a configured channel such as h3-sol, h3-vdn, or fal.
    The fal and self-hosted H3 channels use `model=minimax-h3-ref2va`.
    References must be registered project artifacts.
    Routes never fall back; retain the returned task id.
    """
    record, naming = titles.create(executor, title=title, project_id=project_id, idempotency_key=idempotency_key, model=model,
                               prompt=prompt, duration_seconds=duration_seconds,
                               aspect_ratio=aspect_ratio, references=references, route=route,
                               quality_profile=quality_profile, mode=mode, resolution=resolution, postprocess=postprocess,
                               channel_account_id=channel_account_id, budget_limit=budget_limit, price_ceiling=price_ceiling,
                               audio_guidance=audio_guidance, image_guidance=image_guidance,
                               image_guides=image_guides,
                               workflow_params=workflow_params, upscale=upscale)
    return _result({**naming, "task_id": record.video_task_id, "video_task_id": record.video_task_id,
                    "status": record.status, "created_at": record.created_at, "error": record.error, "delivery": record.delivery})


@mcp.tool(name="video.cancel")
@tool_errors
def video_cancel(video_task_id: str = "", task_id: str = "") -> types.CallToolResult:
    """Cancel an in-flight asynchronous video generation task."""
    tid = video_task_id or task_id
    if not tid:
        return _result({"error": "task_id is required"})
    t_store = _active_tasks()
    existing = t_store.get(tid)
    if _is_task_nsfw(existing):
        raise TaskNotFound(f"NOT_FOUND: {tid}")
    if existing.service == "comfyui":
        executor.comfyui.authorize_record(existing, require_write=True)
    if existing.service == "google-flow":
        executor.google_flow.authorize_record(existing)
    if existing.service == "higgsfield":
        executor.higgsfield.authorize_record(existing)
    if existing.service == "fal":
        executor.fal.authorize_record(existing)
    if existing.service == "lipsync" and t_store is executor.tasks:
        authorize(executor.client, existing.project_id)
        record = executor.processing["lipsync"].cancel(tid)
    else:
        record = executor.cancel_generation(existing) if t_store is executor.tasks else existing
    return _result({
        **titles.project(t_store, record),
        "task_id": record.video_task_id,
        "video_task_id": record.video_task_id,
        "status": record.status,
        "stage": record.runtime_stage,
        "cancel_pending": record.runtime_stage == "canceling",
        "service": record.service,
        "runtime_route": record.runtime_route,
        "error": record.error,
    })


@mcp.tool(name="video.status")
@tool_errors
def video_status(video_task_id: str = "", task_id: str = "") -> types.CallToolResult:
    tid = video_task_id or task_id
    if not tid:
        return _result({"error": "task_id is required"})
    t_store = _active_tasks()
    existing = t_store.get(tid)
    if _is_task_nsfw(existing):
        raise TaskNotFound(f"NOT_FOUND: {tid}")
    if existing.service == 'comfyui':
        executor.comfyui.authorize_record(existing, require_write=False)
    if existing.service == 'google-flow':
        executor.google_flow.authorize_record(existing)
    if existing.service == 'higgsfield':
        executor.higgsfield.authorize_record(existing)
    if existing.service == 'fal':
        executor.fal.authorize_record(existing)
    if existing.service in {'interpolate', 'regenerate_2k', 'character_swap', 'lipsync'} or existing.request.get('postprocess'):
        authorize(executor.client, existing.project_id)
    record = executor.status(tid) if t_store is executor.tasks else existing
    if _is_task_nsfw(record):
        raise TaskNotFound(f"NOT_FOUND: {tid}")
    gpu_queue = public_gpu_queue(record)
    admission_reconciliation = public_admission_reconciliation(record)
    return _result({**titles.project(t_store, record), "task_id": record.video_task_id, "video_task_id": record.video_task_id, "status": record.status,
                    "created_at": record.created_at, "updated_at": record.updated_at, "error": record.error,
                    "timing": record.timing, "runtime_metrics": record.runtime_metrics, "delivery": record.delivery,
                    "collection": public_collection(record),
                    "service": record.service, "runtime_route": record.runtime_route,
                    "execution_instance_id": record.execution_instance_id, "stage": record.runtime_stage,
                    "needs_attention": record.runtime_stage in {"submission_unknown", "canceling"} or (record.status == "queued" and bool(record.error)),
                    "progress": (record.provider_state or {}).get("progress"),
                    **({"gpu_queue": gpu_queue} if gpu_queue is not None else {}),
                    **({"admission_reconciliation": admission_reconciliation} if admission_reconciliation is not None else {})})


@mcp.tool(name="video.result")
@tool_errors
def video_result(video_task_id: str = "", task_id: str = "", execution_proof_only: bool = False) -> types.CallToolResult:
    tid = video_task_id or task_id
    if not tid:
        return _result({"error": "task_id is required"})
    t_store = _active_tasks()
    a_store = _active_artifacts()
    existing = t_store.get(tid)
    if _is_task_nsfw(existing):
        raise TaskNotFound(f"NOT_FOUND: {tid}")
    if execution_proof_only:
        from .execution_proof import proof
        actor = authorize(executor.client, existing.project_id, require_write=False)
        if existing.authority and existing.authority.get('organization_id') != actor['organization_id']:
            raise PermissionError('execution_proof_access_denied')
        return _result({'execution_proof': proof(existing, a_store)})
    if existing.service == 'comfyui':
        executor.comfyui.authorize_record(existing, require_write=False)
    if existing.service == 'google-flow':
        executor.google_flow.authorize_record(existing)
    if existing.service == 'higgsfield':
        executor.higgsfield.authorize_record(existing)
    if existing.service == 'fal':
        executor.fal.authorize_record(existing)
    if existing.service in {'interpolate', 'regenerate_2k', 'character_swap', 'lipsync'} or existing.request.get('postprocess'):
        authorize(executor.client, existing.project_id)
    if t_store is executor.tasks:
        record = executor.result(tid)
    else:
        if existing.status not in {"succeeded", "completed"}:
            raise ExecutionError(f"task_not_completed:{existing.status}")
        record = existing
    if _is_task_nsfw(record):
        raise TaskNotFound(f"NOT_FOUND: {tid}")
    if record.service == 'google-flow':
        return _result({**executor.google_flow.public_result(record), **titles.project(t_store, record)})
    if record.service == 'higgsfield':
        return _result({**executor.higgsfield.public_result(record), **titles.project(t_store, record)})
    if record.service in executor.processing:
        res = executor.processing[record.service].public_result(record)
        res["task_id"] = record.video_task_id
        return _result({**res, **titles.project(t_store, record)})
    if record.service == "depth":
        res = executor.depth.public_result(record)
        res["task_id"] = record.video_task_id
        return _result({**res, **titles.project(t_store, record)})
    artifact = a_store.get(record.artifact_id, record.project_id)
    value = {"task_id": record.video_task_id, "video_task_id": record.video_task_id, "artifact_id": artifact.artifact_id,
             "status": "succeeded", "project_id": record.project_id,
             "model": record.request.get("model", "minimax-h3-ref2va"), "runtime_version": record.runtime_version or executor.runtime_version,
             "runtime_route": record.runtime_route, "service": record.service,
             "input_digest": record.input_digest,
             "media": {**(record.media or {}), "sha256": artifact.sha256, "size": artifact.size}, "delivery": record.delivery,
             "download_path": a_store.download_path(artifact.artifact_id)}
    return _result({**value, **titles.project(t_store, record)})


@mcp.tool(name="video.list")
def video_list(
    project_id: str | None = None,
    service: str | None = None,
    route: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> types.CallToolResult:
    """List video generation tasks (NSFW-rated tasks are unconditionally excluded from MCP)."""
    t_store = _active_tasks()
    r_store = _active_ratings()
    records, total = t_store.list_tasks(
        project_id=project_id,
        service=service,
        route=route,
        status=status,
        limit=limit,
        offset=offset,
        include_nsfw=False,
        ratings_store=r_store,
    )
    _ui_only = {"content_rating", "content_rating_detail", "can_clear_rating", "can_mark_rating"}
    return _result({
        "tasks": [
            {
                **{k: v for k, v in public_task(r, t_store, r_store).items() if k not in _ui_only},
                "task_id": r.video_task_id,
            }
            for r in records
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    })


@mcp.tool(name="video.rate_content")
def video_rate_content(
    version_id: str,
    rating: str,
    reason: str,
    rubric_version: str = "rubric v1.0",
    project_id: str | None = None,
) -> types.CallToolResult:
    """Record an add-only content rating (general | nsfw) for a generated video version."""
    resolved_vid, resolved_project_id, resolved_task_id = _resolve_rating_target(
        version_id, project_id=project_id
    )
    try:
        res = _active_ratings().rate_content(
            version_id=resolved_vid,
            project_id=resolved_project_id,
            task_id=resolved_task_id,
            rating=rating,
            reason=reason,
            rubric_version=rubric_version,
            rated_by="mcp:video.rate_content",
        )
    except ContentRatingError as e:
        raise ValueError(f"{e.code}: {e}") from e
    return _result(res)


@mcp.tool(name="video.depth")
def video_depth(source_artifact_id: str, project_id: str, idempotency_key: str = "",
                model: str = "video-depth-anything", output_format: str = "mp4") -> types.CallToolResult:
    import uuid
    actual_idem = idempotency_key or f"idem_{uuid.uuid4().hex[:12]}"
    record = executor.depth.generate(project_id=project_id, idempotency_key=actual_idem,
                                     source_artifact_id=source_artifact_id, model=model, output_format=output_format)
    res = executor.depth.public_status(record)
    res["task_id"] = record.video_task_id
    return _result({**res, **titles.project(executor.tasks, record)})


@mcp.tool(name="video.sr")
def video_sr(source_artifact_id: str, target_width: int, target_height: int,
             project_id: str, idempotency_key: str = "",
             quality_mode: str = "standard", backend: str = "seedvr2", seed: int = 666) -> types.CallToolResult:
    import uuid
    actual_idem = idempotency_key or f"idem_{uuid.uuid4().hex[:12]}"
    adapter = executor.processing["sr"]
    record = adapter.generate(project_id=project_id, idempotency_key=actual_idem,
        source_artifact_id=source_artifact_id, target_width=target_width, target_height=target_height,
        quality_mode=quality_mode, backend=backend, seed=seed)
    res = adapter.public_status(record)
    res["task_id"] = record.video_task_id
    return _result({**res, **titles.project(executor.tasks, record)})


@mcp.tool(name="video.interpolate")
def video_interpolate(source_artifact_id: str, project_id: str, idempotency_key: str,
                      target_fps_num: int = 48, target_fps_den: int = 1,
                      backend: str = "rife") -> types.CallToolResult:
    authorize(executor.client, project_id)
    adapter = executor.processing["interpolate"]
    record = adapter.generate(project_id=project_id, idempotency_key=idempotency_key,
        source_artifact_id=source_artifact_id, target_fps_num=target_fps_num,
        target_fps_den=target_fps_den, backend=backend)
    res = adapter.public_status(record)
    res["task_id"] = record.video_task_id
    return _result({**res, **titles.project(executor.tasks, record)})


@mcp.tool(name="video.regenerate_2k")
@tool_errors
def video_regenerate_2k(
    project_id: str,
    idempotency_key: str,
    source_artifact_id: str,
    profile: str = "open_h3_2k_ic",
    grid_mode: str = "budget_1440p",
    prompt: str | None = None,
    seed: int = 7,
    title: str | None = None,
    postprocess: dict | None = None,
) -> types.CallToolResult:
    """Regenerate a 24 FPS H3 video into strict aspect-preserving 2K via Open-H3-Regenerate-2K-IC."""
    return _result(
        _submit_regenerate_2k(
            source_artifact_id=source_artifact_id,
            project_id=project_id,
            idempotency_key=idempotency_key,
            profile=profile,
            grid_mode=grid_mode,
            prompt=prompt,
            seed=seed,
            title=title,
            postprocess=postprocess,
        )
    )


@mcp.tool(name="video.title_update")
def video_title_update(video_task_id: str, title: str, expected_title_revision: StrictInt) -> types.CallToolResult:
    """Rename an authorized project task with optimistic concurrency; never resubmit generation."""
    try:
        naming = titles.rename(executor, video_task_id, title, expected_title_revision)
        return _result({'video_task_id': video_task_id, **naming})
    except titles.TitleConflict:
        return types.CallToolResult(isError=True, content=[types.TextContent(type='text', text='title_conflict: read the latest task before editing')])


for _name in ('video.title_update', 'video.list', 'video.rate_content', 'video.cancel'):
    _registered_tool = mcp._tool_manager.get_tool(_name)
    VIDEO_TOOLS_SCHEMA.append({'name': _name, 'description': _registered_tool.description,
                               'inputSchema': _registered_tool.parameters})


for _entry in VIDEO_TOOLS_SCHEMA:
    if _entry['name'] in {'video.create', 'video.interpolate'}:
        _registered = mcp._tool_manager.get_tool(_entry['name'])
        _entry['inputSchema'] = _registered.parameters


def transport_security_from_environment() -> TransportSecuritySettings:
    hosts = [x.strip() for x in os.environ.get("VIDEO_MCP_ALLOWED_HOSTS", "").split(",") if x.strip()]
    origins = [x.strip() for x in os.environ.get("VIDEO_MCP_ALLOWED_ORIGINS", "").split(",") if x.strip()]
    return TransportSecuritySettings(enable_dns_rebinding_protection=True,
                                     allowed_hosts=hosts or ["127.0.0.1:*", "localhost:*", "[::1]:*"],
                                     allowed_origins=origins or ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"])


async def health(request: Request) -> JSONResponse:
    artifacts.ensure_ready(); tasks.ensure_ready(); ratings.ensure_ready()
    return JSONResponse({"status": "ok", "contract_version": "v1", "version": __version__,
                         "channels": {"higgsfield": executor.higgsfield.channel_health(), "google-flow": executor.google_flow.channel_health()}})


async def artifact_content(request: Request) -> Response:
    try:
        artifact_id = request.path_params["artifact_id"]
        a_store = _active_artifacts()
        t_store = _active_tasks()
        r_store = _active_ratings()
        record = a_store.get(artifact_id)
        if not include_nsfw_from_headers(request.headers):
            if r_store.is_nsfw(artifact_id):
                return JSONResponse({"error": "artifact_not_found"}, status_code=404)
            linked_task = t_store.find_by_artifact_or_task(artifact_id)
            if linked_task is not None and t_store.is_record_nsfw(linked_task, r_store):
                return JSONResponse({"error": "artifact_not_found"}, status_code=404)
        etag = f'"{record.sha256}"'
        cache_headers = {
            "Cache-Control": "private, max-age=2592000, immutable",
            "ETag": etag,
        }
        if _etag_matches(request.headers.get("if-none-match"), etag):
            return Response(status_code=304, headers=cache_headers)
        return FileResponse(
            a_store.content_path(record),
            media_type=record.media_type,
            filename=record.filename,
            headers=cache_headers,
        )
    except (ValueError, ArtifactNotFound):
        return JSONResponse({"error": "artifact_not_found"}, status_code=404)


async def internal_clear_rating(request: Request) -> JSONResponse:
    version_id = request.path_params["version_id"]
    actor_user_id = (
        request.headers.get("X-User-Id")
        or request.headers.get("X-VF-User-Id")
        or ""
    ).strip()
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not actor_user_id:
        actor_user_id = str(body.get("actor_user_id") or "").strip()
    note = str(body.get("note") or body.get("reason") or "manual_clear").strip()
    project_id = body.get("project_id")
    try:
        resolved_vid, resolved_project_id, resolved_task_id = _resolve_rating_target(
            version_id, project_id=project_id
        )
        res = _active_ratings().clear_rating(
            version_id=resolved_vid,
            project_id=resolved_project_id,
            task_id=resolved_task_id,
            actor_user_id=actor_user_id,
            note=note,
        )
        return JSONResponse(res)
    except TaskNotFound:
        return JSONResponse({"error": "not_found"}, status_code=404)
    except ContentRatingError as e:
        return JSONResponse({"error": str(e), "code": e.code}, status_code=400)


async def internal_mark_rating(request: Request) -> JSONResponse:
    version_id = request.path_params["version_id"]
    actor_user_id = (
        request.headers.get("X-User-Id")
        or request.headers.get("X-VF-User-Id")
        or ""
    ).strip()
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not actor_user_id:
        actor_user_id = str(body.get("actor_user_id") or "").strip()
    reason = str(body.get("reason") or body.get("note") or "manual_mark_nsfw").strip()
    project_id = body.get("project_id")
    try:
        resolved_vid, resolved_project_id, resolved_task_id = _resolve_rating_target(
            version_id, project_id=project_id
        )
        res = _active_ratings().mark_rating(
            version_id=resolved_vid,
            project_id=resolved_project_id,
            task_id=resolved_task_id,
            actor_user_id=actor_user_id,
            reason=reason,
        )
        return JSONResponse(res)
    except TaskNotFound:
        return JSONResponse({"error": "not_found"}, status_code=404)
    except ContentRatingError as e:
        return JSONResponse({"error": str(e), "code": e.code}, status_code=400)


class BearerAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        raw_path = request.url.path
        norm_path = raw_path[len("/video"):] if raw_path.startswith("/video/") else raw_path
        if norm_path == "/health" or norm_path.startswith(("/runtime-artifacts/", "/artifacts/")):
            return await call_next(request)
        if norm_path in {"/dashboard", "/dashboard/"} or norm_path.startswith(("/dashboard/frontend/", "/dashboard/static/", "/dashboard/tasks/")):
            response = await call_next(request)
            response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Referrer-Policy"] = "no-referrer"
            return response
        token = os.environ.get("VIDEO_MCP_BEARER_TOKEN", "").strip()
        if norm_path.startswith(("/api/", "/internal/")) and not token:
            return JSONResponse({"error": "authentication_not_configured"}, status_code=503)
        if token and not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {token}"):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        caller = {'user_id': request.headers.get('X-User-Id', ''),
                  'organization_id': request.headers.get('X-Organization-Id', '')}
        context_token = CALLER.set(caller if token else None)
        try:
            response = await call_next(request)
        finally:
            CALLER.reset(context_token)
        if norm_path in {"/mcp", "/api/tasks", "/api/artifacts/import"} or norm_path.endswith("/result"):
            dashboard.resources.record_request(response.status_code >= 400)
        return response


@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    artifacts.ensure_ready(); tasks.ensure_ready(); ratings.ensure_ready()
    await registry.start()
    async def delivery_poll():
        while True:
            try:
                await asyncio.to_thread(executor.delivery.sync)
            except Exception:
                logging.getLogger(__name__).warning("Delivery reconciliation deferred")
            await asyncio.sleep(2)
    async def collection_poll():
        while True:
            try:
                await asyncio.to_thread(executor.collector.sync)
            except Exception:
                logging.getLogger(__name__).warning("Result reconciliation deferred")
            await asyncio.sleep(2)
    async def gpu_poll():
        while True:
            try:
                await asyncio.to_thread(executor.reconcile_gpu)
            except Exception:
                logging.getLogger(__name__).warning("GPU lease reconciliation deferred")
            await asyncio.sleep(20)
    async with mcp.session_manager.run():
        dashboard.recover_incomplete_submissions()
        pollers = [asyncio.create_task(coro) for coro in (dashboard.poll(), dashboard.resources.poll_protocol(), delivery_poll(), collection_poll(), gpu_poll())]
        try:
            yield
        finally:
            executor.collector.close()
            await registry.stop()
            for poller in pollers:
                poller.cancel()
            for poller in pollers:
                with contextlib.suppress(asyncio.CancelledError):
                    await poller


app = Starlette(routes=[*dashboard.routes(), Route("/health", health),
                        Route("/artifacts/{artifact_id:str}/content", artifact_content),
                        Route("/video/artifacts/{artifact_id:str}/content", artifact_content),
                        Route("/runtime-artifacts/{artifact_id:str}/content", artifact_content),
                        Route("/internal/content-ratings/{version_id:str}/clear", internal_clear_rating, methods=["POST"]),
                        Route("/video/internal/content-ratings/{version_id:str}/clear", internal_clear_rating, methods=["POST"]),
                        Route("/internal/content-ratings/{version_id:str}/mark", internal_mark_rating, methods=["POST"]),
                        Route("/video/internal/content-ratings/{version_id:str}/mark", internal_mark_rating, methods=["POST"]),
                        Mount("/", app=mcp.streamable_http_app(json_response=True, stateless_http=True,
                                                               transport_security=transport_security_from_environment()))],
                lifespan=lifespan)
app.state.tasks = tasks
app.state.artifacts = artifacts
app.state.ratings = ratings
app.add_middleware(BearerAuthMiddleware)
