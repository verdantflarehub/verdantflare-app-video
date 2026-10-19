"""Google AI Studio (Veo 3.1) video channel adapter via google-channel:8001."""
from __future__ import annotations

import base64
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from fractions import Fraction
import hashlib
import io
import json
import os
import sqlite3
import subprocess
import time
from urllib.parse import quote, urlsplit

import httpx
from PIL import Image, ImageOps

from ..artifacts import ArtifactNotFound
from ..tasks import TaskConflict

RATIOS = ["16:9", "9:16"]
MODELS = {
    "veo-3.1-lite-generate-preview",
    "veo-3.1-fast-generate-preview",
    "veo-3.1-generate-preview",
}
DEFAULT_MODEL = "veo-3.1-lite-generate-preview"
VERSION = "google-channel-v1"


class GoogleVideoAdapter:
    def __init__(self, executor):
        self.executor = executor
        self.channel_url = (os.getenv("VIDEO_GOOGLE_CHANNEL_URL") or os.getenv("IMAGE_GOOGLE_CHANNEL_URL") or "").rstrip("/")
        self.channel_token = (os.getenv("VIDEO_GOOGLE_CHANNEL_TOKEN") or os.getenv("IMAGE_GOOGLE_CHANNEL_TOKEN") or "").strip()
        self.model = os.getenv("GOOGLE_VIDEO_MODEL", DEFAULT_MODEL)
        self.enabled = bool(self.channel_url and len(self.channel_token) >= 32)
        self.profile = hashlib.sha256((self.channel_url + "\0" + VERSION).encode()).hexdigest()

    def configured(self):
        p = urlsplit(self.channel_url)
        return bool(
            self.enabled
            and p.scheme in {"http", "https"}
            and p.hostname
            and not (p.username or p.password or p.query or p.fragment)
            and self.model in MODELS
        )

    def capability(self):
        return {
            "route": "google",
            "provider": "google",
            "model": self.model,
            "models": sorted(MODELS),
            "enabled": self.enabled,
            "configured": self.configured(),
            "readiness": "not_checked",
            "verification_status": "not_recorded",
            "adapter_version": VERSION,
            "modes": ["text_to_video", "image_to_video"],
            "duration_seconds": {"minimum": 4, "maximum": 8},
            "aspect_ratios": RATIOS,
            "resolutions": ["720p", "1080p"],
            "limit_scope": "adapter",
            "poll": True,
            "callback": False,
            "cancel": False,
            "progress": "optional",
            "runtime_metrics": False,
        }

    @contextmanager
    def locked(self):
        self.executor.tasks.ensure_ready()
        with closing(sqlite3.connect(self.executor.tasks.root / "google-lock.db", timeout=120)) as conn:
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                yield

    def source(self, project_id, artifact_id):
        asset = self.executor.artifacts.get(artifact_id, project_id)
        if asset.size > 10 * 1024 * 1024:
            raise ValueError("source_size_exceeded")
        data = self.executor.artifacts.content_path(asset).read_bytes()
        if len(data) != asset.size or hashlib.sha256(data).hexdigest() != asset.sha256:
            raise ValueError("source_integrity_failed")
        try:
            with Image.open(io.BytesIO(data)) as im:
                if im.format not in {"PNG", "JPEG", "WEBP"} or getattr(im, "n_frames", 1) != 1:
                    raise ValueError
                if im.width * im.height > 16 * 1024 * 1024:
                    raise ValueError
                mime = Image.MIME[im.format]
                im.verify()
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                width, height = ImageOps.exif_transpose(im).size
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise ValueError("source_integrity_failed") from exc
        return data, {"sha256": asset.sha256, "width": width, "height": height, "mime": mime}

    def normalize(self, *, project_id, model, prompt, duration_seconds, aspect_ratio,
                  references, mode=None, resolution=None, quality_profile=None):
        if quality_profile is not None:
            raise ValueError("unsupported_parameter")
        model = model or self.model
        if model not in MODELS:
            raise ValueError("model_route_mismatch")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("invalid_prompt")
        if type(duration_seconds) is not int or not 1 <= duration_seconds <= 15:
            raise ValueError("unsupported_duration")
        resolved_duration = 4 if duration_seconds <= 5 else (6 if duration_seconds <= 6 else 8)
        if not isinstance(references, dict) or set(references) - {"images", "videos", "audios"}:
            raise ValueError("invalid_references")
        refs = {k: references.get(k, []) for k in ("images", "videos", "audios")}
        if any(not isinstance(v, list) for v in refs.values()) or refs["videos"] or refs["audios"] or len(refs["images"]) > 1:
            raise ValueError("unsupported_references")
        resolved_mode = "image_to_video" if refs["images"] else "text_to_video"
        if mode is not None and mode != resolved_mode:
            raise ValueError("mode_reference_mismatch")
        resolution = resolution or "720p"
        if resolution not in {"480p", "720p", "1080p"}:
            raise ValueError("unsupported_resolution")
        resolved_resolution = "720p" if resolution == "480p" else resolution
        info = None
        if refs["images"]:
            ref = refs["images"][0]
            if not isinstance(ref, dict) or set(ref) != {"artifact_id", "purpose"} or ref["purpose"] not in {"first_frame", "reference"}:
                raise ValueError("first_frame_reference_required")
            _, info = self.source(project_id, ref["artifact_id"])
            aspect_ratio = aspect_ratio or ("9:16" if info["height"] > info["width"] else "16:9")
        else:
            aspect_ratio = aspect_ratio or "16:9"
        if aspect_ratio not in RATIOS:
            raise ValueError("unsupported_aspect_ratio")
        request = dict(
            project_id=project_id,
            model=model,
            prompt=prompt.strip(),
            duration_seconds=resolved_duration,
            aspect_ratio=aspect_ratio,
            references=refs,
            mode=resolved_mode,
            resolution=resolved_resolution,
            route="google",
            source=info,
            endpoint_profile_id=self.profile,
        )
        digest = "sha256:" + hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        return request, digest

    def _resolve_actor(self, project_id):
        from ..delivery import CALLER, authorize
        caller = CALLER.get() or {}
        if caller.get("user_id") and caller.get("organization_id") and os.environ.get("VIDEO_PROJECT_SERVICE_URL") and os.environ.get("VIDEO_PROJECT_SERVICE_TOKEN"):
            return authorize(self.executor.client, project_id)
        org_id = str(caller.get("organization_id") or os.environ.get("CHANNEL_DISCOVERY_SCOPE_ID") or os.environ.get("STATION_ORGANIZATION_ID") or "01a0fbf9-0a33-7c31-90d0-a51e3b0a9781").strip()
        user_id = str(caller.get("user_id") or "video-mcp").strip()
        return {"organization_id": org_id, "user_id": user_id}

    def generate(self, *, idempotency_key, channel_account_id=None, budget_limit=None, **kwargs):
        request, digest = self.normalize(**kwargs)
        actor = self._resolve_actor(request["project_id"])
        provider_state = {
            "organization_id": actor["organization_id"],
            "budget_limit": str(budget_limit or os.getenv("VIDEO_GOOGLE_DEFAULT_BUDGET_LIMIT", "2")),
        }
        if channel_account_id is not None:
            provider_state["channel_account_id"] = str(channel_account_id)
        with self.locked():
            old = self.executor.tasks.find_idempotency(request["project_id"], idempotency_key)
            if old:
                if old.input_digest != digest:
                    raise TaskConflict("idempotency_conflict")
                return old
            if not self.enabled or not self.configured():
                raise ValueError("provider_not_configured")
            rec = self.executor.tasks.create(
                project_id=request["project_id"],
                idempotency_key=idempotency_key,
                input_digest=digest,
                request=request,
                runtime_task_id="",
                status="queued",
                authority=actor,
            )
            rec = self.executor.tasks.update(
                rec,
                service="google",
                runtime_route="google",
                runtime_version=VERSION,
                runtime_stage="prepared",
                provider_state=provider_state,
            )
            try:
                return self.submit(rec)
            except (OSError, ValueError):
                return self.error(rec, "source_integrity_failed", terminal=True)

    def save(self, rec, **values):
        return self.executor.tasks.update(rec, **values)

    def error(self, rec, code, *, terminal=False, stage=None, delay=30):
        state = dict(rec.provider_state or {})
        state["next_poll_at"] = time.time() + delay
        return self.save(
            rec,
            status="failed" if terminal else "running",
            error={"code": code, "message": code},
            provider_state=state,
            runtime_stage=stage or ("failed" if terminal else rec.runtime_stage),
        )

    def _headers(self, rec):
        state = rec.provider_state or {}
        org = state.get("organization_id") or (rec.authority or {}).get("organization_id") or ""
        return {
            "Authorization": "Bearer " + self.channel_token,
            "X-Organization-Id": str(org),
            "X-Caller-Id": "video-mcp",
        }

    def submit(self, rec):
        p = rec.request
        if not self.enabled:
            return rec
        state = dict(rec.provider_state or {})
        payload = {
            "prompt": p["prompt"],
            "duration_seconds": p["duration_seconds"],
            "resolution": p["resolution"],
            "aspect_ratio": p["aspect_ratio"],
            "mode": p["mode"],
        }
        if p["mode"] == "image_to_video":
            data, info = self.source(rec.project_id, p["references"]["images"][0]["artifact_id"])
            if info != p["source"]:
                return self.error(rec, "source_integrity_failed", terminal=True)
            payload["image"] = {"mime_type": info["mime"], "data": base64.b64encode(data).decode()}
        rec = self.save(rec, status="running", runtime_stage="submitting")
        try:
            r = self.executor.client.post(
                self.channel_url + "/v1/calls",
                headers=self._headers(rec),
                json={
                    "idempotency_key": rec.video_task_id,
                    "task_id": rec.video_task_id,
                    "model_id": p["model"],
                    "capability": "video",
                    "payload": payload,
                    "account_id": state.get("channel_account_id"),
                    "budget_limit": state.get("budget_limit") or "2",
                },
                timeout=60,
            )
            if r.status_code in {400, 401, 403, 404, 409, 422}:
                code = "provider_rejected"
                try:
                    code = r.json().get("error") or code
                except Exception:
                    pass
                return self.error(rec, code, terminal=True)
            if not r.is_success:
                return self.error(rec, f"provider_http_{r.status_code}", stage="submission_unknown")
            body = r.json()
            call_id = body["call_id"]
            if body.get("accepted") is False or body.get("state") in {"rejected", "failed"}:
                return self.error(rec, body.get("error") or "provider_rejected", terminal=True)
            if body.get("accepted") is None or body.get("state") == "acceptance_unknown":
                return self.save(
                    rec,
                    runtime_task_id=call_id,
                    runtime_stage="submission_unknown",
                    dispatched_at=datetime.now(UTC).isoformat(),
                    error={"code": "submission_unknown", "message": "submission_unknown"},
                )
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return self.error(rec, "submission_unknown", stage="submission_unknown")
        return self.save(
            rec,
            runtime_task_id=call_id,
            runtime_stage="remote_pending",
            dispatched_at=datetime.now(UTC).isoformat(),
            error=None,
        )

    def output_path(self, rec):
        return self.executor.tasks.root / (rec.video_task_id + ".google.mp4")

    def status(self, record):
        with self.locked():
            rec = self.executor.tasks.get(record.video_task_id)
            if rec.status in {"succeeded", "failed", "cancelled"}:
                return rec
            if rec.request.get("endpoint_profile_id") != self.profile:
                return self.error(rec, "provider_configuration_changed", delay=300)
            state = dict(rec.provider_state or {})
            if rec.dispatched_at and (datetime.now(UTC) - datetime.fromisoformat(rec.dispatched_at)).total_seconds() > 86400:
                return self.error(rec, "observation_window_exceeded", terminal=True)
            if time.time() < state.get("next_poll_at", 0):
                return rec
            if rec.runtime_stage == "prepared":
                try:
                    return self.submit(rec)
                except (OSError, ValueError):
                    return self.error(rec, "source_integrity_failed", terminal=True)
            output = self.output_path(rec)
            if output.exists() and state.get("result_sha256"):
                return self.collect(rec, state)
            try:
                path = (
                    "/v1/calls/" + quote(rec.runtime_task_id, safe="")
                    if rec.runtime_task_id
                    else "/v1/calls/by-idempotency/" + quote(rec.video_task_id, safe="")
                )
                r = self.executor.client.get(self.channel_url + path, headers=self._headers(rec), timeout=30)
                if r.status_code == 404 and not rec.runtime_task_id:
                    return self.error(rec, "submission_unknown", stage="submission_unknown", delay=60)
                r.raise_for_status()
                data = r.json()
                if not rec.runtime_task_id and isinstance(data.get("call_id"), str):
                    rec = self.save(rec, runtime_task_id=data["call_id"])
                cstate = data.get("state")
                if data.get("accepted") is False or cstate in {"rejected", "failed"}:
                    return self.error(rec, data.get("error") or "provider_rejected", terminal=True)
                if cstate in {"acceptance_unknown", "queued", "in_queue", "in_progress"}:
                    state["poll_count"] = int(state.get("poll_count", 0)) + 1
                    delay = min(10, 2 ** min(3, state["poll_count"]))
                    state["next_poll_at"] = time.time() + delay
                    return self.save(rec, provider_state=state, status="running", error=None, runtime_stage="remote_pending")
                if cstate != "completed":
                    return self.error(rec, "provider_protocol_error")
                state.update(next_poll_at=0, progress=100)
                return self.collect(self.save(rec, runtime_stage="collecting", provider_state=state), state)
            except (httpx.HTTPError, ValueError, TypeError, KeyError, OSError):
                return self.error(rec, "provider_protocol_error")

    def collect(self, rec, state):
        output = self.output_path(rec)
        try:
            if not output.exists() or not state.get("result_sha256"):
                pending = output.with_suffix(".part")
                digest, count = hashlib.sha256(), 0
                with self.executor.client.stream(
                    "GET",
                    self.channel_url + "/v1/calls/" + quote(rec.runtime_task_id, safe="") + "/result",
                    headers=self._headers(rec),
                    timeout=180,
                ) as response:
                    response.raise_for_status()
                    with pending.open("wb") as f:
                        for chunk in response.iter_bytes():
                            count += len(chunk)
                            if count > 1024 ** 3:
                                raise ValueError("result_size_exceeded")
                            digest.update(chunk)
                            f.write(chunk)
                        f.flush()
                        os.fsync(f.fileno())
                if not count:
                    raise ValueError("result_integrity_failed")
                os.replace(pending, output)
                state["result_sha256"] = digest.hexdigest()
                rec = self.save(rec, provider_state=state)
            with output.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != state["result_sha256"]:
                    return self.error(rec, "result_integrity_failed", terminal=True)
            media = self.executor.grok.probe(output)
            artifact_id = "art_" + hashlib.sha256(rec.video_task_id.encode()).hexdigest()[:32]
            try:
                asset = self.executor.artifacts.get(artifact_id, rec.project_id)
                if asset.sha256 != state["result_sha256"]:
                    raise ValueError("result_integrity_failed")
            except ArtifactNotFound:
                with output.open("rb") as f:
                    asset = self.executor.artifacts.create_from_chunks(
                        project_id=rec.project_id,
                        operation="google",
                        filename="google.mp4",
                        media_type="video/mp4",
                        artifact_id=artifact_id,
                        chunks=iter(lambda: f.read(1024 * 1024), b""),
                        expected_sha256=state["result_sha256"],
                    )
            state["next_poll_at"] = 0
            return self.save(
                rec,
                status="succeeded",
                artifact_id=asset.artifact_id,
                media=media,
                runtime_stage="completed",
                provider_state=state,
                error=None,
            )
        except (OSError, ValueError, KeyError, TypeError, StopIteration, ZeroDivisionError, httpx.HTTPError, subprocess.SubprocessError):
            return self.error(rec, "result_collection_failed", stage="collect_retry")

    def result(self, record):
        rec = self.status(record)
        if rec.status != "succeeded" or not rec.artifact_id:
            from ..executor import ExecutionError
            raise ExecutionError("video task has not succeeded")
        return rec
