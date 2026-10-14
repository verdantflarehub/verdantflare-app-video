"""Higgsfield team route: trusted Studio actor, internal CLI gateway, durable recovery."""
from __future__ import annotations

import base64
from datetime import UTC, datetime
import hashlib
import json
import os
import ssl
import time
from urllib.parse import urlsplit
import uuid

import httpx

from ..tasks import TaskConflict

MODELS = {"grok_video_v15_lite": (2, 15), "seedance_2_5": (4, 15)}
RATIOS = {"16:9", "9:16", "4:3", "3:4", "1:1"}


class HiggsfieldAdapter:
    def __init__(self, executor):
        self.executor = executor
        self.url = os.getenv("HIGGSFIELD_GATEWAY_URL", "").rstrip("/")
        self.token = os.getenv("HIGGSFIELD_INTERNAL_TOKEN", "")
        self.binding = os.getenv("HIGGSFIELD_BINDING_ID", "")
        self.organization = os.getenv("HIGGSFIELD_ORGANIZATION_ID", "")
        self.members = frozenset(filter(None, os.getenv("HIGGSFIELD_ALLOWED_USERS", "").split(",")))
        self.enabled = os.getenv("HIGGSFIELD_ENABLED", "false").lower() == "true"
        trust = ssl.create_default_context()
        if os.getenv("VIDEO_PROJECT_CA_FILE"):
            trust.load_verify_locations(os.environ["VIDEO_PROJECT_CA_FILE"])
        self.client = httpx.Client(trust_env=False, follow_redirects=False, timeout=120, verify=trust)
        self.profile = hashlib.sha256(json.dumps([self.url, self.binding, self.organization]).encode()).hexdigest()

    def configured(self):
        parsed = urlsplit(self.url)
        return bool(self.binding and self.organization and len(self.token) >= 32 and
                    parsed.scheme in {"http", "https"} and parsed.hostname and
                    not (parsed.username or parsed.password or parsed.query or parsed.fragment))

    def capabilities(self):
        return [{"route": "higgsfield", "provider": "higgsfield", "model": model,
                 "enabled": self.enabled, "configured": self.configured(), "readiness": "not_checked",
                 "limit_scope": "adapter", "duration_seconds": {"minimum": bounds[0], "maximum": bounds[1]},
                 "resolutions": ["720p"] if model == "seedance_2_5" else ["480p", "720p"],
                 "aspect_ratios": sorted(RATIOS | ({"21:9"} if model == "seedance_2_5" else set())),
                 "modes": ["text_to_video"] if model == "seedance_2_5" else ["text_to_video", "image_to_video"],
                 "poll": True, "cancel": False, "callback": False, "verification_status": "not_recorded"}
                for model, bounds in MODELS.items()]

    def authorize(self, project_id, authority=None):
        from ..delivery import CALLER
        from .higgsfield_artifact import uuid7
        actor = authority if authority is not None else CALLER.get()
        if (not actor or actor.get("organization_id") != self.organization or
                not actor.get("user_id") or (self.members and actor["user_id"] not in self.members)):
            raise PermissionError("higgsfield_account_access_denied")
        url = os.getenv("VIDEO_PROJECT_SERVICE_URL", "").rstrip("/")
        token = os.getenv("VIDEO_PROJECT_SERVICE_TOKEN", "")
        if not url or not token:
            raise ValueError("project_authorization_not_configured")
        response = self.client.post(url + "/internal/v1/project/open",
            headers={"Authorization": "Bearer " + token, "X-User-Id": actor["user_id"],
                     "X-Organization-Id": actor["organization_id"], "X-Request-Id": uuid7()},
            json={"project_id": project_id, "require_write": True}, timeout=15)
        if response.status_code in {401, 403, 404}:
            raise PermissionError("project_access_denied")
        response.raise_for_status()
        return dict(actor)

    def authorize_record(self, record):
        actor = self.authorize(record.project_id)
        if not record.authority or actor["organization_id"] != record.authority.get("organization_id"):
            raise PermissionError("higgsfield_task_access_denied")
        return actor

    def normalize(self, *, project_id, model, prompt, duration_seconds, aspect_ratio, references,
                  mode=None, resolution=None, quality_profile=None):
        model = model or "seedance_2_5"
        if model not in MODELS or quality_profile is not None:
            raise ValueError("unsupported_higgsfield_model_or_parameter")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 10000 or "\x00" in prompt:
            raise ValueError("invalid_prompt")
        low, high = MODELS[model]
        if type(duration_seconds) is not int or not low <= duration_seconds <= high:
            raise ValueError("unsupported_duration")
        if not isinstance(references, dict) or set(references) - {"images", "videos", "audios"}:
            raise ValueError("invalid_references")
        refs = {k: references.get(k, []) for k in ("images", "videos", "audios")}
        if any(not isinstance(v, list) for v in refs.values()) or refs["videos"] or refs["audios"] or len(refs["images"]) > 1:
            raise ValueError("unsupported_references")
        resolved_mode = "image_to_video" if refs["images"] else "text_to_video"
        if mode is not None and mode != resolved_mode:
            raise ValueError("mode_reference_mismatch")
        source = None
        if refs["images"]:
            if model != "grok_video_v15_lite":
                raise ValueError("unsupported_references")
            ref = refs["images"][0]
            if not isinstance(ref, dict) or set(ref) != {"artifact_id", "purpose"} or ref["purpose"] != "first_frame":
                raise ValueError("first_frame_reference_required")
            _, source = self.executor.grok.source(project_id, ref["artifact_id"])
        if source and aspect_ratio is None:
            aspect_ratio = next((ratio for ratio in sorted(RATIOS) if
                abs(source["width"] * int(ratio.split(":")[1]) - source["height"] * int(ratio.split(":")[0])) <=
                max(map(int, ratio.split(":")))), None)
            if aspect_ratio is None:
                raise ValueError("unsupported_source_aspect_ratio")
        aspect_ratio = aspect_ratio or "16:9"
        if aspect_ratio not in RATIOS | ({"21:9"} if model == "seedance_2_5" else set()):
            raise ValueError("unsupported_aspect_ratio")
        if source:
            a, b = map(int, aspect_ratio.split(":"))
            if abs(source["width"] * b - source["height"] * a) > max(a, b):
                raise ValueError("aspect_ratio_conflict")
        resolution = resolution or "720p"
        if resolution not in ({"720p"} if model == "seedance_2_5" else {"480p", "720p"}):
            raise ValueError("unsupported_resolution")
        value = dict(project_id=project_id, model=model, prompt=prompt, duration_seconds=duration_seconds,
                     aspect_ratio=aspect_ratio, references=refs, mode=resolved_mode, resolution=resolution,
                     source=source, route="higgsfield", binding_id=self.binding, gateway_profile=self.profile)
        digest = "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        return value, digest

    def generate(self, *, idempotency_key, **kwargs):
        actor = self.authorize(kwargs["project_id"])
        request, digest = self.normalize(**kwargs)
        old = self.executor.tasks.find_idempotency(request["project_id"], idempotency_key)
        if old:
            self.authorize_record(old)
            if old.input_digest != digest:
                raise TaskConflict("idempotency_conflict")
            return old
        if not self.enabled or not self.configured():
            raise ValueError("higgsfield_not_configured")
        record = self.executor.tasks.create(project_id=request["project_id"], idempotency_key=idempotency_key,
            input_digest=digest, request=request, runtime_task_id="", status="queued", authority=actor,
            runtime_route="higgsfield", runtime_version="higgsfield-cli-1.1.26")
        return self.executor.tasks.update(record, service="higgsfield", runtime_stage="prepared", provider_state={})

    def payload(self, record):
        p = record.request
        generation = {k: p[k] for k in ("model", "prompt", "duration_seconds", "aspect_ratio", "resolution", "mode")}
        if p["source"]:
            data, source = self.executor.grok.source(record.project_id, p["references"]["images"][0]["artifact_id"])
            if source != p["source"]:
                raise ValueError("source_integrity_failed")
            generation["source"] = {"content_base64": base64.b64encode(data).decode(), "sha256": source["sha256"],
                                    "format": source["mime"].split("/")[1]}
        return {"request_id": record.video_task_id, "binding_id": p["binding_id"], "project_id": record.project_id,
                "organization_id": record.authority["organization_id"], "user_id": record.authority["user_id"],
                "generation": generation}

    def headers(self):
        return {"Authorization": "Bearer " + self.token}

    def defer(self, record, code):
        # Collection can persist an upload receipt before a later HTTP failure.
        # Never overwrite that durable progress with the caller's stale record.
        record = self.executor.tasks.get(record.video_task_id)
        state = dict(record.provider_state or {})
        state["next_poll_at"] = time.time() + 15
        return self.executor.tasks.update(record, provider_state=state, error={"code": code, "message": code})

    def status(self, record):
        if record.status in {"succeeded", "failed", "cancelled"}:
            return record
        if record.request.get("gateway_profile") != self.profile:
            return self.defer(record, "higgsfield_binding_changed")
        if (record.provider_state or {}).get("next_poll_at", 0) > time.time():
            return record
        try:
            self.authorize(record.project_id, record.authority)
            if record.runtime_stage == "prepared":
                if not self.enabled:
                    return record
                # The gateway is independently idempotent by our durable task ID.
                response = self.client.post(self.url + "/v1/jobs", headers=self.headers(), json=self.payload(record))
            else:
                response = self.client.get(self.url + "/v1/jobs/" + record.video_task_id, headers=self.headers())
            if response.status_code in {400, 403, 409}:
                return self.defer(record, "higgsfield_request_blocked")
            response.raise_for_status()
            data = response.json()
            if data.get("request_id") != record.video_task_id or data.get("binding_id") != self.binding:
                raise ValueError("gateway_identity_mismatch")
            state = dict(record.provider_state or {})
            state.update(next_poll_at=time.time() + 10, quoted_credits=data.get("quoted_credits"))
            identity = data.get("remote_id")
            if identity and str(uuid.UUID(identity)) != identity:
                raise ValueError("gateway_identity_mismatch")
            record = self.executor.tasks.update(record, runtime_task_id=identity or "", status="running",
                runtime_stage=data["stage"], provider_state=state,
                dispatched_at=data.get("created_at") if identity else None,
                error={"code": "submission_unknown", "message": "submission_unknown"} if data["stage"] == "submission_unknown" else None)
            if data["status"] == "failed":
                return self.executor.tasks.update(record, status="failed",
                    error={"code": "provider_generation_failed", "message": "provider_generation_failed"})
            if data["status"] == "completed":
                from .higgsfield_artifact import collect
                return collect(self, record, data["media"])
            return record
        except (httpx.HTTPError, ValueError, KeyError, TypeError, OSError, PermissionError):
            return self.defer(record, "higgsfield_recovery_pending")

    def result(self, record):
        from ..executor import ExecutionError
        record = self.status(record)
        if record.status != "succeeded" or not (record.provider_state or {}).get("content_ref"):
            raise ExecutionError("higgsfield_result_not_ready")
        return record

    def public_result(self, record):
        state = record.provider_state
        return {"task_id": record.video_task_id, "video_task_id": record.video_task_id, "status": record.status,
                "model": record.request["model"], "runtime_route": "higgsfield", "media": record.media,
                "content_ref": state["content_ref"], "download_path": state["download_path"],
                "quoted_credits": state.get("quoted_credits"), "input_digest": record.input_digest}
