"""Dynamic ComfyUI published workflow adapter for Video MCP (`本地` local channel)."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from fractions import Fraction
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any
from urllib.parse import quote, urlsplit
import uuid

import httpx

from ..artifacts import ArtifactNotFound
from ..channel_errors import ChannelError
from ..delivery import CALLER, authorize
from ..tasks import TaskConflict, TaskRecord

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
MODEL_ID_RE = re.compile(r"^comfyui:([a-z0-9][a-z0-9-]{1,46}[a-z0-9])@([0-9]+\.[0-9]+\.[0-9]+)$")
CLUSTER_HOST_SUFFIXES = (".svc.cluster.local", ".svc")
FORBIDDEN_WORKFLOW_KEYS = frozenset(
    {
        "raw_workflow",
        "workflow_json",
        "canvas_json",
        "prompt_graph",
        "node_overrides",
        "class_type",
    }
)


def _is_uuid(value: object) -> bool:
    return isinstance(value, str) and bool(UUID_RE.fullmatch(value.strip()))


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_origin(raw: str) -> str:
    value = (raw or "").strip().rstrip("/")
    if not value:
        return ""
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        return ""
    default_port = 443 if parsed.scheme == "https" else 80
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname.lower()
    if parsed.port and parsed.port != default_port:
        return f"{parsed.scheme}://{host}:{parsed.port}"
    return f"{parsed.scheme}://{host}"


def is_allowed_comfyui_origin(
    raw_origin: str,
    *,
    allowed_origins: set[str] | frozenset[str] | None = None,
    allow_loopback: bool = False,
) -> bool:
    origin = _normalize_origin(raw_origin)
    if not origin:
        return False
    explicit = {_normalize_origin(x) for x in (allowed_origins or set()) if _normalize_origin(x)}
    if origin in explicit:
        return True
    parsed = urlsplit(origin)
    host = (parsed.hostname or "").lower()
    if host in {"localhost", "127.0.0.1", "::1"}:
        return bool(allow_loopback)
    try:
        ip_val = ipaddress.ip_address(host)
        if ip_val.is_loopback:
            return bool(allow_loopback)
        return False
    except ValueError:
        pass
    if any(host.endswith(suffix) for suffix in CLUSTER_HOST_SUFFIXES):
        return host.startswith("comfyui-manager.") or host == "comfyui-manager"
    return False


def build_parameter_schema(spec: dict[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for slot in spec.get("exposed_inputs") or []:
        if not isinstance(slot, dict):
            continue
        name = str(slot.get("name") or "").strip()
        ptype = str(slot.get("param_type") or slot.get("type") or "").strip()
        if not name or not ptype:
            continue
        prop: dict[str, Any] = {}
        if ptype == "integer":
            prop["type"] = "integer"
            if isinstance(slot.get("minimum"), int):
                prop["minimum"] = slot["minimum"]
            if isinstance(slot.get("maximum"), int):
                prop["maximum"] = slot["maximum"]
        elif ptype == "number":
            prop["type"] = "number"
            if isinstance(slot.get("minimum"), (int, float)):
                prop["minimum"] = slot["minimum"]
            if isinstance(slot.get("maximum"), (int, float)):
                prop["maximum"] = slot["maximum"]
        elif ptype == "boolean":
            prop["type"] = "boolean"
        elif ptype == "enum":
            prop["type"] = "string"
            if isinstance(slot.get("enum_values"), list):
                prop["enum"] = list(slot["enum_values"])
        elif ptype in {"artifact_image", "artifact_mask", "artifact_video"}:
            prop["type"] = "string"
            prop["x-artifact-kind"] = ptype
        else:
            prop["type"] = "string"
            if isinstance(slot.get("min_length"), int):
                prop["minLength"] = slot["min_length"]
            if isinstance(slot.get("max_length"), int):
                prop["maxLength"] = slot["max_length"]
        if "default_value" in slot and slot["default_value"] is not None:
            prop["default"] = slot["default_value"]
        properties[name] = prop
        if slot.get("required") is True and "default_value" not in slot:
            required.append(name)
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


class ComfyUIVideoAdapter:
    """Adapter bridging Video MCP `local` / `comfyui` route to ComfyUI Headless API."""

    def __init__(self, executor) -> None:
        self.executor = executor
        self._lock = threading.RLock()
        self.reload_from_environment()

    def reload_from_environment(self) -> None:
        raw_origin = os.getenv("COMFYUI_INTERNAL_ORIGIN", "").strip()
        self.origin = _normalize_origin(raw_origin)
        self.token = os.getenv("COMFYUI_INTERNAL_TOKEN", "").strip()
        raw_allowed = os.getenv("COMFYUI_ALLOWED_ORIGINS", "")
        self.allowed_origins = frozenset(
            _normalize_origin(item)
            for item in raw_allowed.split(",")
            if _normalize_origin(item)
        )
        self.allow_loopback = _truthy_env("COMFYUI_ALLOW_LOOPBACK_ORIGIN")
        self.default_organization_id = (
            os.getenv("COMFYUI_ORGANIZATION_ID")
            or os.getenv("STATION_ORGANIZATION_ID")
            or ""
        ).strip()
        self.default_user_id = (
            os.getenv("COMFYUI_SERVICE_USER_ID")
            or os.getenv("STATION_SERVICE_USER_ID")
            or "0196ed00-0000-7000-8000-000000000001"
        ).strip()
        self.artifact_service_url = _normalize_origin(
            os.getenv("VIDEO_ARTIFACT_SERVICE_URL")
            or os.getenv("IMAGE_ARTIFACT_SERVICE_URL")
            or os.getenv("VIDEO_PROJECT_SERVICE_URL")
            or ""
        )
        self.artifact_service_token = (
            os.getenv("VIDEO_ARTIFACT_SERVICE_TOKEN")
            or os.getenv("IMAGE_ARTIFACT_SERVICE_TOKEN")
            or os.getenv("VIDEO_PROJECT_SERVICE_TOKEN")
            or ""
        ).strip()

    @contextmanager
    def locked(self):
        with self._lock:
            yield

    def configured(self) -> bool:
        return bool(
            self.origin
            and len(self.token) >= 16
            and is_allowed_comfyui_origin(
                self.origin,
                allowed_origins=self.allowed_origins,
                allow_loopback=self.allow_loopback,
            )
        )

    def _resolve_discovery_org(self, explicit_org: str | None = None) -> str | None:
        if explicit_org is not None:
            cand = str(explicit_org).strip()
            return cand.lower() if _is_uuid(cand) else None
        caller = CALLER.get()
        if caller is not None:
            org = str(caller.get("organization_id") or "").strip()
            return org.lower() if _is_uuid(org) else None
        if _is_uuid(self.default_organization_id):
            return self.default_organization_id.lower()
        return None

    def _resolve_actor(self, project_id: str, saved_authority: dict[str, Any] | None = None) -> dict[str, str]:
        if saved_authority and _is_uuid(saved_authority.get("organization_id")):
            uid = str(saved_authority.get("user_id") or self.default_user_id).strip()
            return {
                "organization_id": str(saved_authority["organization_id"]).strip().lower(),
                "user_id": uid.lower() if _is_uuid(uid) else self.default_user_id.lower(),
            }
        if os.getenv("VIDEO_PROJECT_SERVICE_URL", "").strip():
            actor = authorize(self.executor.client, project_id)
            return {
                "organization_id": str(actor["organization_id"]).strip().lower(),
                "user_id": str(actor["user_id"]).strip().lower(),
            }
        caller = CALLER.get()
        if caller is not None:
            org = str(caller.get("organization_id") or "").strip()
            uid = str(caller.get("user_id") or self.default_user_id).strip()
            if not _is_uuid(org) or not _is_uuid(uid):
                raise PermissionError("verified Studio organization_id and user_id required")
            return {"organization_id": org.lower(), "user_id": uid.lower()}
        if _is_uuid(self.default_organization_id) and _is_uuid(self.default_user_id):
            return {
                "organization_id": self.default_organization_id.lower(),
                "user_id": self.default_user_id.lower(),
            }
        raise PermissionError("organization_id context required for ComfyUI local channel")

    def discover_models(self, organization_id: str | None = None) -> list[dict[str, Any]]:
        if not self.configured():
            return []
        org_id = self._resolve_discovery_org(organization_id)
        if not org_id:
            return []
        try:
            response = self.executor.client.post(
                f"{self.origin}/internal/v1/comfyui/published-channels",
                headers={"Authorization": f"Bearer {self.token}"},
                json={"organization_id": org_id, "mount_channel": "video"},
                timeout=10,
            )
            if response.status_code != 200:
                return []
            payload = response.json()
        except (httpx.HTTPError, ValueError, TypeError):
            return []
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            return []
        discovered: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in items:
            if not isinstance(raw, dict):
                continue
            if str(raw.get("organization_id") or "").strip().lower() != org_id:
                continue
            mount = str(raw.get("mount_channel") or raw.get("output_modality") or "").strip().lower()
            if mount != "video":
                continue
            slug = str(raw.get("endpoint_slug") or "").strip()
            ver = str(raw.get("version") or "").strip()
            spec = raw.get("spec")
            if not slug or not ver or not isinstance(spec, dict) or spec.get("schema_version") != "v1":
                continue
            model_id = f"comfyui:{slug}@{ver}"
            if not MODEL_ID_RE.fullmatch(model_id) or model_id in seen:
                continue
            seen.add(model_id)
            display_name = str(
                raw.get("display_name") or spec.get("display_name") or f"{slug} ({ver})"
            ).strip()
            discovered.append(
                {
                    "published_api_id": str(raw.get("published_api_id") or f"{slug}-{ver}"),
                    "entry_id": f"comfyui-{slug}-{ver}",
                    "model_id": model_id,
                    "endpoint_slug": slug,
                    "version": ver,
                    "display_name": display_name,
                    "organization_id": org_id,
                    "project_id": str(raw.get("project_id") or "").strip().lower(),
                    "channel_id": "local",
                    "source_badge": "本地",
                    "vram_budget_mib": raw.get("vram_budget_mib") or spec.get("vram_budget_mib"),
                    "reproducibility_level": raw.get("reproducibility_level") or spec.get("reproducibility_level"),
                    "seed_policy": raw.get("seed_policy") or spec.get("seed_policy"),
                    "spec": spec,
                    "parameter_schema": build_parameter_schema(spec),
                }
            )
        return discovered

    def capabilities(self) -> list[dict[str, Any]]:
        items = self.discover_models()
        routes: list[dict[str, Any]] = []
        for item in items:
            routes.append(
                {
                    "route": "comfyui",
                    "channel": "local",
                    "channel_label": "本地",
                    "source_badge": "本地",
                    "model": item["model_id"],
                    "display_name": item["display_name"],
                    "configured": True,
                    "readiness": "ready",
                    "limit_scope": "workflow_spec",
                    "parameter_schema": item["parameter_schema"],
                }
            )
        return routes

    def _validate_and_map_inputs(
        self,
        *,
        project_id: str,
        prompt: str,
        duration_seconds: int,
        aspect_ratio: str | None,
        references: dict[str, list[dict[str, str]]] | None,
        workflow_params: dict[str, Any] | None,
        spec: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, list[dict[str, str]]]]:
        if workflow_params is not None and not isinstance(workflow_params, dict):
            raise ValueError("workflow_params must be an object")
        supplied = dict(workflow_params or {})
        for forbidden in FORBIDDEN_WORKFLOW_KEYS:
            if forbidden in supplied:
                raise ValueError("unsupported_parameter")

        exposed_list = spec.get("exposed_inputs")
        if not isinstance(exposed_list, list):
            raise ValueError("invalid_workflow_spec")
        exposed_by_name: dict[str, dict[str, Any]] = {}
        for item in exposed_list:
            if isinstance(item, dict) and isinstance(item.get("name"), str) and item["name"].strip():
                exposed_by_name[item["name"].strip()] = item

        unknown_keys = set(supplied) - set(exposed_by_name)
        if unknown_keys:
            raise ValueError("unsupported_parameter")

        refs = references or {}
        limits = {"images": 9, "videos": 3, "audios": 3}
        if not isinstance(refs, dict) or set(refs) - limits.keys():
            raise ValueError("invalid_reference")
        normalized_refs: dict[str, list[dict[str, str]]] = {}
        local_by_kind: dict[str, list[str]] = {"images": [], "videos": [], "audios": []}
        reference_mapping: list[dict[str, str]] = []
        for kind, limit in limits.items():
            raw_list = refs.get(kind, [])
            if not isinstance(raw_list, list) or len(raw_list) > limit:
                raise ValueError("unsupported_references")
            normalized_refs[kind] = []
            expected_prefix = {"images": "image/", "videos": "video/", "audios": "audio/"}[kind]
            for entry in raw_list:
                if not isinstance(entry, dict) or set(entry) != {"artifact_id", "purpose"}:
                    raise ValueError("invalid_reference")
                art_id = str(entry.get("artifact_id") or "").strip()
                purpose = str(entry.get("purpose") or "").strip()
                if not art_id or not purpose:
                    raise ValueError("invalid_reference")
                if _is_uuid(art_id):
                    local_by_kind[kind].append(art_id.lower())
                    normalized_refs[kind].append({"artifact_id": art_id.lower(), "purpose": purpose})
                else:
                    art = self.executor.artifacts.get(art_id, project_id)
                    if not art.media_type.startswith(expected_prefix):
                        raise ValueError("invalid_reference")
                    local_by_kind[kind].append(art.artifact_id)
                    normalized_refs[kind].append({"artifact_id": art.artifact_id, "purpose": purpose})

        # Auto-bind standard MCP parameters into exposed workflow slots when not explicitly supplied.
        if "prompt" in exposed_by_name and "prompt" not in supplied and isinstance(prompt, str) and prompt.strip():
            supplied["prompt"] = prompt.strip()
        if "duration_seconds" in exposed_by_name and "duration_seconds" not in supplied and isinstance(duration_seconds, int):
            supplied["duration_seconds"] = duration_seconds
        if "aspect_ratio" in exposed_by_name and "aspect_ratio" not in supplied and isinstance(aspect_ratio, str) and aspect_ratio:
            supplied["aspect_ratio"] = aspect_ratio

        img_cursor = 0
        vid_cursor = 0
        for name, slot in exposed_by_name.items():
            ptype = str(slot.get("param_type") or slot.get("type") or "").strip()
            if name in supplied:
                continue
            if ptype in {"artifact_image", "artifact_mask"} and img_cursor < len(local_by_kind["images"]):
                chosen = local_by_kind["images"][img_cursor]
                img_cursor += 1
                supplied[name] = chosen
            elif ptype == "artifact_video" and vid_cursor < len(local_by_kind["videos"]):
                chosen = local_by_kind["videos"][vid_cursor]
                vid_cursor += 1
                supplied[name] = chosen

        resolved_params: dict[str, Any] = {}
        for name, slot in exposed_by_name.items():
            ptype = str(slot.get("param_type") or slot.get("type") or "").strip()
            required = bool(slot.get("required", False))
            if name not in supplied or supplied[name] is None:
                if "default_value" in slot and slot["default_value"] is not None:
                    val = slot["default_value"]
                elif required:
                    raise ValueError(f"missing_required_workflow_param:{name}")
                else:
                    continue
            else:
                val = supplied[name]

            if ptype == "integer":
                if isinstance(val, bool) or not isinstance(val, int):
                    raise ValueError(f"invalid_workflow_param_type:{name}")
                if isinstance(slot.get("minimum"), int) and val < slot["minimum"]:
                    raise ValueError(f"workflow_param_out_of_range:{name}")
                if isinstance(slot.get("maximum"), int) and val > slot["maximum"]:
                    raise ValueError(f"workflow_param_out_of_range:{name}")
                resolved_params[name] = val
            elif ptype == "number":
                if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
                    raise ValueError(f"invalid_workflow_param_type:{name}")
                fval = float(val)
                if isinstance(slot.get("minimum"), (int, float)) and fval < float(slot["minimum"]):
                    raise ValueError(f"workflow_param_out_of_range:{name}")
                if isinstance(slot.get("maximum"), (int, float)) and fval > float(slot["maximum"]):
                    raise ValueError(f"workflow_param_out_of_range:{name}")
                resolved_params[name] = fval
            elif ptype == "boolean":
                if not isinstance(val, bool):
                    raise ValueError(f"invalid_workflow_param_type:{name}")
                resolved_params[name] = val
            elif ptype == "enum":
                allowed = slot.get("enum_values") if isinstance(slot.get("enum_values"), list) else []
                if not isinstance(val, str) or (allowed and val not in allowed):
                    raise ValueError(f"invalid_workflow_enum:{name}")
                resolved_params[name] = val
            elif ptype == "string":
                if not isinstance(val, str):
                    raise ValueError(f"invalid_workflow_param_type:{name}")
                min_len = slot.get("min_length")
                max_len = slot.get("max_length")
                if isinstance(min_len, int) and len(val) < min_len:
                    raise ValueError(f"workflow_param_length:{name}")
                if isinstance(max_len, int) and len(val) > max_len:
                    raise ValueError(f"workflow_param_length:{name}")
                resolved_params[name] = val
            elif ptype in {"artifact_image", "artifact_mask", "artifact_video"}:
                if not isinstance(val, str) or not val.strip():
                    raise ValueError(f"invalid_workflow_artifact:{name}")
                sval = val.strip()
                if _is_uuid(sval):
                    resolved_params[name] = sval.lower()
                else:
                    art = self.executor.artifacts.get(sval, project_id)
                    expected_prefix = "video/" if ptype == "artifact_video" else "image/"
                    if not art.media_type.startswith(expected_prefix):
                        raise ValueError(f"invalid_workflow_artifact_media:{name}")
                    resolved_params[name] = art.artifact_id
                reference_mapping.append({"slot": name, "artifact_id": resolved_params[name], "kind": ptype})
            else:
                raise ValueError(f"unsupported_workflow_slot_type:{ptype}")

        return resolved_params, reference_mapping, normalized_refs

    def normalize(
        self,
        *,
        project_id: str,
        prompt: str,
        model: str | None,
        duration_seconds: int = 5,
        aspect_ratio: str | None = None,
        references: dict[str, list[dict[str, str]]] | None = None,
        mode: str | None = None,
        resolution: str | None = None,
        quality_profile: str | None = None,
        workflow_params: dict[str, Any] | None = None,
        cached_entry: dict[str, Any] | None = None,
        saved_authority: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, str]]:
        if not self.configured():
            raise ChannelError("provider_not_configured", source="comfyui", accepted=False)
        if not isinstance(model, str) or not MODEL_ID_RE.fullmatch(model.strip()):
            raise ValueError("invalid_comfyui_model_id")
        model = model.strip()
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("invalid_prompt")
        if type(duration_seconds) is not int or not 1 <= duration_seconds <= 60:
            raise ValueError("unsupported_duration")

        actor = self._resolve_actor(project_id, saved_authority=saved_authority)
        if cached_entry is not None:
            entry = cached_entry
        else:
            live_models = {m["model_id"]: m for m in self.discover_models(actor["organization_id"])}
            entry = live_models.get(model)
            if entry is None:
                raise ChannelError("channel_unavailable", source="comfyui", accepted=False)

        resolved_params, reference_mapping, normalized_refs = self._validate_and_map_inputs(
            project_id=project_id,
            prompt=prompt,
            duration_seconds=duration_seconds,
            aspect_ratio=aspect_ratio,
            references=references,
            workflow_params=workflow_params,
            spec=entry["spec"],
        )
        request = {
            "project_id": project_id,
            "route": "comfyui",
            "channel": "local",
            "model": model,
            "prompt": prompt.strip(),
            "duration_seconds": duration_seconds,
            "aspect_ratio": aspect_ratio or "16:9",
            "mode": mode or ("image_to_video" if normalized_refs.get("images") else "text_to_video"),
            "resolution": resolution,
            "quality_profile": quality_profile,
            "references": normalized_refs,
            "workflow_params": resolved_params,
            "reference_mapping": reference_mapping,
            "resolved_output": {
                "model": model,
                "endpoint_slug": entry["endpoint_slug"],
                "version": entry["version"],
                "channel": "local",
                "source_badge": "本地",
            },
        }
        canonical = json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        digest = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return request, digest, entry, actor

    def preflight(
        self,
        *,
        project_id: str,
        prompt: str,
        model: str | None,
        duration_seconds: int = 5,
        aspect_ratio: str | None = None,
        references: dict[str, list[dict[str, str]]] | None = None,
        mode: str | None = None,
        resolution: str | None = None,
        quality_profile: str | None = None,
        workflow_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request, digest, entry, _ = self.normalize(
            project_id=project_id,
            prompt=prompt,
            model=model,
            duration_seconds=duration_seconds,
            aspect_ratio=aspect_ratio,
            references=references,
            mode=mode,
            resolution=resolution,
            quality_profile=quality_profile,
            workflow_params=workflow_params,
        )
        return {
            "status": "validated",
            "generation_verified": False,
            "can_submit": True,
            "route": "comfyui",
            "channel": "local",
            "source_badge": "本地",
            "display_name": entry["display_name"],
            "input_digest": digest,
            "resolved_output": request["resolved_output"],
            "reference_mapping": request["reference_mapping"],
        }

    def _effective_comfy_project_id(self, rec: TaskRecord, state: dict[str, Any]) -> str:
        if _is_uuid(rec.project_id):
            return rec.project_id.lower()
        wf_proj = str(state.get("workflow_project_id") or "").strip()
        if _is_uuid(wf_proj):
            return wf_proj.lower()
        raise ChannelError("invalid_request", source="comfyui", accepted=False)

    def _upload_local_artifact_to_studio(
        self,
        *,
        local_artifact_id: str,
        local_project_id: str,
        studio_project_id: str,
        actor: dict[str, str],
    ) -> str:
        if _is_uuid(local_artifact_id):
            return local_artifact_id.lower()
        if not self.artifact_service_url or not self.artifact_service_token:
            raise ChannelError("provider_not_configured", source="comfyui", accepted=False)
        art = self.executor.artifacts.get(local_artifact_id, local_project_id)
        raw = self.executor.artifacts.content_path(art).read_bytes()
        headers = {
            "Authorization": f"Bearer {self.artifact_service_token}",
            "X-Organization-Id": actor["organization_id"],
            "X-User-Id": actor["user_id"],
            "X-Request-Id": str(uuid.uuid4()),
        }
        init_resp = self.executor.client.post(
            f"{self.artifact_service_url}/v2/artifacts/uploads",
            headers=headers,
            json={
                "project_id": studio_project_id,
                "filename": art.filename,
                "mime_type": art.media_type,
                "size_bytes": len(raw),
                "sha256": art.sha256,
                "idempotency_key": f"vid-in-{local_artifact_id}-{art.sha256[:16]}",
            },
            timeout=30,
        )
        if init_resp.status_code not in {200, 201}:
            raise ChannelError("reference_upload_failed", source="comfyui", accepted=False)
        init_data = init_resp.json()
        upload_id = str(init_data.get("upload_id") or "").strip()
        if not upload_id:
            raise ChannelError("upload_protocol_error", source="comfyui", accepted=False)

        put_resp = self.executor.client.put(
            f"{self.artifact_service_url}/v2/artifacts/uploads/{quote(upload_id, safe='')}/content",
            headers={
                **headers,
                "Content-Type": art.media_type,
                "X-Request-Id": str(uuid.uuid4()),
            },
            content=raw,
            timeout=120,
        )
        if put_resp.status_code not in {200, 201, 204}:
            raise ChannelError("reference_upload_failed", source="comfyui", accepted=False)

        commit_resp = self.executor.client.post(
            f"{self.artifact_service_url}/v2/artifacts/uploads/{quote(upload_id, safe='')}/commit",
            headers={**headers, "X-Request-Id": str(uuid.uuid4())},
            json={"project_id": studio_project_id},
            timeout=30,
        )
        if commit_resp.status_code not in {200, 201}:
            raise ChannelError("reference_upload_failed", source="comfyui", accepted=False)
        commit_data = commit_resp.json()
        studio_id = str(commit_data.get("artifact_id") or "").strip().lower()
        if not _is_uuid(studio_id):
            raise ChannelError("upload_protocol_error", source="comfyui", accepted=False)
        return studio_id

    def _call_headless(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        actor: dict[str, str],
        timeout: float = 30.0,
    ) -> tuple[int, dict[str, Any]]:
        resp = self.executor.client.post(
            f"{self.origin}/internal/apps/comfyui-api/call",
            headers={
                "Authorization": f"Bearer {self.token}",
                "X-Organization-Id": actor["organization_id"],
                "X-User-Id": actor["user_id"],
                "X-Request-Id": str(uuid.uuid4()),
            },
            json={"name": tool_name, "arguments": arguments},
            timeout=timeout,
        )
        try:
            body = resp.json()
        except ValueError:
            body = {}
        return resp.status_code, body if isinstance(body, dict) else {}

    def generate(
        self,
        *,
        project_id: str,
        idempotency_key: str,
        model: str | None,
        prompt: str,
        duration_seconds: int = 5,
        aspect_ratio: str | None = None,
        references: dict[str, list[dict[str, str]]] | None = None,
        quality_profile: str | None = None,
        mode: str | None = None,
        resolution: str | None = None,
        workflow_params: dict[str, Any] | None = None,
    ) -> TaskRecord:
        with self.locked():
            existing = self.executor.tasks.find_idempotency(project_id, idempotency_key)
            if existing is not None:
                cached_entry = (existing.provider_state or {}).get("_comfyui_entry")
                if existing.service == "comfyui" and isinstance(cached_entry, dict):
                    _, digest, _, _ = self.normalize(
                        project_id=project_id,
                        prompt=prompt,
                        model=model,
                        duration_seconds=duration_seconds,
                        aspect_ratio=aspect_ratio,
                        references=references,
                        mode=mode,
                        resolution=resolution,
                        quality_profile=quality_profile,
                        workflow_params=workflow_params,
                        cached_entry=cached_entry,
                        saved_authority=existing.authority,
                    )
                    if existing.input_digest != digest:
                        raise TaskConflict("idempotency key already exists with different input")
                    return existing
                raise TaskConflict("idempotency key already exists with different input")

            request, digest, entry, actor = self.normalize(
                project_id=project_id,
                prompt=prompt,
                model=model,
                duration_seconds=duration_seconds,
                aspect_ratio=aspect_ratio,
                references=references,
                mode=mode,
                resolution=resolution,
                quality_profile=quality_profile,
                workflow_params=workflow_params,
            )
            pstate: dict[str, Any] = {
                "display_name": entry["display_name"],
                "source_badge": "本地",
                "endpoint_slug": entry["endpoint_slug"],
                "version": entry["version"],
                "workflow_project_id": entry["project_id"],
                "uploaded_refs": {},
                "_comfyui_entry": entry,
            }
            rec = self.executor.tasks.create(
                project_id=project_id,
                idempotency_key=idempotency_key,
                input_digest=digest,
                request=request,
                runtime_task_id="",
                status="queued",
                authority=actor,
                runtime_endpoint=self.origin,
                runtime_version=f"comfyui-api-v1:{entry['endpoint_slug']}@{entry['version']}",
                runtime_route="comfyui",
            )
            rec = self.executor.tasks.update(
                rec,
                service="comfyui",
                runtime_stage="prepared",
                provider_state=pstate,
            )
            return self._submit(rec)

    def _submit(self, rec: TaskRecord) -> TaskRecord:
        state = dict(rec.provider_state or {})
        actor = self._resolve_actor(rec.project_id, saved_authority=rec.authority)
        studio_project_id = self._effective_comfy_project_id(rec, state)
        entry = state.get("_comfyui_entry") or {}
        spec = entry.get("spec") or {}
        exposed_types = {
            str(s.get("name")): str(s.get("param_type") or s.get("type") or "")
            for s in (spec.get("exposed_inputs") or [])
            if isinstance(s, dict) and s.get("name")
        }
        uploaded_refs: dict[str, str] = dict(state.get("uploaded_refs") or {})
        invoke_params: dict[str, Any] = {}
        try:
            for key, val in (rec.request.get("workflow_params") or {}).items():
                if exposed_types.get(key) in {"artifact_image", "artifact_mask", "artifact_video"} and isinstance(val, str):
                    if val not in uploaded_refs:
                        uploaded_refs[val] = self._upload_local_artifact_to_studio(
                            local_artifact_id=val,
                            local_project_id=rec.project_id,
                            studio_project_id=studio_project_id,
                            actor=actor,
                        )
                        state["uploaded_refs"] = dict(uploaded_refs)
                        rec = self.executor.tasks.update(rec, provider_state=state)
                    invoke_params[key] = uploaded_refs[val]
                else:
                    invoke_params[key] = val
        except ChannelError as exc:
            return self.executor.tasks.update(
                rec,
                status="failed",
                runtime_stage="failed",
                error={"code": exc.code, "message": str(exc)},
                provider_state=state,
            )

        rec = self.executor.tasks.update(rec, runtime_stage="submitting", provider_state=state)
        try:
            code, body = self._call_headless(
                tool_name="api.invoke",
                arguments={
                    "project_id": studio_project_id,
                    "idempotency_key": f"vid-{rec.video_task_id}",
                    "endpoint_slug": state["endpoint_slug"],
                    "version": state["version"],
                    "params": invoke_params,
                },
                actor=actor,
                timeout=30.0,
            )
        except (httpx.HTTPError, OSError):
            return self.executor.tasks.update(
                rec,
                status="queued",
                runtime_stage="submission_unknown",
                error={"code": "submission_unknown", "message": "ComfyUI invoke outcome unknown; do not resubmit"},
                provider_state=state,
            )

        if 400 <= code < 500:
            err_code = str(body.get("code") or "provider_rejected")
            return self.executor.tasks.update(
                rec,
                status="failed",
                runtime_stage="failed",
                error={"code": err_code, "message": str(body.get("message") or err_code)},
                provider_state=state,
            )
        if code >= 500 or not isinstance(body.get("task_id"), str) or not body["task_id"].strip():
            return self.executor.tasks.update(
                rec,
                status="queued",
                runtime_stage="submission_unknown",
                error={"code": "submission_unknown", "message": "ComfyUI invoke outcome unknown; do not resubmit"},
                provider_state=state,
            )

        remote_id = body["task_id"].strip()
        remote_state = str(body.get("state") or "queued")
        mapped_status = (
            "queued"
            if remote_state in {"accepted", "validated", "queued"}
            else "running"
        )
        rec = self.executor.tasks.update(
            rec,
            runtime_task_id=remote_id,
            status=mapped_status,
            runtime_stage="remote_pending",
            dispatched_at=datetime.now(UTC).isoformat(),
            error=None,
            provider_state=state,
        )
        if remote_state in {"succeeded", "failed", "cancelled"}:
            return self._apply_remote_snapshot(rec, body)
        return rec

    def _output_path(self, rec: TaskRecord) -> Path:
        return self.executor.tasks.root / f"{rec.video_task_id}.comfyui.mp4"

    @staticmethod
    def probe(path: Path) -> dict[str, Any]:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        info = json.loads(result.stdout)
        video = next(x for x in info["streams"] if x["codec_type"] == "video")
        audio = next((x for x in info["streams"] if x["codec_type"] == "audio"), {})
        fps_raw = video.get("avg_frame_rate") or video.get("r_frame_rate") or "24/1"
        fps = float(Fraction(fps_raw)) if fps_raw != "0/0" else 24.0
        if not 0 < fps <= 120 or int(video["width"]) * int(video["height"]) > 4096 * 2160:
            raise ValueError("result_integrity_failed")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
            check=True,
            capture_output=True,
            timeout=180,
        )
        return {
            "width": int(video["width"]),
            "height": int(video["height"]),
            "frame_rate": fps,
            "duration_ms": round(float(info["format"]["duration"]) * 1000),
            "video_codec": video["codec_name"],
            "audio_codec": audio.get("codec_name"),
            "audio_channels": audio.get("channels"),
            "audio_sample_rate": audio.get("sample_rate"),
        }

    def _collect_output(
        self,
        rec: TaskRecord,
        result_items: list[dict[str, Any]],
        state: dict[str, Any],
    ) -> TaskRecord:
        if not result_items or not isinstance(result_items[0], dict):
            return self.executor.tasks.update(
                rec,
                status="failed",
                runtime_stage="failed",
                error={"code": "provider_protocol_error", "message": "ComfyUI succeeded without output artifact"},
                provider_state=state,
            )
        primary = result_items[0]
        studio_artifact_id = str(primary.get("artifact_id") or "").strip().lower()
        expected_sha256 = str(primary.get("sha256") or "").strip().lower()
        expected_size = int(primary.get("size_bytes") or 0)
        media_type = str(primary.get("mime_type") or "video/mp4").strip() or "video/mp4"
        filename = str(primary.get("filename") or f"{rec.video_task_id}.mp4").strip() or f"{rec.video_task_id}.mp4"
        if not _is_uuid(studio_artifact_id) or len(expected_sha256) != 64:
            return self.executor.tasks.update(
                rec,
                status="failed",
                runtime_stage="failed",
                error={"code": "provider_protocol_error", "message": "Invalid ComfyUI output artifact receipt"},
                provider_state=state,
            )
        actor = self._resolve_actor(rec.project_id, saved_authority=rec.authority)
        studio_project_id = self._effective_comfy_project_id(rec, state)
        output = self._output_path(rec)
        try:
            if not output.exists() or state.get("result_sha256") != expected_sha256:
                if not self.artifact_service_url or not self.artifact_service_token:
                    raise ValueError("artifact_service_not_configured")
                pending = output.with_suffix(".part")
                digest = hashlib.sha256()
                size = 0
                with self.executor.client.stream(
                    "GET",
                    f"{self.artifact_service_url}/v2/artifacts/{quote(studio_artifact_id, safe='')}/content",
                    params={"project_id": studio_project_id, "artifact_id": studio_artifact_id},
                    headers={
                        "Authorization": f"Bearer {self.artifact_service_token}",
                        "X-Organization-Id": actor["organization_id"],
                        "X-User-Id": actor["user_id"],
                        "X-Request-Id": str(uuid.uuid4()),
                    },
                    timeout=180.0,
                ) as resp:
                    resp.raise_for_status()
                    with pending.open("wb") as handle:
                        for chunk in resp.iter_bytes():
                            size += len(chunk)
                            if size > 1024**3:
                                raise ValueError("result_size_exceeded")
                            digest.update(chunk)
                            handle.write(chunk)
                        handle.flush()
                actual_sha = digest.hexdigest()
                if actual_sha != expected_sha256 or (expected_size > 0 and size != expected_size):
                    pending.unlink(missing_ok=True)
                    return self.executor.tasks.update(
                        rec,
                        status="failed",
                        runtime_stage="failed",
                        error={"code": "result_integrity_failed", "message": "ComfyUI output SHA-256 mismatch"},
                        provider_state=state,
                    )
                os.replace(pending, output)
                state["result_sha256"] = actual_sha
                state["studio_output_artifact_id"] = studio_artifact_id
                rec = self.executor.tasks.update(rec, provider_state=state)

            media = self.probe(output)
            media["sha256"] = expected_sha256
            media["studio_artifact_id"] = studio_artifact_id
            local_art_id = "art_" + hashlib.sha256(rec.video_task_id.encode("utf-8")).hexdigest()[:32]
            try:
                asset = self.executor.artifacts.get(local_art_id, rec.project_id)
                if asset.sha256 != expected_sha256:
                    raise ValueError("result_integrity_failed")
            except ArtifactNotFound:
                with output.open("rb") as stream:
                    asset = self.executor.artifacts.create_from_chunks(
                        project_id=rec.project_id,
                        operation="comfyui",
                        filename=filename,
                        media_type=media_type if media_type.startswith("video/") else "video/mp4",
                        artifact_id=local_art_id,
                        chunks=iter(lambda: stream.read(1024 * 1024), b""),
                        expected_sha256=expected_sha256,
                    )
            state["worker_settled"] = True
            state["progress"] = 100
            return self.executor.tasks.update(
                rec,
                status="succeeded",
                artifact_id=asset.artifact_id,
                media=media,
                runtime_stage="completed",
                provider_state=state,
                error=None,
            )
        except (httpx.HTTPError, OSError, ValueError, subprocess.SubprocessError, StopIteration, KeyError):
            return self.executor.tasks.update(
                rec,
                status="running",
                runtime_stage="collect_retry",
                error={"code": "result_collection_failed", "message": "ComfyUI output collection deferred; retry status"},
                provider_state=state,
            )

    def _apply_remote_snapshot(self, rec: TaskRecord, data: dict[str, Any]) -> TaskRecord:
        state = dict(rec.provider_state or {})
        remote_state = str(data.get("state") or "")
        instance_id = data.get("instance_id") if _is_uuid(data.get("instance_id")) else rec.execution_instance_id
        if remote_state in {"accepted", "validated", "queued"}:
            return self.executor.tasks.update(
                rec,
                status="queued",
                runtime_stage="queued",
                execution_instance_id=instance_id,
                provider_state=state,
                error=None,
            )
        if remote_state in {"dispatching", "running", "collecting", "reconciling"}:
            return self.executor.tasks.update(
                rec,
                status="running",
                runtime_stage=remote_state,
                execution_instance_id=instance_id,
                provider_state=state,
                error=None,
            )
        if remote_state == "cancelled":
            state["worker_settled"] = True
            return self.executor.tasks.update(
                rec,
                status="cancelled",
                runtime_stage="cancelled",
                execution_instance_id=instance_id,
                provider_state=state,
                error={"code": "cancelled", "message": "ComfyUI workflow task was cancelled"},
            )
        if remote_state == "failed":
            state["worker_settled"] = True
            return self.executor.tasks.update(
                rec,
                status="failed",
                runtime_stage="failed",
                execution_instance_id=instance_id,
                provider_state=state,
                error={
                    "code": "provider_generation_failed",
                    "message": str(data.get("error") or "ComfyUI workflow execution failed"),
                },
            )
        if remote_state == "succeeded":
            rec = self.executor.tasks.update(
                rec,
                execution_instance_id=instance_id,
                runtime_stage="collecting",
                provider_state=state,
            )
            results = data.get("result") if isinstance(data.get("result"), list) else []
            return self._collect_output(rec, results, state)
        return rec

    def status(self, record: TaskRecord) -> TaskRecord:
        with self.locked():
            rec = self.executor.tasks.get(record.video_task_id)
            if rec.status in {"succeeded", "failed", "cancelled"}:
                return rec
            state = dict(rec.provider_state or {})
            actor = self._resolve_actor(rec.project_id, saved_authority=rec.authority)
            studio_project_id = self._effective_comfy_project_id(rec, state)
            if not rec.runtime_task_id:
                if rec.runtime_stage == "submission_unknown":
                    try:
                        code, body = self._call_headless(
                            tool_name="api.invoke",
                            arguments={
                                "project_id": studio_project_id,
                                "idempotency_key": f"vid-{rec.video_task_id}",
                                "endpoint_slug": state["endpoint_slug"],
                                "version": state["version"],
                                "params": {
                                    k: state.get("uploaded_refs", {}).get(v, v)
                                    for k, v in (rec.request.get("workflow_params") or {}).items()
                                },
                            },
                            actor=actor,
                            timeout=20.0,
                        )
                        if code == 200 and isinstance(body.get("task_id"), str) and body["task_id"].strip():
                            rec = self.executor.tasks.update(
                                rec,
                                runtime_task_id=body["task_id"].strip(),
                                dispatched_at=rec.dispatched_at or datetime.now(UTC).isoformat(),
                                error=None,
                            )
                            return self._apply_remote_snapshot(rec, body)
                    except (httpx.HTTPError, OSError):
                        pass
                return rec

            try:
                code, body = self._call_headless(
                    tool_name="task.get",
                    arguments={"project_id": studio_project_id, "task_id": rec.runtime_task_id},
                    actor=actor,
                    timeout=20.0,
                )
            except (httpx.HTTPError, OSError):
                return rec
            if code == 404:
                state["missing_count"] = int(state.get("missing_count", 0)) + 1
                if state["missing_count"] >= 3:
                    return self.executor.tasks.update(
                        rec,
                        status="failed",
                        runtime_stage="failed",
                        provider_state=state,
                        error={"code": "remote_not_found", "message": "ComfyUI task not found"},
                    )
                return self.executor.tasks.update(rec, provider_state=state)
            if code != 200:
                return rec
            return self._apply_remote_snapshot(rec, body)

    def result(self, record: TaskRecord) -> TaskRecord:
        rec = self.status(record)
        if rec.status != "succeeded" or not rec.artifact_id:
            from ..executor import ExecutionError

            raise ExecutionError("video task has not succeeded")
        return rec

    def cancel(self, record: TaskRecord) -> TaskRecord:
        with self.locked():
            rec = self.executor.tasks.get(record.video_task_id)
            if rec.status in {"succeeded", "failed", "cancelled"}:
                return rec
            state = dict(rec.provider_state or {})
            if not rec.runtime_task_id:
                state["worker_settled"] = True
                return self.executor.tasks.update(
                    rec,
                    status="cancelled",
                    runtime_stage="cancelled",
                    provider_state=state,
                    error={"code": "cancelled", "message": "ComfyUI workflow task was cancelled"},
                )
            actor = self._resolve_actor(rec.project_id, saved_authority=rec.authority)
            studio_project_id = self._effective_comfy_project_id(rec, state)
            try:
                code, body = self._call_headless(
                    tool_name="task.cancel",
                    arguments={
                        "project_id": studio_project_id,
                        "idempotency_key": f"cancel-{rec.video_task_id}",
                        "task_id": rec.runtime_task_id,
                    },
                    actor=actor,
                    timeout=20.0,
                )
                if code == 200 and isinstance(body, dict):
                    state["cancel_delivered"] = True
                    rec = self.executor.tasks.update(rec, provider_state=state)
                    return self._apply_remote_snapshot(rec, body)
            except (httpx.HTTPError, OSError):
                pass
            state["_cancel_requested"] = True
            return self.executor.tasks.update(rec, provider_state=state)
