"""Persisted Grok REST jobs; remote completion includes local result collection."""
from __future__ import annotations

import base64
import hashlib
import io
import ipaddress
import json
import os
import random
import socket
import sqlite3
import subprocess
import time
from contextlib import contextmanager, closing
from datetime import UTC, datetime
from fractions import Fraction
from urllib.parse import quote, urlsplit

import httpx
from PIL import Image, ImageOps

from ..tasks import TaskConflict
from ..artifacts import ArtifactNotFound

RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4"]
MODELS = {"grok-imagine-video", "grok-imagine-video-1.5", "grok-imagine-video-1.5-lite"}
VERSION = "grok-video-v1"
CHANNEL_VERSION = "grok-channel-v1"


class GrokVideoAdapter:
    def __init__(self, executor):
        self.executor = executor
        self.channel_url = (os.getenv("VIDEO_GROK_CHANNEL_URL") or os.getenv("IMAGE_GROK_CHANNEL_URL") or "").rstrip("/")
        self.channel_token = (os.getenv("VIDEO_GROK_CHANNEL_TOKEN") or os.getenv("IMAGE_GROK_CHANNEL_TOKEN") or "").strip()
        self.channel_mode = bool(self.channel_url and len(self.channel_token) >= 32)
        self.base = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1").rstrip("/")
        self.key = os.getenv("XAI_API_KEY", "")
        self.model = os.getenv("XAI_VIDEO_MODEL", "grok-imagine-video" if self.channel_mode else "grok-imagine-video-1.5")
        self.enabled = self.channel_mode or (os.getenv("XAI_VIDEO_ENABLED", "false").lower() == "true")
        profile_seed = (self.channel_url + "\0" + CHANNEL_VERSION) if self.channel_mode else (self.base + "\0" + self.key)
        self.profile = hashlib.sha256(profile_seed.encode()).hexdigest()
        self.download_client = httpx.Client(follow_redirects=False, timeout=120)
        self.hosts = {"vidgen.x.ai", "assets.grok.com", urlsplit(self.base).hostname}
        self.hosts.update(x.strip() for x in os.getenv("XAI_RESULT_HOSTS", "").split(",") if x.strip())

    def configured(self):
        if self.channel_mode:
            p = urlsplit(self.channel_url)
            return bool(p.scheme in {"http", "https"} and p.hostname and not
                        (p.username or p.password or p.query or p.fragment) and self.model in MODELS)
        p = urlsplit(self.base)
        return bool(self.key and p.scheme == "https" and p.hostname and not
                    (p.username or p.password or p.query or p.fragment) and self.model in MODELS)

    def capability(self):
        return {"route": "grok", "provider": "xai", "model": self.model,
                "models": sorted(MODELS) if self.channel_mode else [self.model],
                "enabled": self.enabled, "configured": self.configured(),
                "readiness": "not_checked", "verification_status": "not_recorded",
                "adapter_version": CHANNEL_VERSION if self.channel_mode else VERSION,
                "modes": ["text_to_video", "image_to_video"],
                "duration_seconds": {"minimum": 1, "maximum": 15}, "aspect_ratios": RATIOS,
                "resolutions": ["480p", "720p"], "limit_scope": "adapter",
                "poll": True, "callback": False, "cancel": False, "progress": "optional",
                "runtime_metrics": False}

    @contextmanager
    def locked(self):
        self.executor.tasks.ensure_ready()
        with closing(sqlite3.connect(self.executor.tasks.root / "grok-lock.db", timeout=120)) as conn:
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
        if not isinstance(references, dict) or set(references) - {"images", "videos", "audios"}:
            raise ValueError("invalid_references")
        refs = {k: references.get(k, []) for k in ("images", "videos", "audios")}
        if any(not isinstance(v, list) for v in refs.values()) or refs["videos"] or refs["audios"] or len(refs["images"]) > 1:
            raise ValueError("unsupported_references")
        resolved_mode = "image_to_video" if refs["images"] else "text_to_video"
        if mode is not None and mode != resolved_mode:
            raise ValueError("mode_reference_mismatch")
        resolution = resolution or "720p"
        if resolution not in {"480p", "720p"}:
            raise ValueError("unsupported_resolution")
        info = None
        if refs["images"]:
            ref = refs["images"][0]
            if not isinstance(ref, dict) or set(ref) != {"artifact_id", "purpose"} or ref["purpose"] != "first_frame":
                raise ValueError("first_frame_reference_required")
            _, info = self.source(project_id, ref["artifact_id"])
            if aspect_ratio is not None:
                if aspect_ratio not in RATIOS:
                    raise ValueError("unsupported_aspect_ratio")
                a, b = map(int, aspect_ratio.split(":"))
                if abs(info["width"] * b - info["height"] * a) > max(a, b):
                    raise ValueError("aspect_ratio_conflict")
            else:
                aspect_ratio = str(Fraction(info["width"], info["height"])).replace("/", ":")
                if ":" not in aspect_ratio:
                    aspect_ratio += ":1"
        else:
            aspect_ratio = aspect_ratio or "16:9"
            if aspect_ratio not in RATIOS:
                raise ValueError("unsupported_aspect_ratio")
        request = dict(project_id=project_id, model=model, prompt=prompt,
                       duration_seconds=duration_seconds, aspect_ratio=aspect_ratio,
                       references=refs, mode=resolved_mode, resolution=resolution, route="grok",
                       source=info, endpoint_profile_id=self.profile)
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
        actor = None
        provider_state = {}
        if self.channel_mode:
            actor = self._resolve_actor(request["project_id"])
            provider_state["organization_id"] = actor["organization_id"]
            if channel_account_id is not None:
                provider_state["channel_account_id"] = str(channel_account_id)
            provider_state["budget_limit"] = str(budget_limit or os.getenv("VIDEO_GROK_DEFAULT_BUDGET_LIMIT", "2"))
        with self.locked():
            old = self.executor.tasks.find_idempotency(request["project_id"], idempotency_key)
            if old:
                if old.input_digest != digest:
                    raise TaskConflict("idempotency_conflict")
                return old
            if not self.enabled or not self.configured():
                raise ValueError("provider_not_configured")
            rec = self.executor.tasks.create(project_id=request["project_id"], idempotency_key=idempotency_key,
                input_digest=digest, request=request, runtime_task_id="", status="queued", authority=actor)
            rec = self.executor.tasks.update(rec, service="grok", runtime_route="grok",
                runtime_version=CHANNEL_VERSION if self.channel_mode else VERSION,
                runtime_stage="prepared", provider_state=provider_state)
            if self.channel_mode:
                try:
                    return self.submit(rec)
                except (OSError, ValueError):
                    return self.error(rec, "source_integrity_failed", terminal=True)
            return rec

    def save(self, rec, **values):
        return self.executor.tasks.update(rec, **values)

    def error(self, rec, code, *, terminal=False, stage=None, delay=30):
        state = dict(rec.provider_state or {})
        state["next_poll_at"] = time.time() + delay
        return self.save(rec, status="failed" if terminal else "running",
            error={"code": code, "message": code}, provider_state=state,
            runtime_stage=stage or ("failed" if terminal else rec.runtime_stage))

    def _channel_headers(self, rec):
        state = rec.provider_state or {}
        org = state.get("organization_id") or (rec.authority or {}).get("organization_id") or ""
        return {"Authorization": "Bearer " + self.channel_token,
                "X-Organization-Id": str(org), "X-Caller-Id": "video-mcp"}

    def submit(self, rec):
        p = rec.request
        if not self.enabled:
            return rec  # Disabling the channel stops undispatched work only.
        if self.channel_mode:
            state = dict(rec.provider_state or {})
            payload = {
                "prompt": p["prompt"],
                "duration": p["duration_seconds"],
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
                r = self.executor.client.post(self.channel_url + "/v1/calls",
                    headers=self._channel_headers(rec),
                    json={
                        "idempotency_key": rec.video_task_id,
                        "task_id": rec.video_task_id,
                        "model_id": p["model"],
                        "capability": "video",
                        "payload": payload,
                        "account_id": state.get("channel_account_id"),
                        "budget_limit": state.get("budget_limit") or "2",
                    }, timeout=60)
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
                    return self.save(rec, runtime_task_id=call_id, runtime_stage="submission_unknown",
                                     dispatched_at=datetime.now(UTC).isoformat(),
                                     error={"code": "submission_unknown", "message": "submission_unknown"})
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                return self.error(rec, "submission_unknown", stage="submission_unknown")
            return self.save(rec, runtime_task_id=call_id, runtime_stage="remote_pending",
                dispatched_at=datetime.now(UTC).isoformat(), error=None)
        payload = {"model": p["model"], "prompt": p["prompt"], "duration": p["duration_seconds"], "resolution": p["resolution"]}
        if p["mode"] == "image_to_video":
            data, info = self.source(rec.project_id, p["references"]["images"][0]["artifact_id"])
            if info != p["source"]:
                return self.error(rec, "source_integrity_failed", terminal=True)
            payload["image"] = {"url": "data:" + info["mime"] + ";base64," + base64.b64encode(data).decode()}
        else:
            payload["aspect_ratio"] = p["aspect_ratio"]
        rec = self.save(rec, status="running", runtime_stage="submitting")
        try:
            r = self.executor.client.post(self.base + "/videos/generations", json=payload,
                headers={"Authorization": "Bearer " + self.key}, timeout=60)
            if r.status_code in {401, 403, 400, 422}:
                return self.error(rec, "provider_auth_failed" if r.status_code in {401,403} else "provider_rejected", terminal=True)
            if not r.is_success:
                return self.error(rec, f"provider_http_{r.status_code}", stage="submission_unknown")
            identity = r.json()["request_id"]
            if not isinstance(identity, str) or not identity or len(identity) > 256:
                raise ValueError
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return self.error(rec, "submission_unknown", stage="submission_unknown")
        return self.save(rec, runtime_task_id=identity, runtime_stage="remote_pending",
            dispatched_at=datetime.now(UTC).isoformat(), error=None)

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
            if self.channel_mode:
                return self._channel_status(rec, state)
            if not rec.runtime_task_id:
                return self.error(rec, (rec.error or {}).get("code", "submission_unknown"), stage="submission_unknown", delay=300)
            output = self.output_path(rec)
            if output.exists() and state.get("result_sha256"):
                return self.collect(rec)
            try:
                r = self.executor.client.get(self.base + "/videos/" + quote(rec.runtime_task_id, safe=""),
                    headers={"Authorization": "Bearer " + self.key}, timeout=20)
                if r.status_code == 202:
                    data = {"status": "pending"}
                elif r.status_code == 404:
                    state["missing_count"] = int(state.get("missing_count", 0)) + 1
                    rec = self.save(rec, provider_state=state)
                    return self.error(rec, "remote_not_found", terminal=state["missing_count"] >= 5)
                elif r.status_code in {401, 403}:
                    return self.error(rec, "provider_auth_failed", delay=300)
                elif r.status_code == 429:
                    delay = r.headers.get("retry-after", "30")
                    return self.error(rec, "provider_rate_limited", delay=min(3600, max(2, int(delay))) if delay.isdigit() else 30)
                else:
                    r.raise_for_status()
                    data = r.json()
                status = data["status"]
                if status == "pending":
                    state["poll_count"] = int(state.get("poll_count", 0)) + 1
                    delay = min(15, 2 ** min(4, state["poll_count"]))
                    if rec.dispatched_at and (datetime.now(UTC)-datetime.fromisoformat(rec.dispatched_at)).total_seconds() > 600:
                        delay = 60
                    state["next_poll_at"] = time.time() + delay + random.random()
                    progress = data.get("progress")
                    state["progress"] = progress if type(progress) is int and 0 <= progress < 100 else None
                    return self.save(rec, provider_state=state, status="running", error=None, runtime_stage="remote_pending")
                if status in {"failed", "expired"}:
                    return self.error(rec, "provider_rejected" if status == "failed" else "result_expired", terminal=True)
                if status != "done" or not isinstance(data.get("video"), dict):
                    return self.error(rec, "provider_protocol_error")
                if data["video"].get("respect_moderation") is False:
                    return self.error(rec, "provider_rejected", terminal=True)
                url = data["video"].get("url")
                url, _ = self.result_request(url, rec.runtime_task_id)
                state.update(result_url=url, next_poll_at=0, resolved_model=data.get("model"),
                             usage=data.get("usage"), progress=100)
                return self.collect(self.save(rec, runtime_stage="collecting", provider_state=state))
            except (httpx.HTTPError, ValueError, TypeError, KeyError, OSError):
                return self.error(rec, "provider_protocol_error")

    def _channel_status(self, rec, state):
        output = self.output_path(rec)
        if output.exists() and state.get("result_sha256"):
            return self._channel_collect(rec, state)
        try:
            path = ("/v1/calls/" + quote(rec.runtime_task_id, safe="")) if rec.runtime_task_id else ("/v1/calls/by-idempotency/" + quote(rec.video_task_id, safe=""))
            r = self.executor.client.get(self.channel_url + path, headers=self._channel_headers(rec), timeout=30)
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
            return self._channel_collect(self.save(rec, runtime_stage="collecting", provider_state=state), state)
        except (httpx.HTTPError, ValueError, TypeError, KeyError, OSError):
            return self.error(rec, "provider_protocol_error")

    def _channel_collect(self, rec, state):
        output = self.output_path(rec)
        try:
            if not output.exists() or not state.get("result_sha256"):
                pending = output.with_suffix(".part")
                digest, count = hashlib.sha256(), 0
                with self.executor.client.stream("GET", self.channel_url + "/v1/calls/" + quote(rec.runtime_task_id, safe="") + "/result",
                                                 headers=self._channel_headers(rec), timeout=120) as response:
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
            media = self.probe(output)
            artifact_id = "art_" + hashlib.sha256(rec.video_task_id.encode()).hexdigest()[:32]
            try:
                asset = self.executor.artifacts.get(artifact_id, rec.project_id)
                if asset.sha256 != state["result_sha256"]:
                    raise ValueError("result_integrity_failed")
            except ArtifactNotFound:
                with output.open("rb") as f:
                    asset = self.executor.artifacts.create_from_chunks(project_id=rec.project_id,
                        operation="grok", filename="grok.mp4", media_type="video/mp4", artifact_id=artifact_id,
                        chunks=iter(lambda: f.read(1024 * 1024), b""), expected_sha256=state["result_sha256"])
            state["next_poll_at"] = 0
            return self.save(rec, status="succeeded", artifact_id=asset.artifact_id, media=media,
                             runtime_stage="completed", provider_state=state, error=None)
        except (OSError, ValueError, KeyError, TypeError, StopIteration, ZeroDivisionError, httpx.HTTPError, subprocess.SubprocessError):
            return self.error(rec, "result_collection_failed", stage="collect_retry")

    def output_path(self, rec):
        return self.executor.tasks.root / (rec.video_task_id + ".grok.mp4")

    def allowed_url(self, url):
        if not isinstance(url, str):
            raise ValueError("result_url_invalid")
        p = urlsplit(url)
        if p.scheme != "https" or p.hostname not in self.hosts or p.username or p.password or p.fragment or p.port not in {None,443}:
            raise ValueError("result_url_denied")
        for item in socket.getaddrinfo(p.hostname, 443, type=socket.SOCK_STREAM):
            if not ipaddress.ip_address(item[4][0]).is_global:
                raise ValueError("result_url_denied")
        return url

    def result_request(self, url, request_id):
        # A compatible gateway may return its protected content route. Build the
        # target from trusted configuration and the persisted task identity;
        # never attach credentials to an arbitrary provider-supplied URL.
        expected = self.base + "/videos/" + quote(request_id, safe="") + "/content"
        headers = {"Accept": "video/*"}
        if request_id and url in {expected, urlsplit(expected).path}:
            headers["Authorization"] = "Bearer " + self.key
            return self.allowed_url(expected), headers
        return self.allowed_url(url), headers

    def collect(self, rec):
        state = dict(rec.provider_state or {})
        output = self.output_path(rec)
        try:
            if not output.exists() or not state.get("result_sha256"):
                url, headers = self.result_request(state["result_url"], rec.runtime_task_id)
                pending = output.with_suffix(".part")
                digest, count = hashlib.sha256(), 0
                with self.download_client.stream("GET", url, headers=headers) as response:
                    response.raise_for_status()
                    if response.headers.get("content-type", "").split(";")[0] not in {"video/mp4", "application/octet-stream"}:
                        raise ValueError("result_integrity_failed")
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
            media = self.probe(output)
            if abs(media["duration_ms"] / 1000 - rec.request["duration_seconds"]) > max(2 / media["frame_rate"], .25):
                return self.error(rec, "output_contract_mismatch", terminal=True)
            expected_ratio = Fraction(rec.request["aspect_ratio"].replace(":", "/"))
            if abs(media["width"] - media["height"] * expected_ratio) > 2:
                return self.error(rec, "output_contract_mismatch", terminal=True)
            artifact_id = "art_" + hashlib.sha256(rec.video_task_id.encode()).hexdigest()[:32]
            try:
                asset = self.executor.artifacts.get(artifact_id, rec.project_id)
                if asset.sha256 != state["result_sha256"]:
                    raise ValueError("result_integrity_failed")
            except ArtifactNotFound:
                with output.open("rb") as f:
                    asset = self.executor.artifacts.create_from_chunks(project_id=rec.project_id,
                        operation="grok", filename="grok.mp4", media_type="video/mp4", artifact_id=artifact_id,
                        chunks=iter(lambda: f.read(1024 * 1024), b""), expected_sha256=state["result_sha256"])
            state.pop("result_url", None)
            state["next_poll_at"] = 0
            return self.save(rec, status="succeeded", artifact_id=asset.artifact_id, media=media,
                             runtime_stage="completed", provider_state=state, error=None)
        except (OSError, ValueError, KeyError, TypeError, StopIteration, ZeroDivisionError, httpx.HTTPError, subprocess.SubprocessError):
            return self.error(rec, "result_collection_failed", stage="collect_retry")

    @staticmethod
    def probe(path):
        result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                                check=True, capture_output=True, text=True, timeout=60)
        info = json.loads(result.stdout)
        video = next(x for x in info["streams"] if x["codec_type"] == "video")
        audio = next((x for x in info["streams"] if x["codec_type"] == "audio"), {})
        fps = float(Fraction(video["avg_frame_rate"]))
        if not 0 < fps <= 120 or video["width"] * video["height"] > 4096 * 2160:
            raise ValueError("result_integrity_failed")
        subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
                       check=True, capture_output=True, timeout=180)
        return {"width": video["width"], "height": video["height"], "frame_rate": fps,
                "duration_ms": round(float(info["format"]["duration"]) * 1000),
                "video_codec": video["codec_name"], "audio_codec": audio.get("codec_name"),
                "audio_channels": audio.get("channels"), "audio_sample_rate": audio.get("sample_rate")}

    def result(self, record):
        rec = self.status(record)
        if rec.status != "succeeded" or not rec.artifact_id:
            from ..executor import ExecutionError
            raise ExecutionError("video task has not succeeded")
        return rec
