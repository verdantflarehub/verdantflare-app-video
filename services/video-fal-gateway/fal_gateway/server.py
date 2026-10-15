"""Two authenticated internal listeners; no public route or browser credential storage."""
from __future__ import annotations

import asyncio
import hmac
import json
import os
from pathlib import Path
import re
import uuid
from urllib.parse import urlsplit

import httpx
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
import uvicorn

from . import __version__
from .dashboard import DashboardAssets
from .store import Conflict, Store

MODEL = "minimax/h3/reference-to-video"
QUEUE = "https://queue.fal.run"
CHECK = "https://api.fal.ai/v1/models/requests/by-endpoint"
IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}")
PAYLOAD_FIELDS = {"duration", "resolution", "enable_safety_checker", "sync_mode",
                  "prompt_expansion_mode", "aspect_ratio", "prompt",
                  "reference_image_urls", "reference_video_urls", "reference_audio_urls"}
ASPECT_RATIOS = {"adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"}


class APIError(Exception):
    def __init__(self, code: str, status: int = 400):
        self.code, self.status = code, status


def respond(value: dict, status: int = 200):
    return JSONResponse(value, status_code=status, headers={"Cache-Control": "no-store"})


async def body(request: Request, limit: int = 32768) -> dict:
    chunks = bytearray()
    async for chunk in request.stream():
        chunks.extend(chunk)
        if len(chunks) > limit:
            raise APIError("request_too_large", 413)
    try:
        def unique(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("duplicate_key")
                value[key] = item
            return value
        result = json.loads(chunks, object_pairs_hook=unique)
    except (ValueError, UnicodeDecodeError):
        raise APIError("invalid_request") from None
    if not isinstance(result, dict):
        raise APIError("invalid_request")
    return result


def number(value, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise APIError("invalid_request")
    return value


def identity(value: str) -> str:
    if not isinstance(value, str) or not IDENTITY.fullmatch(value):
        raise APIError("invalid_identity", 403)
    return value


def authenticate(request: Request, token: str, *, manage: bool) -> str:
    headers = request.headers.getlist("authorization")
    if len(token) < 32 or len(headers) != 1 or not hmac.compare_digest(headers[0], "Bearer " + token):
        raise APIError("unauthorized", 401)
    if manage:
        identity(request.headers.get("x-user-id", ""))
    return identity(request.headers.get("x-organization-id", ""))


class Gateway:
    def __init__(self, store: Store, client: httpx.Client | None = None):
        self.store = store
        self.client = client or httpx.Client(follow_redirects=False, timeout=20)

    def check(self, key: str) -> str:
        try:
            response = self.client.get(CHECK, params={"endpoint_id": MODEL, "limit": 1},
                                       headers={"Authorization": "Key " + key}, timeout=15)
        except httpx.HTTPError:
            return "unavailable"
        if response.status_code in {401, 403}:
            return "invalid"
        if response.status_code != 200:
            return "unavailable"
        try:
            data = response.json()
            return "active" if isinstance(data.get("items"), list) else "unavailable"
        except (ValueError, AttributeError):
            return "unavailable"

    @staticmethod
    def validate_payload(value) -> dict:
        if not isinstance(value, dict) or not value or set(value) - PAYLOAD_FIELDS:
            raise APIError("invalid_payload")
        if not isinstance(value.get("prompt"), str) or not value["prompt"].strip():
            raise APIError("invalid_payload")
        if (value.get("resolution") != "480P" or value.get("enable_safety_checker") is not True
                or value.get("sync_mode") is not False or value.get("prompt_expansion_mode") != "disabled"
                or value.get("aspect_ratio") not in ASPECT_RATIOS):
            raise APIError("unsupported_profile")
        number(value.get("duration"), 5, 15)
        for field in ("reference_image_urls", "reference_video_urls", "reference_audio_urls"):
            refs = value.get(field, [])
            if not isinstance(refs, list) or len(refs) > {"reference_image_urls": 9,
                    "reference_video_urls": 3, "reference_audio_urls": 3}[field]:
                raise APIError("invalid_payload")
            prefix = {"reference_image_urls": "data:image/", "reference_video_urls": "data:video/",
                      "reference_audio_urls": "data:audio/"}[field]
            if any(not isinstance(item, str) or not item.startswith(prefix) for item in refs):
                raise APIError("invalid_payload")
        if not any(value.get(field) for field in ("reference_image_urls", "reference_video_urls", "reference_audio_urls")):
            raise APIError("reference_required")
        if sum(len(value.get(field, [])) for field in ("reference_image_urls", "reference_video_urls", "reference_audio_urls")) > 12:
            raise APIError("invalid_payload")
        return value

    def submit(self, organization_id: str, operation_id: str, payload: dict) -> dict:
        item, fresh = self.store.reserve_submission(organization_id, operation_id, MODEL, payload, str(uuid.uuid4()))
        if not fresh:
            return {"id": item["id"], "state": item["state"]}
        key = self.store.decrypt(organization_id, item["credential_version"])
        try:
            response = self.client.post(QUEUE + "/" + MODEL, json=payload,
                                        headers={"Authorization": "Key " + key}, timeout=60)
        except httpx.ConnectError:
            self.store.reject_submission(item["id"])
            return {"id": item["id"], "state": "rejected"}
        except httpx.HTTPError:
            return {"id": item["id"], "state": "submission_unconfirmed"}
        if response.status_code in {400, 401, 403, 404, 422, 429}:
            self.store.reject_submission(item["id"])
            return {"id": item["id"], "state": "rejected"}
        if not response.is_success:
            return {"id": item["id"], "state": "submission_unconfirmed"}
        try:
            remote = response.json()["request_id"]
            if not isinstance(remote, str) or not REQUEST_ID.fullmatch(remote):
                raise ValueError
        except (ValueError, KeyError, TypeError):
            return {"id": item["id"], "state": "submission_unconfirmed"}
        self.store.confirm_submission(item["id"], remote)
        return {"id": item["id"], "state": "queued"}

    def remote(self, organization_id: str, submission_id: str, result: bool = False) -> dict:
        item = self.store.submission(organization_id, submission_id)
        if not item["provider_request_id"]:
            return {"id": item["id"], "state": item["state"]}
        key = self.store.decrypt(organization_id, item["credential_version"])
        url = QUEUE + "/minimax/h3/requests/" + item["provider_request_id"] + ("" if result else "/status")
        try:
            response = self.client.get(url, headers={"Authorization": "Key " + key}, timeout=30)
        except httpx.HTTPError:
            raise APIError("provider_unavailable", 503) from None
        if response.status_code in {401, 403}:
            raise APIError("provider_auth_failed", 503)
        if not response.is_success:
            raise APIError("provider_unavailable", 503)
        try:
            data = response.json()
            if result:
                video = data["video"]["url"]
                host = urlsplit(video).hostname if isinstance(video, str) else None
                if (not isinstance(video, str) or urlsplit(video).scheme != "https" or
                        not host or not (host == "fal.media" or host.endswith(".fal.media")) or
                        urlsplit(video).username or urlsplit(video).password):
                    raise ValueError
                return {"id": item["id"], "state": "completed", "video_url": video}
            state = data["status"]
            if state not in {"IN_QUEUE", "IN_PROGRESS", "COMPLETED"}:
                raise ValueError
            return {"id": item["id"], "state": state.lower(), "metrics": data.get("metrics") if isinstance(data.get("metrics"), dict) else None,
                    "failed": bool(data.get("error") or data.get("error_type"))}
        except (ValueError, KeyError, TypeError):
            raise APIError("provider_response_invalid", 503) from None


def create_apps(gateway: Gateway | None = None, management_token: str | None = None,
                execution_token: str | None = None):
    if gateway is None:
        gateway = Gateway(Store(os.environ.get("FAL_STATE_DB", "/data/fal/state.sqlite3"),
                                os.environ["FAL_ENCRYPTION_KEY"]))
    management_token = management_token if management_token is not None else os.environ.get("FAL_MANAGEMENT_TOKEN", "")
    execution_token = execution_token if execution_token is not None else os.environ.get("FAL_EXECUTION_TOKEN", "")
    dashboard_path = Path(os.environ.get("FAL_DASHBOARD_DIR", "/app/dashboard"))
    assets = DashboardAssets(dashboard_path) if dashboard_path.exists() else None

    async def manage(request: Request):
        try:
            path = request.url.path
            if path.startswith("/internal/"):
                headers = request.headers.getlist("authorization")
                if len(management_token) < 32 or len(headers) != 1 or not hmac.compare_digest(headers[0], "Bearer " + management_token):
                    raise APIError("unauthorized", 401)
                if assets is None:
                    raise APIError("dashboard_unavailable", 503)
                if path == "/internal/dashboard-declaration":
                    data, mime = assets.declaration, "application/json"
                elif path == "/internal/dashboard-index":
                    data, mime = assets.index, "application/json"
                else:
                    try:
                        data, mime = assets.asset(request.path_params["asset"])
                    except (KeyError, ValueError, OSError):
                        raise APIError("dashboard_asset_unavailable", 404) from None
                return Response(data, headers={"Content-Type": mime, "Cache-Control": "no-store",
                                               "X-Content-Type-Options": "nosniff"})
            org = authenticate(request, management_token, manage=True)
            if path in {"/v1/overview", "/v1/authorization", "/v1/limits"} and request.method == "GET":
                return respond(gateway.store.overview(org))
            if path == "/v1/usage" and request.method == "GET":
                return respond(gateway.store.usage(org))
            if path == "/v1/authorization" and request.method == "PUT":
                data = await body(request)
                key = data.get("api_key")
                if not isinstance(key, str) or not 16 <= len(key) <= 4096 or any(c.isspace() for c in key):
                    raise APIError("invalid_key")
                revision = number(data.get("expected_revision"), 0, 2**31 - 1)
                version = gateway.store.candidate(org, key, revision)
                state = await run_in_threadpool(gateway.check, key)
                if state == "active":
                    return respond(gateway.store.activate(org, version, revision))
                gateway.store.mark_candidate(org, version, state)
                return respond({"status": state, "active": gateway.store.overview(org)},
                               422 if state == "invalid" else 503)
            if path == "/v1/authorization" and request.method == "DELETE":
                revision = number((await body(request)).get("expected_revision"), 0, 2**31 - 1)
                return respond(gateway.store.revoke(org, revision))
            if path == "/v1/authorization/check" and request.method == "POST":
                current = gateway.store.overview(org)
                if not current["version"]:
                    raise APIError("authorization_required", 409)
                key = gateway.store.decrypt(org, current["version"])
                state = await run_in_threadpool(gateway.check, key)
                return respond(gateway.store.recheck(org, state))
            if path == "/v1/limits" and request.method == "PUT":
                data = await body(request)
                if type(data.get("enabled")) is not bool:
                    raise APIError("invalid_request")
                return respond(gateway.store.set_limits(org, data["enabled"],
                    number(data.get("daily_limit"), 0, 1000),
                    number(data.get("expected_revision"), 0, 2**31 - 1)))
            raise APIError("not_found", 404)
        except APIError as error:
            return respond({"error": error.code}, error.status)
        except Conflict as error:
            return respond({"error": str(error)}, 409)

    async def execute(request: Request):
        try:
            org = authenticate(request, execution_token, manage=False)
            if request.url.path == "/v1/submissions" and request.method == "POST":
                data = await body(request, 96 * 1024 * 1024)
                operation_id = identity(data.get("operation_id", ""))
                if data.get("model_id") != MODEL:
                    raise APIError("unsupported_model")
                payload = gateway.validate_payload(data.get("payload"))
                return respond(await run_in_threadpool(gateway.submit, org, operation_id, payload))
            if "submission_id" in request.path_params:
                return respond(await run_in_threadpool(gateway.remote, org,
                    str(request.path_params["submission_id"]), request.url.path.endswith("/result")))
            raise APIError("not_found", 404)
        except APIError as error:
            return respond({"error": error.code}, error.status)
        except Conflict as error:
            return respond({"error": str(error)}, 409)
        except KeyError:
            return respond({"error": "submission_not_found"}, 404)

    async def health(request):
        return respond({"service": "video-fal-gateway", "version": __version__, "status": "ok"})

    management = Starlette(routes=[Route("/health", health),
        Route("/internal/dashboard-declaration", manage), Route("/internal/dashboard-index", manage),
        Route("/internal/dashboard/{asset:path}", manage),
        Route("/v1/overview", manage), Route("/v1/authorization", manage, methods=["GET", "PUT", "DELETE"]),
        Route("/v1/authorization/check", manage, methods=["POST"]),
        Route("/v1/limits", manage, methods=["GET", "PUT"]), Route("/v1/usage", manage)])
    execution = Starlette(routes=[Route("/health", health), Route("/v1/submissions", execute, methods=["POST"]),
        Route("/v1/submissions/{submission_id:uuid}", execute),
        Route("/v1/submissions/{submission_id:uuid}/result", execute)])
    return management, execution


async def main():
    management, execution = create_apps()
    servers = [uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=port, access_log=False))
               for app, port in ((management, 8000), (execution, 8001))]
    await asyncio.gather(*(server.serve() for server in servers))


if __name__ == "__main__":
    asyncio.run(main())
