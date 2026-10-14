"""Separate internal listener for Artifact's native Video write authorization."""
import hmac
import json
import os
import ssl
import uuid

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .tasks import TaskStore, TaskNotFound
from .providers.higgsfield_artifact import source, uuid7


def valid_id(value):
    try:
        parsed = uuid.UUID(value)
        return str(parsed) == value and parsed.version == 7
    except (ValueError, TypeError, AttributeError):
        return False


def authorize(data, tasks, client):
    if not isinstance(data, dict) or set(data) != {"principal", "permission"}:
        return False
    actor, permission = data["principal"], data["permission"]
    if (not isinstance(actor, dict) or set(actor) != {"organization_id", "subject_id", "request_id"} or
            not all(valid_id(v) for v in actor.values()) or not isinstance(permission, dict) or
            set(permission) != {"action", "source"} or permission["action"] != "write"):
        return False
    claimed = permission["source"]
    if not isinstance(claimed, dict) or claimed.get("kind") != "task_output" or claimed.get("service_id") != "video":
        return False
    try:
        record = tasks.get(claimed.get("run_id", ""))
    except (TaskNotFound, ValueError):
        return False
    if (record.service != "higgsfield" or not record.authority or not record.runtime_task_id or
            claimed != source(record) or actor["subject_id"] != record.authority.get("user_id") or
            actor["organization_id"] != record.authority.get("organization_id") or
            actor["organization_id"] != os.getenv("HIGGSFIELD_ORGANIZATION_ID", "")):
        return False
    members = set(filter(None, os.getenv("HIGGSFIELD_ALLOWED_USERS", "").split(",")))
    if members and actor["subject_id"] not in members:
        return False
    media = (record.provider_state or {}).get("produced_media")
    path = tasks.root / (record.video_task_id + ".higgsfield.mp4")
    if (not isinstance(media, dict) or media != record.media or not path.is_file() or path.is_symlink() or
            path.stat().st_size != media.get("size") or record.runtime_stage not in {"collecting", "completed"}):
        return False
    url = os.getenv("VIDEO_PROJECT_SERVICE_URL", "").rstrip("/")
    token = os.getenv("VIDEO_PROJECT_SERVICE_TOKEN", "")
    if not url or not token:
        return False
    response = client.post(url + "/internal/v1/project/open",
        headers={"Authorization": "Bearer " + token, "X-User-Id": actor["subject_id"],
                 "X-Organization-Id": actor["organization_id"], "X-Request-Id": uuid7()},
        json={"project_id": record.project_id, "require_write": True}, timeout=10)
    return response.status_code == 200


async def handle(request: Request):
    token = os.getenv("VIDEO_ARTIFACT_AUTHORITY_TOKEN", "")
    values = request.headers.getlist("authorization")
    if len(token) < 32 or len(values) != 1 or not hmac.compare_digest(values[0], "Bearer " + token):
        return JSONResponse({"allowed": False}, status_code=403)
    try:
        content = bytearray()
        async for chunk in request.stream():
            content.extend(chunk)
            if len(content) > 8192:
                return JSONResponse({"allowed": False}, status_code=400)
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate_key")
                result[key] = value
            return result
        data = json.loads(content, object_pairs_hook=unique)
        from starlette.concurrency import run_in_threadpool
        trust = ssl.create_default_context()
        if os.getenv("VIDEO_PROJECT_CA_FILE"):
            trust.load_verify_locations(os.environ["VIDEO_PROJECT_CA_FILE"])
        with httpx.Client(trust_env=False, follow_redirects=False, verify=trust) as client:
            allowed = await run_in_threadpool(authorize, data, TaskStore.from_environment(), client)
        return JSONResponse({"allowed": allowed})
    except (ValueError, KeyError, TypeError, OSError, httpx.HTTPError):
        return JSONResponse({"allowed": False}, status_code=503)


app = Starlette(routes=[Route("/internal/artifacts/authorize", handle, methods=["POST"])])
