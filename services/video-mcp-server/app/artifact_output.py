"""Recoverable native Artifact prepare/put/commit; never label generated media user_import."""
import hashlib
import os
import re
import secrets
import ssl
import time
from urllib.parse import urlencode
import uuid

import httpx


def uuid7():
    # UUIDv7, RFC 9562: 48-bit timestamp, version, randomness, RFC variant.
    value = (int(time.time() * 1000) << 80) | (7 << 76) | (secrets.randbits(12) << 64)
    value |= (2 << 62) | secrets.randbits(62)
    return str(uuid.UUID(int=value))


def source(record):
    return {"kind": "task_output", "service_id": "video", "project_id": record.project_id,
            "run_id": record.video_task_id, "task_id": record.runtime_task_id}


def validate_media(media):
    if (not isinstance(media, dict) or not isinstance(media.get("sha256"), str) or
            not re.fullmatch(r"[0-9a-f]{64}", media["sha256"]) or
            type(media.get("size")) is not int or not 0 < media["size"] <= 1024 ** 3):
        raise ValueError("invalid_media_metadata")


def publish(tasks, record, path, media):
    """Register verified task output using the original durable upload identity."""
    validate_media(media)
    if path.is_symlink():
        raise ValueError("invalid_media_file")
    state = dict(record.provider_state or {})
    with path.open("rb") as stream:
        if path.stat().st_size != media["size"] or hashlib.file_digest(stream, "sha256").hexdigest() != media["sha256"]:
            raise ValueError("media_integrity_failed")
    # Persist verifiable producer evidence before Artifact asks us for authority.
    state["produced_media"] = media
    state.setdefault("artifact_write_id", uuid7())
    record = tasks.update(record, runtime_stage="collecting", media=media, provider_state=state)
    endpoint = os.getenv("VIDEO_ARTIFACT_SERVICE_URL", "").rstrip("/")
    token = os.getenv("VIDEO_ARTIFACT_SERVICE_TOKEN", "")
    if not endpoint or len(token) < 32:
        raise ValueError("native_artifact_not_configured")
    headers = {"Authorization": "Bearer " + token, "X-User-Id": record.authority["user_id"],
               "X-Organization-Id": record.authority["organization_id"], "X-Request-Id": uuid7()}
    trust = ssl.create_default_context()
    if os.getenv("VIDEO_ARTIFACT_CA_FILE"):
        trust.load_verify_locations(os.environ["VIDEO_ARTIFACT_CA_FILE"])
    with httpx.Client(trust_env=False, follow_redirects=False, timeout=120, verify=trust) as client:
        if "artifact_upload_id" not in state:
            response = client.post(endpoint + "/v2/artifacts/uploads", headers=headers,
                json={"write_id": state["artifact_write_id"], "source": source(record),
                      "sha256": media["sha256"], "size": media["size"], "mime": "video/mp4"})
            response.raise_for_status()
            state["artifact_upload_id"] = str(uuid.UUID(response.json()["upload_id"]))
            record = tasks.update(record, provider_state=state)
        upload = endpoint + "/v2/artifacts/uploads/" + state["artifact_upload_id"]
        response = client.get(upload, headers=headers)
        response.raise_for_status()
        upload_state = response.json()["state"]
        if upload_state not in {"prepared", "committed"}:
            raise ValueError("artifact_upload_state_invalid")
        if upload_state == "prepared":
            with path.open("rb") as stream:
                response = client.put(upload + "/content", headers={**headers, "Content-Type": "video/mp4",
                                      "Content-Length": str(media["size"])}, content=stream)
                response.raise_for_status()
        response = client.post(upload + "/commit", headers=headers)
        response.raise_for_status()
        version = response.json()
        if (version.get("sha256") != media["sha256"] or version.get("size") != media["size"] or
                version.get("source") != source(record) or version.get("organization_id") != record.authority["organization_id"] or
                version.get("created_by") != record.authority["user_id"] or version.get("mime") != "video/mp4"):
            raise ValueError("artifact_receipt_mismatch")
        ref = {key: str(uuid.UUID(version[key])) for key in ("store_id", "artifact_id", "version_id")}
        if any(uuid.UUID(value).version != 7 or value != version[key] for key, value in ref.items()):
            raise ValueError("artifact_receipt_mismatch")
        download = "/v2/artifacts/" + ref["version_id"] + "/content?" + urlencode({
            "store_id": ref["store_id"], "artifact_id": ref["artifact_id"]})
        digest, size = hashlib.sha256(), 0
        with client.stream("GET", endpoint + download, headers=headers) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes(1024 * 1024):
                size += len(chunk)
                if size > media["size"]:
                    raise ValueError("artifact_readback_mismatch")
                digest.update(chunk)
        if size != media["size"] or digest.hexdigest() != media["sha256"]:
            raise ValueError("artifact_readback_mismatch")
    state.update(content_ref=ref, download_path=download, next_poll_at=0)
    return tasks.update(record, status="succeeded", runtime_stage="completed", media=media,
                                         provider_state=state, error=None)
