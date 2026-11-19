"""Shared project authorization, reference checks and native result access."""
import hashlib
import os
import ssl
import uuid

import httpx

from ..channel_errors import check_response


class MembershipAccess:
    def authorize(self, project_id, authority=None):
        actor = self.authorize_member(authority)
        self.project_request("open", {"project_id": project_id, "require_write": True}, actor)
        return actor

    def authorize_member(self, authority=None):
        from ..delivery import CALLER
        actor = authority if authority is not None else CALLER.get()
        if (not actor or actor.get("organization_id") != self.organization or
                not actor.get("user_id") or (self.members and actor["user_id"] not in self.members)):
            raise PermissionError(f"{self.route}_account_access_denied")
        return dict(actor)

    def project_request(self, operation, body, actor=None):
        from ..artifact_output import uuid7
        actor = self.authorize_member(actor)
        if operation not in {"open", "list"}:
            raise ValueError("unsupported_project_operation")
        url = os.getenv("VIDEO_PROJECT_SERVICE_URL", "").rstrip("/")
        token = os.getenv("VIDEO_PROJECT_SERVICE_TOKEN", "")
        if not url or not token:
            raise ValueError("project_authorization_not_configured")
        response = self.client.post(url + "/internal/v1/project/" + operation,
            headers={"Authorization": "Bearer " + token, "X-User-Id": actor["user_id"],
                     "X-Organization-Id": actor["organization_id"], "X-Request-Id": uuid7()},
            json=body, timeout=15)
        if response.status_code in {401, 403, 404}:
            raise PermissionError("project_access_denied")
        check_response(response, self.route, accepted=False)
        return response.json()

    def authorize_record(self, record):
        actor = self.authorize(record.project_id)
        if not record.authority or actor["organization_id"] != record.authority.get("organization_id"):
            raise PermissionError(f"{self.route}_task_access_denied")
        return actor

    def reference_source(self, project_id, artifact_id, kind):
        asset = self.executor.artifacts.get(artifact_id, project_id)
        allowed = {"image/png": "png", "image/jpeg": "jpeg", "image/webp": "webp"} if kind == "image" else {"video/mp4": "mp4"}
        if asset.media_type not in allowed or not 0 < asset.size <= (10 if kind == "image" else 128) * 1024 * 1024:
            raise ValueError("unsupported_reference_media")
        path = self.executor.artifacts.content_path(asset)
        if path.stat().st_size != asset.size:
            raise ValueError("source_integrity_failed")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != asset.sha256:
                raise ValueError("source_integrity_failed")
        return path, {"project_id": project_id, "sha256": asset.sha256, "size": asset.size,
                      "kind": kind, "format": allowed[asset.media_type]}

    def content_path(self, record):
        """Authorize the fixed central version before serving its verified cache."""
        from ..artifact_output import source, uuid7
        actor = self.authorize_record(record)
        if record.status != "succeeded":
            raise ValueError(f"{self.route}_result_not_ready")
        ref = (record.provider_state or {}).get("content_ref", {})
        if set(ref) != {"store_id", "artifact_id", "version_id"} or any(
                str(uuid.UUID(v)) != v or uuid.UUID(v).version != 7 for v in ref.values()):
            raise ValueError("invalid_content_ref")
        endpoint = os.getenv("VIDEO_ARTIFACT_SERVICE_URL", "").rstrip("/")
        token = os.getenv("VIDEO_ARTIFACT_SERVICE_TOKEN", "")
        if not endpoint or len(token) < 32:
            raise ValueError("native_artifact_not_configured")
        trust = ssl.create_default_context()
        if os.getenv("VIDEO_ARTIFACT_CA_FILE"):
            trust.load_verify_locations(os.environ["VIDEO_ARTIFACT_CA_FILE"])
        with httpx.Client(trust_env=False, follow_redirects=False, timeout=15, verify=trust) as client:
            response = client.get(endpoint + "/v2/artifacts/" + ref["version_id"],
                params={"store_id": ref["store_id"], "artifact_id": ref["artifact_id"]},
                headers={"Authorization": "Bearer " + token, "X-User-Id": actor["user_id"],
                         "X-Organization-Id": actor["organization_id"], "X-Request-Id": uuid7()})
            if response.status_code in {401, 403, 404}:
                raise PermissionError("artifact_access_denied")
            check_response(response, self.route)
            version = response.json()
        media = record.media or {}
        if (any(version.get(k) != v for k, v in ref.items()) or version.get("source") != source(record) or
                version.get("sha256") != media.get("sha256") or version.get("size") != media.get("size") or
                version.get("organization_id") != actor["organization_id"] or version.get("mime") != "video/mp4"):
            raise ValueError("artifact_receipt_mismatch")
        path = self.executor.tasks.root / f"{record.video_task_id}.{self.route}.mp4"
        if path.is_symlink() or not path.is_file() or path.stat().st_size != media["size"]:
            raise ValueError("media_integrity_failed")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != media["sha256"]:
                raise ValueError("media_integrity_failed")
        return path

