"""Native-output authorization must preserve old records and fail closed."""
import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from starlette.testclient import TestClient

from app.artifact_authority import app, authorize
from app.artifact_output import source, uuid7
from app.tasks import TaskStore


class ArtifactAuthorityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.org, self.user, self.project = uuid7(), uuid7(), uuid7()
        self.env = patch.dict(os.environ, {
            "HIGGSFIELD_ORGANIZATION_ID": self.org,
            "HIGGSFIELD_ALLOWED_USERS": "",
            "VIDEO_PROJECT_SERVICE_URL": "https://project.internal",
            "VIDEO_PROJECT_SERVICE_TOKEN": "synthetic-project",
            "VIDEO_ARTIFACT_AUTHORITY_TOKEN": "synthetic-authority-" + "x" * 32,
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.tasks = TaskStore(Path(self.temp.name))
        record = self.tasks.create(project_id=self.project, idempotency_key="original",
            input_digest="fixture", request={"route": "higgsfield"}, runtime_task_id=uuid7(),
            status="running", authority={"organization_id": self.org, "user_id": self.user})
        data = b"original-verified-media"
        self.media = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        self.path = self.tasks.root / (record.video_task_id + ".higgsfield.mp4")
        self.path.write_bytes(data)
        self.record = self.tasks.update(record, runtime_stage="collecting", media=self.media,
            provider_state={"produced_media": self.media, "artifact_write_id": uuid7()})
        self.claim = {"principal": {"organization_id": self.org, "subject_id": self.user,
            "request_id": uuid7()}, "permission": {"action": "write", "source": source(self.record)}}
        self.requests = []
        self.project_status = 200
        def respond(request):
            self.requests.append(request)
            return httpx.Response(self.project_status)
        self.client = httpx.Client(transport=httpx.MockTransport(respond))
        self.addCleanup(self.client.close)

    def test_old_record_authorizes_without_modification_or_provider_call(self):
        before = {p.name: p.read_bytes() for p in self.tasks.root.iterdir() if p.is_file()}
        self.assertTrue(authorize(self.claim, self.tasks, self.client))
        self.assertEqual(len(self.requests), 1)
        req = self.requests[0]
        self.assertEqual(str(req.url), "https://project.internal/internal/v1/project/open")
        self.assertEqual(req.headers["X-User-Id"], self.user)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.tasks.root.iterdir() if p.is_file()})

    def test_forged_identity_source_and_unknown_task_are_denied(self):
        mutations = [
            ("principal", "subject_id", uuid7()), ("principal", "organization_id", uuid7()),
            ("source", "project_id", uuid7()), ("source", "task_id", uuid7()),
            ("source", "run_id", "video_task_" + "0" * 32),
            ("source", "service_id", "image"), ("source", "kind", "user_import"),
        ]
        for area, key, value in mutations:
            with self.subTest(area=area, key=key):
                claim = copy.deepcopy(self.claim)
                target = claim["principal"] if area == "principal" else claim["permission"]["source"]
                target[key] = value
                self.assertFalse(authorize(claim, self.tasks, self.client))
        self.assertEqual(self.requests, [])

    def test_current_channel_binding_and_members_are_rechecked(self):
        for changes in ({"HIGGSFIELD_ORGANIZATION_ID": uuid7()}, {"HIGGSFIELD_ALLOWED_USERS": uuid7()}):
            with self.subTest(changes=list(changes)), patch.dict(os.environ, changes):
                self.assertFalse(authorize(self.claim, self.tasks, self.client))
        self.assertEqual(self.requests, [])

    def test_current_project_revocation_and_failure_never_allow(self):
        for status in (403, 404, 500):
            with self.subTest(status=status):
                self.project_status = status
                self.assertFalse(authorize(self.claim, self.tasks, self.client))

    def test_unknown_channel_and_unverified_output_stay_denied(self):
        cases = ({"service": "fal"}, {"authority": None}, {"runtime_task_id": ""},
                 {"provider_state": {}}, {"media": {"size": 1}}, {"runtime_stage": "remote_pending"})
        baseline = {key: getattr(self.record, key) for case in cases for key in case}
        for updates in cases:
            with self.subTest(updates=list(updates)):
                self.tasks.update(self.record, **baseline)
                self.tasks.update(self.record, **updates)
                self.assertFalse(authorize(self.claim, self.tasks, self.client))
        self.assertEqual(self.requests, [])

    def test_missing_resized_and_symlink_media_are_denied(self):
        self.path.write_bytes(b"changed")
        self.assertFalse(authorize(self.claim, self.tasks, self.client))
        self.path.unlink()
        self.assertFalse(authorize(self.claim, self.tasks, self.client))
        target = Path(self.temp.name) / "untrusted.mp4"
        target.write_bytes(b"original-verified-media")
        self.path.symlink_to(target)
        self.assertFalse(authorize(self.claim, self.tasks, self.client))

    def test_http_token_request_bounds_and_dependency_failure(self):
        token = os.environ["VIDEO_ARTIFACT_AUTHORITY_TOKEN"]
        with TestClient(app) as client:
            for headers in ({}, {"Authorization": "Bearer invalid"},
                            [("Authorization", "Bearer " + token), ("Authorization", "Bearer " + token)]):
                response = client.post("/internal/artifacts/authorize", json=self.claim, headers=headers)
                self.assertEqual(response.status_code, 403)
                self.assertFalse(response.json()["allowed"])
            headers = {"Authorization": "Bearer " + token}
            response = client.post("/internal/artifacts/authorize", content=b"x" * 8193, headers=headers)
            self.assertEqual(response.status_code, 400)
            with patch("app.artifact_authority.authorize", side_effect=httpx.ConnectError("unavailable")):
                response = client.post("/internal/artifacts/authorize", json=self.claim, headers=headers)
            self.assertEqual(response.status_code, 503)
            self.assertFalse(response.json()["allowed"])
            response = client.post("/internal/artifacts/authorize", content=b'{"principal":{},"principal":{}}', headers=headers)
            self.assertEqual(response.status_code, 503)
            self.assertFalse(response.json()["allowed"])


if __name__ == "__main__":
    unittest.main()
