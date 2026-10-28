from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

import httpx

from app.artifacts import ArtifactStore
from app.delivery import CALLER
from app.executor import VideoExecutor
from app.dashboard import Dashboard
from app.artifact_compat import read_output_evidence
from app.providers.higgsfield_artifact import collect
from app.artifact_output import source, uuid7
from app.tasks import TaskStore, TaskConflict

ORG = "0199abcd-0000-7000-8000-000000000001"
USER = "0199abcd-0000-7000-8000-000000000002"
PROJECT = "0199abcd-0000-7000-8000-000000000003"
REMOTE = "11111111-1111-4111-8111-111111111111"


class HiggsfieldTest(unittest.TestCase):
    def test_draft_requires_r2v_and_advertises_only_480p(self):
        cap = next(c for c in self.executor.higgsfield.capabilities() if c["model"] == "seedance_2_5_draft")
        self.assertEqual(cap["modes"], ["reference_to_video"])
        self.assertEqual(cap["resolutions"], ["480p"])
        with self.assertRaisesRegex(ValueError, "reference_to_video_required"):
            self.create(model="seedance_2_5_draft", duration_seconds=4)
        refs = {"images": [], "videos": []}
        for group, mime, name in (("images", "image/png", "identity.png"), ("videos", "video/mp4", "motion.mp4")):
            asset = self.executor.artifacts.create_from_chunks(project_id=PROJECT, operation="import",
                filename=name, media_type=mime, chunks=[b"synthetic-media"])
            refs[group].append({"artifact_id": asset.artifact_id, "purpose": "reference"})
        record = self.create(model="seedance_2_5_draft", duration_seconds=4, references=refs)
        self.assertEqual(record.request["mode"], "reference_to_video")
        self.assertEqual(record.request["resolution"], "480p")
        self.assertEqual([v["kind"] for v in record.request["sources"]], ["image", "video"])
        full = self.create(idempotency_key="full-r2v", model="seedance_2_5", duration_seconds=4,
                           resolution="480p", references=refs)
        self.assertEqual(full.request["mode"], "reference_to_video")
        self.assertEqual(full.request["resolution"], "480p")
        self.assertEqual([v["kind"] for v in full.request["sources"]], ["image", "video"])
        from app import server
        with patch.object(server, "executor", self.executor), patch.object(self.executor.higgsfield, "quote", return_value={
                'credits':12, 'can_submit': True, 'blocked_reason':None, 'expires_at':'test-expiry'}) as quote:
            preview = server.video_preflight(project_id=PROJECT, route="higgsfield", model="seedance_2_5_draft",
                prompt="dance", duration_seconds=4, references=refs).structured_content
            self.assertEqual(preview["quoted_credits"], 12)
            self.assertEqual(quote.call_args.args[0]["mode"], "reference_to_video")
            self.assertEqual(len(quote.call_args.args[0]["sources"]), 2)
        for changes in ({"mode": "image_to_video"}, {"resolution": "720p"}):
            with self.assertRaises(ValueError):
                self.create(model="seedance_2_5_draft", duration_seconds=4, references=refs, **changes)
        refs["images"][0]["purpose"] = "first_frame"
        with self.assertRaisesRegex(ValueError, "invalid_reference"):
            self.create(model="seedance_2_5_draft", duration_seconds=4, references=refs)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"HIGGSFIELD_GATEWAY_URL": "http://gateway.internal",
            "HIGGSFIELD_INTERNAL_TOKEN": "synthetic-internal-" + "x" * 32, "HIGGSFIELD_BINDING_ID": "team",
            "HIGGSFIELD_ORGANIZATION_ID": ORG, "HIGGSFIELD_ALLOWED_USERS": "", "HIGGSFIELD_ENABLED": "true",
            "VIDEO_PROJECT_SERVICE_URL": "http://project.internal", "VIDEO_PROJECT_SERVICE_TOKEN": "synthetic-project"})
        self.env.start()
        self.calls = []
        self.gateway_status = "running"
        self.gateway_stage = "remote_pending"
        self.deny_project = False
        self.fail_submit = False
        self.actor = CALLER.set({"organization_id": ORG, "user_id": USER})
        self.client = httpx.Client(transport=httpx.MockTransport(self.respond))
        self.executor = VideoExecutor(ArtifactStore(self.root), TaskStore(self.root), self.client)
        # Directory/lease behavior has independent tests. These tests exercise
        # the adapter after a real registry gate has admitted the exact model.
        gate = patch.object(self.executor, 'require_channel')
        self.registry_gate = gate.start()
        self.addCleanup(gate.stop)
        self.executor.higgsfield.client.close()
        self.executor.higgsfield.client = self.client

    def test_mcp_preflight_rejects_disabled_draft_without_quoting(self):
        from app import server
        from app.channel_errors import ChannelError
        asset = self.executor.artifacts.create_from_chunks(project_id=PROJECT, operation='import',
            filename='identity.png', media_type='image/png', chunks=[b'synthetic-media'])
        self.registry_gate.side_effect = ChannelError('provider_not_found', source='higgsfield', accepted=False)
        with patch.object(server, 'executor', self.executor), patch.object(self.executor.higgsfield, 'quote') as quote:
            result = server.video_preflight(project_id=PROJECT, route='higgsfield', model='seedance_2_5_draft',
                prompt='dance', duration_seconds=4,
                references={'images':[{'artifact_id':asset.artifact_id, 'purpose':'reference'}]})
        self.assertTrue(result.is_error)
        self.assertEqual(result.structured_content['error']['code'], 'provider_not_found')
        self.assertFalse(result.structured_content['error']['accepted'])
        self.registry_gate.assert_called_once_with('higgsfield', 'seedance_2_5_draft')
        quote.assert_not_called()

    def tearDown(self):
        CALLER.reset(self.actor)
        self.client.close()
        self.env.stop()
        self.temp.cleanup()

    def respond(self, req):
        self.calls.append(req)
        if req.url.host == "project.internal":
            self.assertEqual(uuid.UUID(req.headers["X-Request-Id"]).version, 7)
            return httpx.Response(403 if self.deny_project else 200, json={})
        if req.method == "POST":
            self.task_id = json.loads(req.content)["request_id"]
            if self.fail_submit:
                raise httpx.ReadTimeout("synthetic", request=req)
        return httpx.Response(200, json={"request_id": self.task_id, "binding_id": "team",
            "remote_id": REMOTE if self.gateway_stage != "submission_unknown" else None,
            "stage": self.gateway_stage, "status": self.gateway_status, "quoted_credits": 2,
            "media": {"sha256": "0" * 64, "size": 10},
            "created_at": "2026-10-14T00:00:00+00:00"})

    def create(self, **updates):
        args = dict(project_id=PROJECT, idempotency_key="attempt", route="higgsfield", model="grok_video_v15_lite",
                    prompt="paper boat", duration_seconds=2, aspect_ratio="16:9", references={})
        args.update(updates)
        return self.executor.generate(**args)

    def poll(self, record):
        state = dict(record.provider_state or {})
        state["next_poll_at"] = 0
        self.executor.tasks.update(record, provider_state=state)
        return self.executor.status(record.video_task_id)

    def test_create_is_deferred_and_project_authorized(self):
        record = self.create()
        self.assertEqual(record.service, "higgsfield")
        self.assertTrue(all(r.url.host == "project.internal" for r in self.calls))
        self.assertEqual(self.create().video_task_id, record.video_task_id)
        with self.assertRaises(TaskConflict):
            self.create(prompt="different")
        ready = self.poll(record)
        self.assertEqual(ready.runtime_task_id, REMOTE)
        payload = json.loads(next(r.content for r in self.calls if r.url.host == "gateway.internal" and r.method == "POST"))
        self.assertEqual(payload["organization_id"], ORG)
        self.assertNotIn("authorization", payload)

    def test_missing_or_foreign_identity_never_reaches_provider(self):
        for actor in (None, {"user_id": USER, "organization_id": str(uuid.uuid4())}):
            token = CALLER.set(actor)
            try:
                with self.assertRaises(PermissionError):
                    self.create()
            finally:
                CALLER.reset(token)
        self.assertEqual(self.calls, [])

    def test_project_revocation_blocks_status_and_new_submit(self):
        record = self.create()
        self.deny_project = True
        with self.assertRaises(PermissionError):
            self.executor.higgsfield.authorize_record(record)
        observed = self.poll(record)
        self.assertEqual(observed.error["code"], "higgsfield_recovery_pending")
        self.assertTrue(all(r.url.host == "project.internal" for r in self.calls))

    def test_gateway_timeout_keeps_original_id_for_idempotent_recovery(self):
        record = self.create()
        self.fail_submit = True
        record = self.poll(record)
        self.assertEqual(record.runtime_stage, "submission_unknown")
        self.fail_submit = False
        self.poll(record)
        payloads = [json.loads(r.content) for r in self.calls if r.url.host == "gateway.internal" and r.method == "POST"]
        self.assertEqual(len(payloads), 1)
        self.assertTrue(any(r.method == 'GET' and r.url.path.endswith(record.video_task_id) for r in self.calls))

    def test_unknown_provider_submission_only_queries_existing_gateway_task(self):
        self.gateway_stage = "submission_unknown"
        record = self.poll(self.create())
        self.poll(record)
        gateway_calls = [r for r in self.calls if r.url.host == "gateway.internal"]
        self.assertEqual([r.method for r in gateway_calls], ["POST", "GET"])

    def test_status_error_cannot_revoke_known_provider_acceptance(self):
        record = self.poll(self.create())
        self.assertEqual(record.runtime_task_id, REMOTE)
        original = self.executor.higgsfield.client
        def failure(req):
            if req.url.host == 'project.internal':
                return self.respond(req)
            self.assertEqual(req.method, 'GET')
            return httpx.Response(503, json={'accepted': False, 'error': {
                'source': 'higgsfield', 'code': 'provider_unavailable',
                'message': 'Status lookup is temporarily unavailable.'}})
        with httpx.Client(transport=httpx.MockTransport(failure)) as client:
            self.executor.higgsfield.client = client
            observed = self.poll(record)
        self.executor.higgsfield.client = original
        self.assertTrue(observed.error['accepted'])
        self.assertEqual(observed.error['code'], 'provider_unavailable')
        self.assertEqual(observed.runtime_task_id, REMOTE)

    def test_native_output_requires_persisted_media_and_original_identity(self):
        record = self.poll(self.create())
        self.assertIsNone(read_output_evidence(self.executor.tasks, record))
        data = b"verified-media-fixture"
        media = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        path = self.executor.tasks.root / (record.video_task_id + ".higgsfield.mp4")
        path.write_bytes(data)
        record = self.executor.tasks.update(record, runtime_stage="collecting", media=media, provider_state={"produced_media": media})
        self.assertEqual(read_output_evidence(self.executor.tasks, record), (path, media))
        with patch.dict(os.environ, {"HIGGSFIELD_ORGANIZATION_ID": uuid7()}):
            self.assertIsNone(read_output_evidence(self.executor.tasks, record))
        with patch.dict(os.environ, {"HIGGSFIELD_ALLOWED_USERS": uuid7()}):
            self.assertIsNone(read_output_evidence(self.executor.tasks, record))

    def test_collection_failure_preserves_saved_artifact_upload_receipt(self):
        record = self.create()
        self.gateway_status = "completed"
        self.gateway_stage = "completed"
        def collect(adapter, current, media):
            state = dict(current.provider_state)
            state["artifact_upload_id"] = "saved-upload"
            self.executor.tasks.update(current, provider_state=state, runtime_stage="collecting")
            raise httpx.ReadTimeout("synthetic")
        with patch("app.providers.higgsfield_artifact.collect", side_effect=collect):
            observed = self.poll(record)
        self.assertEqual(observed.provider_state["artifact_upload_id"], "saved-upload")
        self.assertEqual(observed.runtime_stage, "collecting")

    def artifact_case(self, loss=None, corrupt=False, foreign=False):
        record = self.poll(self.create())
        data = b"verified-media-fixture"
        media = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        upload_id = uuid7()
        version = dict(store_id=uuid7(), artifact_id=uuid7(), version_id=uuid7(),
                       sha256=media["sha256"], size=media["size"], source=source(record),
                       organization_id=ORG, created_by=USER, mime="video/mp4")
        if foreign:
            version["organization_id"] = uuid7()
        observed = {"prepares": [], "puts": 0, "commits": 0, "downloads": 0, "state": "prepared", "lost": False}

        def reply(req):
            if req.url.host == "gateway.internal":
                observed["downloads"] += 1
                return httpx.Response(200, content=data)
            self.assertEqual(req.headers["X-User-Id"], USER)
            self.assertEqual(req.headers["X-Organization-Id"], ORG)
            self.assertEqual(uuid.UUID(req.headers["X-Request-Id"]).version, 7)
            if req.method == "POST" and req.url.path.endswith("/uploads"):
                body = json.loads(req.content)
                self.assertEqual(body["source"], source(record))
                observed["prepares"].append(body)
                if loss == "prepare" and not observed["lost"]:
                    observed["lost"] = True
                    raise httpx.ReadTimeout("synthetic", request=req)
                return httpx.Response(200, json={"upload_id": upload_id})
            if req.method == "GET" and req.url.path.endswith(upload_id):
                return httpx.Response(200, json={"state": observed["state"]})
            if req.method == "PUT":
                self.assertEqual(req.read(), data)
                observed["puts"] += 1
                if loss == "put" and not observed["lost"]:
                    observed["lost"] = True
                    raise httpx.ReadTimeout("synthetic", request=req)
                return httpx.Response(200, json={"state": "prepared"})
            if req.method == "POST" and req.url.path.endswith("/commit"):
                observed["commits"] += 1
                observed["state"] = "committed"
                if loss == "commit" and not observed["lost"]:
                    observed["lost"] = True
                    raise httpx.ReadTimeout("synthetic", request=req)
                return httpx.Response(200, json=version)
            self.assertEqual(req.url.params["artifact_id"], version["artifact_id"])
            return httpx.Response(200, content=b"bad" if corrupt else data)

        return record, media, observed, reply

    @contextmanager
    def artifact_clients(self, reply):
        client_type = httpx.Client
        original = self.executor.higgsfield.client
        gateway = client_type(transport=httpx.MockTransport(reply))
        self.executor.higgsfield.client = gateway
        try:
            with patch.dict(os.environ, {"VIDEO_ARTIFACT_SERVICE_URL": "http://artifact.internal",
                     "VIDEO_ARTIFACT_SERVICE_TOKEN": "synthetic-artifact-" + "x" * 32}), \
                 patch("app.artifact_output.httpx.Client",
                     side_effect=lambda **kwargs: client_type(transport=httpx.MockTransport(reply), **kwargs)):
                yield
        finally:
            gateway.close()
            self.executor.higgsfield.client = original

    def clear_artifact_fixture(self):
        # Only synthetic fixtures in this test's TemporaryDirectory.
        for path in self.executor.tasks.root.glob("*"):
            if path.is_file():
                path.unlink()

    def test_native_artifact_response_loss_reuses_write_and_upload_identity(self):
        for loss in ("prepare", "put", "commit"):
            with self.subTest(loss=loss):
                record, media, observed, reply = self.artifact_case(loss=loss)
                with self.artifact_clients(reply):
                    with self.assertRaises(httpx.ReadTimeout):
                        collect(self.executor.higgsfield, record, media)
                    durable = self.executor.tasks.get(record.video_task_id)
                    self.assertNotEqual(durable.status, "succeeded")
                    ready = collect(self.executor.higgsfield, durable, media)
                self.assertEqual(ready.status, "succeeded")
                self.assertEqual(len({p["write_id"] for p in observed["prepares"]}), 1)
                self.assertEqual(observed["downloads"], 1)
                self.assertEqual(observed["puts"], 2 if loss == "put" else 1)
                self.assertEqual(ready.provider_state["artifact_upload_id"], durable.provider_state.get(
                    "artifact_upload_id", ready.provider_state["artifact_upload_id"]))
                self.clear_artifact_fixture()

    def test_native_artifact_requires_matching_receipt_and_readback(self):
        for corrupt, foreign, expected in ((True, False, "artifact_readback_mismatch"),
                                            (False, True, "artifact_receipt_mismatch")):
            with self.subTest(expected=expected):
                record, media, observed, reply = self.artifact_case(corrupt=corrupt, foreign=foreign)
                with self.artifact_clients(reply):
                    with self.assertRaisesRegex(ValueError, expected):
                        collect(self.executor.higgsfield, record, media)
                self.assertNotEqual(self.executor.tasks.get(record.video_task_id).status, "succeeded")
                self.clear_artifact_fixture()

    def test_content_reauthorizes_fixed_version_and_detects_tampered_cache(self):
        record, media, observed, reply = self.artifact_case()
        with self.artifact_clients(reply):
            ready = collect(self.executor.higgsfield, record, media)
        version = dict(ready.provider_state["content_ref"], source=source(ready),
                       sha256=ready.media["sha256"], size=ready.media["size"], organization_id=ORG, mime="video/mp4")
        metadata_calls = []
        denied = False
        def metadata(req):
            if req.url.host == "project.internal":
                return self.respond(req)
            metadata_calls.append(req)
            self.assertEqual(req.url.path, "/v2/artifacts/" + version["version_id"])
            self.assertEqual(req.headers["X-User-Id"], USER)
            self.assertEqual(req.headers["X-Organization-Id"], ORG)
            self.assertEqual(req.url.params["artifact_id"], version["artifact_id"])
            return httpx.Response(403 if denied else 200, json=version)
        with self.artifact_clients(metadata):
            path = self.executor.higgsfield.content_path(ready)
            self.assertTrue(path.is_file())
            denied = True
            with self.assertRaises(PermissionError):
                self.executor.higgsfield.content_path(ready)
            denied = False
            version["source"] = dict(source(ready), project_id=uuid7())
            with self.assertRaisesRegex(ValueError, "artifact_receipt_mismatch"):
                self.executor.higgsfield.content_path(ready)
            version["source"] = source(ready)
            path.write_bytes(b"x" * path.stat().st_size)
            with self.assertRaisesRegex(ValueError, "media_integrity_failed"):
                self.executor.higgsfield.content_path(ready)
            self.deny_project = True
            count = len(metadata_calls)
            with self.assertRaises(PermissionError):
                self.executor.higgsfield.content_path(ready)
            self.assertEqual(len(metadata_calls), count)

    def test_dashboard_never_discloses_higgsfield_task_to_foreign_identity(self):
        from types import SimpleNamespace
        record = self.create()
        dashboard = Dashboard(self.executor)
        token = CALLER.set({"user_id": USER, "organization_id": uuid7()})
        try:
            with self.assertRaises(PermissionError):
                dashboard.detail(record.video_task_id)
            with self.assertRaises(PermissionError):
                dashboard.thumbnail(record.video_task_id)
            snapshot = dashboard.snapshot(SimpleNamespace(query_params={}, headers={}, app=SimpleNamespace(state=SimpleNamespace())))
            self.assertEqual(snapshot["tasks"], [])
            self.assertEqual(snapshot["projects"], [])
        finally:
            CALLER.reset(token)


    def test_wan_references_stream_in_order_and_quote_does_not_generate(self):
        refs = {"images": [], "videos": []}
        for group, mime, filename, payload in (("images", "image/png", "a.png", b"a" * 50),
                ("images", "image/png", "b.png", b"b" * 60), ("videos", "video/mp4", "dance.mp4", b"v" * 100)):
            asset = self.executor.artifacts.create_from_chunks(project_id=PROJECT, operation="import",
                filename=filename, media_type=mime, chunks=[payload])
            refs[group].append({"artifact_id": asset.artifact_id, "purpose": "reference"})
        received = []
        def respond(req):
            if req.method == "PUT":
                parts = req.url.path.split("/")
                info = {"project_id": parts[-3], "sha256": parts[-2], "format": parts[-1],
                        "kind": req.headers["X-Media-Kind"], "size": len(req.content)}
                self.assertEqual(hashlib.sha256(req.content).hexdigest(), info["sha256"])
                received.append(info)
                return httpx.Response(200, json={"status": "committed", **info})
            if req.url.path == "/v1/cost":
                self.assertEqual(json.loads(req.content)["references"], received)
                return httpx.Response(200, json={"credits": 0})
            return self.respond(req)
        self.executor.higgsfield.client = httpx.Client(transport=httpx.MockTransport(respond))
        record = self.create(model="wan3_0", duration_seconds=15, aspect_ratio="9:16", resolution="480p", references=refs)
        self.assertEqual(record.request["mode"], "reference_to_video")
        self.assertEqual(self.executor.higgsfield.cost(record.request), 0)
        self.assertEqual([r["kind"] for r in received], ["image", "image", "video"])
        self.assertEqual(record.runtime_stage, "prepared")
        self.assertFalse(any(r.url.path == "/v1/jobs" for r in self.calls))
        self.executor.higgsfield.client.close()

    def test_wan_capability_modes_are_registered_in_mcp_schema(self):
        from app.server import VIDEO_TOOLS_SCHEMA
        from jsonschema import Draft202012Validator
        schema = next(t["inputSchema"] for t in VIDEO_TOOLS_SCHEMA if t["name"] == "video.create")
        caps = next(c for c in self.executor.higgsfield.capabilities() if c["model"] == "wan3_0")
        validator = Draft202012Validator(schema["properties"]["mode"])
        for mode in caps["modes"]:
            self.assertEqual(list(validator.iter_errors(mode)), [])
        self.assertEqual(caps["duration_seconds"]["maximum"], 15)

    def test_wan_rejects_foreign_and_tampered_references_before_upload(self):
        asset = self.executor.artifacts.create_from_chunks(project_id=PROJECT, operation="import",
            filename="a.png", media_type="image/png", chunks=[b"original"])
        refs = {"images": [{"artifact_id": asset.artifact_id, "purpose": "identity"}]}
        with self.assertRaises(Exception):
            self.create(project_id=uuid7(), model="wan3_0", references=refs)
        self.executor.artifacts.content_path(asset).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "source_integrity_failed"):
            self.create(model="wan3_0", references=refs)
        self.assertTrue(all(r.url.host == "project.internal" for r in self.calls))


    def test_channel_health_projects_real_adapter_configuration_without_identity_or_token(self):
        adapter = self.executor.higgsfield
        observed = adapter.channel_health()
        self.assertEqual(observed, {"enabled": True, "configured": True,
            "binding_digest": hashlib.sha256(json.dumps(["team", ORG], separators=(",", ":")).encode()).hexdigest()})
        self.assertNotIn(ORG, json.dumps(observed))
        self.assertNotIn(adapter.token, json.dumps(observed))
        adapter.enabled = False
        self.assertFalse(adapter.channel_health()["enabled"])
        adapter.token = ""
        self.assertFalse(adapter.channel_health()["configured"])
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
