import asyncio
import base64
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from jsonschema import validate

from app import server
from app.artifacts import ArtifactError, ArtifactStore
from app.executor import VideoExecutor
from app.tasks import TaskStore


class SkillContractTest(unittest.TestCase):
    def test_chunk_tools_have_the_same_schema_in_studio_and_native_mcp(self):
        native = {tool.name: tool.input_schema for tool in asyncio.run(server.mcp.list_tools())}
        advertised = {tool["name"]: tool["inputSchema"] for tool in server.registry.tools}
        for name in ("video.import.prepare", "video.import.chunk", "video.import.status", "video.import.commit", "video.preflight"):
            with self.subTest(name=name):
                self.assertEqual(native[name], advertised[name])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env = mock.patch.dict(os.environ, {"H3_RUNTIME_ROUTE": "h3", "H3_RUNTIME_ROUTES": "",
                                               "H3_SOL_RUNTIME_URL": "", "FAL_KEY": ""})
        self.env.start()
        self.addCleanup(self.env.stop)
        root = Path(self.temp.name)
        self.executor = VideoExecutor(ArtifactStore(root), TaskStore(root))
        self.addCleanup(self.executor.client.close)
        self.patch = mock.patch.object(server, "executor", self.executor)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.payload = b"\x89PNG\r\n\x1a\nfixture"
        self.args = dict(project_id="demo", filename="reference.png",
                         expected_sha256=hashlib.sha256(self.payload).hexdigest(),
                         content_base64=base64.b64encode(self.payload).decode())

    def test_import_is_registered_in_video_domain_and_returns_project_reference(self):
        tools = {tool.name for tool in asyncio.run(server.mcp.list_tools())}
        advertised = {tool["name"] for tool in server.registry.tools}
        self.assertTrue({"video.import", "video.capabilities", "video.create"} <= tools & advertised)
        result = server.video_import(**self.args).structured_content
        record = self.executor.artifacts.get(result["artifact"]["artifact_id"], "demo")
        self.assertEqual(self.executor.artifacts.content_path(record).read_bytes(), self.payload)
        self.assertEqual(record.sha256, self.args["expected_sha256"])
        with self.assertRaises(ArtifactError):
            self.executor.artifacts.get(record.artifact_id, "other")

    def test_import_rejects_bad_hash_payload_and_ambiguous_source(self):
        for updates in ({"expected_sha256": "0" * 64}, {"content_base64": "%%%"},
                        {"content_base64": base64.b64encode(b"not an image").decode()},
                        {"content_base64": "A" * (4194304 + 4)},
                        {"source_url": "https://example.invalid/asset.png"}):
            with self.subTest(updates=list(updates)), self.assertRaises((ValueError, ArtifactError)):
                server.video_import(**{**self.args, **updates})
        self.assertEqual(list(self.executor.artifacts.artifacts_root.glob("art_*")), [])

    def test_capabilities_do_not_claim_configuration_is_readiness(self):
        with mock.patch.object(self.executor.client, "get", side_effect=AssertionError("network")):
            result = server.video_capabilities().structured_content
        self.assertEqual(result["default_route"], "h3")
        self.assertIn("16:9", result["model_specification"]["aspect_ratios"])
        routes = {r["route"]: r for r in result["routes"]}
        self.assertFalse(routes["fal"]["configured"])
        self.assertEqual(routes["h3"]["readiness"], "not_checked")
        self.assertEqual(routes["h3"]["aspect_ratios"], ["9:16"])

    def test_schema_accepts_15_seconds_but_unsupported_combination_creates_no_task(self):
        artifact = server.video_import(**self.args).structured_content["artifact"]
        request = dict(project_id="demo", idempotency_key="shot/v1", prompt="A shot",
                       duration_seconds=15, aspect_ratio="16:9", route="h3",
                       references={"images": [{"artifact_id": artifact["artifact_id"], "purpose": "identity"}]})
        schema = next(t["inputSchema"] for t in server.registry.tools if t["name"] == "video.create")
        validate(request, schema)
        with mock.patch.object(self.executor.client, "post", side_effect=AssertionError("submitted")):
            with self.assertRaisesRegex(ValueError, "only 9:16"):
                server.video_create(**request)
        self.assertIsNone(self.executor.tasks.find_idempotency("demo", "shot/v1"))

    def test_create_uses_host_default_and_preserves_idempotency(self):
        artifact = server.video_import(**self.args).structured_content["artifact"]
        args = dict(project_id="demo", idempotency_key="shot/v1", prompt="A shot", duration_seconds=15, aspect_ratio="9:16",
                    references={"images": [{"artifact_id": artifact["artifact_id"], "purpose": "identity"}]})
        import httpx
        def reply(request):
            return httpx.Response(200, json={"id": "runtime-task"})
        with httpx.Client(transport=httpx.MockTransport(reply)) as client:
            self.executor.client = client
            first = server.video_create(**args).structured_content
            second = server.video_create(**args).structured_content
        self.assertEqual(first["task_id"], second["task_id"])
        record = self.executor.tasks.get(first["task_id"])
        self.assertEqual(record.request["route"], "h3")
        self.assertEqual(record.request["duration_seconds"], 15)
