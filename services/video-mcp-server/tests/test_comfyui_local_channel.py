"""Tests for ComfyUI published workflow dynamic projection and execution in Video MCP."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import httpx
from PIL import Image

from app import server
from app.artifacts import ArtifactStore
from app.channel_errors import ChannelError
from app.content_rating import ContentRatingStore
from app.dashboard import public_task, task_presentation
from app.delivery import CALLER
from app.executor import VideoExecutor
from app.providers.comfyui import is_allowed_comfyui_origin
from app.tasks import TaskConflict, TaskStore

ORG_A = "0196ed00-0000-7000-8000-0000000000aa"
ORG_B = "0196ed00-0000-7000-8000-0000000000bb"
USER_A = "0196ed00-0000-7000-8000-000000000001"
PROJ_A = "0196ed00-0000-7000-8000-000000000111"
STUDIO_IN_ART = "0196ed00-0000-7000-8000-000000000222"
STUDIO_OUT_ART = "0196ed00-0000-7000-8000-000000000333"
REMOTE_TASK_ID = "0196ed00-0000-7000-8000-000000000444"
INSTANCE_ID = "0196ed00-0000-7000-8000-000000000555"


def _video_workflow_spec() -> dict[str, object]:
    return {
        "schema_version": "v1",
        "endpoint_slug": "wan-i2v-local",
        "version": "1.0.0",
        "display_name": "Wan 2.1 图生视频工作流",
        "output_modality": "video",
        "mount_channel": "video",
        "reproducibility_level": "strict",
        "seed_policy": "exposed",
        "vram_budget_mib": 18432,
        "exposed_inputs": [
            {
                "name": "prompt",
                "param_type": "string",
                "node_id": "6",
                "input_key": "text",
                "required": True,
                "min_length": 1,
                "max_length": 2000,
            },
            {
                "name": "first_frame",
                "param_type": "artifact_image",
                "node_id": "12",
                "input_key": "image",
                "required": True,
            },
            {
                "name": "steps",
                "param_type": "integer",
                "node_id": "3",
                "input_key": "steps",
                "required": False,
                "default_value": 24,
                "minimum": 4,
                "maximum": 50,
            },
            {
                "name": "motion_bucket",
                "param_type": "enum",
                "node_id": "9",
                "input_key": "motion",
                "required": False,
                "default_value": "smooth",
                "enum_values": ["calm", "smooth", "dynamic"],
            },
        ],
        "outputs": [
            {
                "name": "video",
                "node_id": "20",
                "output_kind": "video",
                "mime_type": "video/mp4",
            }
        ],
    }


class ComfyUIVideoLocalChannelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.published = True
        self.remote_state = "running"
        self.cancel_calls: list[dict[str, object]] = []
        self.invoke_calls: list[dict[str, object]] = []
        self.upload_commits: list[str] = []
        self.video_bytes = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isomsynthetic-comfyui-video"
        self.video_sha256 = hashlib.sha256(self.video_bytes).hexdigest()

        self.env = mock.patch.dict(
            os.environ,
            {
                "COMFYUI_INTERNAL_ORIGIN": "http://comfyui-manager.verdantflare-app.svc.cluster.local:8091",
                "COMFYUI_INTERNAL_TOKEN": "comfy-internal-token-1234567890",
                "COMFYUI_ORGANIZATION_ID": ORG_A,
                "VIDEO_ARTIFACT_SERVICE_URL": "http://station-core.verdantflare-station.svc.cluster.local:8080",
                "VIDEO_ARTIFACT_SERVICE_TOKEN": "station-artifact-token-1234567890",
                "VIDEO_PROJECT_SERVICE_URL": "",
                "H3_RUNTIME_ROUTE": "h3",
                "H3_RUNTIME_ROUTES": "",
                "FAL_KEY": "",
            },
            clear=False,
        )
        self.env.start()
        self.addCleanup(self.env.stop)

        def transport_handler(req: httpx.Request) -> httpx.Response:
            path = req.url.path
            if path == "/internal/v1/comfyui/published-channels":
                if req.headers.get("Authorization") != "Bearer comfy-internal-token-1234567890":
                    return httpx.Response(401, json={"error": "unauthorized"})
                body = json.loads(req.content.decode("utf-8"))
                if body.get("organization_id") != ORG_A or body.get("mount_channel") != "video" or not self.published:
                    return httpx.Response(200, json={"items": []})
                return httpx.Response(
                    200,
                    json={
                        "items": [
                            {
                                "published_api_id": "0196ed00-0000-7000-8000-000000000999",
                                "catalog_model_id": "comfyui:wan-i2v-local@1.0.0",
                                "engine": "comfyui",
                                "channel_id": "local",
                                "organization_id": ORG_A,
                                "project_id": PROJ_A,
                                "endpoint_slug": "wan-i2v-local",
                                "version": "1.0.0",
                                "display_name": "Wan 2.1 图生视频工作流",
                                "output_modality": "video",
                                "mount_channel": "video",
                                "reproducibility_level": "strict",
                                "seed_policy": "exposed",
                                "vram_budget_mib": 18432,
                                "source_badge": "本地",
                                "spec": _video_workflow_spec(),
                            }
                        ]
                    },
                )
            if path == "/v2/artifacts/uploads" and req.method == "POST":
                return httpx.Response(201, json={"upload_id": "upl-001"})
            if path == "/v2/artifacts/uploads/upl-001/content" and req.method == "PUT":
                return httpx.Response(204)
            if path == "/v2/artifacts/uploads/upl-001/commit" and req.method == "POST":
                self.upload_commits.append(STUDIO_IN_ART)
                return httpx.Response(200, json={"artifact_id": STUDIO_IN_ART})
            if path == f"/v2/artifacts/{STUDIO_OUT_ART}/content" and req.method == "GET":
                return httpx.Response(200, content=self.video_bytes, headers={"content-type": "video/mp4"})
            if path == "/internal/apps/comfyui-api/call" and req.method == "POST":
                payload = json.loads(req.content.decode("utf-8"))
                name = payload.get("name")
                args = payload.get("arguments") or {}
                if name == "api.invoke":
                    self.invoke_calls.append(args)
                    return httpx.Response(
                        200,
                        json={
                            "task_id": REMOTE_TASK_ID,
                            "instance_id": INSTANCE_ID,
                            "project_id": args["project_id"],
                            "state": "queued",
                        },
                    )
                if name == "task.get":
                    result_list = (
                        [
                            {
                                "artifact_id": STUDIO_OUT_ART,
                                "version_id": "0196ed00-0000-7000-8000-000000000334",
                                "node_id": "20",
                                "filename": "wan_out.mp4",
                                "mime_type": "video/mp4",
                                "size_bytes": len(self.video_bytes),
                                "sha256": self.video_sha256,
                            }
                        ]
                        if self.remote_state == "succeeded"
                        else None
                    )
                    return httpx.Response(
                        200,
                        json={
                            "task_id": REMOTE_TASK_ID,
                            "instance_id": INSTANCE_ID,
                            "project_id": args["project_id"],
                            "state": self.remote_state,
                            "result": result_list,
                        },
                    )
                if name == "task.cancel":
                    self.cancel_calls.append(args)
                    self.remote_state = "cancelled"
                    return httpx.Response(
                        200,
                        json={
                            "task_id": REMOTE_TASK_ID,
                            "instance_id": INSTANCE_ID,
                            "project_id": args["project_id"],
                            "state": "cancelled",
                            "cancel_requested": True,
                        },
                    )
            return httpx.Response(404, json={"error": f"unhandled:{path}"})

        self.client = httpx.Client(transport=httpx.MockTransport(transport_handler))
        self.addCleanup(self.client.close)
        self.executor = VideoExecutor(ArtifactStore(self.root), TaskStore(self.root), self.client)
        self.executor.comfyui.probe = lambda _p: {  # type: ignore[method-assign]
            "width": 1280,
            "height": 720,
            "frame_rate": 24.0,
            "duration_ms": 5000,
            "video_codec": "h264",
            "audio_codec": None,
            "audio_channels": None,
            "audio_sample_rate": None,
        }
        self.server_patch = mock.patch.object(server, "executor", self.executor)
        self.server_patch.start()
        self.addCleanup(self.server_patch.stop)
        for attr, val in (
            ("tasks", self.executor.tasks),
            ("artifacts", self.executor.artifacts),
            ("ratings", ContentRatingStore(self.executor.tasks.db_path, domain="video")),
        ):
            p = mock.patch.object(server.app.state, attr, val)
            p.start()
            self.addCleanup(p.stop)

        buf = io.BytesIO()
        Image.new("RGB", (64, 36), "navy").save(buf, "PNG")
        self.ref_image = self.executor.artifacts.create_from_chunks(
            project_id=PROJ_A,
            operation="video.import",
            filename="frame0.png",
            media_type="image/png",
            chunks=(buf.getvalue(),),
        )

    def test_origin_allowlist_enforces_internal_cluster_boundary(self) -> None:
        self.assertTrue(
            is_allowed_comfyui_origin(
                "http://comfyui-manager.verdantflare-app.svc.cluster.local:8091"
            )
        )
        self.assertFalse(is_allowed_comfyui_origin("https://evil.example.com"))
        self.assertFalse(is_allowed_comfyui_origin("http://127.0.0.1:8091", allow_loopback=False))
        self.assertTrue(is_allowed_comfyui_origin("http://127.0.0.1:8091", allow_loopback=True))

    def test_dynamic_catalog_org_isolation_and_chinese_local_badge(self) -> None:
        snapshot_before = copy.deepcopy(self.executor.catalog.snapshot())

        tok_a = CALLER.set({"organization_id": ORG_A, "user_id": USER_A})
        try:
            view_a = self.executor.catalog_view()
            caps_a = self.executor.capabilities()
        finally:
            CALLER.reset(tok_a)

        local_ch = next(ch for ch in view_a["channels"] if ch["id"] == "local")
        self.assertEqual(local_ch["display_name"], "本地")
        entry = next(m for m in local_ch["models"] if m["model_id"] == "comfyui:wan-i2v-local@1.0.0")
        self.assertEqual(entry["source_badge"], "本地")
        self.assertEqual(entry["declaration_source"], "comfyui_published_api")
        self.assertTrue(view_a["execution"]["local"][entry["id"]]["can_submit"])
        self.assertTrue(view_a["execution"]["comfyui"][entry["id"]]["can_submit"])
        comfy_route = next(r for r in caps_a["routes"] if r["route"] == "comfyui")
        self.assertEqual(comfy_route["source_badge"], "本地")
        self.assertEqual(self.executor.catalog.snapshot(), snapshot_before)

        # Cross-org isolation + malformed caller org fail-closed
        for caller in (
            {"organization_id": ORG_B, "user_id": USER_A},
            {"organization_id": "not-a-uuid", "user_id": USER_A},
        ):
            tok = CALLER.set(caller)
            try:
                isolated = self.executor.catalog_view()
            finally:
                CALLER.reset(tok)
            local_models = [
                m
                for ch in isolated["channels"]
                if ch["id"] == "local"
                for m in ch["models"]
                if m["model_id"].startswith("comfyui:")
            ]
            self.assertEqual(local_models, [])

    def test_end_to_end_artifact_bridge_presentation_and_unpublish_drain(self) -> None:
        tok = CALLER.set({"organization_id": ORG_A, "user_id": USER_A})
        try:
            pre = server.video_preflight(
                project_id=PROJ_A,
                model="comfyui:wan-i2v-local@1.0.0",
                route="local",
                prompt="A lantern drifting across dusk water",
                references={"images": [{"artifact_id": self.ref_image.artifact_id, "purpose": "first_frame"}]},
                workflow_params={"steps": 28, "motion_bucket": "dynamic"},
            ).structured_content
            self.assertEqual(pre["status"], "validated")
            self.assertEqual(pre["source_badge"], "本地")

            created = server.video_create(
                project_id=PROJ_A,
                idempotency_key="idem-video-comfy-01",
                model="comfyui:wan-i2v-local@1.0.0",
                route="local",
                prompt="A lantern drifting across dusk water",
                references={"images": [{"artifact_id": self.ref_image.artifact_id, "purpose": "first_frame"}]},
                workflow_params={"steps": 28, "motion_bucket": "dynamic"},
            ).structured_content
            task_id = created["video_task_id"]
            self.assertEqual(len(self.upload_commits), 1)
            self.assertEqual(self.invoke_calls[0]["params"]["first_frame"], STUDIO_IN_ART)
            self.assertEqual(self.invoke_calls[0]["params"]["steps"], 28)

            # Idempotent replay returns same task without re-invoking
            replayed = server.video_create(
                project_id=PROJ_A,
                idempotency_key="idem-video-comfy-01",
                model="comfyui:wan-i2v-local@1.0.0",
                route="local",
                prompt="A lantern drifting across dusk water",
                references={"images": [{"artifact_id": self.ref_image.artifact_id, "purpose": "first_frame"}]},
                workflow_params={"steps": 28, "motion_bucket": "dynamic"},
            ).structured_content
            self.assertEqual(replayed["video_task_id"], task_id)
            self.assertEqual(len(self.invoke_calls), 1)

            # Unpublish workflow while task is still running -> new submission rejected immediately,
            # while in-flight task drains and collects output cleanly!
            self.published = False
            with self.assertRaises(ChannelError) as ctx:
                self.executor.generate(
                    project_id=PROJ_A,
                    idempotency_key="idem-video-comfy-after-unpublish",
                    model="comfyui:wan-i2v-local@1.0.0",
                    route="comfyui",
                    prompt="New submission after archive",
                    duration_seconds=5,
                    aspect_ratio="16:9",
                    references={"images": [{"artifact_id": self.ref_image.artifact_id, "purpose": "first_frame"}]},
                )
            self.assertEqual(ctx.exception.code, "channel_unavailable")

            self.remote_state = "succeeded"
            status_res = server.video_status(video_task_id=task_id).structured_content
            self.assertEqual(status_res["status"], "succeeded")

            result_res = server.video_result(video_task_id=task_id).structured_content
            self.assertEqual(result_res["status"], "succeeded")
            self.assertEqual(result_res["media"]["sha256"], self.video_sha256)
            saved_art = self.executor.artifacts.get(result_res["artifact_id"], PROJ_A)
            self.assertEqual(self.executor.artifacts.content_path(saved_art).read_bytes(), self.video_bytes)

            # Presentation keeps Chinese `本地` label and saved display_name even after unpublish
            rec = self.executor.tasks.get(task_id)
            pres = task_presentation(rec, self.executor.catalog_view())
            self.assertEqual(pres["channel_id"], "local")
            self.assertEqual(pres["channel_label"], "本地")
            self.assertEqual(pres["model_label"], "Wan 2.1 图生视频工作流")
            pub = public_task(rec, self.executor.tasks, catalog=self.executor.catalog_view())
            self.assertEqual(pub["presentation"]["channel_label"], "本地")
        finally:
            CALLER.reset(tok)

    def test_exposed_inputs_reject_unexposed_keys_and_forward_cancel(self) -> None:
        tok = CALLER.set({"organization_id": ORG_A, "user_id": USER_A})
        try:
            for bad_params in (
                {"unexposed_node_param": 123},
                {"raw_workflow": {"1": {"class_type": "KSampler"}}},
                {"steps": 999},
                {"motion_bucket": "illegal"},
            ):
                with self.subTest(bad_params=bad_params):
                    err_res = server.video_create(
                        project_id=PROJ_A,
                        idempotency_key="idem-bad",
                        model="comfyui:wan-i2v-local@1.0.0",
                        route="comfyui",
                        prompt="Valid prompt",
                        references={"images": [{"artifact_id": self.ref_image.artifact_id, "purpose": "first_frame"}]},
                        workflow_params=bad_params,
                    ).model_dump(by_alias=True)
                    self.assertTrue(err_res["isError"])
            self.assertEqual(self.invoke_calls, [])

            created = server.video_create(
                project_id=PROJ_A,
                idempotency_key="idem-cancel-01",
                model="comfyui:wan-i2v-local@1.0.0",
                route="comfyui",
                prompt="Cancelable task",
                references={"images": [{"artifact_id": self.ref_image.artifact_id, "purpose": "first_frame"}]},
            ).structured_content
            with self.assertRaises(TaskConflict):
                self.executor.generate(
                    project_id=PROJ_A,
                    idempotency_key="idem-cancel-01",
                    model="comfyui:wan-i2v-local@1.0.0",
                    route="comfyui",
                    prompt="Mutated prompt",
                    duration_seconds=5,
                    aspect_ratio="16:9",
                    references={"images": [{"artifact_id": self.ref_image.artifact_id, "purpose": "first_frame"}]},
                )

            cancelled = server.video_cancel(video_task_id=created["video_task_id"]).structured_content
            self.assertEqual(cancelled["status"], "cancelled")
            self.assertEqual(len(self.cancel_calls), 1)
            self.assertEqual(self.cancel_calls[0]["task_id"], REMOTE_TASK_ID)
        finally:
            CALLER.reset(tok)


if __name__ == "__main__":
    unittest.main()
