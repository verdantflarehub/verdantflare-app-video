from __future__ import annotations

import io
import json
import shutil
import unittest
import wave
from unittest import mock

import httpx
from PIL import Image

from app.artifacts import ArtifactError
from app.dashboard import task_presentation
from app.executor import VideoExecutor
from app.providers.fal import FalAdapter, FAL_LIP_PUBLIC_MODEL, FAL_LIP_MODEL_ID
from app.tasks import TaskConflict
import test_fal_route as fixtures


class LipSyncTest(unittest.TestCase):
    tearDown = fixtures.FalRouteTest.tearDown
    inputs = fixtures.FalRouteTest.inputs

    def setUp(self):
        fixtures.FalRouteTest.setUp(self)
        output = io.BytesIO()
        with wave.open(output, "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            audio.writeframes(b"\0\0" * 80000)
        self.audio = self.artifacts.create_from_chunks(project_id="demo", operation="test",
            filename="speech.wav", media_type="audio/wav", chunks=[output.getvalue()])

    def lip_inputs(self, **updates):
        content = io.BytesIO()
        Image.new("RGB", (48, 96)).save(content, format="PNG")
        image = self.artifacts.create_from_chunks(project_id="demo", operation="test",
            filename="face.png", media_type="image/png", chunks=[content.getvalue()])
        return self.inputs(model=FAL_LIP_PUBLIC_MODEL, aspect_ratio="adaptive", mode="lip_sync",
            references={"images": [{"artifact_id": image.artifact_id, "purpose": "face"}],
                        "audios": [{"artifact_id": self.audio.artifact_id, "purpose": "exact speech"}],
                        "videos": []}, **updates)

    def executor(self, timeout=False):
        self.sent = []
        def respond(request):
            self.sent.append(request)
            if request.method == "POST":
                if timeout:
                    raise httpx.ReadTimeout("interrupted", request=request)
                return httpx.Response(200, json={"call_id": "call-1", "state": "queued", "accepted": True})
            return httpx.Response(200, json={"call_id": "call-1", "state": "acceptance_unknown", "accepted": None})
        self.client = httpx.Client(transport=httpx.MockTransport(respond))
        self.addCleanup(self.client.close)
        return VideoExecutor(self.artifacts, self.tasks, self.client)

    @mock.patch.object(FalAdapter, "lip_sync_declared", return_value=True)
    @mock.patch.object(FalAdapter, "_reference_duration", return_value=5.0)
    def test_exact_payload_idempotency_and_task_labels(self, *_):
        ex = self.executor(); args = self.lip_inputs(resolution="480p")
        first = ex.generate(**args)
        self.assertEqual(first.request["mode"], "lip_sync")
        self.assertEqual(first.request["endpoint_id"], FAL_LIP_MODEL_ID)
        self.assertEqual(first.request["source_audio_duration_seconds"], 5.0)
        self.assertEqual(first.request["prompt_usage"], "task_description_only")
        payload = json.loads(next(r.content for r in self.sent if r.method == "POST"))
        self.assertEqual(payload["model_id"], FAL_LIP_MODEL_ID)
        self.assertEqual(payload["budget_limit"], "1.00")
        self.assertEqual(set(payload["payload"]), {"image_url", "audio_url", "resolution",
            "enable_transcription", "enable_safety_checker"})
        self.assertEqual(payload["payload"]["resolution"], "480P")
        self.assertTrue(payload["payload"]["image_url"].startswith("data:image/png;base64,"))
        self.assertTrue(payload["payload"]["audio_url"].startswith("data:audio/wav;base64,"))
        self.assertEqual(ex.generate(**args).video_task_id, first.video_task_id)
        self.assertEqual(sum(r.method == "POST" for r in self.sent), 1)
        with self.assertRaises(TaskConflict):
            ex.generate(**dict(args, resolution="768p"))
        presentation = task_presentation(first)
        self.assertEqual(presentation["model_label"], "MiniMax H3 Max Lip Sync")
        self.assertEqual(presentation["generation_code"], "A2V")
        self.assertEqual(presentation["channel_label"], "fal.ai")

    @mock.patch.object(FalAdapter, "lip_sync_declared", return_value=True)
    def test_duration_mismatch_rejected_before_task_and_submission(self, *_):
        ex = self.executor(); args = self.lip_inputs()
        for duration in [4.99, 15.01, 5.051, float("nan")]:
            with self.subTest(seconds=duration), mock.patch.object(FalAdapter, "_reference_duration", return_value=duration):
                with self.assertRaises(ValueError):ex.generate(**args)
                self.assertIsNone(self.tasks.find_idempotency("demo", args["idempotency_key"]))
        self.assertEqual(self.sent, [])

    @mock.patch.object(FalAdapter, "_reference_duration", return_value=5.0)
    def test_undeclared_channel_fails_closed_but_existing_task_can_recover(self, *_):
        ex = self.executor(); args = self.lip_inputs()
        with mock.patch.object(FalAdapter, "lip_sync_declared", return_value=False):
            with self.assertRaises(ValueError):ex.generate(**args)
        self.assertIsNone(self.tasks.find_idempotency("demo", args["idempotency_key"]))
        with mock.patch.object(FalAdapter, "lip_sync_declared", return_value=True):first = ex.generate(**args)
        with mock.patch.object(FalAdapter, "lip_sync_declared", return_value=False):
            self.assertEqual(ex.generate(**args).video_task_id, first.video_task_id)
        self.assertEqual(sum(r.method == "POST" for r in self.sent), 1)

    @mock.patch.object(FalAdapter, "lip_sync_declared", return_value=True)
    @mock.patch.object(FalAdapter, "_reference_duration", return_value=5.0)
    def test_reject_extra_refs_foreign_project_ratio_and_resolution(self, *_):
        ex = self.executor(); args = self.lip_inputs()
        invalid = [dict(args, aspect_ratio="9:16"), dict(args, resolution="1080p"),
                   dict(args, mode="reference_to_video"), dict(args, project_id="other"),
                   dict(args, references=dict(args["references"], videos=[{"artifact_id": self.video.artifact_id, "purpose": "motion"}])),
                   dict(args, references=dict(args["references"], audios=[]))]
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises((ValueError, ArtifactError)):ex.generate(**value)
        self.assertEqual(self.sent, [])

    @mock.patch.object(FalAdapter, "lip_sync_declared", return_value=True)
    @mock.patch.object(FalAdapter, "_reference_duration", return_value=5.0)
    def test_corrupt_image_rejected_before_task(self, *_):
        ex = self.executor(); args = self.lip_inputs()
        image = self.artifacts.get(args["references"]["images"][0]["artifact_id"], "demo")
        self.artifacts.content_path(image).write_bytes(b"changed")
        with self.assertRaises(ArtifactError):ex.generate(**args)
        self.assertIsNone(self.tasks.find_idempotency("demo", args["idempotency_key"]))
        self.assertEqual(self.sent, [])

    @mock.patch.object(FalAdapter, "lip_sync_declared", return_value=True)
    @mock.patch.object(FalAdapter, "_reference_duration", return_value=5.0)
    def test_unknown_acceptance_never_posts_again(self, *_):
        ex = self.executor(timeout=True); args = self.lip_inputs()
        first = ex.generate(**args)
        self.assertEqual(first.runtime_stage, "submission_unknown")
        self.assertEqual(ex.generate(**args).video_task_id, first.video_task_id)
        self.assertEqual(sum(r.method == "POST" for r in self.sent), 1)

    def test_live_directory_only_enables_exact_lip_endpoint(self):
        ex = self.executor()
        item = {"capability_id": "video.h3max.lip_sync", "type": "video",
                "model_id": "minimax/h3-max", "endpoint_id": FAL_LIP_MODEL_ID,
                "display_name": "Lip Sync Max", "execution_state": "adapter_connected_unverified",
                "output_resolutions": ["480P", "768P", "1080P", "2K"]}
        directory = {"schema_version": 1, "channel_id": "fal", "version": "a" * 64,
                     "capabilities": [item]}
        real_client = httpx.Client
        def response(request):
            self.assertEqual(request.url.path, "/v1/capabilities")
            self.assertEqual(request.headers["X-Caller-Id"], "video-mcp")
            return httpx.Response(200, json=directory)
        with mock.patch("app.providers.fal.httpx.Client", side_effect=lambda **kwargs:
                        real_client(transport=httpx.MockTransport(response), **kwargs)):
            self.assertTrue(ex.fal.lip_sync_declared())
            item["execution_state"] = "catalog_only"
            self.assertFalse(ex.fal.lip_sync_declared())
            item.update(execution_state="adapter_connected_unverified",
                        endpoint_id="minimax/h3-max/recast")
            self.assertIsNone(ex.fal.channel_capabilities())
            self.assertFalse(ex.fal.lip_sync_declared())

    @unittest.skipUnless(shutil.which("ffprobe"), "requires ffprobe in the dev service image")
    @mock.patch.object(FalAdapter, "lip_sync_declared", return_value=True)
    def test_real_wav_duration_before_submission(self, *_):
        ex = self.executor()
        for seconds in (4.5, 5.0, 5.1):
            content = io.BytesIO()
            with wave.open(content, "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                audio.writeframes(b"\0\0" * int(seconds * 16000))
            self.audio = self.artifacts.create_from_chunks(project_id="demo", operation="test",
                filename="speech.wav", media_type="audio/wav", chunks=[content.getvalue()])
            args = self.lip_inputs(idempotency_key=f"real-wav/{seconds}")
            if seconds == 5.0:
                task = ex.generate(**args)
                self.assertEqual(task.request["source_audio_duration_seconds"], 5.0)
            else:
                with self.assertRaises(ValueError):ex.generate(**args)
                self.assertIsNone(self.tasks.find_idempotency("demo", args["idempotency_key"]))
        self.assertEqual(sum(r.method == "POST" for r in self.sent), 1)


if __name__ == "__main__":unittest.main()
