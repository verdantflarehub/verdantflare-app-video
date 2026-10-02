from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from app.artifacts import ArtifactStore
from app.executor import VideoExecutor
from app.providers.fal import FalError
from app.tasks import TaskStore


class FalRouteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.artifacts = ArtifactStore(root)
        self.tasks = TaskStore(root)
        self.image = self.artifacts.create_from_chunks(
            project_id="demo",
            operation="test",
            filename="start.png",
            media_type="image/png",
            chunks=(b"\x89PNG\r\n\x1a\nfixture",),
        )
        self.video = self.artifacts.create_from_chunks(
            project_id="demo",
            operation="test",
            filename="motion.mp4",
            media_type="video/mp4",
            chunks=(b"video-fixture",),
        )
        self.audio = self.artifacts.create_from_chunks(
            project_id="demo",
            operation="test",
            filename="rhythm.wav",
            media_type="audio/wav",
            chunks=(b"audio-fixture",),
        )
        self.environment = mock.patch.dict(
            os.environ,
            {
                "FAL_KEY": "test-fal-key",
                "FAL_QUEUE_BASE_URL": "https://queue.fal.run",
                "FAL_MODEL_ID": "attacker/override-must-be-ignored",
                "FAL_RUNTIME_VERSION": "attacker-version",
                "FAL_RESOLUTION": "4K",
                "FAL_RESULT_ORIGINS": "https://attacker.example",
            },
        )
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()
        self.temporary.cleanup()

    def inputs(self, **updates):
        value = {
            "project_id": "demo",
            "idempotency_key": "shot-001/v1",
            "model": "minimax-h3-ref2va",
            "prompt": "Slow cinematic camera movement",
            "duration_seconds": 5,
            "aspect_ratio": "9:16",
            "references": {
                "images": [{"artifact_id": self.image.artifact_id, "purpose": "start-frame"}],
                "videos": [],
                "audios": [],
            },
            "route": "fal",
        }
        value.update(updates)
        return value

    def test_submit_poll_and_result_preserve_provider_boundary(self) -> None:
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            if request.method == "POST":
                self.assertEqual(request.url.path, "/minimax/h3/reference-to-video")
                self.assertEqual(request.headers["Authorization"], "Key test-fal-key")
                body = json.loads(request.content)
                self.assertEqual(body["duration"], 5)
                self.assertEqual(body["resolution"], "480P")
                self.assertEqual(body["aspect_ratio"], "9:16")
                self.assertTrue(body["enable_safety_checker"])
                self.assertTrue(body["reference_image_urls"][0].startswith("data:image/png;base64,"))
                self.assertTrue(body["prompt"].startswith("Image 1 is the approved start-frame reference."))
                self.assertFalse({"image_url", "end_image_url", "target_audio_url"} & body.keys())
                return httpx.Response(200, json={"request_id": "fal-request-001"})
            if request.url.host == "queue.fal.run" and request.url.path.endswith("/status"):
                return httpx.Response(
                    200,
                    json={"status": "COMPLETED", "metrics": {"inference_time": 12.5}},
                )
            if request.url.host == "queue.fal.run":
                return httpx.Response(
                    200, json={"video": {"url": "https://v3.fal.media/files/result.mp4"}}
                )
            self.assertEqual(request.url.host, "v3.fal.media")
            self.assertNotIn("Authorization", request.headers)
            return httpx.Response(200, content=b"video-bytes", headers={"content-type": "video/mp4"})

        client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
        executor = VideoExecutor(self.artifacts, self.tasks, client)
        submitted = executor.generate(**self.inputs())
        self.assertEqual((submitted.service, submitted.runtime_route), ("fal", "fal"))
        self.assertEqual(submitted.runtime_task_id, "fal-request-001")
        self.assertNotIn("test-fal-key", submitted.model_dump_json())

        status = executor.status(submitted.video_task_id)
        self.assertEqual(status.status, "succeeded")
        self.assertEqual(status.runtime_metrics, {"runtime_total_seconds": 12.5})
        probe = mock.Mock(
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "codec_type": "video",
                            "codec_name": "h264",
                            "width": 768,
                            "height": 1344,
                            "r_frame_rate": "24/1",
                        },
                        {
                            "codec_type": "audio",
                            "codec_name": "aac",
                            "sample_rate": "48000",
                            "channels": 2,
                        },
                    ],
                    "format": {"duration": "5.0"},
                }
            )
        )
        with mock.patch("app.providers.fal.subprocess.run", return_value=probe):
            result = executor.result(submitted.video_task_id)
        self.assertIsNotNone(result.artifact_id)
        self.assertEqual(result.media["frame_rate"], 24)
        self.assertEqual([call.method for call in calls], ["POST", "GET", "GET", "GET"])
        client.close()

    def test_missing_key_fails_closed_without_network_or_fallback(self) -> None:
        calls = []
        with mock.patch.dict(os.environ, {"FAL_KEY": ""}):
            client = httpx.Client(
                transport=httpx.MockTransport(lambda request: calls.append(request) or httpx.Response(500))
            )
            executor = VideoExecutor(self.artifacts, self.tasks, client)
            record = executor.generate(**self.inputs(idempotency_key="missing-key"))
            self.assertEqual((record.status, record.error["code"]), ("failed", "provider_auth_not_configured"))
            self.assertEqual(record.service, "fal")
            self.assertFalse(calls)
            client.close()

    def test_auth_failure_is_redacted_and_idempotent(self) -> None:
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(401, text="test-fal-key upstream detail")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        executor = VideoExecutor(self.artifacts, self.tasks, client)
        inputs = self.inputs(idempotency_key="auth-failure")
        first = executor.generate(**inputs)
        second = executor.generate(**inputs)
        self.assertEqual(first.video_task_id, second.video_task_id)
        self.assertEqual(first.error["code"], "provider_auth_failed")
        self.assertNotIn("test-fal-key", first.model_dump_json())
        self.assertEqual(len(calls), 1)
        client.close()

    def test_changed_reference_fails_before_provider_submission(self) -> None:
        self.artifacts.content_path(self.image).write_bytes(b"changed-after-registration")
        calls = []
        client = httpx.Client(
            transport=httpx.MockTransport(lambda request: calls.append(request) or httpx.Response(200))
        )
        executor = VideoExecutor(self.artifacts, self.tasks, client)
        record = executor.generate(**self.inputs(idempotency_key="changed-reference"))
        self.assertEqual((record.status, record.error["code"]), ("failed", "source_integrity_failed"))
        self.assertFalse(calls)
        client.close()

    def test_multimodal_reference_payload_matches_fal_schema(self) -> None:
        submitted = {}

        def handler(request):
            submitted.update(json.loads(request.content))
            return httpx.Response(200, json={"request_id": "fal-multimodal-001"})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        executor = VideoExecutor(self.artifacts, self.tasks, client)
        timed_probe = mock.Mock(stdout=json.dumps({"format": {"duration": "5.0"}}))
        references = {
            "images": [{"artifact_id": self.image.artifact_id, "purpose": "character identity"}],
            "videos": [{"artifact_id": self.video.artifact_id, "purpose": "camera motion"}],
            "audios": [{"artifact_id": self.audio.artifact_id, "purpose": "dialogue rhythm"}],
        }
        with mock.patch("app.providers.fal.subprocess.run", return_value=timed_probe):
            record = executor.generate(
                **self.inputs(
                    idempotency_key="multimodal",
                    aspect_ratio="adaptive",
                    references=references,
                )
            )
        self.assertEqual(record.runtime_task_id, "fal-multimodal-001")
        self.assertEqual(len(submitted["reference_image_urls"]), 1)
        self.assertEqual(len(submitted["reference_video_urls"]), 1)
        self.assertEqual(len(submitted["reference_audio_urls"]), 1)
        self.assertIn("Video 1 is the approved camera motion reference", submitted["prompt"])
        self.assertIn("Audio 1 is the approved dialogue rhythm reference", submitted["prompt"])
        client.close()

    def test_model_reference_and_result_origin_are_restricted(self) -> None:
        executor = VideoExecutor(self.artifacts, self.tasks)
        with self.assertRaises(ValueError):
            executor.generate(**self.inputs(model="minimax-h3-max"))
        with self.assertRaises(ValueError):
            executor.generate(
                **self.inputs(
                    idempotency_key="bad-video-ref",
                    references={
                        "images": [],
                        "videos": [{"artifact_id": self.image.artifact_id, "purpose": "motion"}],
                        "audios": [],
                    },
                )
            )
        with self.assertRaises(ValueError):
            executor.generate(**self.inputs(idempotency_key="too-short", duration_seconds=4))
        audio_only, _ = executor.fal.normalize(
            project_id="demo",
            model="minimax-h3-ref2va",
            prompt="Audio 1 guides the generated scene",
            duration_seconds=5,
            aspect_ratio="adaptive",
            references={
                "images": [],
                "videos": [],
                "audios": [{"artifact_id": self.audio.artifact_id, "purpose": "sound design"}],
            },
        )
        self.assertEqual(len(audio_only["references"]["audios"]), 1)
        with self.assertRaises(ValueError):
            executor.fal.normalize(
                project_id="demo",
                model="minimax-h3-ref2va",
                prompt="No reference",
                duration_seconds=5,
                aspect_ratio="adaptive",
                references={"images": [], "videos": [], "audios": []},
            )
        with self.assertRaisesRegex(FalError, "result_download_failed"):
            executor.fal._allowed_result_url("https://attacker.example/result.mp4")


if __name__ == "__main__":
    unittest.main()
