import json
import os
import unittest
import httpx
from pathlib import Path
import sys
from unittest.mock import patch

import test_sol_route
from app.executor import VideoExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "video-minimax-h3-singularity" / "src"))
from h3_singularity.contract import resolve_geometry, resolve_timing, validate_request


class SingularityRouteTest(unittest.TestCase):
    def setUp(self):
        test_sol_route.SolRouteTest.setUp(self)
        self.singularity_env = patch.dict(os.environ, {
            "H3_RUNTIME_URL": "http://singularity.example:8000",
            "H3_RUNTIME_ROUTE": "h3-singularity",
            "H3_SINGULARITY_RUNTIME_TOKEN": "test-singularity-token",
        })
        self.singularity_env.start()
        self.addCleanup(self.singularity_env.stop)
        self.executor = VideoExecutor(self.assets, self.tasks, self.http)
        self.media_patch = patch("app.media_validation.validate_reference_media", return_value={"type": "image", "width": 768, "height": 1344, "has_audio": False})
        self.media_patch.start()
        self.addCleanup(self.media_patch.stop)

    def tearDown(self):
        test_sol_route.SolRouteTest.tearDown(self)

    def test_ref2va_uses_four_nfe_bearer_and_idempotency(self):
        row = self.executor.generate(**self.kw, route="h3-singularity")
        self.assertEqual(row.service, "h3-singularity")
        health, submit = self.calls[-2:]
        self.assertEqual(health.url.host, "singularity.example")
        self.assertEqual(health.headers["Authorization"], "Bearer test-singularity-token")
        body = json.loads(submit.content)
        self.assertEqual(submit.url.host, "singularity.example")
        self.assertEqual(submit.headers["Authorization"], "Bearer test-singularity-token")
        self.assertEqual(body["num_inference_steps"], 4)
        self.assertEqual(body["quality_profile"], "du-0")
        self.assertEqual(body["idempotency_key"], row.video_task_id)

    def test_deferred_quality_profiles_are_rejected_before_submission(self):
        for profile in ("du-1", "du-2", "du-3", "hr-refine-tile-v1", "hr-refine-global-v2"):
            with self.subTest(profile=profile), self.assertRaisesRegex(ValueError, "only du-0"):
                self.executor.generate(**self.kw, route="h3-singularity", quality_profile=profile)
        self.assertFalse(self.calls)

    def test_landscape_15_seconds_preserves_prompt_and_input_hash(self):
        args = {**self.kw, "aspect_ratio": "16:9", "duration_seconds": 15}
        row = self.executor.generate(**args, route="h3-singularity")
        body = json.loads(self.calls[-1].content)
        self.assertEqual(body["prompt"], args["prompt"].strip())
        self.assertEqual(body["target"]["aspect_ratio"], "16:9")
        self.assertEqual(row.request["resolved_output"]["width"], 1344)
        self.assertEqual(row.request["resolved_output"]["height"], 768)
        self.assertEqual(row.request["resolved_output"]["frames"], 362)
        self.assertEqual(len(body["conditions"][0]["sha256"]), 64)
        self.assertGreater(body["conditions"][0]["size"], 0)
        self.assertEqual(row.request["reference_mapping"][0]["tag"], "<Picture 1>")
        validate_request(body, self.executor.runtime_artifact_url)

    def test_mcp_and_runtime_resolve_identical_native_grids(self):
        from app.singularity import ASPECT_RATIOS, output_spec
        for ratio in ASPECT_RATIOS:
            for seconds in range(4, 16):
                with self.subTest(ratio=ratio, seconds=seconds):
                    actual = output_spec(ratio, seconds, {"width": 1920, "height": 1080})
                    expected = {**resolve_geometry(ratio, (1920, 1080)), **resolve_timing(seconds)}
                    self.assertEqual(actual, expected)

    def test_dual_quality_profile_is_restricted_to_singularity(self):
        with self.assertRaisesRegex(ValueError, "require h3-singularity"):
            self.executor.generate(**self.kw, route="h3", quality_profile="du-1")
        self.assertFalse(self.calls)

    def test_preflight_matches_create_without_submitting(self):
        from app import server
        with patch.object(server, "executor", self.executor):
            args = {**self.kw, "aspect_ratio": "16:9", "duration_seconds": 15, "route": "h3-singularity"}
            preflight = server.video_preflight(**{k: v for k, v in args.items() if k != "idempotency_key"}).structured_content
            self.assertFalse(self.calls)
            self.assertIsNone(self.tasks.find_idempotency(args["project_id"], args["idempotency_key"]))
            row = self.executor.generate(**args)
            self.assertEqual(preflight["resolved_output"], row.request["resolved_output"])
            self.assertEqual(preflight["reference_mapping"], row.request["reference_mapping"])
            self.assertFalse(preflight["gpu_verified"])

    def test_authoritative_rejection_is_not_left_queued(self):
        def handler(request):
            return httpx.Response(200, json={"ready": True}) if request.url.path == "/health" else httpx.Response(422)
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            self.executor.client = client
            row = self.executor.generate(**self.kw, route="h3-singularity")
            self.assertEqual(row.status, "failed")
            self.assertEqual(row.error["code"], "runtime_rejected")

    def test_lost_submission_response_recovers_original_runtime_identity(self):
        posts = []
        identity = "singularity_" + "c" * 32

        def handler(request):
            if request.url.path == "/health":
                return httpx.Response(200, json={"ready": True})
            if request.method == "POST":
                posts.append(json.loads(request.content))
                raise httpx.ReadTimeout("response lost", request=request)
            if "/by-idempotency/" in request.url.path:
                self.assertTrue(request.url.path.endswith(posts[0]["idempotency_key"]))
                return httpx.Response(200, json={"id": identity})
            self.assertTrue(request.url.path.endswith(identity))
            return httpx.Response(200, json={"status": "in_progress", "stage": "generating"})

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            self.executor.client = client
            row = self.executor.generate(**self.kw, route="h3-singularity")
            self.assertEqual(row.error["code"], "submission_unconfirmed")
            self.assertEqual(row.status, "queued")
            recovered = self.executor.status(row.video_task_id)
            self.assertEqual(recovered.runtime_task_id, identity)
            self.assertEqual(recovered.status, "running")
            replay = self.executor.generate(**self.kw, route="h3-singularity")
            self.assertEqual(replay.video_task_id, row.video_task_id)
            self.assertEqual(len(posts), 1)

    def test_restart_recovers_a_crash_between_dispatch_and_response_persistence(self):
        from app.dashboard import Dashboard
        identity = "singularity_" + "d" * 32
        # A process crash can leave the durable reservation with no error or
        # runtime identity even though the GPU service accepted the request.
        request, digest = self.executor._normalize(self.kw["project_id"], self.kw["model"], self.kw["prompt"],
            self.kw["duration_seconds"], self.kw["aspect_ratio"], self.kw["references"], "h3-singularity")
        record = self.tasks.create(project_id=self.kw["project_id"], idempotency_key=self.kw["idempotency_key"],
            input_digest=digest, request=request, runtime_task_id="", status="queued")
        def handler(request):
            self.assertEqual(request.method, "GET")
            if request.url.path == "/health": return httpx.Response(200, json={"ready":True})
            if "/by-idempotency/" in request.url.path: return httpx.Response(200, json={"id":identity})
            return httpx.Response(200, json={"status":"in_progress","stage":"generating"})
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            self.executor.client = client
            dashboard = Dashboard(self.executor)
            dashboard.recover_incomplete_submissions()
            self.assertEqual(self.tasks.get(record.video_task_id).status, "queued")
            dashboard.sync()
            recovered = self.tasks.get(record.video_task_id)
            self.assertEqual(recovered.runtime_task_id, identity)
            self.assertEqual(recovered.status, "running")


if __name__ == "__main__":
    unittest.main()
