import json
import os
import unittest
from unittest.mock import patch

import test_sol_route
from app.executor import VideoExecutor


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

    def test_dual_quality_profile_is_forwarded_and_part_of_idempotency(self):
        row = self.executor.generate(**self.kw, route="h3-singularity", quality_profile="du-1")
        self.assertEqual(row.request["quality_profile"], "du-1")
        body = json.loads(self.calls[-1].content)
        self.assertEqual(body["quality_profile"], "du-1")
        with self.assertRaises(test_sol_route.TaskConflict):
            self.executor.generate(**self.kw, route="h3-singularity", quality_profile="du-2")

    def test_round_five_du3_profile_is_forwarded(self):
        row = self.executor.generate(**self.kw, route="h3-singularity", quality_profile="du-3")
        self.assertEqual(row.request["quality_profile"], "du-3")
        body = json.loads(self.calls[-1].content)
        self.assertEqual(body["quality_profile"], "du-3")

    def test_round_five_hr_refine_profile_is_forwarded(self):
        row = self.executor.generate(**self.kw, route="h3-singularity", quality_profile="hr-refine-tile-v1")
        self.assertEqual(row.request["quality_profile"], "hr-refine-tile-v1")
        body = json.loads(self.calls[-1].content)
        self.assertEqual(body["quality_profile"], "hr-refine-tile-v1")

    def test_round_five_global_refine_profile_is_forwarded(self):
        row = self.executor.generate(**self.kw, route="h3-singularity", quality_profile="hr-refine-global-v2")
        self.assertEqual(row.request["quality_profile"], "hr-refine-global-v2")
        body = json.loads(self.calls[-1].content)
        self.assertEqual(body["quality_profile"], "hr-refine-global-v2")

    def test_dual_quality_profile_is_restricted_to_singularity(self):
        with self.assertRaisesRegex(ValueError, "require h3-singularity"):
            self.executor.generate(**self.kw, route="h3", quality_profile="du-1")
        self.assertFalse(self.calls)


if __name__ == "__main__":
    unittest.main()
