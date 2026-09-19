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
        self.assertEqual(body["idempotency_key"], row.video_task_id)


if __name__ == "__main__":
    unittest.main()
