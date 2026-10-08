import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import httpx

from app.artifacts import ArtifactStore
from app.executor import ExecutionError, VideoExecutor
from app.singularity import output_spec
from app.tasks import TaskStore


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class SingularityResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.media_temp = tempfile.TemporaryDirectory()
        path = Path(cls.media_temp.name) / "sample.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=size=1344x768:rate=24",
            "-f", "lavfi", "-i", "anullsrc=r=32000:cl=stereo", "-frames:v", "362", "-t", "15.083333",
            "-c:v", "libx264", "-preset", "ultrafast", "-threads", "2", "-c:a", "aac", str(path)], check=True)
        cls.payload = path.read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.media_temp.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.assets = ArtifactStore(self.root)
        self.tasks = TaskStore(self.root)
        self.env = patch.dict(os.environ, {"H3_RUNTIME_ROUTE": "h3-singularity", "H3_RUNTIME_ROUTES": "",
            "H3_RUNTIME_URL": "http://runtime.example", "H3_SINGULARITY_RUNTIME_TOKEN": "test-token"})
        self.env.start()
        self.http = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=self.payload)))
        self.executor = VideoExecutor(self.assets, self.tasks, self.http)

    def tearDown(self):
        self.http.close()
        self.env.stop()
        self.temp.cleanup()

    def record(self, changes=None):
        resolved = {**output_spec("16:9", 15), **(changes or {})}
        row = self.tasks.create(project_id="project-test", idempotency_key="sample", input_digest="test-digest",
            request={"route": "h3-singularity", "duration_seconds": 15, "resolved_output": resolved},
            runtime_task_id="singularity_" + "a" * 32, status="succeeded")
        return self.tasks.update(row, service="h3-singularity",
            runtime_metrics={"result_sha256": hashlib.sha256(self.payload).hexdigest()})

    def test_native_landscape_sample_passes_complete_decode_and_audio_validation(self):
        row = self.executor.result(self.record().video_task_id)
        self.assertEqual(row.media["frames"], 362)
        self.assertEqual((row.media["width"], row.media["height"]), (1344, 768))
        self.assertAlmostEqual(row.media["video_duration_seconds"], 362 / 24, places=4)
        self.assertTrue(row.media["complete_decode_verified"])
        self.assertEqual(row.media["sha256"], hashlib.sha256(self.payload).hexdigest())

    def test_wrong_geometry_or_frame_count_cannot_be_archived(self):
        row = self.record({"frames": 345})
        with self.assertRaisesRegex(ExecutionError, "decoded frame count"):
            self.executor.result(row.video_task_id)
        self.assertFalse(list(self.assets.artifacts_root.glob("art_*")))

    def test_result_bytes_must_match_runtime_hash(self):
        row = self.record()
        self.tasks.update(row, runtime_metrics={"result_sha256": "0" * 64})
        with self.assertRaisesRegex(ExecutionError, "SHA-256"):
            self.executor.result(row.video_task_id)
        self.assertFalse(list(self.assets.artifacts_root.glob("art_*")))


if __name__ == "__main__":
    unittest.main()
