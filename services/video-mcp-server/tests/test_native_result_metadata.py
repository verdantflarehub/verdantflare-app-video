from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from app import server


class NativeResultMetadataTests(unittest.TestCase):
    def test_completed_result_uses_fixed_artifact_metadata_without_changing_identity(self):
        task_id, artifact_id = "video_task_" + "a" * 32, "art_" + "b" * 32
        record = SimpleNamespace(video_task_id=task_id, artifact_id=artifact_id,
            project_id="project-test", status="succeeded", service="google", runtime_route="google",
            request={"model": "gemini-omni-1.1-flash"}, runtime_version="0.1.27", input_digest="input",
            media={"sha256": "stale", "size": 1, "duration_seconds": 4}, delivery=None)
        tasks, artifacts = MagicMock(), MagicMock()
        tasks.get.return_value = record
        artifacts.get.return_value = SimpleNamespace(artifact_id=artifact_id, sha256="c" * 64, size=9000)
        artifacts.download_path.return_value = f"/artifacts/{artifact_id}/content"
        with patch.object(server, "_active_tasks", return_value=tasks), \
                patch.object(server, "_active_artifacts", return_value=artifacts), \
                patch.object(server, "_is_task_nsfw", return_value=False), \
                patch.object(server.titles, "project", return_value={}), \
                patch.object(server.executor, "result") as resume:
            result = server.video_result(video_task_id=task_id).structured_content
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual((result["video_task_id"], result["artifact_id"], result["project_id"]),
                         (task_id, artifact_id, "project-test"))
        self.assertEqual(result["media"], {"sha256": "c" * 64, "size": 9000, "duration_seconds": 4})
        self.assertEqual(result["download_path"], f"/artifacts/{artifact_id}/content")
        self.assertNotIn("content_ref", result)
        artifacts.get.assert_called_once_with(artifact_id, "project-test")
        resume.assert_not_called()

