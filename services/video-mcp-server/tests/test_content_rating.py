from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from starlette.testclient import TestClient

import app.server as server_mod
from app.artifacts import ArtifactStore
from app.content_rating import ContentRatingError, ContentRatingStore
from app.tasks import TaskNotFound, TaskStore


class TestVideoContentRatingStore(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "coordination.sqlite3"
        self.store = ContentRatingStore(self.db_path, domain="video")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_add_only_and_human_override_lifecycle(self) -> None:
        # 1. Rate general
        r1 = self.store.rate_content(
            version_id="art_11111111111111111111111111111111",
            project_id="proj-1",
            rating="general",
            reason="clean landscape shot",
            rubric_version="rubric v1.0",
        )
        self.assertEqual(r1["rating"], "general")
        self.assertFalse(r1["ignored"])

        # 2. Upgrade general -> nsfw
        r2 = self.store.rate_content(
            version_id="art_11111111111111111111111111111111",
            project_id="proj-1",
            rating="nsfw",
            reason="explicit_nudity",
            rubric_version="rubric v1.0",
        )
        self.assertEqual(r2["rating"], "nsfw")
        self.assertFalse(r2["ignored"])
        self.assertTrue(self.store.is_nsfw("art_11111111111111111111111111111111"))

        # 3. Attempt to downgrade nsfw -> general via MCP (must be ignored)
        r3 = self.store.rate_content(
            version_id="art_11111111111111111111111111111111",
            project_id="proj-1",
            rating="general",
            reason="attempt to clear via mcp",
            rubric_version="rubric v1.0",
        )
        self.assertEqual(r3["rating"], "nsfw")
        self.assertTrue(r3["ignored"])
        self.assertTrue(self.store.is_nsfw("art_11111111111111111111111111111111"))

        # 4. Clear by human
        r4 = self.store.clear_rating(
            version_id="art_11111111111111111111111111111111",
            project_id="proj-1",
            actor_user_id="user-admin",
            note="false positive mannequin",
        )
        self.assertEqual(r4["rating"], "general")
        self.assertEqual(r4["source"], "human")
        self.assertTrue(r4["human_cleared"])
        self.assertFalse(self.store.is_nsfw("art_11111111111111111111111111111111"))

        # 4b. AI attempts to re-rate nsfw after human clear -> ignored!
        r4b = self.store.rate_content(
            version_id="art_11111111111111111111111111111111",
            project_id="proj-1",
            rating="nsfw",
            reason="ai re-rate attempt",
            rubric_version="rubric v1.0",
        )
        self.assertEqual(r4b["rating"], "general")
        self.assertTrue(r4b["ignored"])
        self.assertFalse(self.store.is_nsfw("art_11111111111111111111111111111111"))

        # 5. Mark by human
        r5 = self.store.mark_rating(
            version_id="art_11111111111111111111111111111111",
            project_id="proj-1",
            actor_user_id="user-admin",
            reason="manual 18+ mark",
        )
        self.assertEqual(r5["rating"], "nsfw")
        self.assertEqual(r5["source"], "human")
        self.assertTrue(self.store.is_nsfw("art_11111111111111111111111111111111"))

        events = self.store.list_events("art_11111111111111111111111111111111")
        self.assertEqual(
            [e["event_type"] for e in events],
            ["rate", "rate", "rate", "clear", "ignored_after_clear", "rate"],
        )

    def test_validation_rejects_invalid_rating_or_reason(self) -> None:
        with self.assertRaises(ContentRatingError):
            self.store.rate_content(
                version_id="art_11111111111111111111111111111111",
                project_id="proj-1",
                rating="r18",
                reason="bad enum",
            )
        with self.assertRaises(ContentRatingError):
            self.store.rate_content(
                version_id="art_11111111111111111111111111111111",
                project_id="proj-1",
                rating="nsfw",
                reason="   ",
            )


class TestVideoMcpAndHttpContentRating(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        os.environ["VIDEO_MCP_BEARER_TOKEN"] = "test-secret-token"
        self.orig_tasks = server_mod.tasks
        self.orig_artifacts = server_mod.artifacts
        self.orig_ratings = server_mod.ratings

        self.tasks = TaskStore(root)
        self.artifacts = ArtifactStore(root)
        self.ratings = ContentRatingStore(self.tasks.db_path, domain="video")
        self.tasks.ensure_ready()
        self.artifacts.ensure_ready()
        self.ratings.ensure_ready()

        server_mod.tasks = self.tasks
        server_mod.artifacts = self.artifacts
        server_mod.ratings = self.ratings
        server_mod.app.state.tasks = self.tasks
        server_mod.app.state.artifacts = self.artifacts
        server_mod.app.state.ratings = self.ratings

        # Seed 1 general video task and 1 NSFW candidate video task
        art_gen = self.artifacts.create_from_chunks(
            project_id="proj-a",
            operation="video.result",
            filename="general.mp4",
            media_type="video/mp4",
            chunks=[b"\x00\x00\x00\x18ftypmp42generalvideo"],
        )
        t_gen = self.tasks.create(
            project_id="proj-a",
            idempotency_key="idem-gen",
            input_digest="sha256:gen",
            request={"model": "minimax-h3-ref2va", "prompt": "sunny mountain"},
            runtime_task_id="rt-gen",
            status="succeeded",
            runtime_route="h3",
        )
        self.task_gen = self.tasks.update(
            t_gen,
            status="succeeded",
            artifact_id=art_gen.artifact_id,
            media={"duration_ms": 5000, "width": 768, "height": 1344},
        )
        self.art_gen = art_gen

        art_nsfw = self.artifacts.create_from_chunks(
            project_id="proj-a",
            operation="video.result",
            filename="nsfw.mp4",
            media_type="video/mp4",
            chunks=[b"\x00\x00\x00\x18ftypmp42nsfwvideo"],
        )
        t_nsfw = self.tasks.create(
            project_id="proj-a",
            idempotency_key="idem-nsfw",
            input_digest="sha256:nsfw",
            request={"model": "minimax-h3-ref2va", "prompt": "adult sequence test"},
            runtime_task_id="rt-nsfw",
            status="succeeded",
            runtime_route="h3",
        )
        self.task_nsfw = self.tasks.update(
            t_nsfw,
            status="succeeded",
            artifact_id=art_nsfw.artifact_id,
            media={"duration_ms": 5000, "width": 768, "height": 1344},
        )
        self.art_nsfw = art_nsfw

    def tearDown(self) -> None:
        server_mod.tasks = self.orig_tasks
        server_mod.artifacts = self.orig_artifacts
        server_mod.ratings = self.orig_ratings
        server_mod.app.state.tasks = self.orig_tasks
        server_mod.app.state.artifacts = self.orig_artifacts
        server_mod.app.state.ratings = self.orig_ratings
        self._tmp.cleanup()

    def test_mcp_rate_content_hides_from_mcp_and_http_supports_local_switch_and_clear(self) -> None:
        # Initially both tasks are visible in MCP list
        initial_list = json.loads(server_mod.video_list(project_id="proj-a").content[0].text)
        self.assertEqual(initial_list["total"], 2)

        # Rate nsfw video artifact via MCP tool
        rate_res = json.loads(
            server_mod.video_rate_content(
                version_id=self.art_nsfw.artifact_id,
                rating="nsfw",
                reason="explicit_nudity",
                rubric_version="rubric v1.0",
            ).content[0].text
        )
        self.assertEqual(rate_res["rating"], "nsfw")
        self.assertFalse(rate_res["ignored"])

        # Attempting to downgrade via MCP general must not clear nsfw
        downgrade_res = json.loads(
            server_mod.video_rate_content(
                version_id=self.art_nsfw.artifact_id,
                rating="general",
                reason="try downgrade",
                rubric_version="rubric v1.0",
            ).content[0].text
        )
        self.assertEqual(downgrade_res["rating"], "nsfw")
        self.assertTrue(downgrade_res["ignored"])

        # MCP list now hides the nsfw task and decrements total (no placeholder)
        after_list = json.loads(server_mod.video_list(project_id="proj-a").content[0].text)
        self.assertEqual(after_list["total"], 1)
        self.assertEqual([t["task_id"] for t in after_list["tasks"]], [self.task_gen.video_task_id])

        # MCP status and result raise TaskNotFound for the nsfw task
        with self.assertRaises(TaskNotFound):
            server_mod.video_status(self.task_nsfw.video_task_id)
        with self.assertRaises(TaskNotFound):
            server_mod.video_result(self.task_nsfw.video_task_id)

        # Unknown or cross-project version_id raises TaskNotFound
        with self.assertRaises(TaskNotFound):
            server_mod.video_rate_content(
                version_id="art_00000000000000000000000000000000",
                rating="general",
                reason="missing",
            )
        with self.assertRaises(TaskNotFound):
            server_mod.video_rate_content(
                version_id=self.art_nsfw.artifact_id,
                rating="general",
                reason="wrong project",
                project_id="proj-other",
            )

        headers = {"Authorization": "Bearer test-secret-token"}
        client = TestClient(server_mod.app)

        # HTTP /api/tasks and /api/dashboard without X-VF-Include-NSFW hide the nsfw task
        resp_default = client.get("/api/tasks?project_id=proj-a", headers=headers)
        self.assertEqual(resp_default.status_code, 200)
        self.assertEqual(resp_default.json()["total"], 1)

        resp_dash_default = client.get("/api/dashboard?project_id=proj-a", headers=headers)
        self.assertEqual(resp_dash_default.status_code, 200)
        self.assertEqual(resp_dash_default.json()["total"], 1)

        resp_stats_default = client.get("/api/tasks/stats?project_id=proj-a", headers=headers)
        self.assertEqual(resp_stats_default.status_code, 200)
        self.assertEqual(resp_stats_default.json()["total"], 1)

        # HTTP /api/tasks/{id} without X-VF-Include-NSFW returns 404
        resp_detail_hidden = client.get(f"/api/tasks/{self.task_nsfw.video_task_id}", headers=headers)
        self.assertEqual(resp_detail_hidden.status_code, 404)

        # HTTP /artifacts/{id}/content without X-VF-Include-NSFW returns 404
        resp_art_hidden = client.get(f"/artifacts/{self.art_nsfw.artifact_id}/content")
        self.assertEqual(resp_art_hidden.status_code, 404)

        # HTTP /api/tasks with X-VF-Include-NSFW: 1 includes the nsfw task with UI fields
        resp_inc = client.get(
            "/api/tasks?project_id=proj-a",
            headers={**headers, "X-VF-Include-NSFW": "1"},
        )
        self.assertEqual(resp_inc.status_code, 200)
        inc_data = resp_inc.json()
        self.assertEqual(inc_data["total"], 2)
        nsfw_item = next(t for t in inc_data["tasks"] if t["task_id"] == self.task_nsfw.video_task_id)
        self.assertEqual(nsfw_item["content_rating"]["rating"], "nsfw")
        self.assertTrue(nsfw_item["can_clear_rating"])
        self.assertFalse(nsfw_item["can_mark_rating"])

        # HTTP /artifacts/{id}/content with X-VF-Include-NSFW: 1 returns 200
        resp_art_shown = client.get(
            f"/artifacts/{self.art_nsfw.artifact_id}/content",
            headers={"X-VF-Include-NSFW": "1"},
        )
        self.assertEqual(resp_art_shown.status_code, 200)

        # Internal human clear endpoint restores general visibility
        resp_clear = client.post(
            f"/internal/content-ratings/{self.art_nsfw.artifact_id}/clear",
            json={"note": "误标撤销"},
            headers={**headers, "X-User-Id": "user-mengsk"},
        )
        self.assertEqual(resp_clear.status_code, 200)
        self.assertEqual(resp_clear.json()["rating"], "general")
        self.assertEqual(resp_clear.json()["source"], "human")
        self.assertTrue(resp_clear.json()["human_cleared"])

        # Now MCP list and status see both tasks again
        restored_list = json.loads(server_mod.video_list(project_id="proj-a").content[0].text)
        self.assertEqual(restored_list["total"], 2)
        status_res = json.loads(server_mod.video_status(self.task_nsfw.video_task_id).content[0].text)
        self.assertEqual(status_res["status"], "succeeded")

        # Internal human mark endpoint marks nsfw and hides from MCP again
        resp_mark = client.post(
            f"/video/internal/content-ratings/{self.art_nsfw.artifact_id}/mark",
            json={"reason": "人工补标 18+"},
            headers={**headers, "X-User-Id": "user-mengsk"},
        )
        self.assertEqual(resp_mark.status_code, 200)
        self.assertEqual(resp_mark.json()["rating"], "nsfw")
        self.assertEqual(resp_mark.json()["source"], "human")
        with self.assertRaises(TaskNotFound):
            server_mod.video_status(self.task_nsfw.video_task_id)


if __name__ == "__main__":
    unittest.main()
