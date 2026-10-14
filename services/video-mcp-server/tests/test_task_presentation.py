"""Presentation follows the actual model/route/mode, never the provider alone."""
import unittest
from types import SimpleNamespace

from app.dashboard import public_task, task_presentation
from app.tasks import TaskRecord


class TaskPresentationTest(unittest.TestCase):
    def present(self, service, model=None, mode=None, route=None):
        return task_presentation(SimpleNamespace(service=service, runtime_route=route,
                                                request={"model": model, "mode": mode}))

    def test_higgsfield_models_and_modes_are_independent_of_channel(self):
        for model, name, mode, code in [
            ("wan3_0", "Wan 3.0", "reference_to_video", "R2V"),
            ("grok_video_v15_lite", "Grok Imagine 1.5 Lite", "image_to_video", "I2V"),
            ("seedance_2_5", "Seedance 2.5", "text_to_video", "T2V"),
            ("seedance_2_5_draft", "Seedance 2.5 Draft", "reference_to_video", "R2V"),
        ]:
            p = self.present("higgsfield", model, mode)
            self.assertTrue(p["is_generation"])
            self.assertEqual((p["model_label"], p["channel_label"], p["generation_code"]),
                             (name, "Higgsfield", code))
            self.assertEqual(p["channel_icon"], "higgsfield")

    def test_h3_local_and_remote_channels_preserve_the_difference(self):
        for route in ("h3", "h3-sol", "h3-vdn", "h3-singularity"):
            p = self.present(route)
            self.assertEqual((p["model_label"], p["channel_label"], p["generation_code"]),
                             ("MiniMax H3", "本地", "R2V"))
        p = self.present("fal", "minimax-h3-ref2va", route="fal")
        self.assertEqual(p["channel_label"], "fal.ai")
        self.assertEqual(p["generation_code"], "R2V")
        self.assertEqual(self.present("h3-sol", route="minimax-h3-sol-ref2va")["channel_label"], "本地")

    def test_frame_mode_requires_explicit_mode_and_unknown_values_do_not_guess(self):
        p = self.present("higgsfield", "future-model", "first_last_frame_to_video")
        self.assertEqual((p["generation_code"], p["generation_label"]), ("F2V", "首尾帧视频"))
        for mode in (None, "future-mode"):
            p = self.present("higgsfield", mode=mode)
            self.assertEqual(p["model_label"], "未记录模型")
            self.assertIsNone(p["generation_code"])
        self.assertIsNone(self.present("h3", mode="unknown")["generation_code"])

    def test_request_route_wins_over_internal_backend(self):
        record = SimpleNamespace(service="h3", runtime_route="h3-singularity",
                                 request={"model": "minimax-h3-ref2va", "route": "fal"})
        p = task_presentation(record)
        self.assertEqual((p["model_label"], p["channel_label"]), ("MiniMax H3", "fal.ai"))
        self.assertIsNone(p["channel_icon"])
        record.request["route"] = "future-route"
        self.assertEqual(task_presentation(record)["channel_label"], "future-route")

    def test_processing_is_not_generation_even_if_request_has_a_mode(self):
        for service in ("depth", "sr", "interpolate", "h3-latent-upscale"):
            p = self.present(service, mode="reference_to_video")
            self.assertFalse(p["is_generation"])
            self.assertIsNone(p["generation_code"])

    def test_public_projection_preserves_model_mode_and_prompt(self):
        record = TaskRecord(video_task_id="video_task_" + "a" * 32, project_id="demo",
                            idempotency_key="test", input_digest="test", service="higgsfield",
                            runtime_task_id="", status="succeeded",
                            created_at="2026-10-14T09:00:00+00:00", updated_at="2026-10-14T09:15:00+00:00",
                            request={"model": "wan3_0", "mode": "reference_to_video",
                                     "route": "higgsfield", "prompt": "Dance at night"})
        value = public_task(record)
        self.assertEqual(value["presentation"]["generation_code"], "R2V")
        self.assertEqual((value["model"], value["mode"], value["prompt"]),
                         ("wan3_0", "reference_to_video", "Dance at night"))
        self.assertNotIn("presentation", record.request)


if __name__ == "__main__":
    unittest.main()
