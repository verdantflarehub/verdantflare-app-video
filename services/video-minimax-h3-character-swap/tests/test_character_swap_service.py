import subprocess
import tempfile
import unittest
from pathlib import Path

from h3_character_swap.contract import plan_swap_windows
from h3_character_swap.errors import RuntimeErrorCode


class MediaTest(unittest.TestCase):
    def test_preview_audio_copy_and_frame_rate(self):
        from h3_character_swap.character_swap import (
            _normalize, probe_audio_packets_sha256, probe_mp4_summary, render_side_by_side_preview,
        )
        for fps, has_audio in ((24, True), (30, False)):
            with self.subTest(fps=fps), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                source = directory / "source.mp4"
                args = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        f"testsrc2=size=192x128:rate={fps}:duration=1"]
                if has_audio:
                    args.extend(["-f", "lavfi", "-i", "sine=duration=1", "-c:a", "aac"])
                subprocess.run([*args, "-c:v", "libx264", str(source)], check=True)
                summary = probe_mp4_summary(source)
                normalized, media = _normalize(source, directory, summary)
                self.assertEqual(media["frames"], 24)
                preview = directory / "preview.mp4"
                render_side_by_side_preview(normalized, normalized, preview, 192, 128)
                checked = probe_mp4_summary(preview)
                self.assertEqual((checked["frames"], checked["audio"]), (24, has_audio))
                if has_audio:
                    self.assertEqual(probe_audio_packets_sha256(source), probe_audio_packets_sha256(preview))


class WindowTest(unittest.TestCase):
    def test_all_admitted_lengths_round_trip(self):
        for length in range(24, 363):
            with self.subTest(length=length):
                source = list(range(length))
                result = []
                for window in plan_swap_windows(length):
                    padded = window.pad(source)
                    self.assertIn(len(padded), (107, 124))
                    self.assertEqual(padded[-1], source[window.end - 1])
                    result.extend(window.trim(padded))
                self.assertEqual(result, source)

    def test_full_length(self):
        windows = plan_swap_windows(362)
        self.assertEqual([(w.start, w.end) for w in windows],
                         [(0, 124), (106, 230), (212, 336), (318, 362)])

    def test_bounds(self):
        for length in (23, 363, 24.0, True):
            with self.assertRaisesRegex(RuntimeErrorCode, "duration_out_of_range"):
                plan_swap_windows(length)


class InputTest(unittest.TestCase):
    def test_http_readiness_and_auth(self):
        from fastapi.testclient import TestClient
        from contextlib import closing
        from h3_character_swap.api import create_app
        from h3_character_swap.queue import Queue
        with tempfile.TemporaryDirectory() as temp, closing(Queue(temp, acquire_lock=False)) as queue:
            with TestClient(create_app(queue, token="test", health_checker=lambda **kw: {"ready": False})) as client:
                response = client.get("/health")
                self.assertEqual(response.status_code, 503)
                self.assertIs(response.json()["ready"], False)
                self.assertEqual(client.post("/v1/character-swap", json={}).status_code, 401)
                for body, headers, status, code in (
                    (b"{", {}, 422, "invalid_json"),
                    (b"[]", {}, 422, "json_object_required"),
                    (b"{}", {"content-length": "invalid"}, 400, "invalid_content_length"),
                    (b"{}", {"content-length": str(2**30)}, 413, "body_too_large"),
                ):
                    with self.subTest(code=code):
                        response = client.post("/v1/character-swap", content=body,
                                               headers={"authorization": "Bearer test", **headers})
                        self.assertEqual((response.status_code, response.json()["detail"]["code"]), (status, code))

    def test_artifact_pair_checks_project_and_digest(self):
        import hashlib
        import json
        import tempfile
        from pathlib import Path
        from h3_character_swap.character_swap import resolve_character_image_artifact_path, resolve_source_artifact_path
        from h3_character_swap.contract import validate_swap_request

        request = {"project_id": "dev", "idempotency_key": "video_task_" + "a" * 32,
                   "prompt": "Replace the Man in <Video 1> with the Woman in <Picture 1>."}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for role, kind, mime, char in (("source", "video", "video/mp4", "b"),
                                           ("character_image", "image", "image/png", "c")):
                key = "art_" + char * 32
                directory = root / "artifacts" / key
                directory.mkdir(parents=True)
                blob = kind.encode()
                digest = hashlib.sha256(blob).hexdigest()
                (directory / "media").write_bytes(blob)
                metadata = dict(project_id="dev", artifact_id=key, kind=kind, media_type=mime,
                                filename="media", size=len(blob), sha256=digest)
                (directory / "metadata.json").write_text(json.dumps(metadata))
                request[role + "_artifact_id"] = key
                request[role + "_sha256"] = digest
            validated = validate_swap_request(request)
            for role, resolve in (("source", resolve_source_artifact_path),
                                  ("character_image", resolve_character_image_artifact_path)):
                self.assertTrue(resolve(validated, root).is_file())
                with self.assertRaisesRegex(RuntimeErrorCode, role + "_not_found"):
                    resolve({**validated, "project_id": "other"}, root)
                with self.assertRaisesRegex(RuntimeErrorCode, role + "_integrity_failed"):
                    resolve({**validated, role + "_sha256": "1" * 64}, root)
                path = resolve(validated, root)
                path.write_bytes(b"x" * path.stat().st_size)
                with self.assertRaisesRegex(RuntimeErrorCode, role + "_integrity_failed"):
                    resolve(validated, root)
            for override, code in (({"seed_mode": "random"}, "unsupported_seed_mode"),
                                   ({"prompt": "use <Picture 2>"}, "multi_character_single_pass_unsupported"),
                                   ({"lora_strength": True}, "invalid_lora_strength"),
                                   ({"character_lora_version": "v2"}, "unsupported_character_lora_version")):
                with self.assertRaisesRegex(RuntimeErrorCode, code):
                    validate_swap_request({**request, **override})

    def test_vram_threshold_is_hard_floor(self):
        from unittest.mock import patch
        from h3_character_swap.worker import check_free_vram
        gpu = "GPU-99999999-8888-7777-6666-555555555555"
        with patch("h3_character_swap.worker._assigned_gpu_from_environ", return_value=gpu):
            with patch("subprocess.check_output", return_value=gpu + ", 28671"):
                with self.assertRaisesRegex(RuntimeErrorCode, "gpu_memory_occupancy_changed"):
                    check_free_vram(1)
            with patch("subprocess.check_output", return_value=gpu + ", 28672"):
                self.assertEqual(check_free_vram(28672), (gpu, 28672))
