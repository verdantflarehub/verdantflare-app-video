import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from h3_singularity.api import _download_references
from h3_singularity.contract import ASPECT_RATIOS, resolve_geometry, resolve_timing, validate_request
from h3_singularity.errors import RuntimeErrorCode


BASE = "http://video-mcp-server:8000"


def payload():
    return {"idempotency_key": "video_task_" + "a" * 32, "model": "MiniMaxAI/MiniMax-H3", "task": "ref2va",
            "prompt": "subject_definitions: ...", "seconds": 15, "target": {"aspect_ratio": "16:9"},
            "conditions": [{"type": "image", "role": "reference", "uri": BASE + "/runtime-artifacts/art_" + "b" * 32 + "/content",
                            "size": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}]}


class ContractTests(unittest.TestCase):
    def test_six_canvases_and_adaptive_follow_upstream_grid(self):
        expected = {"21:9": (1536, 672), "16:9": (1344, 768), "4:3": (1024, 768),
                    "1:1": (768, 768), "3:4": (768, 1024), "9:16": (768, 1344)}
        for ratio, size in expected.items():
            with self.subTest(ratio=ratio):
                result = resolve_geometry(ratio)
                self.assertEqual((result["width"], result["height"]), size)
                self.assertEqual(result["requested_aspect_ratio"], ratio)
        self.assertEqual(resolve_geometry("16:9")["actual_aspect_ratio"], "7:4")
        result = resolve_geometry("adaptive", (1920, 1080))
        self.assertEqual((result["width"], result["height"]), (1344, 768))
        with self.assertRaises(RuntimeErrorCode):
            resolve_geometry("adaptive")

    def test_every_duration_rounds_up_to_native_bucket(self):
        for seconds in range(4, 16):
            result = resolve_timing(seconds)
            self.assertEqual(result["frames"] % 17, 5)
            self.assertGreaterEqual(result["video_duration_seconds"], seconds)
            self.assertLess(result["video_duration_seconds"] - seconds, 17 / 24)
        self.assertEqual(resolve_timing(15)["frames"], 362)
        for invalid in (True, 3, 16, 15.0, "15"):
            with self.assertRaises(RuntimeErrorCode):
                resolve_timing(invalid)

    def test_audio_only_is_valid_with_explicit_canvas(self):
        request = payload()
        request["conditions"][0]["type"] = "audio"
        validate_request(request, BASE)
        request["target"]["aspect_ratio"] = "adaptive"
        with self.assertRaisesRegex(RuntimeErrorCode, "visual_reference"):
            validate_request(request, BASE)

    def test_invalid_counts_fields_and_profile_rejected(self):
        for mutation in (lambda r: r.update(unknown=True), lambda r: r.update(quality_profile="du-3"),
                         lambda r: r.update(num_outputs_per_prompt=True), lambda r: r.update(flow_shift=float("nan")),
                         lambda r: r.update(audio_flow_shift=6),
                         lambda r: r.update(seconds=True), lambda r: r.update(conditions=r["conditions"] * 10),
                         lambda r: r["conditions"][0].pop("sha256"),
                         lambda r: r["target"].update(width=768, height=1344)):
            request = payload()
            mutation(request)
            with self.assertRaises(RuntimeErrorCode):
                validate_request(request, BASE)
        request = payload()
        request["conditions"] *= 9
        request["conditions"] += [{**request["conditions"][0], "type": "video"}] * 3
        validate_request(request, BASE)
        request["conditions"] += [{**request["conditions"][0], "type": "audio"}]
        with self.assertRaises(RuntimeErrorCode):
            validate_request(request, BASE)

    def test_arbitrary_sources_query_and_traversal_are_rejected(self):
        for uri in ("http://169.254.169.254/latest", BASE + "/health", payload()["conditions"][0]["uri"] + "?token=x",
                    BASE + "/runtime-artifacts/../secret", "http://video-mcp-server:8000@evil.invalid/content"):
            request = payload()
            request["conditions"][0]["uri"] = uri
            with self.assertRaisesRegex(RuntimeErrorCode, "source_not_allowed"):
                validate_request(request, BASE)

    def test_download_verifies_bytes_and_does_not_follow_redirects(self):
        for response, expected_error in ((httpx.Response(200, content=b"abc"), None),
                                        (httpx.Response(200, content=b"bad"), "hash_mismatch"),
                                        (httpx.Response(200, content=b"ab"), "size_mismatch"),
                                        (httpx.Response(302, headers={"location": "http://evil.invalid"}), "download_failed")):
            with self.subTest(error=expected_error), tempfile.TemporaryDirectory() as directory:
                client = httpx.Client(transport=httpx.MockTransport(lambda _: response), follow_redirects=False)
                with patch("h3_singularity.api.httpx.Client", return_value=client):
                    if expected_error:
                        with self.assertRaisesRegex(RuntimeErrorCode, expected_error):
                            _download_references(Path(directory), payload()["conditions"])
                    else:
                        files = _download_references(Path(directory), payload()["conditions"])
                        self.assertEqual(files["images"][0].read_bytes(), b"abc")


if __name__ == "__main__":
    unittest.main()
