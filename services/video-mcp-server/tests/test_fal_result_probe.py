import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from app.providers.fal import FalAdapter, FalError


class FalResultProbeTests(unittest.TestCase):
    def test_probe_passes_readable_payload_path_and_cleans_it(self):
        payload = b"controlled downloaded media bytes"
        paths = []

        def probe(command, **kwargs):
            path = Path(os.fsdecode(command[-1]))
            paths.append(path)
            self.assertEqual(path.read_bytes(), payload)
            return subprocess.CompletedProcess(command, 0, json.dumps({
                "format": {"duration": "0.2"},
                "streams": [{"codec_type": "video", "codec_name": "h264",
                             "width": 64, "height": 64, "r_frame_rate": "25/1"}],
            }))

        with mock.patch("app.providers.fal.subprocess.run", side_effect=probe):
            media = FalAdapter._probe(payload)
        self.assertEqual(media["duration_ms"], 200)
        self.assertEqual(media["frame_rate"], 25)
        self.assertEqual(len(paths), 1)
        self.assertFalse(paths[0].exists())

    @unittest.skipUnless(shutil.which("ffprobe") and shutil.which("ffmpeg"),
                         "requires dev image media tools")
    def test_real_video_and_audio_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.mp4"
            subprocess.run([
                "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                "color=c=black:s=64x64:r=25:d=0.2", "-f", "lavfi", "-i",
                "anullsrc=r=48000:cl=mono", "-t", "0.2", "-c:v", "mpeg4",
                "-c:a", "aac", str(path),
            ], check=True, capture_output=True, timeout=30)
            media = FalAdapter._probe(path.read_bytes())
        self.assertEqual((media["width"], media["height"]), (64, 64))
        self.assertEqual(media["frame_rate"], 25)
        self.assertEqual(media["audio_codec"], "aac")
        self.assertEqual(media["audio_sample_rate"], 48000)
        self.assertAlmostEqual(media["duration_ms"], 200, delta=30)

    @unittest.skipUnless(shutil.which("ffprobe"), "requires dev image media tools")
    def test_invalid_media_is_rejected(self):
        with self.assertRaisesRegex(FalError, "result_integrity_failed"):
            FalAdapter._probe(b"not a video")
