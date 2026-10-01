import shutil
import tempfile
from pathlib import Path
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("PyTorch is required for media tensor tests") from error

from h3_singularity.media import mux_mp4


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg and ffprobe required")
class MediaTest(unittest.TestCase):
    def test_comfy_audio_object_preserves_duration_when_resampling(self):
        frames = torch.full((24, 16, 16, 3), 0.5)
        t = torch.arange(48000, dtype=torch.float32) / 48000
        audio = {"waveform": (0.1 * torch.sin(2 * torch.pi * 440 * t))[None, None, :], "sample_rate": 48000}
        with tempfile.TemporaryDirectory() as directory:
            result = mux_mp4(frames, audio, Path(directory) / "video.mp4")
        self.assertEqual(result["frames"], 24)
        self.assertEqual(result["video_codec"], "h264")
        self.assertEqual(result["audio_codec"], "aac")
        self.assertEqual(result["audio_sample_rate"], 32000)
        self.assertEqual(result["audio_channels"], 2)
        self.assertAlmostEqual(result["duration_seconds"], 1.0, delta=0.05)

    def test_tensor_audio_remains_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            result = mux_mp4(torch.zeros(24, 16, 16, 3), torch.zeros(1, 2, 32000), Path(directory) / "video.mp4")
        self.assertAlmostEqual(result["duration_seconds"], 1.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
