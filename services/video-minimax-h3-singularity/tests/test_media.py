import shutil
import tempfile
from pathlib import Path
import unittest
import subprocess

import numpy as np

try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("PyTorch is required for media tensor tests") from error

from h3_singularity.media import load_audio, load_video, align_video_reference, mux_mp4


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg and ffprobe required")
class MediaTest(unittest.TestCase):
    def test_video_timestamps_and_paired_sound_at_different_source_rates(self):
        with tempfile.TemporaryDirectory() as directory:
            for fps in (24, 30, 60):
                for audio_first in (False, True):
                    with self.subTest(fps=fps, audio_first=audio_first):
                        path = Path(directory) / f"ref-{fps}-{audio_first}.mp4"
                        maps = ["1:a", "0:v"] if audio_first else ["0:v", "1:a"]
                        subprocess.run([
                            "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                            f"testsrc2=size=32x32:rate={fps}:duration=2", "-f", "lavfi", "-i",
                            "sine=frequency=440:sample_rate=48000:duration=2",
                            "-map", maps[0], "-map", maps[1], "-c:v", "libx264", "-c:a", "aac", str(path),
                        ], check=True)
                        frames, sound, target_fps = load_video(path)
                        self.assertEqual(target_fps, 24)
                        self.assertEqual(tuple(frames.shape), (48, 32, 32, 3))
                        self.assertIsNotNone(sound)
                        self.assertEqual(sound["waveform"].shape[-1], 96000)
                        self.assertGreater(sound["waveform"].square().mean().item(), 0.001)
                        aligned, paired = align_video_reference(frames, sound, 362)
                        self.assertEqual(len(aligned), 39)
                        self.assertEqual(paired["waveform"].shape[-1], 78000)

    def test_pcm_formats_preserve_amplitude_and_stereo_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            for codec in ("pcm_u8", "pcm_s16le", "pcm_s32le", "pcm_f32le"):
                with self.subTest(codec=codec):
                    path = Path(directory) / f"{codec}.wav"
                    subprocess.run([
                        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        "aevalsrc=0.25*sin(2*PI*440*t)|0.5*sin(2*PI*880*t):s=48000:d=2",
                        "-c:a", codec, str(path),
                    ], check=True)
                    audio = load_audio(path)
                    self.assertEqual(tuple(audio["waveform"].shape), (1, 2, 96000))
                    self.assertEqual(audio["sample_rate"], 48000)
                    rms = audio["waveform"].square().mean(dim=-1).sqrt()[0]
                    self.assertAlmostEqual(rms[0].item(), 0.25 / np.sqrt(2), delta=0.01)
                    self.assertAlmostEqual(rms[1].item(), 0.5 / np.sqrt(2), delta=0.01)

    def test_video_audio_offset_is_not_shifted_to_time_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "delayed.mp4"
            subprocess.run([
                "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                "color=size=32x32:rate=24:duration=2", "-itsoffset", "0.5", "-f", "lavfi", "-i",
                "sine=frequency=440:sample_rate=48000:duration=1", "-c:v", "libx264", "-c:a", "aac", str(path),
            ], check=True)
            _, audio, _ = load_video(path)
            samples = audio["waveform"]
            self.assertEqual(samples.shape[-1], 96000)
            self.assertLess(samples[..., :20000].abs().max().item(), 0.001)
            self.assertGreater(samples[..., 30000:60000].square().mean().item(), 0.001)

    def test_video_display_rotation_is_applied_before_adaptive_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            source, rotated = Path(directory) / "source.mp4", Path(directory) / "rotated.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x96:rate=24:duration=2",
                            "-c:v", "libx264", str(source)], check=True)
            subprocess.run(["ffmpeg", "-v", "error", "-display_rotation", "90", "-i", str(source), "-c", "copy", str(rotated)], check=True)
            original, _, _ = load_video(source)
            displayed, _, _ = load_video(rotated)
            self.assertEqual(tuple(displayed.shape), (48, 64, 96, 3))
            self.assertTrue(torch.equal(displayed, torch.rot90(original, 1, (1, 2))))

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
