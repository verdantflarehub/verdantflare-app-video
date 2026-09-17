from pathlib import Path
import subprocess
import tempfile
import unittest
import torch
from h3_latent_upscaler.media import finish_video, probe


class MediaTest(unittest.TestCase):
    def test_real_encoding_retains_audio_and_frame_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=blue:s=32x32:r=24:d=1',
                            '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=32000:duration=1',
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(source)], check=True)
            frames = torch.zeros(24, 64, 64, 3)
            frames[:, :, :, 2] = 1
            result = finish_video(frames, source, root)
            self.assertEqual(result['frames'], 24)
            self.assertTrue(result['audio'])
            self.assertEqual(probe(root / 'preview.mp4')['width'], 128)
            def packets(path):
                return subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:a', '-c', 'copy', '-f', 'adts', '-'])
            self.assertEqual(packets(source), packets(root / 'output.mp4'))
