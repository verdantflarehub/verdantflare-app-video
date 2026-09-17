from pathlib import Path
import tempfile
import unittest
import torch
from safetensors.torch import save_file
from h3_latent_upscaler.resources import ResourceError
from h3_latent_upscaler.tensors import canonical_audio_from_diffusers, load_av


class TensorsTest(unittest.TestCase):
    def test_stereo_channel_order_preserved(self):
        audio = torch.arange(2 * 32 * 5, dtype=torch.float32).reshape(2, 32, 5)
        converted = canonical_audio_from_diffusers(audio)
        self.assertTrue(torch.equal(converted[0, :, 0], audio[0]))
        self.assertTrue(torch.equal(converted[0, :, 1], audio[1]))

    def test_bundle_geometry_and_normalization_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video, audio = root / 'video.safetensors', root / 'audio.safetensors'
            save_file({'samples': torch.zeros(1, 24, 37, 2, 4)}, video)
            save_file({'samples': torch.zeros(1, 32, 2, 207)}, audio)
            manifest = {'representation': 'h3-normalized-av/v1',
                        'media': {'fps': 24, 'width': 64, 'height': 32, 'frames': 124}}
            resources = {'manifest': manifest, 'files': {'video_latent': video, 'audio_latent': audio}}
            self.assertEqual(load_av(resources)[0].shape[2], 37)
            manifest['representation'] = 'raw-vae'
            with self.assertRaisesRegex(ResourceError, 'unsupported_latent_representation'):
                load_av(resources)
            manifest['representation'] = 'h3-normalized-av/v1'
            manifest['media']['frames'] = 125
            with self.assertRaisesRegex(ResourceError, 'latent_media_mismatch'):
                load_av(resources)

    def test_nonfinite_audio_rejected(self):
        with self.assertRaisesRegex(ResourceError, 'nonfinite_audio'):
            canonical_audio_from_diffusers(torch.full((2, 32, 5), float('nan')))
