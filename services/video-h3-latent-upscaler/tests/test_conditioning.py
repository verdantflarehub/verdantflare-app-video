import json
from pathlib import Path
import tempfile
import unittest
import torch
from safetensors.torch import save_file
from h3_latent_upscaler.conditioning import load_conditioning
from h3_latent_upscaler.resources import ResourceError
from h3_latent_upscaler.tensors import decoded_video_frames


class ConditioningTests(unittest.TestCase):
    def fixture(self, root):
        video = torch.arange(24 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 24, 2, 3, 4)
        # Stereo channels have disjoint values to expose interleaved/channel-major mistakes.
        audio = torch.arange(2 * 5 * 32, dtype=torch.float32).reshape(10, 32)
        tensors = {'prompt_embeds': torch.ones(3, 8), 'text_token_tags': torch.tensor([0, 1, 2]),
                   'video': video, 'audio': audio}
        geometry = {'condition_latents': ['video'], 'audio_condition_latents': ['audio'],
                    'normalized_references': [{'kind': 'video', 'has_audio': True}]}
        files = {'conditions': root / 'conditions.safetensors', 'geometry': root / 'geometry.json'}
        save_file(tensors, files['conditions'])
        files['geometry'].write_text(json.dumps(geometry))
        return {'files': files}, tensors, geometry

    def test_reference_video_and_stereo_keep_axis_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            resources, original, _ = self.fixture(Path(tmp))
            embeddings, metadata = load_conditioning(resources)[0]
            self.assertEqual(embeddings.shape, (1, 3, 8))
            ref = metadata['minimax_refs'][0]
            self.assertEqual(ref['kind'], 'video_audio')
            self.assertTrue(torch.equal(ref['latent'], original['video']))
            for channel in range(2):
                for time in range(5):
                    self.assertTrue(torch.equal(ref['audio_latent'][0, :, channel, time],
                                                original['audio'][channel * 5 + time]))

    def test_missing_and_extra_references_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            resources, _, geometry = self.fixture(Path(tmp))
            for refs, code in [([], 'unmatched_reference_conditions'),
                               (geometry['normalized_references'] * 2, 'incomplete_reference_conditions')]:
                resources['files']['geometry'].write_text(json.dumps({**geometry, 'normalized_references': refs}))
                with self.assertRaisesRegex(ResourceError, code):
                    load_conditioning(resources)

    def test_video_vae_batch_axis_is_removed_without_reordering_frames(self):
        frames = torch.arange(1 * 5 * 4 * 6 * 3, dtype=torch.float32).reshape(1, 5, 4, 6, 3)
        actual = decoded_video_frames(frames, count=5, width=6, height=4)
        self.assertTrue(torch.equal(actual[3], frames[0, 3]))
        with self.assertRaisesRegex(ResourceError, 'decoded_batch_mismatch'):
            decoded_video_frames(frames.repeat(2, 1, 1, 1, 1), count=10, width=6, height=4)
        with self.assertRaisesRegex(ResourceError, 'invalid_decoded_video_shape'):
            decoded_video_frames(frames, count=6, width=6, height=4)
