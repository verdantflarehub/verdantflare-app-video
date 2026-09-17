"""Exercise the actual vendored tensor helpers without importing GPU ComfyUI."""
import ast
import math
from pathlib import Path
import unittest

import torch


path = Path(__file__).resolve().parents[1] / 'src/h3_latent_upscaler/vendor/split_upscale.py'
tree = ast.parse(path.read_text())
helpers = {'trim_keyframe', 'frames_for_tokens', 'reanchor_conditioning', 'crop_keyframes_to_tile'}
module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in helpers], type_ignores=[])
namespace = {'torch': torch, 'math': math, 'FRAME_PER_TOKEN': (1, 4, 4, 4, 4), 'FRAME_RESCALE': 5 / 3}
exec(compile(module, str(path), 'exec'), namespace)


class SplitGeometryTest(unittest.TestCase):
    def test_audio_end_uses_same_units_as_start(self):
        audio = torch.arange(100).reshape(1, 1, 1, 100)
        result = namespace['trim_keyframe']({'resolved_frame_index': 0, 'audio_latent': audio}, 6, 24)
        self.assertTrue(torch.equal(result['audio_latent'], audio[..., 10:40]))

    def test_resize_preserves_channel_and_time_identity(self):
        latent = torch.arange(24 * 2, dtype=torch.float32).reshape(1, 24, 2, 1, 1).expand(1, 24, 2, 2, 2).clone()
        cond = [[None, {'minimax_keyframes': [{'resolved_frame_index': 0, 'latent': latent}]}]]
        resized = namespace['reanchor_conditioning'](cond, 0, 5, (4, 4))[0][1]['minimax_keyframes'][0]['latent']
        self.assertEqual(tuple(resized.shape), (1, 24, 2, 4, 4))
        self.assertTrue(torch.equal(resized[..., 0, 0], latent[..., 0, 0]))
        self.assertEqual(tuple(cond[0][1]['minimax_keyframes'][0]['latent'].shape), (1, 24, 2, 2, 2))

    def test_mismatched_keyframe_resize_is_spatial_only(self):
        latent = torch.arange(48, dtype=torch.float32).reshape(1, 24, 2, 1, 1).expand(1, 24, 2, 2, 2).clone()
        cond = [[None, {'minimax_keyframes': [{'resolved_frame_index': 0, 'latent': latent}]}]]
        cropped = namespace['crop_keyframes_to_tile'](cond, 4, 4, 1, 1, 2, 2)[0][1]['minimax_keyframes'][0]['latent']
        self.assertEqual(tuple(cropped.shape), (1, 24, 2, 2, 2))
        self.assertTrue(torch.equal(cropped, latent))


if __name__ == '__main__':
    unittest.main()
