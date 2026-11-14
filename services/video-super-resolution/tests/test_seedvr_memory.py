import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile

import torch
from rotary_embedding_torch import RotaryEmbedding

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seedvr_memory import compact_rope_freqs, mapped_dit_offload


class CompactRopeTest(unittest.TestCase):
    def test_matches_dense_language_positions_for_multiple_samples(self):
        rope = RotaryEmbedding(dim=42, freqs_for='lang', theta=10000)
        module = SimpleNamespace(rope=rope)
        shapes = torch.tensor([[3, 7, 5], [5, 4, 8]])
        texts = torch.tensor([[11], [9]])
        actual_video, actual_text = compact_rope_freqs(module, shapes, texts)
        dense = rope.get_axial_freqs(16, 8, 8)
        dense_text = rope.get_axial_freqs(16)
        expected_video, expected_text = [], []
        for (f, h, w), length in zip(shapes.tolist(), texts[:, 0].tolist()):
            expected_video.append(dense[length:length + f, :h, :w].reshape(-1, 126))
            expected_text.append(dense_text[:length].repeat(1, 3))
        torch.testing.assert_close(actual_video, torch.cat(expected_video), rtol=0, atol=0)
        torch.testing.assert_close(actual_text, torch.cat(expected_text), rtol=0, atol=0)

    def test_rejects_geometry_outside_upstream_grid(self):
        rope = RotaryEmbedding(dim=42, freqs_for='lang')
        for shape in ([[3, 129, 5]], [[1020, 4, 4]]):
            with self.assertRaises(ValueError):
                compact_rope_freqs(SimpleNamespace(rope=rope), torch.tensor(shape), torch.tensor([[11]]))


class MappedOffloadTest(unittest.TestCase):
    def test_cpu_offload_restores_checkpoint_storage_without_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.pt'
            original = torch.nn.Linear(4, 3).eval().requires_grad_(False)
            torch.save(original.state_dict(), path)
            checkpoint = torch.load(path, mmap=True, weights_only=True)
            model = torch.nn.Linear(4, 3).eval().requires_grad_(False)
            model.load_state_dict(checkpoint, assign=True)
            mapped_dit_offload(model)
            # Simulate storage replaced during an inference phase.
            model.weight.data = torch.zeros_like(model.weight, dtype=torch.bfloat16)
            self.assertIs(model.to('cpu'), model)
            self.assertEqual(model.weight.data_ptr(), checkpoint['weight'].data_ptr())
            torch.testing.assert_close(model(torch.ones(1, 4)), original(torch.ones(1, 4)))
            with self.assertRaises(ValueError):
                model.to('meta')


if __name__ == '__main__':
    unittest.main()
