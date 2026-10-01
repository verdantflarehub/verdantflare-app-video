import unittest

try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("PyTorch is required for HR refine tests") from error

from h3_singularity.hr_refine import (
    h3_resize_video,
    spatial_tiles,
    temporal_windows,
    tile_blend_mask,
)


class HrRefineTest(unittest.TestCase):
    def test_temporal_windows_cover_h3_grid(self):
        windows = temporal_windows(102, 73, 22)
        self.assertEqual(windows[0][0], 0)
        self.assertEqual(windows[-1][1], 102)
        self.assertTrue(all(left < right for left, right, _, _ in windows))
        self.assertTrue(all(windows[i][1] >= windows[i + 1][0] for i in range(len(windows) - 1)))

    def test_spatial_tiles_cover_canvas_without_gaps(self):
        tiles = spatial_tiles(84, 48, 42, 24, 8)
        coverage = torch.zeros(84, 48, dtype=torch.int32)
        for r0, r1, c0, c1, *_ in tiles:
            coverage[r0:r1, c0:c1] += 1
        self.assertTrue(torch.all(coverage > 0))
        self.assertEqual(len(tiles), 9)

    def test_resize_preserves_time_channels_and_finiteness(self):
        source = torch.randn(1, 24, 102, 70, 40, dtype=torch.float32)
        mean = torch.zeros(1, 24, 1, 1, 1)
        std = torch.ones(1, 24, 1, 1, 1)
        result = h3_resize_video(source, 84, 48, mean, std)
        self.assertEqual(tuple(result.shape), (1, 24, 102, 84, 48))
        self.assertTrue(torch.isfinite(result).all())

    def test_blend_mask_is_bounded(self):
        mask = tile_blend_mask(10, 12, 3, 4, torch.device("cpu"), torch.float32)
        self.assertEqual(tuple(mask.shape), (1, 1, 1, 10, 12))
        self.assertGreaterEqual(float(mask.min()), 0.0)
        self.assertLessEqual(float(mask.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
