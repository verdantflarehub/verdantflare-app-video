import ctypes
import unittest
import uuid
from unittest.mock import Mock, patch

from video_interpolation.gpu_guard import assigned_gpu, verify_gpu


GPU = 'GPU-29a56c24-484e-d51d-fac9-b141be9e24c5'


class GPUAllocationTest(unittest.TestCase):
    def setUp(self):
        self.torch = Mock()
        self.torch.cuda.is_available.return_value = True
        self.torch.cuda.device_count.return_value = 1
        driver = Mock()
        driver.cuInit.return_value = 0
        driver.cuDeviceGet.return_value = 0

        def device_uuid(raw, device):
            ctypes.memmove(raw, uuid.UUID(GPU[4:]).bytes, 16)
            return 0

        driver.cuDeviceGetUuid.side_effect = device_uuid
        self.library = patch('video_interpolation.gpu_guard.ctypes.CDLL', return_value=driver)
        self.library.start()
        self.addCleanup(self.library.stop)

    def test_native_assignment_matches_cuda_identity(self):
        self.assertEqual(verify_gpu(self.torch, '', 'time-slicing', GPU), GPU)

    def test_duplicate_base_image_environment_uses_concrete_assignment(self):
        raw = b'NVIDIA_VISIBLE_DEVICES=all\0NVIDIA_VISIBLE_DEVICES=' + GPU.encode() + b'\0'
        with patch('video_interpolation.gpu_guard.Path.read_bytes', return_value=raw):
            self.assertEqual(verify_gpu(self.torch, '', 'time-slicing'), GPU)

    def test_original_environment_must_have_one_concrete_assignment(self):
        for raw in (b'NVIDIA_VISIBLE_DEVICES=all\0',
                    ('NVIDIA_VISIBLE_DEVICES='+GPU+'\0NVIDIA_VISIBLE_DEVICES='+GPU+'\0').encode()):
            with self.subTest(raw=raw), patch('video_interpolation.gpu_guard.Path.read_bytes', return_value=raw), self.assertRaises(RuntimeError):
                assigned_gpu()

    def test_native_assignment_rejects_unbound_or_mismatched_devices(self):
        for visible in ('', 'all', '0', GPU + ',' + GPU, 'GPU-' + '0' * 36):
            with self.subTest(visible=visible), self.assertRaises(RuntimeError):
                verify_gpu(self.torch, '', 'time-slicing', visible)

    def test_legacy_allocation_remains_verified(self):
        self.assertEqual(verify_gpu(self.torch, GPU + ',NVIDIA,24564,100:;'), GPU)
        with self.assertRaises(RuntimeError):
            verify_gpu(self.torch, '')

    def test_native_mode_does_not_accept_legacy_or_multiple_devices(self):
        with self.assertRaises(RuntimeError):
            verify_gpu(self.torch, GPU + ',NVIDIA,24564,100:;', 'time-slicing', GPU)
        self.torch.cuda.device_count.return_value = 2
        with self.assertRaises(RuntimeError):
            verify_gpu(self.torch, '', 'time-slicing', GPU)


if __name__ == '__main__':
    unittest.main()
