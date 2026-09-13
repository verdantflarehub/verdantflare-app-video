import unittest
from types import SimpleNamespace
from gpu_guard import verify_gpu

UUID = '7bffc292-2b42-42a7-cdad-1e93405a3017'


def torch_fixture(count=1, available=True):
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available,
        device_count=lambda: count, get_device_properties=lambda n: SimpleNamespace(uuid=UUID)))


class GPUGuardTest(unittest.TestCase):
    def test_matching_full_card(self):
        self.assertEqual(verify_gpu(torch_fixture(), f'GPU-{UUID},NVIDIA,24564,100:;'), 'GPU-' + UUID)

    def test_rejects_cpu_and_multiple_cards(self):
        for fixture in (torch_fixture(available=False), torch_fixture(count=2)):
            with self.assertRaisesRegex(RuntimeError, 'exactly_one_cuda_gpu_required'):
                verify_gpu(fixture, f'GPU-{UUID},NVIDIA,24564,100:;')

    def test_rejects_missing_or_sliced_allocation(self):
        for allocation in ('', f'GPU-{UUID},NVIDIA,12000,50:;'):
            with self.assertRaisesRegex(RuntimeError, 'exclusive_hami_allocation_required'):
                verify_gpu(torch_fixture(), allocation)

    def test_rejects_actual_uuid_different_from_scheduler(self):
        with self.assertRaisesRegex(RuntimeError, 'hami_cuda_uuid_mismatch'):
            verify_gpu(torch_fixture(), 'GPU-2c7a056b-3248-c027-3c7c-a799f47ab495,NVIDIA,24564,100:;')
