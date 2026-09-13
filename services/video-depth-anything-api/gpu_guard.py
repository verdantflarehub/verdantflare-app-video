"""Reject HAMI/native device-plugin disagreement before loading any model."""
import re


def verify_gpu(torch, allocation):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('exactly_one_cuda_gpu_required')
    match = re.fullmatch(r'(GPU-[0-9a-f-]{36}),NVIDIA,24564,100:;?', allocation)
    if not match:
        raise RuntimeError('exclusive_hami_allocation_required')
    actual = str(torch.cuda.get_device_properties(0).uuid)
    if not actual.startswith('GPU-'):
        actual = 'GPU-' + actual
    if actual != match.group(1):
        raise RuntimeError('hami_cuda_uuid_mismatch')
    return actual
