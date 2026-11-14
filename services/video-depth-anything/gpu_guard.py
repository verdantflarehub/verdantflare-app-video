"""Reject HAMI/native device-plugin disagreement before loading any model."""
import re


def verify_gpu(torch, allocation, mode='hami'):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('exactly_one_cuda_gpu_required')
    actual = str(torch.cuda.get_device_properties(0).uuid)
    if not actual.startswith('GPU-'):
        actual = 'GPU-' + actual
    if not re.fullmatch(r'GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', actual):
        raise RuntimeError('invalid_cuda_gpu_uuid')
    if mode == 'native':
        # Native device-plugin visibility is the allocation boundary. A shared
        # allocation is not evidence of exclusive ownership of the physical GPU.
        if allocation:
            raise RuntimeError('conflicting_gpu_allocation_modes')
        return actual
    if mode != 'hami':
        raise RuntimeError('unsupported_gpu_mode')
    match = re.fullmatch(r'(GPU-[0-9a-f-]{36}),NVIDIA,24564,100:;?', allocation)
    if not match:
        raise RuntimeError('exclusive_hami_allocation_required')
    if actual != match.group(1):
        raise RuntimeError('hami_cuda_uuid_mismatch')
    return actual
