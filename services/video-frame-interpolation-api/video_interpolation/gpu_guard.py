"""Validate one assigned CUDA device for native sharing or legacy HAMI."""
import ctypes
import re
import uuid


def verify_gpu(torch, allocation, sharing_strategy='none', visible_devices=''):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('exactly_one_cuda_gpu_required')
    if sharing_strategy == 'time-slicing':
        match = re.fullmatch(r'GPU-[0-9a-f-]{36}', visible_devices)
        if allocation or not match:
            raise RuntimeError('native_single_gpu_allocation_required')
        expected = visible_devices
    elif sharing_strategy == 'none':
        match = re.fullmatch(r'(GPU-[0-9a-f-]{36}),NVIDIA,24564,100:;?', allocation)
        if not match:
            raise RuntimeError('exclusive_hami_allocation_required')
        expected = match.group(1)
    else:
        raise RuntimeError('unsupported_gpu_sharing_strategy')
    # Read the CUDA identity instead of assuming the host's GPU ordinal.
    driver = ctypes.CDLL('libcuda.so.1')
    device = ctypes.c_int()
    raw = (ctypes.c_ubyte * 16)()
    if driver.cuInit(0) or driver.cuDeviceGet(ctypes.byref(device), 0) or driver.cuDeviceGetUuid(ctypes.byref(raw), device):
        raise RuntimeError('cuda_device_identity_unavailable')
    actual = 'GPU-' + str(uuid.UUID(bytes=bytes(raw)))
    if actual != expected:
        raise RuntimeError('allocated_cuda_uuid_mismatch')
    return actual
