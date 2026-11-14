"""Validate the actual CUDA device against the exclusive HAMI allocation."""
import ctypes
import re
import uuid


def verify_gpu(torch, allocation):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('exactly_one_cuda_gpu_required')
    match = re.fullmatch(r'(GPU-[0-9a-f-]{36}),NVIDIA,24564,100:;?', allocation)
    if not match:
        raise RuntimeError('exclusive_hami_allocation_required')
    # Torch 2.3 does not expose UUID in device properties. Query the CUDA driver.
    driver = ctypes.CDLL('libcuda.so.1')
    device = ctypes.c_int()
    raw = (ctypes.c_ubyte * 16)()
    if driver.cuInit(0) or driver.cuDeviceGet(ctypes.byref(device), 0) or driver.cuDeviceGetUuid(ctypes.byref(raw), device):
        raise RuntimeError('cuda_device_identity_unavailable')
    actual = 'GPU-' + str(uuid.UUID(bytes=bytes(raw)))
    if actual != match.group(1):
        raise RuntimeError('hami_cuda_uuid_mismatch')
    return actual
