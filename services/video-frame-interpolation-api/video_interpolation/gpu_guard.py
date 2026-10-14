"""Validate one assigned CUDA device for native sharing or legacy HAMI."""
import ctypes
from pathlib import Path
import re
import subprocess
import uuid


def assigned_gpu():
    # CUDA base images declare "all"; the device plugin appends its UUID. libc
    # getenv/os.environ can retain the first duplicate, so inspect the original
    # OCI process environment and require exactly one concrete assigned UUID.
    try:
        values = [entry.split(b'=', 1)[1].decode('ascii')
                  for entry in Path('/proc/self/environ').read_bytes().split(b'\0')
                  if entry.startswith(b'NVIDIA_VISIBLE_DEVICES=')]
    except (OSError, UnicodeError) as exc:
        raise RuntimeError('native_gpu_assignment_unavailable') from exc
    concrete = [value for value in values if value not in ('all', 'void', 'none', '')]
    if not concrete:
        # NVIDIA Container Toolkit can consume the OCI assignment and replace
        # this variable with "void" after injecting the assigned devices.
        # NVML sees the injected set independently of CUDA_VISIBLE_DEVICES.
        try:
            concrete = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                text=True, timeout=10, stderr=subprocess.DEVNULL).strip().splitlines()
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError('native_gpu_assignment_unavailable') from exc
    if len(concrete) != 1 or not re.fullmatch(r'GPU-[0-9a-f-]{36}', concrete[0]):
        raise RuntimeError('native_single_gpu_allocation_required')
    return concrete[0]


def verify_gpu(torch, allocation, sharing_strategy='none', visible_devices=None):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('exactly_one_cuda_gpu_required')
    if sharing_strategy == 'time-slicing':
        if visible_devices is None:
            visible_devices = assigned_gpu()
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
