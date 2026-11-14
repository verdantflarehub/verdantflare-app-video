"""Isolated single-task Open-H3-Regenerate-2K-IC GPU worker subprocess."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .errors import RuntimeErrorCode
from .queue import Queue
from .regenerate import resolve_source_artifact_path, run_regenerate_2k


GPU_UUID_RE = re.compile(r"GPU-[0-9a-f-]{36}")


def _assigned_gpu_from_environ() -> str | None:
    try:
        entries = [
            entry.split(b"=", 1)[1].decode("ascii", errors="ignore")
            for entry in Path("/proc/self/environ").read_bytes().split(b"\0")
            if entry.startswith(b"NVIDIA_VISIBLE_DEVICES=")
        ]
    except OSError:
        entries = [os.environ.get("NVIDIA_VISIBLE_DEVICES", "")]
    concrete = [v.strip() for v in entries if v.strip() and v.strip() not in {"all", "void", "none"}]
    if len(concrete) == 1 and GPU_UUID_RE.fullmatch(concrete[0]):
        return concrete[0]
    if len(concrete) > 1:
        raise RuntimeErrorCode("gpu_memory_occupancy_changed")
    return None


def check_free_vram(min_free_mib: int | None = None) -> tuple[str, int]:
    """Verify assigned physical GPU UUID and enforce free VRAM gate before loading CUDA weights."""
    required_mib = (
        int(os.environ.get("REGENERATE_2K_MIN_FREE_VRAM_MIB", "28672"))
        if min_free_mib is None
        else int(min_free_mib)
    )
    expected_uuid = _assigned_gpu_from_environ()
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeErrorCode("gpu_memory_occupancy_changed") from exc

    rows: list[tuple[str, int]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2 or not GPU_UUID_RE.fullmatch(parts[0]):
            raise RuntimeErrorCode("gpu_memory_occupancy_changed")
        try:
            free_val = int(parts[1])
        except ValueError as exc:
            raise RuntimeErrorCode("gpu_memory_occupancy_changed") from exc
        rows.append((parts[0], free_val))

    if len(rows) != 1:
        raise RuntimeErrorCode("gpu_memory_occupancy_changed")
    actual_uuid, free_mib = rows[0]
    if expected_uuid is not None and actual_uuid != expected_uuid:
        raise RuntimeErrorCode(
            "gpu_memory_occupancy_changed",
            metrics={
                "expected_gpu_uuid": expected_uuid,
                "gpu_uuid": actual_uuid,
                "free_vram_mib_before": free_mib,
                "min_free_vram_mib": required_mib,
            },
        )
    if free_mib < required_mib:
        raise RuntimeErrorCode(
            "gpu_memory_occupancy_changed",
            metrics={
                "gpu_uuid": actual_uuid,
                "free_vram_mib_before": free_mib,
                "min_free_vram_mib": required_mib,
            },
        )
    return actual_uuid, free_mib


def _preload_sm120_cuda13_and_comfy() -> dict:
    """Preload CUDA 13 shared libraries and initialize comfy-aimdo + comfy-kitchen BEFORE torch import."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for cuda13_lib in (
        "/usr/local/cuda/lib64/libcudart.so.13",
        "/usr/local/cuda/lib64/libcublas.so.13",
        "/usr/local/cuda/lib64/libcublasLt.so.13",
    ):
        if os.path.exists(cuda13_lib):
            ctypes.CDLL(cuda13_lib, mode=ctypes.RTLD_GLOBAL)

    comfy_root = os.environ.get("COMFY_ROOT", "/opt/comfy")
    if comfy_root not in sys.path:
        sys.path.insert(0, comfy_root)

    aimdo_mod = None
    try:
        import comfy_aimdo.control as _aimdo_ctl

        _aimdo_ctl.init(simple_vram_headroom=None, nvml_pressure=True)
        aimdo_mod = _aimdo_ctl
    except Exception:
        aimdo_mod = None

    import torch  # noqa: F401
    import comfy_kitchen as ck
    import comfy.quant_ops  # noqa: F401
    import comfy.memory_management
    import comfy.model_management as model_management
    import comfy.model_patcher
    import folder_paths  # noqa: F401
    import nodes  # noqa: F401
    import comfy_extras.nodes_minimax_h3  # noqa: F401
    import comfy_extras.nodes_custom_sampler  # noqa: F401
    import comfy_extras.nodes_audio  # noqa: F401

    ck.registry.enable("cuda")
    ck_cuda_active = not ck.list_backends().get("cuda", {}).get("disabled", True)
    aimdo_active = False
    headroom_gib = float(os.environ.get("REGENERATE_2K_AIMDO_HEADROOM_GIB", "8.0"))
    if aimdo_mod is not None:
        aimdo_ok = aimdo_mod.init_devices(
            (d.index, int(headroom_gib * 1024**3)) for d in model_management.get_all_torch_devices()
        )
        if aimdo_ok:
            comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
            comfy.memory_management.aimdo_enabled = True
            aimdo_active = True

    return {
        "comfy_kitchen_cuda_enabled": ck_cuda_active,
        "dynamic_vram_aimdo_enabled": aimdo_active,
        "aimdo_headroom_gib": headroom_gib,
    }


def run_worker_task(video_task_id: str, task_root: Path | None = None) -> int:
    root = Path(task_root or os.environ.get("REGENERATE_2K_TASK_ROOT", "/data/projects/regenerate-2k/tasks"))
    queue = Queue(root, acquire_lock=False)
    try:
        task = queue.get_internal(video_task_id)
        request = task["request"]
        directory = root / video_task_id
        try:
            directory.mkdir(parents=True, exist_ok=False)
            artifact_root = Path(os.environ.get("REGENERATE_2K_ARTIFACT_ROOT", "/source/video"))
            source_mp4 = resolve_source_artifact_path(request, artifact_root)
            gpu_uuid, free_mib = check_free_vram()
            _preload_sm120_cuda13_and_comfy()
            result = run_regenerate_2k(
                request,
                source_mp4,
                directory,
                lambda stage: queue.update(video_task_id, stage=stage, worker_settled=0),
                gpu_uuid=gpu_uuid,
                free_vram_mib_before=free_mib,
            )
            (directory / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            queue.update(
                video_task_id,
                status="succeeded",
                stage="succeeded",
                runtime_metrics=result.get("runtime_metrics"),
                result=result,
                worker_settled=0,
            )
            return 0
        except Exception as exc:
            code = getattr(exc, "code", type(exc).__name__)
            metrics = dict(getattr(exc, "runtime_metrics", None) or {})
            queue.update(
                video_task_id,
                status="failed",
                stage="engine_failed",
                runtime_metrics=metrics or None,
                error=code,
                worker_settled=0,
            )
            return 1
    finally:
        queue.close()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        sys.stderr.write("Usage: python -m h3_regenerate_2k.worker <video_task_id>\n")
        return 2
    return run_worker_task(args[0])


if __name__ == "__main__":
    raise SystemExit(main())
