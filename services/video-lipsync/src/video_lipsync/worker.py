"""Isolated single-task Video LipSync GPU worker subprocess."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys

from .errors import RuntimeErrorCode
from .lipsync import run_lipsync
from .queue import Queue


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
    """Verify assigned physical GPU UUID and enforce free VRAM gate BEFORE importing CUDA/torch."""
    required_mib = (
        int(os.environ.get("LIPSYNC_MIN_FREE_VRAM_MIB", "28672"))
        if min_free_mib is None
        else int(min_free_mib)
    )
    required_mib = max(28672, required_mib)
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


def run_worker_task(task_id: str, task_root: str | Path) -> int:
    queue = Queue(task_root, acquire_lock=False)
    try:
        task = queue.get_internal(task_id)
        if task["cancel_requested"] or task["status"] == "cancelled":
            queue.mark_settled(task_id, status="cancelled", stage="cancelled")
            return 0
        gpu_uuid, free_mib = check_free_vram()
        queue.update(task_id, stage="inferring")
        artifact_root = Path(os.environ.get("LIPSYNC_ARTIFACT_ROOT", "/data/projects/video"))
        result = run_lipsync(
            task["request"],
            artifact_root=artifact_root,
            task_dir=Path(task_root) / task_id,
            gpu_uuid=gpu_uuid,
            free_vram_mib_before=free_mib,
        )
        queue.update(
            task_id,
            status="succeeded",
            stage="completed",
            result=result,
            runtime_metrics=result.get("runtime_metrics"),
            worker_settled=True,
        )
        return 0
    except RuntimeErrorCode as exc:
        queue.update(
            task_id,
            status="failed",
            stage=exc.code,
            error=exc.code,
            runtime_metrics=exc.runtime_metrics,
            worker_settled=True,
        )
        return 1
    except Exception:
        queue.update(
            task_id,
            status="failed",
            stage="inference_failed",
            error="inference_failed",
            worker_settled=True,
        )
        return 1
    finally:
        queue.close()


def main() -> int:
    if len(sys.argv) < 2:
        return 2
    task_id = sys.argv[1]
    task_root = os.environ.get("LIPSYNC_TASK_ROOT", "/data/projects/lipsync/tasks")
    return run_worker_task(task_id, task_root)


if __name__ == "__main__":
    raise SystemExit(main())
