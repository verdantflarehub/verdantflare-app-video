"""Isolated single-task GPU worker subprocess for Singularity (du-0)."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

from .errors import RuntimeErrorCode
from .queue import Queue


GPU_UUID_RE = re.compile(r"GPU-[0-9a-f-]{36}")


def read_vm_hwm_bytes() -> int:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


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
        int(os.environ.get("SINGULARITY_MIN_FREE_VRAM_MIB", "28672"))
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


def run_worker_task(task_id: str, task_root: Path | None = None) -> int:
    root = Path(task_root or os.environ.get("SINGULARITY_TASK_ROOT", "/data/projects/singularity/tasks"))
    queue = Queue(root, acquire_lock=False)
    try:
        task = queue.get(task_id)
        request = task["request"]
        directory = root / task_id
        from .api import _download_references

        download_started = time.perf_counter()
        download_seconds = None
        try:
            directory.mkdir(parents=True, exist_ok=False)
            queue.update(task_id, stage="downloading", worker_settled=0)
            files = _download_references(directory, request.get("conditions", []))
            download_seconds = time.perf_counter() - download_started
            gpu_uuid, free_mib = check_free_vram()
            queue.update(task_id, stage="warming", worker_settled=0)

            os.environ.setdefault("H3_CLIP_DEVICE", "cuda:0")
            os.environ.setdefault("H3_VAE_DEVICE", "cuda:0")
            os.environ.setdefault("SINGULARITY_LOW_VRAM_MODE", "auto")
            from .engine import Engine

            engine = Engine()
            result = engine.generate(
                request,
                files,
                directory / "output.mp4",
                lambda stage: queue.update(task_id, stage=stage, worker_settled=0),
            )
            metrics = dict(result.pop("runtime_metrics", {}) or {})
            metrics["reference_download_seconds"] = download_seconds
            metrics.setdefault("gpu_uuid", gpu_uuid)
            metrics.setdefault("free_vram_mib_before", free_mib)
            metrics.setdefault("vm_hwm_bytes", read_vm_hwm_bytes())
            queue.update(
                task_id,
                status="completed",
                stage="completed",
                execution_instance_id=result.pop("execution_instance_id", str(uuid.uuid4())),
                runtime_metrics=metrics,
                result=result,
                worker_settled=0,
            )
            return 0
        except Exception as exc:
            code = getattr(exc, "code", type(exc).__name__)
            metrics = dict(getattr(exc, "runtime_metrics", None) or {})
            if download_seconds is None:
                metrics["reference_download_seconds"] = time.perf_counter() - download_started
            queue.update(
                task_id,
                status="failed",
                stage="engine_failed",
                runtime_metrics=metrics,
                error=code,
                worker_settled=0,
            )
            return 1
    finally:
        queue.close()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        sys.stderr.write("Usage: python -m h3_singularity.worker <task_id>\n")
        return 2
    return run_worker_task(args[0])


if __name__ == "__main__":
    raise SystemExit(main())
