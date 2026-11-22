from __future__ import annotations

import logging
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.artifacts import ArtifactStore
from app.executor import VideoExecutor
from app.gpu_lease import EngineConcurrencyGate, GpuCoordinator, LocalGpuStageSlot
from app.tasks import TaskStore


def _make_store(tmp_path: Path) -> TaskStore:
    store = TaskStore(tmp_path)
    store.ensure_ready()
    return store


def test_from_environment_disables_global_stage_slot_and_legacy_lease_when_scheduler_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _make_store(tmp_path)
    monkeypatch.setenv("STATION_GPU_SCHEDULER_URL", "http://station-runtime:5056")
    monkeypatch.setenv("GPU_TASK_ROLE", "h3")

    with caplog.at_level(logging.WARNING):
        coord = GpuCoordinator.from_environment()
        slot = LocalGpuStageSlot.from_environment(store)
        gate = EngineConcurrencyGate.from_environment(store)

    assert coord.enabled is False
    assert slot.enabled is False
    assert gate.enabled is True
    assert any("Ignoring legacy GPU_TASK_ROLE=h3" in rec.message for rec in caplog.records)

    monkeypatch.delenv("STATION_GPU_SCHEDULER_URL", raising=False)
    legacy_coord = GpuCoordinator.from_environment()
    legacy_slot = LocalGpuStageSlot.from_environment(store)
    legacy_gate = EngineConcurrencyGate.from_environment(store)

    assert legacy_coord.enabled is True
    assert legacy_slot.enabled is True
    assert legacy_gate.enabled is False


def test_engine_concurrency_gate_allows_cross_engine_and_blocks_same_engine(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    gate = EngineConcurrencyGate(store, enabled=True, default_limit=1)

    t_regen = store.create(
        project_id="proj-1",
        idempotency_key="k-regen",
        input_digest="sha256:" + "1" * 64,
        request={"service": "regenerate_2k"},
        runtime_task_id="rt-1",
        status="running",
    )
    t_regen = store.update(t_regen, service="regenerate_2k")

    t_fasth3 = store.create(
        project_id="proj-1",
        idempotency_key="k-fasth3",
        input_digest="sha256:" + "2" * 64,
        request={"service": "h3-fasth3"},
        runtime_task_id="rt-2",
        status="running",
    )
    t_fasth3 = store.update(t_fasth3, service="h3-fasth3")

    t_rife_1 = store.create(
        project_id="proj-1",
        idempotency_key="k-rife-1",
        input_digest="sha256:" + "3" * 64,
        request={"service": "interpolate"},
        runtime_task_id="rt-3",
        status="running",
    )
    t_rife_1 = store.update(t_rife_1, service="interpolate")

    t_rife_2 = store.create(
        project_id="proj-1",
        idempotency_key="k-rife-2",
        input_digest="sha256:" + "4" * 64,
        request={"service": "interpolate"},
        runtime_task_id="rt-4",
        status="queued",
    )
    t_rife_2 = store.update(t_rife_2, service="interpolate")

    ident_gpu1 = {"boot_id": "boot-gpu1"}
    ident_gpu0 = {"boot_id": "boot-gpu0"}
    ident_rife = {"boot_id": "boot-rife-1"}

    # Cross-engine tasks acquire concurrently without blocking each other.
    assert gate.try_acquire(t_regen.video_task_id, "regenerate_2k", runtime_identity=ident_gpu1) is True
    assert gate.try_acquire(t_fasth3.video_task_id, "h3-fasth3", runtime_identity=ident_gpu0) is True

    # Same engine instance (service + boot_id) enforces limit=1.
    assert gate.try_acquire(t_rife_1.video_task_id, "interpolate", runtime_identity=ident_rife) is True
    assert gate.try_acquire(t_rife_2.video_task_id, "interpolate", runtime_identity=ident_rife) is False

    # Different boot_id (restarted engine pod) is isolated.
    assert gate.try_acquire(t_rife_2.video_task_id, "interpolate", runtime_identity={"boot_id": "boot-rife-2"}) is True
    gate.release(t_rife_2.video_task_id, force=True)

    # Once holder settles and releases, second task acquires immediately.
    store.update(t_rife_1, status="succeeded", provider_state={"worker_settled": True})
    gate.release(t_rife_1.video_task_id)
    assert gate.try_acquire(t_rife_2.video_task_id, "interpolate", runtime_identity=ident_rife) is True


def test_engine_slot_rejection_skips_scheduler_and_preserves_attempt_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATION_GPU_SCHEDULER_URL", "http://station-runtime:5056")
    monkeypatch.setenv("STATION_GPU_SCHEDULER_TOKEN", "test-token")
    monkeypatch.setenv("GPU_TASK_ROLE", "h3")

    store = _make_store(tmp_path)
    artifacts = ArtifactStore(tmp_path / "artifacts")
    executor = VideoExecutor(artifacts, store)
    executor.scheduler = MagicMock()
    executor.scheduler.enabled = True

    rife_ident = {
        "pod_uid": "pod-rife-1",
        "container_id": "cid-rife-1",
        "boot_id": "boot-rife-1",
        "visible_gpu_uuids": ["GPU-1"],
    }
    executor.probe_runtime_identity = MagicMock(return_value=rife_ident)

    t1 = store.create(
        project_id="proj-1",
        idempotency_key="rife-1",
        input_digest="sha256:" + "a" * 64,
        request={"service": "interpolate"},
        runtime_task_id="rt-rife-1",
        status="running",
    )
    t1 = store.update(t1, service="interpolate")
    assert executor.engine_gate.try_acquire(t1.video_task_id, "interpolate", runtime_identity=rife_ident) is True

    t2 = store.create(
        project_id="proj-1",
        idempotency_key="rife-2",
        input_digest="sha256:" + "b" * 64,
        request={"service": "interpolate"},
        runtime_task_id="",
        status="queued",
        provider_state={"_gpu_attempt_id": "att-fixed-123", "gpu_dispatched": False},
    )
    t2 = store.update(t2, service="interpolate")

    # Attempting to acquire engine slot returns False without calling :5056 scheduler or rotating attempt_id.
    assert executor.acquire_engine_slot(t2.video_task_id, "interpolate", runtime_identity=rife_ident, record=t2) is False
    deferred = executor.defer_local_gpu_queue(t2, waiting_reason="engine_slot")

    assert deferred.status == "queued"
    assert (deferred.provider_state or {}).get("waiting_reason") == "engine_slot"
    assert (deferred.provider_state or {}).get("_gpu_attempt_id") == "att-fixed-123"
    executor.scheduler.acquire.assert_not_called()
    executor.scheduler.release.assert_not_called()


def test_rollback_backoff_skips_rapid_dispatch_until_wakeup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATION_GPU_SCHEDULER_URL", "http://station-runtime:5056")
    monkeypatch.setenv("STATION_GPU_SCHEDULER_TOKEN", "test-token")

    store = _make_store(tmp_path)
    artifacts = ArtifactStore(tmp_path / "artifacts")
    executor = VideoExecutor(artifacts, store)
    executor.scheduler = MagicMock()
    executor.scheduler.enabled = True
    executor.scheduler.release.return_value = {"state": "released"}

    rec = store.create(
        project_id="proj-1",
        idempotency_key="backoff-1",
        input_digest="sha256:" + "c" * 64,
        request={"service": "h3-fasth3"},
        runtime_task_id="",
        status="queued",
        provider_state={
            "gpu_dispatched": False,
            "_gpu_lease_owned": True,
            "_gpu_lease": {
                "lease_id": "lease-1",
                "state": "leased",
                "fencing_token": 10,
                "version": 1,
            },
        },
    )
    rec = store.update(rec, service="h3-fasth3")

    rolled = executor.rollback_unstarted_scheduler_grant(rec)
    pstate = rolled.provider_state or {}
    assert pstate.get("_gpu_retry_failures") == 1
    assert float(pstate.get("_gpu_retry_after_epoch") or 0) > time.time()

    # Without wakeup, dispatch_queued_local_tasks respects backoff and skips calling status().
    executor._dispatch_wakeup.clear()
    executor.status = MagicMock()
    executor.dispatch_queued_local_tasks()
    executor.status.assert_not_called()

    # Releasing any local slot sets wakeup event, allowing immediate dispatch.
    executor.release_local_slots("video_task_" + "0" * 32, force=True)
    executor.dispatch_queued_local_tasks()
    executor.status.assert_called_once_with(rec.video_task_id)


def test_reconcile_gpu_multi_lease_isolation_without_legacy_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATION_GPU_SCHEDULER_URL", "http://station-runtime:5056")
    monkeypatch.setenv("STATION_GPU_SCHEDULER_TOKEN", "test-token")
    monkeypatch.setenv("GPU_TASK_ROLE", "h3")

    store = _make_store(tmp_path)
    artifacts = ArtifactStore(tmp_path / "artifacts")
    executor = VideoExecutor(artifacts, store)
    assert executor.gpu.enabled is False
    assert executor.stage_slot.enabled is False

    executor.gpu.try_acquire = MagicMock(side_effect=AssertionError("legacy K8s lease must not be touched"))
    executor.gpu.renew_existing = MagicMock(side_effect=AssertionError("legacy K8s lease must not be touched"))
    executor.gpu.release = MagicMock(side_effect=AssertionError("legacy K8s lease must not be touched"))

    executor.scheduler = MagicMock()
    executor.scheduler.enabled = True
    executor.scheduler.heartbeat.return_value = {
        "lease_id": "lease-a",
        "state": "running",
        "fencing_token": 101,
        "version": 2,
    }
    executor.scheduler.release.return_value = {"lease_id": "lease-b", "state": "released"}

    # Task A running on GPU 0 (unsettled) -> should only heartbeat.
    task_a = store.create(
        project_id="proj-1",
        idempotency_key="multi-a",
        input_digest="sha256:" + "d" * 64,
        request={"service": "h3-fasth3"},
        runtime_task_id="fasth3_" + "1" * 32,
        status="running",
        provider_state={
            "gpu_dispatched": True,
            "worker_settled": False,
            "_gpu_lease_owned": True,
            "_gpu_active_service": "h3-fasth3",
            "_gpu_lease": {
                "lease_id": "lease-a",
                "state": "running",
                "fencing_token": 101,
                "version": 1,
                "gpu_uuid": "GPU-0",
            },
        },
    )
    task_a = store.update(task_a, service="h3-fasth3")

    # Task B succeeded & settled on GPU 1 -> should only release lease-b.
    task_b = store.create(
        project_id="proj-1",
        idempotency_key="multi-b",
        input_digest="sha256:" + "e" * 64,
        request={"service": "regenerate_2k"},
        runtime_task_id="rt-b",
        status="succeeded",
        provider_state={
            "gpu_dispatched": True,
            "worker_settled": True,
            "_gpu_lease_owned": True,
            "_gpu_active_service": "regenerate_2k",
            "gpu_epoch": 5,
            "reclaim_proof": {
                "reclaimed": True,
                "active_tasks": 0,
                "proof_source": "receiver_guard",
                "token_sealed": True,
                "sealed_gpu_epoch": 5,
                "supervisor_pid": 56,
                "supervisor_starttime_ticks": 12345,
                "sealed_pod_uid": "pod-b",
                "sealed_container_id": "cid-b",
                "sealed_boot_id": "boot-b",
                "revoked_lease_id": "lease-b",
                "consumed_fencing_token": 202,
            },
            "_gpu_lease": {
                "lease_id": "lease-b",
                "state": "running",
                "fencing_token": 202,
                "version": 3,
                "pod_uid": "pod-b",
                "container_id": "cid-b",
                "boot_id": "boot-b",
                "gpu_uuid": "GPU-1",
            },
        },
    )
    task_b = store.update(task_b, service="regenerate_2k")

    executor.reconcile_gpu()

    executor.scheduler.heartbeat.assert_called_once_with("lease-a", fencing_token=101, expected_version=1)
    assert executor.scheduler.release.call_count == 1
    assert executor.scheduler.release.call_args.args[0] == "lease-b"
    assert (store.get(task_a.video_task_id).provider_state or {}).get("_gpu_lease_released") is not True
    assert (store.get(task_b.video_task_id).provider_state or {}).get("_gpu_lease_released") is True
