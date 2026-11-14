"""Shared GPU task lease and on-demand scale.

Approved 2026-10-19 for 5090 Music3, H3, and IndexTTS2. Write this role's
Lease first, then read peers. A live conflicting peer means release and retry.
H3 and IndexTTS2 do not exclude each other. Music3 and IndexTTS2 scale to 1
while held and back to 0 after the idle window. H3 stays resident.
"""

from __future__ import annotations

import os
import random
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

LEASES = {
    "music3": "gpu-task-music3",
    "h3": "gpu-task-h3",
    "indextts2": "gpu-task-indextts2",
}
CONFLICTS = {
    "music3": ("h3", "indextts2"),
    "h3": ("music3",),
    "indextts2": ("music3",),
}


class LeaseBusy(RuntimeError):
    def __init__(self, holder: str | None = None) -> None:
        self.holder = holder or ""
        super().__init__("waiting_gpu_lease")


def format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000Z")


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def replicas_from_scale(payload: dict) -> int:
    # Kubernetes omits spec.replicas when the desired count is zero.
    spec = payload.get("spec") or {}
    if "replicas" not in spec:
        return 0
    return int(spec["replicas"])


def held_by_other(lease: dict | None, holder: str, now: datetime) -> str | None:
    spec = (lease or {}).get("spec") or {}
    identity = spec.get("holderIdentity") or ""
    if not identity or identity == holder:
        return None
    renew = spec.get("renewTime")
    duration = int(spec.get("leaseDurationSeconds") or 60)
    if not renew:
        return identity
    if parse_time(renew) + timedelta(seconds=duration) > now:
        return identity
    return None


class MemoryApi:
    """In-memory Lease and scale API for tests."""

    def __init__(self) -> None:
        self.leases: dict[str, dict] = {}
        self.scales: dict[str, int] = {}

    def get_lease(self, name: str) -> dict | None:
        item = self.leases.get(name)
        return None if item is None else {
            "metadata": dict(item["metadata"]),
            "spec": dict(item["spec"]),
        }

    def replace_lease(self, name: str, spec: dict, resource_version: str | None) -> int:
        current = self.leases.get(name)
        current_version = None if current is None else current["metadata"].get("resourceVersion")
        if current is not None and current_version != resource_version:
            return 409
        version = str(int(current_version or "0") + 1)
        self.leases[name] = {"metadata": {"name": name, "resourceVersion": version}, "spec": dict(spec)}
        return 200

    def get_scale(self, name: str) -> int:
        return self.scales.get(name, 0)

    def patch_scale(self, name: str, replicas: int) -> None:
        self.scales[name] = replicas


class K8sApi:
    def __init__(
        self,
        namespace: str,
        host: str = "https://kubernetes.default.svc",
        token_path: str = "/var/run/secrets/kubernetes.io/serviceaccount/token",
        ca_path: str = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt",
    ) -> None:
        self.namespace = namespace
        self.host = host.rstrip("/")
        self.token_path = token_path
        self.ca_path = ca_path

    def _request(self, method: str, path: str, **kwargs: object) -> httpx.Response:
        token = Path(self.token_path).read_text(encoding="utf-8").strip()
        headers = {"Authorization": f"Bearer {token}"}
        extra = kwargs.pop("headers", None)
        if isinstance(extra, dict):
            headers.update(extra)
        with httpx.Client(verify=self.ca_path, timeout=10, trust_env=False) as client:
            return client.request(method, self.host + path, headers=headers, **kwargs)

    def get_lease(self, name: str) -> dict | None:
        path = f"/apis/coordination.k8s.io/v1/namespaces/{self.namespace}/leases/{name}"
        response = self._request("GET", path)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def replace_lease(self, name: str, spec: dict, resource_version: str | None) -> int:
        path = f"/apis/coordination.k8s.io/v1/namespaces/{self.namespace}/leases/{name}"
        body = {
            "apiVersion": "coordination.k8s.io/v1",
            "kind": "Lease",
            "metadata": {"name": name, "namespace": self.namespace},
            "spec": spec,
        }
        if resource_version:
            body["metadata"]["resourceVersion"] = resource_version
            response = self._request("PUT", path, json=body)
        else:
            response = self._request("POST", f"/apis/coordination.k8s.io/v1/namespaces/{self.namespace}/leases", json=body)
            # A competing creator must be re-observed, never overwritten.
        if response.status_code in {200, 201}:
            return response.status_code
        return response.status_code

    def get_scale(self, name: str) -> int:
        path = f"/apis/apps/v1/namespaces/{self.namespace}/deployments/{name}/scale"
        response = self._request("GET", path)
        response.raise_for_status()
        return replicas_from_scale(response.json())

    def patch_scale(self, name: str, replicas: int) -> None:
        path = f"/apis/apps/v1/namespaces/{self.namespace}/deployments/{name}/scale"
        response = self._request(
            "PATCH",
            path,
            json={"spec": {"replicas": replicas}},
            headers={"Content-Type": "application/merge-patch+json"},
        )
        response.raise_for_status()


class GpuCoordinator:
    def __init__(
        self,
        api: object | None,
        role: str,
        holder: str,
        deployment: str | None = None,
        enabled: bool = False,
        idle_seconds: int = 600,
        queue_seconds: int = 1200,
        ready_seconds: int = 900,
        lease_seconds: int = 90,
        now=None,
        sleep=None,
        monotonic=None,
    ) -> None:
        self.api = api
        self.role = role
        self.holder = holder
        self.deployment = deployment or None
        self.enabled = enabled
        self.idle_seconds = idle_seconds
        self.queue_seconds = queue_seconds
        self.ready_seconds = ready_seconds
        self.lease_seconds = lease_seconds
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._sleep = sleep or time.sleep
        self._monotonic = monotonic or time.monotonic
        self.in_flight = 0
        self.idle_at: float | None = None
        self._lock = threading.Lock()

    @property
    def lease_name(self) -> str:
        return LEASES[self.role]

    @classmethod
    def from_environment(cls) -> "GpuCoordinator":
        role = os.environ.get("GPU_TASK_ROLE", "").strip()
        if role not in CONFLICTS:
            return cls(api=None, role="", holder="", enabled=False)
        namespace = os.environ.get("GPU_TASK_NAMESPACE", "").strip()
        namespace_file = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
        if not namespace and namespace_file.is_file():
            namespace = namespace_file.read_text(encoding="utf-8").strip()
        namespace = namespace or "verdantflare-app"
        holder = os.environ.get("GPU_TASK_HOLDER", "").strip() or f"{role}/{os.environ.get('HOSTNAME', 'local')}"
        deployment = os.environ.get("GPU_TASK_DEPLOYMENT", "").strip() or None
        return cls(
            api=K8sApi(namespace),
            role=role,
            holder=holder,
            deployment=deployment,
            enabled=True,
            idle_seconds=int(os.environ.get("GPU_IDLE_SECONDS", "600")),
            queue_seconds=int(os.environ.get("GPU_QUEUE_SECONDS", "1200")),
            ready_seconds=int(os.environ.get("GPU_READY_SECONDS", "900")),
            lease_seconds=int(os.environ.get("GPU_LEASE_SECONDS", "90")),
        )

    def peer_holder(self) -> str | None:
        if not self.enabled:
            return None
        now = self._now()
        for peer in CONFLICTS[self.role]:
            holder = held_by_other(self.api.get_lease(LEASES[peer]), self.holder, now)
            if holder:
                return holder
        return None

    def api_ok(self) -> bool:
        if not self.enabled:
            return True
        try:
            self.api.get_lease(self.lease_name)
            return True
        except Exception:
            return False

    def replicas(self) -> int | None:
        if not self.enabled or not self.deployment:
            return None
        try:
            return int(self.api.get_scale(self.deployment))
        except Exception:
            return None

    def scale_to(self, replicas: int) -> None:
        if not self.enabled or not self.deployment:
            return
        if self.api.get_scale(self.deployment) != replicas:
            self.api.patch_scale(self.deployment, replicas)

    def _spec(self, holder: str, acquire_time: str | None = None) -> dict:
        now = format_time(self._now())
        return {
            "holderIdentity": holder,
            "leaseDurationSeconds": self.lease_seconds,
            "acquireTime": acquire_time or now,
            "renewTime": now,
        }

    def _write(self, holder: str) -> bool:
        current = self.api.get_lease(self.lease_name)
        identity = ((current or {}).get("spec") or {}).get("holderIdentity")
        if not holder and identity != self.holder:
            return False
        if holder and held_by_other(current, self.holder, self._now()):
            return False
        version = None if current is None else current.get("metadata", {}).get("resourceVersion")
        acquire_time = None
        if current and (current.get("spec") or {}).get("holderIdentity") == self.holder:
            acquire_time = (current.get("spec") or {}).get("acquireTime")
        return self.api.replace_lease(self.lease_name, self._spec(holder, acquire_time), version) in {200, 201}

    def try_acquire(self) -> bool:
        if not self.enabled:
            return True
        with self._lock:
            now = self._now()
            current = self.api.get_lease(self.lease_name)
            if held_by_other(current, self.holder, now):
                return False
            if held_by_other(current, "", now) == self.holder:
                # Keep an already admitted role alive while a peer briefly
                # writes its own claim before observing this role.
                return self._write(self.holder)
            if not self._write(self.holder):
                return False
            if self.peer_holder():
                self._write("")
                return False
            return True

    def acquire_for_queue(self, *, deadline: float | None = None) -> None:
        if not self.enabled:
            return
        deadline = (self._monotonic() + self.queue_seconds) if deadline is None else deadline
        delay = 0.2
        while True:
            if self.try_acquire():
                with self._lock:
                    self.in_flight += 1
                    self.idle_at = None
                return
            if self._monotonic() >= deadline:
                raise LeaseBusy(self.peer_holder())
            delay = min(5.0, delay * 2)
            self._sleep(delay * (0.5 + random.random()))

    def renew(self) -> None:
        if not self.enabled:
            return
        self.try_acquire()

    def renew_existing(self) -> None:
        """Maintain a live lease without admitting a not-yet-started call."""
        if not self.enabled:
            return
        with self._lock:
            current = self.api.get_lease(self.lease_name)
            if held_by_other(current, "", self._now()) == self.holder:
                self._write(self.holder)

    def release(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            current = self.api.get_lease(self.lease_name)
            identity = ((current or {}).get("spec") or {}).get("holderIdentity")
            if identity in {None, "", self.holder}:
                self._write("")
            if self.in_flight:
                self.in_flight -= 1
            if self.in_flight == 0:
                self.idle_at = self._monotonic()

    def maybe_scale_down(self) -> None:
        if not self.enabled or not self.deployment or self.in_flight or self.idle_at is None:
            return
        if self._monotonic() - self.idle_at < self.idle_seconds:
            return
        current = self.api.get_lease(self.lease_name)
        if ((current or {}).get("spec") or {}).get("holderIdentity") == self.holder:
            return
        if self.replicas() not in {None, 0}:
            self.scale_to(0)

    def hold(self) -> "_Hold":
        return _Hold(self)


class _Hold:
    def __init__(self, coord: GpuCoordinator) -> None:
        self.coord = coord
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_Hold":
        self.coord.acquire_for_queue()
        if self.coord.enabled:
            self._thread = threading.Thread(target=self._renew, daemon=True)
            self._thread.start()
        return self

    def _renew(self) -> None:
        interval = max(5, self.coord.lease_seconds // 3)
        while not self._stop.wait(interval):
            try:
                self.coord.renew()
            except Exception:
                pass

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._stop.set()
        self.coord.release()
        return False


class LocalGpuStageSlot:
    """Single-slot local GPU stage gate in coordination.sqlite3 across P1, P2, and P3."""

    SLOT_ID = "local_gpu_5090"

    def __init__(
        self,
        tasks,
        *,
        enabled: bool = False,
        queue_seconds: int = 1200,
        sleep=None,
        monotonic=None,
    ) -> None:
        self.tasks = tasks
        self.enabled = enabled
        self.queue_seconds = queue_seconds
        self._sleep = sleep or time.sleep
        self._monotonic = monotonic or time.monotonic

    @classmethod
    def from_environment(cls, tasks) -> "LocalGpuStageSlot":
        role = os.environ.get("GPU_TASK_ROLE", "").strip()
        return cls(
            tasks,
            enabled=role in CONFLICTS,
            queue_seconds=int(os.environ.get("GPU_QUEUE_SECONDS", "1200")),
        )

    @contextmanager
    def _connection(self):
        self.tasks.ensure_ready()
        existing = getattr(self.tasks._local, "connection", None)
        db = existing or sqlite3.connect(self.tasks.db_path, timeout=30)
        try:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS local_gpu_stage_slot (
                    slot_id TEXT PRIMARY KEY,
                    holder_task_id TEXT NOT NULL,
                    service TEXT NOT NULL,
                    acquired_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            if existing is None:
                db.execute("BEGIN IMMEDIATE")
            yield db
            if existing is None:
                db.commit()
        except Exception:
            if existing is None:
                try:
                    db.rollback()
                except sqlite3.Error:
                    pass
            raise
        finally:
            if existing is None:
                db.close()

    @staticmethod
    def is_task_settled(record) -> bool:
        if record is None:
            return True
        if record.status not in {"succeeded", "failed", "cancelled", "canceled"}:
            return False
        if record.status == "failed" and (record.error or {}).get("code") in {
            "submission_unconfirmed",
            "runtime_task_lost",
        }:
            return False
        if (record.provider_state or {}).get("worker_settled") is False:
            return False
        return True

    def _holder_record(self, holder_task_id: str):
        if not holder_task_id:
            return None
        from .tasks import TaskNotFound
        try:
            return self.tasks.get(holder_task_id)
        except (TaskNotFound, ValueError):
            return None

    def current_holder(self) -> tuple[str, str] | None:
        if not self.enabled:
            return None
        with self._connection() as db:
            row = db.execute(
                "SELECT holder_task_id, service FROM local_gpu_stage_slot WHERE slot_id = ?",
                (self.SLOT_ID,),
            ).fetchone()
            if not row or not row[0]:
                return None
            record = self._holder_record(row[0])
            if self.is_task_settled(record):
                db.execute(
                    "UPDATE local_gpu_stage_slot SET holder_task_id = '', service = '', updated_at = ? WHERE slot_id = ?",
                    (time.time(), self.SLOT_ID),
                )
                return None
            return str(row[0]), str(row[1])

    def try_acquire(self, task_id: str, service: str) -> bool:
        if not self.enabled:
            return True
        now = time.time()
        with self._connection() as db:
            row = db.execute(
                "SELECT holder_task_id, service, acquired_at FROM local_gpu_stage_slot WHERE slot_id = ?",
                (self.SLOT_ID,),
            ).fetchone()
            if row and row[0] and row[0] != task_id:
                holder = self._holder_record(row[0])
                if not self.is_task_settled(holder):
                    return False
            acquired_at = float(row[2]) if row and row[0] == task_id else now
            db.execute(
                """
                INSERT INTO local_gpu_stage_slot (slot_id, holder_task_id, service, acquired_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(slot_id) DO UPDATE SET
                    holder_task_id = excluded.holder_task_id,
                    service = excluded.service,
                    acquired_at = excluded.acquired_at,
                    updated_at = excluded.updated_at
                """,
                (self.SLOT_ID, task_id, service, acquired_at, now),
            )
            return True

    def acquire_for_queue(
        self,
        task_id: str,
        service: str,
        *,
        deadline: float | None = None,
        reconcile_holder=None,
    ) -> None:
        if not self.enabled:
            return
        deadline = (self._monotonic() + self.queue_seconds) if deadline is None else deadline
        delay = 0.2
        while True:
            if self.try_acquire(task_id, service):
                return
            current = self.current_holder()
            if current and reconcile_holder is not None and current[0] != task_id:
                try:
                    reconcile_holder(current[0], current[1])
                except Exception:
                    pass
                if self.try_acquire(task_id, service):
                    return
            if self._monotonic() >= deadline:
                holder_name = current[0] if current else "local_gpu_stage_slot"
                raise LeaseBusy(holder_name)
            delay = min(2.0, delay * 1.5)
            self._sleep(delay * (0.5 + 0.5 * random.random()))

    def release(self, task_id: str | None = None, *, force: bool = False) -> None:
        if not self.enabled:
            return
        with self._connection() as db:
            row = db.execute(
                "SELECT holder_task_id FROM local_gpu_stage_slot WHERE slot_id = ?",
                (self.SLOT_ID,),
            ).fetchone()
            if not row or not row[0]:
                return
            if task_id is not None and row[0] != task_id:
                return
            if not force:
                holder = self._holder_record(row[0])
                if not self.is_task_settled(holder):
                    return
            db.execute(
                "UPDATE local_gpu_stage_slot SET holder_task_id = '', service = '', updated_at = ? WHERE slot_id = ?",
                (time.time(), self.SLOT_ID),
            )
