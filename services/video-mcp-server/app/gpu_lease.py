"""Shared GPU task lease and on-demand scale.

Approved 2026-10-19 for 5090 Music3, H3, and IndexTTS2. Write this role's
Lease first, then read peers. A live conflicting peer means release and retry.
H3 and IndexTTS2 do not exclude each other. Music3 and IndexTTS2 scale to 1
while held and back to 0 after the idle window. H3 stays resident.
"""

from __future__ import annotations

import json
import logging
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
    "moss-image": "gpu-task-moss-image",
}
CONFLICTS = {
    "music3": ("h3", "indextts2", "moss-image"),
    "h3": ("music3",),
    "indextts2": ("music3",),
    "moss-image": ("music3",),
}

SCHEDULER_CONSUMER_SPECS: dict[str, dict[str, object]] = {
    "h3": {"consumer_kind": "h3", "stage_id": "generation", "reclaim_mode": "epoch_unload"},
    "h3-vdn": {"consumer_kind": "h3", "stage_id": "generation", "reclaim_mode": "epoch_unload"},
    "h3-sol": {"consumer_kind": "h3", "stage_id": "generation", "reclaim_mode": "epoch_unload"},
    "h3-fasth3": {"consumer_kind": "h3", "stage_id": "generation", "reclaim_mode": "epoch_unload"},
    "h3-singularity": {"consumer_kind": "h3", "stage_id": "generation", "reclaim_mode": "epoch_unload"},
    "regenerate_2k": {"consumer_kind": "video.regenerate_2k", "stage_id": "regenerate_2k", "reclaim_mode": "epoch_unload"},
    "h3-latent-upscale": {"consumer_kind": "video.regenerate_2k", "stage_id": "regenerate_2k", "reclaim_mode": "epoch_unload"},
    "interpolate": {"consumer_kind": "rife", "stage_id": "postprocess_rife", "reclaim_mode": "warm_idle_settled"},
    "sr": {"consumer_kind": "sr", "stage_id": "sr", "reclaim_mode": "warm_idle_settled"},
    "depth": {"consumer_kind": "depth", "stage_id": "depth", "reclaim_mode": "warm_idle_settled"},
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


SCHEDULER_GRANTED_STATES = frozenset({"leased", "running", "granted"})


def is_lease_granted(lease_doc: dict | None) -> bool:
    return isinstance(lease_doc, dict) and str(lease_doc.get("state") or "") in SCHEDULER_GRANTED_STATES


class SchedulerError(RuntimeError):
    def __init__(self, code: str, message: str = "", status_code: int = 500) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message or code)


def _fsync_directory(dir_path: Path) -> None:
    if os.name == "nt":
        return
    dfd = os.open(str(dir_path), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


class SchedulerLeaseClient:
    """Station Runtime :5056 GPU scheduler lease client (acquire, heartbeat, stage handoff, downgrade, release)."""

    def __init__(
        self,
        *,
        base_url: str = "",
        token: str = "",
        holder_app_id: str = "video-mcp",
        enabled: bool = False,
        ttl_seconds: int = 90,
        client: httpx.Client | None = None,
        pending_releases_path: str | Path | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.holder_app_id = holder_app_id
        self.enabled = bool(enabled and self.base_url)
        self.ttl_seconds = ttl_seconds
        self._client = client
        self._journal_lock = threading.RLock()
        self._version_lock = threading.RLock()
        self._observed_versions: dict[tuple[str, int], int] = {}
        raw_journal = (
            str(pending_releases_path).strip()
            if pending_releases_path is not None
            else os.environ.get("STATION_GPU_PENDING_RELEASES_PATH", "").strip()
        )
        self._pending_releases_path: Path | None = Path(raw_journal) if raw_journal else None
        self._pending_releases: dict[str, dict[str, object]] = {}
        self._load_pending_releases_locked()

    @classmethod
    def from_environment(
        cls,
        *,
        holder_app_id: str = "video-mcp",
        pending_releases_path: str | Path | None = None,
    ) -> "SchedulerLeaseClient":
        url = os.environ.get("STATION_GPU_SCHEDULER_URL", "").strip()
        token = os.environ.get("STATION_GPU_SCHEDULER_TOKEN", "").strip()
        env_journal = os.environ.get("STATION_GPU_PENDING_RELEASES_PATH", "").strip()
        resolved_journal = env_journal or pending_releases_path
        return cls(
            base_url=url,
            token=token,
            holder_app_id=os.environ.get("STATION_GPU_HOLDER_APP_ID", holder_app_id).strip() or holder_app_id,
            enabled=bool(url and token),
            ttl_seconds=int(os.environ.get("GPU_LEASE_SECONDS", "90")),
            pending_releases_path=resolved_journal,
        )

    def configure_journal(self, path: str | Path | None) -> None:
        if path is None:
            return
        with self._journal_lock:
            if self._pending_releases_path is not None:
                return
            self._pending_releases_path = Path(path)
            existing_mem = dict(self._pending_releases)
            self._load_pending_releases_locked()
            if existing_mem:
                self._pending_releases.update(existing_mem)
                self._persist_pending_releases_locked()

    def _load_pending_releases_locked(self) -> None:
        if self._pending_releases_path is None or not os.path.lexists(self._pending_releases_path):
            return
        if os.path.islink(self._pending_releases_path) or not self._pending_releases_path.is_file():
            raise SchedulerError("JOURNAL_CORRUPTED", "pending releases journal is not a regular file")
        try:
            payload = json.loads(self._pending_releases_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SchedulerError("JOURNAL_CORRUPTED", f"failed reading pending releases journal: {exc}") from exc
        if not isinstance(payload, dict):
            raise SchedulerError("JOURNAL_CORRUPTED", "pending releases journal root must be an object")
        loaded: dict[str, dict[str, object]] = {}
        for k, v in payload.items():
            if not isinstance(k, str) or not k.strip() or not isinstance(v, dict):
                raise SchedulerError("JOURNAL_CORRUPTED", "invalid pending release journal entry")
            loaded[k.strip()] = dict(v)
        self._pending_releases = loaded

    def _persist_pending_releases_locked(self) -> None:
        if self._pending_releases_path is None:
            return
        self._pending_releases_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._pending_releases_path.with_name(
            f"{self._pending_releases_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        data = json.dumps(self._pending_releases, separators=(",", ":"), sort_keys=True).encode("utf-8")
        try:
            fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                view = memoryview(data)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError("short_write_zero_bytes")
                    view = view[written:]
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(str(tmp_path), str(self._pending_releases_path))
            _fsync_directory(self._pending_releases_path.parent)
        except OSError as exc:
            try:
                if os.path.lexists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass
            raise SchedulerError("JOURNAL_PERSIST_FAILED", str(exc), status_code=503) from exc

    @staticmethod
    def _is_confirmed_release_receipt(lease_id: str, data: object) -> bool:
        if not isinstance(data, dict):
            return False
        if str(data.get("lease_id") or "").strip() != lease_id:
            return False
        return str(data.get("state") or "").strip().lower() in {"released", "expired"}

    @staticmethod
    def _is_concrete_identity_value(value: object) -> bool:
        if not isinstance(value, str):
            return False
        s = value.strip().lower()
        if not s or s in {"unknown", "none", "null", "n/a"}:
            return False
        return not (s.startswith("unknown-") or s.startswith("unknown_") or s.startswith("local-"))

    def _request(self, method: str, path: str, *, json_body: dict | None = None) -> tuple[int, dict]:
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            if self._client is not None:
                resp = self._client.request(method, self.base_url + path, headers=headers, json=json_body)
            else:
                with httpx.Client(timeout=15.0, trust_env=False) as client:
                    resp = client.request(method, self.base_url + path, headers=headers, json=json_body)
        except SchedulerError:
            raise
        except Exception as exc:
            raise SchedulerError("TRANSPORT_ERROR", str(exc), status_code=503) from exc
        try:
            data = resp.json() if resp.content else {}
        except ValueError:
            data = {}
        if resp.status_code >= 400:
            code = ""
            msg = ""
            if isinstance(data, dict):
                if isinstance(data.get("error_code"), str) and data["error_code"]:
                    code = data["error_code"]
                if isinstance(data.get("message"), str) and data["message"]:
                    msg = data["message"]
                err = data.get("error")
                if isinstance(err, dict):
                    if not code and isinstance(err.get("code"), str):
                        code = err["code"]
                    if not msg and isinstance(err.get("message"), str):
                        msg = err["message"]
            raise SchedulerError(
                code or f"HTTP_{resp.status_code}",
                msg or resp.text,
                status_code=resp.status_code,
            )
        return resp.status_code, data if isinstance(data, dict) else {}

    def get_pending_release(self, lease_id: str) -> dict[str, object] | None:
        with self._journal_lock:
            entry = self._pending_releases.get(lease_id)
            return dict(entry) if isinstance(entry, dict) else None

    def flush_pending_releases(self, *, include_quarantined: bool = False) -> int:
        if not self.enabled:
            return 0
        with self._journal_lock:
            if not self._pending_releases:
                return 0
            snapshot = [
                (lid, dict(b))
                for lid, b in self._pending_releases.items()
                if include_quarantined or not bool(b.get("_quarantined_400"))
            ]
        if not snapshot:
            return 0
        flushed = 0
        for lease_id, body in snapshot:
            retry_body = {
                k: v
                for k, v in body.items()
                if k != "expected_version" and not str(k).startswith("_")
            }
            try:
                _, data = self._request(
                    "POST",
                    f"/internal/v1/gpu-scheduler/leases/{lease_id}/release",
                    json_body=retry_body,
                )
                if self._is_confirmed_release_receipt(lease_id, data):
                    self._record_lease_decision(data)
                    with self._journal_lock:
                        self._pending_releases.pop(lease_id, None)
                        self._persist_pending_releases_locked()
                    flushed += 1
            except SchedulerError as err:
                if err.status_code == 400:
                    logging.error(
                        "video_gpu_release_quarantined_400 lease_id=%s code=%s message=%s",
                        lease_id,
                        err.code,
                        err.message,
                    )
                    with self._journal_lock:
                        cur = self._pending_releases.get(lease_id)
                        if isinstance(cur, dict):
                            upd = dict(cur)
                            upd["_quarantined_400"] = True
                            upd["_last_error"] = f"{err.code}:{err.message}"
                            self._pending_releases[lease_id] = upd
                            self._persist_pending_releases_locked()
                continue
            except Exception:
                continue
        return flushed

    def acquire(
        self,
        task_id: str,
        service: str,
        *,
        attempt_id: str,
        stage_id: str | None = None,
        runtime_identity: dict | None = None,
        bind_identity: bool = True,
        k8s_slots_required: int = 0,
        idempotency_key: str | None = None,
    ) -> dict:
        self.flush_pending_releases()
        spec = SCHEDULER_CONSUMER_SPECS.get(service, SCHEDULER_CONSUMER_SPECS["h3"])
        resolved_stage = stage_id or str(spec["stage_id"])
        ident = runtime_identity or {}
        body: dict[str, object] = {
            "idempotency_key": idempotency_key or f"acq:{task_id}:{attempt_id}:{resolved_stage}",
            "holder_app_id": self.holder_app_id,
            "consumer_kind": str(spec["consumer_kind"]),
            "task_id": task_id,
            "attempt_id": attempt_id,
            "stage_id": resolved_stage,
            "ttl_seconds": self.ttl_seconds,
        }
        if k8s_slots_required > 0:
            body["k8s_slots_required"] = int(k8s_slots_required)
        if bind_identity:
            for key in ("pod_uid", "container_id", "boot_id"):
                val = ident.get(key)
                if self._is_concrete_identity_value(val):
                    body[key] = str(val).strip()
        visible = ident.get("visible_gpu_uuids")
        if isinstance(visible, list) and visible:
            body["candidate_gpu_uuids"] = [str(u) for u in visible if str(u)]
        _, data = self._request("POST", "/internal/v1/gpu-scheduler/leases/acquire", json_body=body)
        return self._record_lease_decision(data)

    def _record_lease_decision(self, data: dict) -> dict:
        if not isinstance(data, dict) or not data.get("lease_id"):
            return data
        lid = str(data["lease_id"]).strip()
        tok = int(data.get("fencing_token") or 0)
        ver = int(data.get("version") or 0)
        state = str(data.get("state") or "").strip().lower()
        with self._version_lock:
            if state in {"released", "expired"}:
                stale_keys = [k for k in self._observed_versions if k[0] == lid]
                for k in stale_keys:
                    self._observed_versions.pop(k, None)
            elif lid and tok > 0 and ver > 0:
                prev = self._observed_versions.get((lid, tok), 0)
                if ver > prev:
                    self._observed_versions[(lid, tok)] = ver
        return data

    def resolve_expected_version(self, lease_id: str, fencing_token: int, supplied_version: int) -> int:
        lid = str(lease_id or "").strip()
        tok = int(fencing_token or 0)
        sup = int(supplied_version or 0)
        if not lid or tok <= 0:
            return sup
        with self._version_lock:
            obs = self._observed_versions.get((lid, tok), 0)
        return max(sup, obs)

    def bind_target(
        self,
        lease_id: str,
        *,
        fencing_token: int,
        target_identity: dict,
        expected_version: int = 0,
        idempotency_key: str | None = None,
    ) -> dict:
        pod_uid = str(target_identity.get("pod_uid") or "").strip()
        container_id = str(target_identity.get("container_id") or "").strip()
        boot_id = str(target_identity.get("boot_id") or "").strip()
        visible = target_identity.get("visible_gpu_uuids")
        actual_gpus = [str(u) for u in visible if str(u)] if isinstance(visible, list) else []
        body: dict[str, object] = {
            "idempotency_key": idempotency_key or f"bind:{lease_id}:{fencing_token}:{pod_uid}",
            "fencing_token": int(fencing_token),
            "pod_uid": pod_uid,
            "container_id": container_id,
            "boot_id": boot_id,
            "actual_gpu_uuids": actual_gpus,
            "ttl_seconds": self.ttl_seconds,
        }
        eff_version = self.resolve_expected_version(lease_id, int(fencing_token), int(expected_version))
        if eff_version > 0:
            body["expected_version"] = eff_version
        _, data = self._request("POST", f"/internal/v1/gpu-scheduler/leases/{lease_id}/bind-target", json_body=body)
        return self._record_lease_decision(data)

    def get_lease(self, lease_id: str) -> dict:
        _, data = self._request("GET", f"/internal/v1/gpu-scheduler/leases/{lease_id}")
        return data

    def heartbeat(
        self,
        lease_id: str,
        *,
        fencing_token: int,
        expected_version: int = 0,
        idempotency_key: str | None = None,
    ) -> dict:
        body: dict[str, object] = {
            "idempotency_key": idempotency_key or f"hb:{lease_id}:{fencing_token}:{int(time.time() // 15)}",
            "fencing_token": int(fencing_token),
            "ttl_seconds": self.ttl_seconds,
        }
        eff_version = self.resolve_expected_version(lease_id, int(fencing_token), int(expected_version))
        if eff_version > 0:
            body["expected_version"] = eff_version
        _, data = self._request("POST", f"/internal/v1/gpu-scheduler/leases/{lease_id}/heartbeat", json_body=body)
        return self._record_lease_decision(data)

    def stage_handoff(
        self,
        lease_id: str,
        *,
        fencing_token: int,
        new_stage_id: str,
        reclaim_mode: str,
        reclaim_proof: dict,
        current_identity: dict | None = None,
        next_identity: dict | None = None,
        gpu_epoch: int = 0,
        expected_version: int = 0,
        idempotency_key: str | None = None,
    ) -> dict:
        cur = current_identity or {}
        nxt = next_identity or {}
        body: dict[str, object] = {
            "idempotency_key": idempotency_key or f"handoff:{lease_id}:{new_stage_id}:{fencing_token}",
            "fencing_token": int(fencing_token),
            "ttl_seconds": self.ttl_seconds,
            "new_stage_id": new_stage_id,
            "worker_settled": True,
            "reclaim_mode": reclaim_mode,
            "reclaim_proof": reclaim_proof,
        }
        eff_version = self.resolve_expected_version(lease_id, int(fencing_token), int(expected_version))
        if eff_version > 0:
            body["expected_version"] = eff_version
        if gpu_epoch > 0:
            body["gpu_epoch"] = int(gpu_epoch)
        if isinstance(cur.get("supervisor_pid"), int) and cur["supervisor_pid"] > 0:
            body["supervisor_pid"] = int(cur["supervisor_pid"])
        if isinstance(cur.get("supervisor_starttime_ticks"), int) and cur["supervisor_starttime_ticks"] > 0:
            body["supervisor_starttime_ticks"] = int(cur["supervisor_starttime_ticks"])
        for key in ("pod_uid", "container_id", "boot_id"):
            val = cur.get(key)
            if isinstance(val, str) and val and not val.startswith("unknown-"):
                body[key] = val
        for src_key, dst_key in (
            ("pod_uid", "next_pod_uid"),
            ("container_id", "next_container_id"),
            ("boot_id", "next_boot_id"),
        ):
            val = nxt.get(src_key)
            if isinstance(val, str) and val and not val.startswith("unknown-"):
                body[dst_key] = val
        _, data = self._request("POST", f"/internal/v1/gpu-scheduler/leases/{lease_id}/heartbeat", json_body=body)
        return self._record_lease_decision(data)

    def downgrade(
        self,
        lease_id: str,
        *,
        fencing_token: int,
        downgrade_to_mib: int = 4096,
        new_stage_id: str = "postprocess_rife",
        reclaim_mode: str = "epoch_unload",
        reclaim_proof: dict,
        current_identity: dict | None = None,
        next_identity: dict | None = None,
        gpu_epoch: int = 0,
        expected_version: int = 0,
        idempotency_key: str | None = None,
    ) -> dict:
        cur = current_identity or {}
        nxt = next_identity or {}
        body: dict[str, object] = {
            "idempotency_key": idempotency_key or f"downgrade:{lease_id}:{new_stage_id}:{fencing_token}",
            "fencing_token": int(fencing_token),
            "ttl_seconds": self.ttl_seconds,
            "downgrade_to_mib": int(downgrade_to_mib),
            "new_stage_id": new_stage_id,
            "worker_settled": True,
            "reclaim_mode": reclaim_mode,
            "reclaim_proof": reclaim_proof,
        }
        eff_version = self.resolve_expected_version(lease_id, int(fencing_token), int(expected_version))
        if eff_version > 0:
            body["expected_version"] = eff_version
        if gpu_epoch > 0:
            body["gpu_epoch"] = int(gpu_epoch)
        if isinstance(cur.get("supervisor_pid"), int) and cur["supervisor_pid"] > 0:
            body["supervisor_pid"] = int(cur["supervisor_pid"])
        if isinstance(cur.get("supervisor_starttime_ticks"), int) and cur["supervisor_starttime_ticks"] > 0:
            body["supervisor_starttime_ticks"] = int(cur["supervisor_starttime_ticks"])
        for key in ("pod_uid", "container_id", "boot_id"):
            val = cur.get(key)
            if isinstance(val, str) and val and not val.startswith("unknown-"):
                body[key] = val
        for src_key, dst_key in (
            ("pod_uid", "next_pod_uid"),
            ("container_id", "next_container_id"),
            ("boot_id", "next_boot_id"),
        ):
            val = nxt.get(src_key)
            if isinstance(val, str) and val and not val.startswith("unknown-"):
                body[dst_key] = val
        try:
            _, data = self._request("POST", f"/internal/v1/gpu-scheduler/leases/{lease_id}/downgrade", json_body=body)
            return self._record_lease_decision(data)
        except SchedulerError as exc:
            if exc.status_code == 409 and exc.code == "LOCK_DOWNGRADE_DISABLED_BEFORE_CUTOVER":
                fallback_idem = (
                    f"handoff-fallback:{idempotency_key}"
                    if idempotency_key
                    else f"handoff-fallback:{lease_id}:{new_stage_id}:{fencing_token}"
                )
                fallback = self.stage_handoff(
                    lease_id,
                    fencing_token=fencing_token,
                    new_stage_id=new_stage_id,
                    reclaim_mode=reclaim_mode,
                    reclaim_proof=reclaim_proof,
                    current_identity=current_identity,
                    next_identity=next_identity,
                    gpu_epoch=gpu_epoch,
                    expected_version=eff_version,
                    idempotency_key=fallback_idem,
                )
                fallback["downgrade_skipped_before_cutover"] = True
                return fallback
            raise

    def release(
        self,
        lease_id: str,
        *,
        fencing_token: int,
        worker_settled: bool = True,
        reclaim_mode: str = "",
        reclaim_proof: dict | None = None,
        current_identity: dict | None = None,
        gpu_epoch: int = 0,
        expected_version: int = 0,
        idempotency_key: str | None = None,
    ) -> dict:
        cur = current_identity or {}
        body: dict[str, object] = {
            "idempotency_key": idempotency_key or f"rel:{lease_id}:{fencing_token}",
            "fencing_token": int(fencing_token),
            "worker_settled": bool(worker_settled),
        }
        eff_version = self.resolve_expected_version(lease_id, int(fencing_token), int(expected_version))
        if eff_version > 0:
            body["expected_version"] = eff_version
        if reclaim_mode:
            body["reclaim_mode"] = reclaim_mode
        if reclaim_proof is not None:
            body["reclaim_proof"] = reclaim_proof
        if gpu_epoch > 0:
            body["gpu_epoch"] = int(gpu_epoch)
        if isinstance(cur.get("supervisor_pid"), int) and cur["supervisor_pid"] > 0:
            body["supervisor_pid"] = int(cur["supervisor_pid"])
        if isinstance(cur.get("supervisor_starttime_ticks"), int) and cur["supervisor_starttime_ticks"] > 0:
            body["supervisor_starttime_ticks"] = int(cur["supervisor_starttime_ticks"])
        for key in ("pod_uid", "container_id", "boot_id"):
            val = cur.get(key)
            if self._is_concrete_identity_value(val):
                body[key] = str(val).strip()
        retry_body = {k: v for k, v in body.items() if k != "expected_version"}
        with self._journal_lock:
            self._pending_releases[lease_id] = retry_body
            self._persist_pending_releases_locked()
        try:
            _, data = self._request("POST", f"/internal/v1/gpu-scheduler/leases/{lease_id}/release", json_body=body)
        except SchedulerError as err:
            if err.status_code == 400:
                logging.error(
                    "video_gpu_release_quarantined_400 lease_id=%s code=%s message=%s",
                    lease_id,
                    err.code,
                    err.message,
                )
                with self._journal_lock:
                    cur = self._pending_releases.get(lease_id)
                    if isinstance(cur, dict):
                        upd = dict(cur)
                        upd["_quarantined_400"] = True
                        upd["_last_error"] = f"{err.code}:{err.message}"
                        self._pending_releases[lease_id] = upd
                        self._persist_pending_releases_locked()
            raise
        if not self._is_confirmed_release_receipt(lease_id, data):
            raise SchedulerError(
                "LEASE_NOT_RELEASED",
                f"lease {lease_id} release receipt unconfirmed (lease_id={data.get('lease_id')!r}, state={data.get('state')!r})",
                status_code=409,
            )
        with self._journal_lock:
            self._pending_releases.pop(lease_id, None)
            self._persist_pending_releases_locked()
        return self._record_lease_decision(data)

    @classmethod
    def build_grant_headers(
        cls,
        lease_doc: dict | None,
        *,
        parent_task_id: str = "",
        stage_id: str = "",
        target_identity: dict | None = None,
    ) -> dict[str, str]:
        if not isinstance(lease_doc, dict) or not lease_doc.get("lease_id"):
            return {}
        ident = target_identity or {}
        headers: dict[str, str] = {
            "X-VF-GPU-Lease-ID": str(lease_doc["lease_id"]),
            "X-VF-GPU-Fencing-Token": str(lease_doc.get("fencing_token") or 0),
            "X-VF-GPU-Attempt-ID": str(lease_doc.get("attempt_id") or "att-1"),
            "X-VF-GPU-Stage-ID": str(stage_id or lease_doc.get("stage_id") or "default"),
        }
        if parent_task_id:
            headers["X-VF-GPU-Parent-Task-ID"] = parent_task_id
        pod_uid = str(ident.get("pod_uid") or lease_doc.get("pod_uid") or "").strip()
        container_id = str(ident.get("container_id") or lease_doc.get("container_id") or "").strip()
        boot_id = str(ident.get("boot_id") or lease_doc.get("boot_id") or "").strip()
        if isinstance(lease_doc.get("gpu_uuids"), list) and lease_doc["gpu_uuids"]:
            gpu_uuid = ",".join(str(u) for u in lease_doc["gpu_uuids"] if str(u))
        else:
            gpu_uuid = str(lease_doc.get("gpu_uuid") or "").strip()
        expires_at = str(lease_doc.get("expires_at") or "").strip()
        if cls._is_concrete_identity_value(pod_uid):
            headers["X-VF-GPU-Pod-UID"] = pod_uid
        if cls._is_concrete_identity_value(container_id):
            headers["X-VF-GPU-Container-ID"] = container_id
        if cls._is_concrete_identity_value(boot_id):
            headers["X-VF-GPU-Boot-ID"] = boot_id
        if gpu_uuid:
            headers["X-VF-GPU-UUID"] = gpu_uuid
        if expires_at:
            headers["X-VF-GPU-Expires-At"] = expires_at
        return headers
