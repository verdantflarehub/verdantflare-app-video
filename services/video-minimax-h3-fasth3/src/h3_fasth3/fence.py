"""Zero-CUDA supervisor subreaper, shm cleanup, and 6-tuple epoch lease fence."""

from __future__ import annotations

import ctypes
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

from .fence_scope import ScopedFenceState, gpu_scope


PR_SET_CHILD_SUBREAPER = 36


class StaleFenceError(RuntimeError):
    pass


class FencePersistenceError(RuntimeError):
    pass


def normalize_container_id(raw: str) -> str:
    val = (raw or "").strip()
    for prefix in ("containerd://", "docker://", "cri-o://"):
        if val.startswith(prefix):
            val = val[len(prefix) :].strip()
    return val


def is_unknown_identity_value(raw: str) -> bool:
    val = (raw or "").strip().lower()
    return (not val) or val.startswith("unknown") or val.startswith("local-")


def enable_child_subreaper() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        rc = libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
        return rc == 0
    except OSError:
        return False


def reap_adopted_children() -> int:
    if os.name == "nt" or not hasattr(os, "waitpid") or not hasattr(os, "WNOHANG"):
        return 0
    reaped = 0
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
            if pid <= 0:
                break
            reaped += 1
        except ChildProcessError:
            break
        except OSError:
            break
    return reaped


def cleanup_task_shm(task_id: str) -> int:
    if not task_id or "/" in task_id or "\\" in task_id or ".." in task_id:
        return 0
    shm_root = Path(os.environ.get("VF_DEV_SHM_ROOT", "/dev/shm"))
    if not shm_root.is_dir():
        return 0
    prefix = f"vf_{task_id}"
    removed = 0
    try:
        for entry in shm_root.iterdir():
            if not entry.name.startswith(prefix):
                continue
            try:
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink(missing_ok=True)
                removed += 1
            except OSError:
                continue
    except OSError:
        return removed
    return removed


def _read_proc_starttime_ticks() -> int:
    override = os.environ.get("VF_SUPERVISOR_STARTTIME_TICKS", "").strip()
    if override.isdigit() and int(override) > 0:
        return int(override)
    try:
        stat_raw = Path("/proc/self/stat").read_text(encoding="utf-8")
        rparen = stat_raw.rfind(")")
        if rparen > 0:
            fields = stat_raw[rparen + 2 :].split()
            if len(fields) > 19 and fields[19].isdigit():
                return int(fields[19])
    except OSError:
        pass
    return 1


def _detect_pod_and_container() -> tuple[str, str]:
    pod_uid = os.environ.get("POD_UID", "").strip()
    container_id = os.environ.get("CONTAINER_ID", "").strip()
    if pod_uid and container_id:
        return pod_uid, container_id
    try:
        mountinfo = Path("/proc/self/mountinfo").read_text(encoding="utf-8", errors="replace")
        if not pod_uid:
            m_pod = re.search(r"/pods/([0-9a-fA-F_-]{8,64})/", mountinfo)
            if m_pod:
                pod_uid = m_pod.group(1).replace("_", "-")
        if not container_id:
            m_cid = re.search(r"/containers/([0-9a-f]{64})/", mountinfo)
            if m_cid:
                container_id = m_cid.group(1)
    except OSError:
        pass
    if not container_id:
        try:
            cgroup_text = Path("/proc/self/cgroup").read_text(encoding="utf-8", errors="replace")
            m_cg = re.search(r"([0-9a-f]{64})", cgroup_text)
            if m_cg:
                container_id = m_cg.group(1)
        except OSError:
            pass
    return pod_uid or "unknown-pod-uid", container_id or "unknown-container-id"


def _detect_boot_id() -> str:
    env_boot = os.environ.get("NODE_BOOT_ID", "").strip()
    if env_boot:
        return env_boot
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown-boot-id"


def _detect_visible_gpu_uuids() -> list[str]:
    override = os.environ.get("VF_VISIBLE_GPU_UUIDS", "").strip()
    if override:
        return [item.strip() for item in override.split(",") if item.strip()]
    try:
        values = [
            entry.split(b"=", 1)[1].decode("ascii")
            for entry in Path("/proc/self/environ").read_bytes().split(b"\0")
            if entry.startswith(b"NVIDIA_VISIBLE_DEVICES=")
        ]
        concrete = [v for v in values if re.fullmatch(r"GPU-[0-9a-f-]{36}", v)]
        if concrete:
            return concrete
    except (OSError, UnicodeError):
        pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()
        return [line.strip() for line in out if re.fullmatch(r"GPU-[0-9a-f-]{36}", line.strip())]
    except (OSError, subprocess.SubprocessError):
        return []


def read_runtime_identity() -> dict[str, object]:
    pod_uid, container_id = _detect_pod_and_container()
    return {
        "pod_uid": pod_uid,
        "container_id": container_id,
        "boot_id": _detect_boot_id(),
        "supervisor_pid": os.getpid(),
        "supervisor_starttime_ticks": _read_proc_starttime_ticks(),
        "visible_gpu_uuids": _detect_visible_gpu_uuids(),
    }


def _parse_expires_epoch(raw: str) -> float | None:
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _fsync_directory(dir_path: Path) -> None:
    if os.name == "nt":
        return
    dfd = os.open(str(dir_path), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


class EpochLeaseFenceGuard:
    """Supervisor 6-tuple GPU grant fence, fail-closed durable token sealing, and monotonic gpu_epoch proof builder."""

    def __init__(
        self,
        *,
        policy_env: str = "VIDEO_GPU_GRANT_POLICY",
        state_path: str | Path | None = None,
        write_fn=None,
        fsync_fn=None,
        replace_fn=None,
        fsync_dir_fn=None,
        legacy_lease_records: list[dict[str, object]] | None = None,
    ) -> None:
        self._policy_env = policy_env
        raw_state = str(state_path).strip() if state_path is not None else os.environ.get("VF_GPU_GUARD_STATE_PATH", "").strip()
        self._state_path = Path(raw_state) if raw_state else None
        self._write_fn = write_fn or os.write
        self._fsync_fn = fsync_fn or os.fsync
        self._replace_fn = replace_fn or os.replace
        self._fsync_dir_fn = fsync_dir_fn or _fsync_directory
        self._lock = threading.RLock()
        self._corrupted = False
        self._corrupt_reason = ""
        self._min_fencing_token = 0
        self._gpu_epoch = 0
        self._consumed_tokens: set[int] = set()
        self._revoked_leases: set[str] = set()
        self._key_grants: dict[str, dict[str, object]] = {}
        self._task_grants: dict[str, dict[str, object]] = {}
        self._task_epochs: dict[str, int] = {}
        self._attempt_index: dict[tuple[str, str, str], tuple[str, str]] = {}
        self._inflight_tasks: dict[str, dict[str, object]] = {}
        self._executing_tasks: set[str] = set()
        self._scope_state = ScopedFenceState()
        self._legacy_lease_records = legacy_lease_records
        self._load_state_locked()

    def _ensure_healthy_locked(self) -> None:
        if self._corrupted:
            raise FencePersistenceError(self._corrupt_reason or "fence_state_corrupted")

    def _load_state_locked(self) -> None:
        if self._state_path is None or not os.path.lexists(self._state_path):
            return
        try:
            if os.path.islink(self._state_path) or not self._state_path.is_file():
                raise ValueError("state_path_not_regular_file")
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("state_payload_not_dict")
            raw_min = payload.get("min_fencing_token", 0)
            if isinstance(raw_min, bool) or not isinstance(raw_min, int) or raw_min < 0:
                raise ValueError("invalid_min_fencing_token")
            raw_epoch = payload.get("gpu_epoch", 0)
            if isinstance(raw_epoch, bool) or not isinstance(raw_epoch, int) or raw_epoch < 0:
                raise ValueError("invalid_gpu_epoch")
            raw_consumed = payload.get("consumed_tokens", [])
            if not isinstance(raw_consumed, list):
                raise ValueError("invalid_consumed_tokens")
            consumed: set[int] = set()
            for x in raw_consumed:
                if isinstance(x, bool) or not isinstance(x, int) or x <= 0:
                    raise ValueError("invalid_consumed_token_entry")
                consumed.add(x)
            raw_revoked = payload.get("revoked_leases", [])
            if not isinstance(raw_revoked, list):
                raise ValueError("invalid_revoked_leases")
            revoked: set[str] = set()
            for x in raw_revoked:
                if not isinstance(x, str) or not x.strip():
                    raise ValueError("invalid_revoked_lease_entry")
                revoked.add(x.strip())
            key_grants: dict[str, dict[str, object]] = {}
            task_grants: dict[str, dict[str, object]] = {}
            for src, dst in (("key_grants", key_grants), ("task_grants", task_grants)):
                raw_g = payload.get(src, {})
                if not isinstance(raw_g, dict):
                    raise ValueError(f"invalid_{src}")
                for k, v in raw_g.items():
                    if not isinstance(k, str) or not isinstance(v, dict):
                        raise ValueError(f"invalid_{src}_entry")
                    dst[k] = dict(v)
            raw_epochs = payload.get("task_epochs", {})
            if not isinstance(raw_epochs, dict):
                raise ValueError("invalid_task_epochs")
            task_epochs: dict[str, int] = {}
            for k, v in raw_epochs.items():
                if not isinstance(k, str) or isinstance(v, bool) or not isinstance(v, int) or v < 0:
                    raise ValueError("invalid_task_epoch_entry")
                task_epochs[k] = v
            raw_index = payload.get("attempt_index", {})
            if not isinstance(raw_index, dict):
                raise ValueError("invalid_attempt_index")
            index: dict[tuple[str, str, str], tuple[str, str]] = {}
            for k, v in raw_index.items():
                parts = str(k).split("\x1f")
                if len(parts) != 3 or not isinstance(v, list) or len(v) != 2 or not all(isinstance(s, str) for s in v):
                    raise ValueError("invalid_attempt_index_entry")
                index[(parts[0], parts[1], parts[2])] = (v[0], v[1])
            raw_inflight = payload.get("inflight_tasks", {})
            if not isinstance(raw_inflight, dict):
                raise ValueError("invalid_inflight_tasks")
            scopes = ScopedFenceState.load(
                payload, grants=[*key_grants.values(), *task_grants.values()],
                revoked=revoked, consumed=consumed, minimum=raw_min,
                legacy_records=self._legacy_lease_records,
            )
            cur_pid = os.getpid()
            cur_ticks = _read_proc_starttime_ticks()
            inflight: dict[str, dict[str, object]] = {}
            needs_repersist = payload.get("schema_version", 1) == 1
            for tid, meta in raw_inflight.items():
                if not isinstance(tid, str) or not isinstance(meta, dict):
                    raise ValueError("invalid_inflight_entry")
                owner_pid = int(meta.get("owner_pid") or 0)
                owner_ticks = int(meta.get("owner_ticks") or 0)
                if owner_pid == cur_pid and owner_ticks == cur_ticks:
                    inflight[tid] = dict(meta)
                else:
                    lid = str(meta.get("lease_id") or "")
                    tok = int(meta.get("fencing_token") or 0)
                    if lid:
                        revoked.add(lid)
                    if tok > 0:
                        source = scopes.leases.get(lid)
                        if source is None or source["fencing_token"] != tok:
                            raise ValueError("orphan_fence_scope_unresolved")
                        scopes = scopes.advance(lid, tok, source["gpu_uuid"], seal=True)
                        consumed.add(tok)
                        if tok > raw_min:
                            raw_min = tok
                    raw_epoch = max(1, raw_epoch + 1)
                    task_epochs.setdefault(tid, raw_epoch)
                    needs_repersist = True
            self._min_fencing_token = raw_min
            self._gpu_epoch = raw_epoch
            self._consumed_tokens = consumed
            self._revoked_leases = revoked
            self._key_grants = key_grants
            self._task_grants = task_grants
            self._task_epochs = task_epochs
            self._attempt_index = index
            self._inflight_tasks = inflight
            self._scope_state = scopes
            if needs_repersist:
                self._commit_state_locked(
                    min_fencing_token=raw_min,
                    gpu_epoch=raw_epoch,
                    consumed_tokens=consumed,
                    revoked_leases=revoked,
                    key_grants=key_grants,
                    task_grants=task_grants,
                    task_epochs=task_epochs,
                    attempt_index=index,
                    inflight_tasks=inflight,
                    scope_state=scopes,
                )
        except Exception as exc:
            self._corrupted = True
            self._corrupt_reason = f"fence_state_corrupted:{type(exc).__name__}:{exc}"

    def _commit_state_locked(
        self,
        *,
        min_fencing_token: int,
        gpu_epoch: int,
        consumed_tokens: set[int],
        revoked_leases: set[str],
        key_grants: dict[str, dict[str, object]],
        task_grants: dict[str, dict[str, object]],
        task_epochs: dict[str, int],
        attempt_index: dict[tuple[str, str, str], tuple[str, str]],
        inflight_tasks: dict[str, dict[str, object]],
        scope_state: ScopedFenceState | None = None,
    ) -> None:
        self._ensure_healthy_locked()
        scopes = scope_state if scope_state is not None else self._scope_state
        if self._state_path is not None:
            tmp_path = self._state_path.with_name(
                f"{self._state_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            try:
                self._state_path.parent.mkdir(parents=True, exist_ok=True)
                serialized_index = {
                    f"{p}\x1f{a}\x1f{s}": [kid, dig]
                    for (p, a, s), (kid, dig) in attempt_index.items()
                }
                data = json.dumps(
                    {
                        "schema_version": 2,
                        "scope_state": scopes.payload(),
                        "min_fencing_token": min_fencing_token,
                        "gpu_epoch": gpu_epoch,
                        "consumed_tokens": sorted(consumed_tokens),
                        "revoked_leases": sorted(revoked_leases),
                        "key_grants": key_grants,
                        "task_grants": task_grants,
                        "task_epochs": task_epochs,
                        "attempt_index": serialized_index,
                        "inflight_tasks": inflight_tasks,
                    },
                    separators=(",", ":"),
                ).encode("utf-8")
                fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                try:
                    view = memoryview(data)
                    while view:
                        written = self._write_fn(fd, view)
                        if written <= 0:
                            raise OSError("short_write_zero_bytes")
                        view = view[written:]
                    self._fsync_fn(fd)
                finally:
                    os.close(fd)
                self._replace_fn(str(tmp_path), str(self._state_path))
                self._fsync_dir_fn(self._state_path.parent)
            except Exception as exc:
                try:
                    if os.path.lexists(tmp_path):
                        os.unlink(tmp_path)
                except OSError:
                    pass
                self._corrupted = True
                self._corrupt_reason = f"fence_state_persist_failed:{type(exc).__name__}:{exc}"
                raise FencePersistenceError(self._corrupt_reason) from exc

        self._min_fencing_token = min_fencing_token
        self._gpu_epoch = gpu_epoch
        self._consumed_tokens = consumed_tokens
        self._revoked_leases = revoked_leases
        self._key_grants = key_grants
        self._task_grants = task_grants
        self._task_epochs = task_epochs
        self._attempt_index = attempt_index
        self._inflight_tasks = inflight_tasks
        self._scope_state = scopes

    def _resolve_scope_locked(self, lease_id, token, *, grant=None, identity=None, expected_pod_uid="", expected_container_id="", expected_boot_id="") -> str:
        if not lease_id or isinstance(token, bool) or not isinstance(token, int) or token <= 0:
            raise StaleFenceError("missing_lease_or_token")
        if grant is not None and (grant.get("lease_id") != lease_id or grant.get("fencing_token") != token):
            raise StaleFenceError("lease_fence_conflict")
        previous = self._scope_state.leases.get(lease_id)
        if previous is not None:
            if previous["fencing_token"] != token:
                raise StaleFenceError("lease_fence_conflict")
            return previous["gpu_uuid"]
        # An unseen lease may be sealed before dispatch. Only a caller-bound
        # Pod identity establishes the current physical card; never guess it.
        ident = identity or {}
        visible = ident.get("visible_gpu_uuids") or []
        if (any(is_unknown_identity_value(x) for x in (expected_pod_uid, expected_container_id, expected_boot_id))
                or expected_pod_uid != ident.get("pod_uid")
                or normalize_container_id(expected_container_id) != normalize_container_id(str(ident.get("container_id") or ""))
                or expected_boot_id != ident.get("boot_id") or len(visible) != 1):
            raise StaleFenceError("fence_scope_unresolved")
        try:
            return gpu_scope(visible[0])
        except ValueError as exc:
            raise StaleFenceError("fence_scope_unresolved") from exc

    @property
    def policy(self) -> str:
        raw = os.environ.get(self._policy_env, "disabled").strip().lower()
        if raw in ("enforce", "require", "enforcing"):
            return "enforce"
        if raw in ("warn", "accept"):
            return "warn"
        return "disabled"

    @property
    def current_epoch(self) -> int:
        with self._lock:
            self._ensure_healthy_locked()
            return self._gpu_epoch

    def extract_and_validate(
        self,
        headers,
        *,
        key_id: str,
        request_digest: str,
        identity: dict[str, object] | None = None,
    ) -> tuple[dict[str, object] | None, str | None]:
        with self._lock:
            self._ensure_healthy_locked()
        norm = {str(k).lower(): str(v).strip() for k, v in headers.items()}
        policy = self.policy
        ident = identity or read_runtime_identity()
        lease_id = norm.get("x-vf-gpu-lease-id", "")
        token_raw = norm.get("x-vf-gpu-fencing-token", "")
        attempt_id = norm.get("x-vf-gpu-attempt-id", "")
        stage_id = norm.get("x-vf-gpu-stage-id", "")
        parent_task_id = norm.get("x-vf-gpu-parent-task-id", "") or key_id
        pod_uid = norm.get("x-vf-gpu-pod-uid", "")
        container_id = norm.get("x-vf-gpu-container-id", "")
        boot_id = norm.get("x-vf-gpu-boot-id", "")
        gpu_uuid = norm.get("x-vf-gpu-uuid", "")
        expires_raw = norm.get("x-vf-gpu-expires-at", "")

        has_any = any((lease_id, token_raw, attempt_id, stage_id, pod_uid, container_id, boot_id, gpu_uuid, expires_raw))
        if not has_any:
            if policy == "enforce":
                raise PermissionError("missing_or_invalid_gpu_grant")
            if policy == "warn":
                logging.warning("gpu_grant_missing_warn key_id=%s", key_id)
            return None, None
        if not lease_id or not token_raw or not attempt_id or not stage_id:
            if policy == "enforce":
                raise PermissionError("missing_or_invalid_gpu_grant")
            return None, None

        try:
            fencing_token = int(token_raw)
        except ValueError as exc:
            raise PermissionError("missing_or_invalid_gpu_grant") from exc
        if fencing_token <= 0:
            raise PermissionError("missing_or_invalid_gpu_grant")

        visible = list(ident.get("visible_gpu_uuids") or [])
        local_pod = str(ident.get("pod_uid") or "")
        local_cid = str(ident.get("container_id") or "")
        local_boot = str(ident.get("boot_id") or "")
        if policy == "enforce":
            if (
                is_unknown_identity_value(local_pod)
                or is_unknown_identity_value(local_cid)
                or is_unknown_identity_value(local_boot)
                or not visible
            ):
                raise PermissionError("gpu_identity_unavailable")
            if not all((pod_uid, container_id, boot_id, gpu_uuid, expires_raw)):
                raise PermissionError("missing_or_invalid_gpu_grant")

        if pod_uid and not is_unknown_identity_value(local_pod) and pod_uid != local_pod:
            raise PermissionError("pod_uid_mismatch")
        if container_id and not is_unknown_identity_value(local_cid):
            if normalize_container_id(container_id) != normalize_container_id(local_cid):
                raise PermissionError("container_id_mismatch")
        if boot_id and not is_unknown_identity_value(local_boot) and boot_id != local_boot:
            raise PermissionError("boot_id_mismatch")
        if gpu_uuid and visible:
            for item in (part.strip() for part in gpu_uuid.split(",") if part.strip()):
                if item not in visible:
                    raise PermissionError("gpu_uuid_mismatch")
        try:
            scope = gpu_scope(gpu_uuid)
        except ValueError as exc:
            raise PermissionError("invalid_fence_gpu_scope") from exc

        expires_epoch = _parse_expires_epoch(expires_raw)
        if expires_raw and expires_epoch is None:
            raise PermissionError("missing_or_invalid_gpu_grant")
        if expires_epoch is not None and expires_epoch <= time.time() and policy == "enforce":
            raise StaleFenceError("lease_expired")

        replay_key = (parent_task_id, attempt_id, stage_id)
        with self._lock:
            self._ensure_healthy_locked()
            existing = self._attempt_index.get(replay_key)
            if existing is not None:
                prev_key, prev_digest = existing
                if prev_digest != request_digest:
                    raise ValueError("attempt_stage_conflict")
                previous = self._key_grants.get(prev_key)
                if previous is not None and (previous.get("gpu_uuid") != scope or previous.get("lease_id") != lease_id or previous.get("fencing_token") != fencing_token):
                    raise StaleFenceError("lease_fence_conflict")
                return previous, prev_key

            if key_id in self._key_grants:
                raise ValueError("idempotency_conflict")

            if (
                lease_id in self._revoked_leases
                or self._scope_state.consumed(scope, fencing_token)
                or fencing_token <= self._scope_state.minimum(scope)
            ):
                raise StaleFenceError("stale_or_revoked_fencing_token")

            next_min = max(self._min_fencing_token, fencing_token)
            grant = {
                "lease_id": lease_id,
                "fencing_token": fencing_token,
                "attempt_id": attempt_id,
                "stage_id": stage_id,
                "parent_task_id": parent_task_id,
                "pod_uid": pod_uid or local_pod,
                "container_id": container_id or local_cid,
                "boot_id": boot_id or local_boot,
                "gpu_uuid": scope,
                "expires_epoch": expires_epoch,
                "key_id": key_id,
            }
            next_key_grants = dict(self._key_grants)
            next_key_grants[key_id] = grant
            next_task_grants = dict(self._task_grants)
            next_task_grants[key_id] = grant
            next_index = dict(self._attempt_index)
            next_index[replay_key] = (key_id, request_digest)
            next_inflight = dict(self._inflight_tasks)
            next_inflight[key_id] = {
                "lease_id": lease_id,
                "fencing_token": fencing_token,
                "key_id": key_id,
                "owner_pid": int(ident.get("supervisor_pid") or os.getpid()),
                "owner_ticks": int(ident.get("supervisor_starttime_ticks") or _read_proc_starttime_ticks()),
            }
            self._commit_state_locked(
                min_fencing_token=next_min,
                gpu_epoch=self._gpu_epoch,
                consumed_tokens=set(self._consumed_tokens),
                revoked_leases=set(self._revoked_leases),
                key_grants=next_key_grants,
                task_grants=next_task_grants,
                task_epochs=dict(self._task_epochs),
                attempt_index=next_index,
                inflight_tasks=next_inflight,
                scope_state=self._scope_state.advance(lease_id, fencing_token, scope, seal=False),
            )
            return grant, None

    def is_key_revoked_or_sealed(self, key_id: str) -> bool:
        with self._lock:
            self._ensure_healthy_locked()
            grant = self._key_grants.get(key_id) or self._task_grants.get(key_id)
            if grant is None:
                return False
            lid = str(grant.get("lease_id") or "")
            tok = int(grant.get("fencing_token") or 0)
            return lid in self._revoked_leases or self._scope_state.consumed(gpu_scope(grant.get("gpu_uuid")), tok)

    def bind_task_id(self, key_id: str, runtime_task_id: str) -> None:
        with self._lock:
            self._ensure_healthy_locked()
            grant = self._key_grants.get(key_id)
            if grant is None or not runtime_task_id:
                return
            next_task_grants = dict(self._task_grants)
            next_task_grants[runtime_task_id] = grant
            next_inflight = dict(self._inflight_tasks)
            meta = next_inflight.pop(key_id, None)
            if meta is not None:
                next_inflight[runtime_task_id] = dict(meta)
            self._commit_state_locked(
                min_fencing_token=self._min_fencing_token,
                gpu_epoch=self._gpu_epoch,
                consumed_tokens=set(self._consumed_tokens),
                revoked_leases=set(self._revoked_leases),
                key_grants=dict(self._key_grants),
                task_grants=next_task_grants,
                task_epochs=dict(self._task_epochs),
                attempt_index=dict(self._attempt_index),
                inflight_tasks=next_inflight,
            )

    def allocate_worker_epoch(self, task_id: str) -> int:
        with self._lock:
            self._ensure_healthy_locked()
            grant = self._task_grants.get(task_id)
            if grant is None:
                if self.policy == "enforce":
                    raise StaleFenceError("missing_or_invalid_gpu_grant")
            else:
                lease_id = str(grant["lease_id"])
                token = int(grant["fencing_token"])
                expires_epoch = grant.get("expires_epoch")
                if (
                    lease_id in self._revoked_leases
                    or self._scope_state.consumed(gpu_scope(grant.get("gpu_uuid")), token)
                    or token < self._scope_state.minimum(gpu_scope(grant.get("gpu_uuid")))
                ):
                    raise StaleFenceError("stale_or_revoked_fencing_token")
                if isinstance(expires_epoch, (int, float)) and expires_epoch <= time.time() and self.policy == "enforce":
                    raise StaleFenceError("lease_expired")
            next_epoch = self._gpu_epoch + 1
            next_task_epochs = dict(self._task_epochs)
            next_task_epochs[task_id] = next_epoch
            self._commit_state_locked(
                min_fencing_token=self._min_fencing_token,
                gpu_epoch=next_epoch,
                consumed_tokens=set(self._consumed_tokens),
                revoked_leases=set(self._revoked_leases),
                key_grants=dict(self._key_grants),
                task_grants=dict(self._task_grants),
                task_epochs=next_task_epochs,
                attempt_index=dict(self._attempt_index),
                inflight_tasks=dict(self._inflight_tasks),
            )
            self._executing_tasks.add(task_id)
            return next_epoch

    def seal_task_epoch(self, task_id: str) -> int:
        with self._lock:
            self._executing_tasks.discard(task_id)
            self._ensure_healthy_locked()
            grant = self._task_grants.get(task_id) or self._key_grants.get(task_id)
            next_revoked = set(self._revoked_leases)
            next_consumed = set(self._consumed_tokens)
            next_min = self._min_fencing_token
            key_id = ""
            scopes = self._scope_state
            if grant is not None:
                lid = str(grant.get("lease_id") or "")
                tok = int(grant.get("fencing_token") or 0)
                key_id = str(grant.get("key_id") or "")
                if lid:
                    next_revoked.add(lid)
                if tok > 0:
                    scopes = scopes.advance(lid, tok, gpu_scope(grant.get("gpu_uuid")), seal=True)
                    next_consumed.add(tok)
                    if tok > next_min:
                        next_min = tok
            next_epoch = self._gpu_epoch
            next_task_epochs = dict(self._task_epochs)
            if task_id not in next_task_epochs:
                if next_epoch == 0:
                    next_epoch = 1
                next_task_epochs[task_id] = next_epoch
            next_inflight = dict(self._inflight_tasks)
            next_inflight.pop(task_id, None)
            if key_id:
                next_inflight.pop(key_id, None)
                self._executing_tasks.discard(key_id)
            self._commit_state_locked(
                min_fencing_token=next_min,
                gpu_epoch=next_epoch,
                consumed_tokens=next_consumed,
                revoked_leases=next_revoked,
                key_grants=dict(self._key_grants),
                task_grants=dict(self._task_grants),
                task_epochs=next_task_epochs,
                attempt_index=dict(self._attempt_index),
                inflight_tasks=next_inflight,
                scope_state=scopes,
            )
            return next_task_epochs[task_id]

    def task_epoch(self, task_id: str) -> int | None:
        with self._lock:
            self._ensure_healthy_locked()
            return self._task_epochs.get(task_id)

    def revoke_lease(self, lease_id: str, fencing_token: int, *, identity=None, expected_pod_uid="", expected_container_id="", expected_boot_id="") -> list[str]:
        affected: list[str] = []
        with self._lock:
            self._ensure_healthy_locked()
            scope = self._resolve_scope_locked(lease_id, fencing_token, identity=identity, expected_pod_uid=expected_pod_uid, expected_container_id=expected_container_id, expected_boot_id=expected_boot_id)
            next_revoked = set(self._revoked_leases)
            next_consumed = set(self._consumed_tokens)
            next_min = self._min_fencing_token
            if lease_id:
                next_revoked.add(lease_id)
            if fencing_token > 0:
                next_consumed.add(fencing_token)
                if fencing_token > next_min:
                    next_min = fencing_token
            for tid, grant in self._task_grants.items():
                if grant.get("lease_id") == lease_id or (grant.get("gpu_uuid") == scope and int(grant.get("fencing_token", 0)) <= fencing_token):
                    affected.append(tid)
            self._commit_state_locked(
                min_fencing_token=next_min,
                gpu_epoch=self._gpu_epoch,
                consumed_tokens=next_consumed,
                revoked_leases=next_revoked,
                key_grants=dict(self._key_grants),
                task_grants=dict(self._task_grants),
                task_epochs=dict(self._task_epochs),
                attempt_index=dict(self._attempt_index),
                inflight_tasks=dict(self._inflight_tasks),
                scope_state=self._scope_state.advance(lease_id, fencing_token, scope, seal=True),
            )
        return affected

    def reclaim_and_seal(
        self,
        *,
        task_id: str | None = None,
        lease_id: str = "",
        fencing_token: int = 0,
        active_tasks: int = 0,
        settled: bool = True,
        identity: dict[str, object] | None = None,
        expected_pod_uid: str = "",
        expected_container_id: str = "",
        expected_boot_id: str = "",
    ) -> dict[str, object] | None:
        ident = identity or read_runtime_identity()
        with self._lock:
            self._ensure_healthy_locked()
            grant = self._task_grants.get(task_id) if task_id else None
            if task_id and grant is None and (lease_id or fencing_token):
                raise StaleFenceError("unknown_task_fence")
            epoch = self._task_epochs.get(task_id or "", 0) or max(1, self._gpu_epoch)
            resolved_lease = lease_id or (str(grant["lease_id"]) if grant else "")
            resolved_token = fencing_token or (int(grant["fencing_token"]) if grant else 0)
            if not resolved_lease and resolved_token <= 0 and grant is None:
                return None
            scope = self._resolve_scope_locked(resolved_lease, resolved_token, grant=grant, identity=ident, expected_pod_uid=expected_pod_uid, expected_container_id=expected_container_id, expected_boot_id=expected_boot_id)
            need_commit = (
                (bool(resolved_lease) and resolved_lease not in self._revoked_leases)
                or (resolved_token > 0 and not self._scope_state.consumed(scope, resolved_token))
                or self._gpu_epoch == 0
            )
            if need_commit:
                next_revoked = set(self._revoked_leases)
                next_consumed = set(self._consumed_tokens)
                next_min = self._min_fencing_token
                next_epoch = self._gpu_epoch if self._gpu_epoch > 0 else epoch
                if resolved_lease:
                    next_revoked.add(resolved_lease)
                if resolved_token > 0:
                    next_consumed.add(resolved_token)
                    if resolved_token > next_min:
                        next_min = resolved_token
                self._commit_state_locked(
                    min_fencing_token=next_min,
                    gpu_epoch=next_epoch,
                    consumed_tokens=next_consumed,
                    revoked_leases=next_revoked,
                    key_grants=dict(self._key_grants),
                    task_grants=dict(self._task_grants),
                    task_epochs=dict(self._task_epochs),
                    attempt_index=dict(self._attempt_index),
                    inflight_tasks=dict(self._inflight_tasks),
                    scope_state=self._scope_state.advance(resolved_lease, resolved_token, scope, seal=True),
                )
            effective_active = max(
                int(active_tasks),
                len(self._executing_tasks),
                len(self._inflight_tasks),
            )
            is_sealed = bool(
                (resolved_lease and resolved_lease in self._revoked_leases)
                or (resolved_token > 0 and self._scope_state.consumed(scope, resolved_token))
            )
            is_settled = bool(settled and effective_active == 0)
            consumed = resolved_token if resolved_token > 0 else self._scope_state.minimum(scope)
            return {
                "reclaimed": bool(is_settled and is_sealed),
                "active_tasks": int(effective_active),
                "proof_source": "receiver_guard",
                "token_sealed": is_sealed,
                "sealed_gpu_epoch": int(epoch),
                "supervisor_pid": int(ident["supervisor_pid"]),
                "supervisor_starttime_ticks": int(ident["supervisor_starttime_ticks"]),
                "sealed_pod_uid": str(ident["pod_uid"]),
                "sealed_container_id": str(ident["container_id"]),
                "sealed_boot_id": str(ident["boot_id"]),
                "revoked_lease_id": resolved_lease,
                "consumed_fencing_token": int(consumed),
            }

    def build_epoch_reclaim_proof(
        self,
        *,
        task_id: str | None = None,
        lease_id: str = "",
        fencing_token: int = 0,
        active_tasks: int = 0,
        settled: bool = True,
        identity: dict[str, object] | None = None,
        expected_pod_uid: str = "",
        expected_container_id: str = "",
        expected_boot_id: str = "",
    ) -> dict[str, object] | None:
        ident = identity or read_runtime_identity()
        with self._lock:
            self._ensure_healthy_locked()
            grant = self._task_grants.get(task_id) if task_id else None
            if task_id and grant is None and (lease_id or fencing_token):
                raise StaleFenceError("unknown_task_fence")
            epoch = self._task_epochs.get(task_id or "", 0) or max(1, self._gpu_epoch)
            resolved_lease = lease_id or (str(grant["lease_id"]) if grant else "")
            resolved_token = fencing_token or (int(grant["fencing_token"]) if grant else 0)
            if not resolved_lease and resolved_token <= 0 and grant is None:
                return None
            scope = self._resolve_scope_locked(resolved_lease, resolved_token, grant=grant, identity=ident, expected_pod_uid=expected_pod_uid, expected_container_id=expected_container_id, expected_boot_id=expected_boot_id)
            effective_active = max(
                int(active_tasks),
                len(self._executing_tasks),
                len(self._inflight_tasks),
            )
            is_settled = bool(settled and effective_active == 0)
            if is_settled:
                need_commit = (
                    (bool(resolved_lease) and resolved_lease not in self._revoked_leases)
                    or (resolved_token > 0 and not self._scope_state.consumed(scope, resolved_token))
                    or self._gpu_epoch == 0
                )
                if need_commit:
                    next_revoked = set(self._revoked_leases)
                    next_consumed = set(self._consumed_tokens)
                    next_min = self._min_fencing_token
                    next_epoch = self._gpu_epoch if self._gpu_epoch > 0 else epoch
                    if resolved_lease:
                        next_revoked.add(resolved_lease)
                    if resolved_token > 0:
                        next_consumed.add(resolved_token)
                        if resolved_token > next_min:
                            next_min = resolved_token
                    self._commit_state_locked(
                        min_fencing_token=next_min,
                        gpu_epoch=next_epoch,
                        consumed_tokens=next_consumed,
                        revoked_leases=next_revoked,
                        key_grants=dict(self._key_grants),
                        task_grants=dict(self._task_grants),
                        task_epochs=dict(self._task_epochs),
                        attempt_index=dict(self._attempt_index),
                        inflight_tasks=dict(self._inflight_tasks),
                        scope_state=self._scope_state.advance(resolved_lease, resolved_token, scope, seal=True),
                    )
            is_sealed = bool(
                (resolved_lease and resolved_lease in self._revoked_leases)
                or (resolved_token > 0 and self._scope_state.consumed(scope, resolved_token))
            )
            consumed = resolved_token if resolved_token > 0 else self._scope_state.minimum(scope)
            return {
                "reclaimed": bool(is_settled and is_sealed),
                "active_tasks": int(effective_active),
                "proof_source": "receiver_guard",
                "token_sealed": is_sealed,
                "sealed_gpu_epoch": int(epoch),
                "supervisor_pid": int(ident["supervisor_pid"]),
                "supervisor_starttime_ticks": int(ident["supervisor_starttime_ticks"]),
                "sealed_pod_uid": str(ident["pod_uid"]),
                "sealed_container_id": str(ident["container_id"]),
                "sealed_boot_id": str(ident["boot_id"]),
                "revoked_lease_id": resolved_lease,
                "consumed_fencing_token": int(consumed),
            }
