"""P1b tests for :5056 GPU scheduler grant headers, P1->P2->P3 lock downgrade, fallback, and cross-card handoff."""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import httpx

from app.artifacts import ArtifactStore
from app.dashboard import public_task, task_presentation
from app.delivery import CALLER
from app.executor import VideoExecutor
from app.gpu_lease import SchedulerLeaseClient
from app.regenerate import (
    LOCKED_P2_CODE_REVISION,
    LOCKED_P2_WEIGHTS_SHA256,
    preview_geometry,
    resolve_2k_target_geometry,
)
from app.tasks import TaskStore


class FakeSchedulerTransport:
    def __init__(self, *, cutover_downgrade_enabled: bool = True):
        self.cutover_downgrade_enabled = cutover_downgrade_enabled
        self.calls: list[tuple[str, str, dict]] = []
        self.leases: dict[str, dict] = {}
        self._seq = 0
        self.force_queued = False
        self.force_draining = False
        self.fail_next_releases = 0

    def handle(self, request: httpx.Request) -> httpx.Response:
        import json

        path = request.url.path
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        self.calls.append((request.method, path, body))

        if path == "/internal/v1/gpu-scheduler/leases/acquire":
            self._seq += 1
            lid = f"0196ed00-0000-7000-8000-{self._seq:012d}"
            cand = body.get("candidate_gpu_uuids") or ["GPU-0000"]
            peak = 29696 if body.get("consumer_kind") in {"h3", "video.regenerate_2k"} else (
                4096 if body.get("consumer_kind") == "rife" else 8192
            )
            state = "draining" if self.force_draining else ("queued" if self.force_queued else "leased")
            waiting = state in {"queued", "draining"}
            doc = {
                "lease_id": lid,
                "state": state,
                "holder_app_id": body.get("holder_app_id", "video-mcp"),
                "holder_task_id": body.get("task_id", ""),
                "attempt_id": body.get("attempt_id", "att-1"),
                "stage_id": body.get("stage_id", "generation"),
                "node_uid": "node-5090-dev",
                "pod_uid": body.get("pod_uid", ""),
                "container_id": body.get("container_id", ""),
                "boot_id": body.get("boot_id", ""),
                "gpu_uuid": "" if waiting else cand[0],
                "gpu_uuids": [] if waiting else [cand[0]],
                "peak_budget_mib": peak,
                "startup_free_min_mib": peak,
                "fencing_token": 100 + self._seq,
                "priority": "exclusive_batch" if peak >= 20000 else "shared_slice",
                "queue_position": (1 if state == "draining" else 2) if waiting else 0,
                "blocking_reason": (
                    "draining_evictable_peer"
                    if state == "draining"
                    else ("insufficient_vram" if waiting else "none")
                ),
                "blocking_holder_id": (
                    "comfyui/default"
                    if state == "draining"
                    else ("blender/inst-1" if waiting else "")
                ),
                "expires_at": "2099-11-15T12:00:00Z",
                "version": 1,
            }
            self.leases[lid] = doc
            return httpx.Response(200, json=doc)

        if path.startswith("/internal/v1/gpu-scheduler/leases/"):
            rest = path[len("/internal/v1/gpu-scheduler/leases/"):]
            parts = rest.split("/")
            lid = parts[0]
            action = parts[1] if len(parts) > 1 else "get"
            doc = dict(self.leases[lid])
            if action == "get":
                return httpx.Response(200, json=doc)
            if action == "release" and doc.get("state") in {"released", "expired"}:
                return httpx.Response(200, json=doc)
            exp_ver = int(body.get("expected_version") or 0)
            if exp_ver > 0 and exp_ver != int(doc["version"]):
                return httpx.Response(
                    409,
                    json={
                        "error_code": "LEASE_VERSION_CONFLICT",
                        "message": f"expected version {exp_ver}, actual {doc['version']}",
                    },
                )
            if int(body.get("fencing_token") or 0) != int(doc["fencing_token"]):
                return httpx.Response(
                    409,
                    json={"error_code": "FENCING_TOKEN_STALE", "message": "fencing token mismatch"},
                )
            if action == "bind-target":
                doc["pod_uid"] = body["pod_uid"]
                doc["container_id"] = body["container_id"]
                doc["boot_id"] = body["boot_id"]
                doc["version"] = int(doc["version"]) + 1
                self.leases[lid] = doc
                return httpx.Response(200, json=doc)
            if action == "downgrade":
                if not self.cutover_downgrade_enabled:
                    # Emit Station Runtime top-level error_code contract (M1)
                    return httpx.Response(
                        409,
                        json={"error_code": "LOCK_DOWNGRADE_DISABLED_BEFORE_CUTOVER", "message": "disabled"},
                    )
                doc["peak_budget_mib"] = int(body["downgrade_to_mib"])
                doc["stage_id"] = body["new_stage_id"]
                doc["state"] = "running"
                doc["version"] = int(doc["version"]) + 1
                for src, dst in (("next_pod_uid", "pod_uid"), ("next_container_id", "container_id"), ("next_boot_id", "boot_id")):
                    if body.get(src):
                        doc[dst] = body[src]
                self.leases[lid] = doc
                return httpx.Response(200, json=doc)
            if action == "heartbeat":
                if body.get("new_stage_id"):
                    doc["stage_id"] = body["new_stage_id"]
                    for src, dst in (("next_pod_uid", "pod_uid"), ("next_container_id", "container_id"), ("next_boot_id", "boot_id")):
                        if body.get(src):
                            doc[dst] = body[src]
                doc["state"] = "running"
                doc["version"] = int(doc["version"]) + 1
                self.leases[lid] = doc
                return httpx.Response(200, json=doc)
            if action == "release":
                if self.fail_next_releases > 0:
                    self.fail_next_releases -= 1
                    return httpx.Response(503, json={"error_code": "UNKNOWN_NODE_STATE", "message": "verifier busy"})
                doc["state"] = "released"
                doc["version"] = int(doc["version"]) + 1
                self.leases[lid] = doc
                return httpx.Response(200, json=doc)
        return httpx.Response(404, json={"error_code": "NOT_FOUND", "message": "not found"})


class SchedulerP1bFenceAndDowngradeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.artifacts = ArtifactStore(root / "artifacts")
        self.tasks = TaskStore(root / "tasks")
        self.video_bytes = b"\x00\x00\x00\x18ftypmp42fakevideo"
        # Use non-ftyp stub bytes for artifact registration in unit mocks that bypass ffprobe
        self.stub_bytes = b"STUB_MP4_PAYLOAD_FOR_SCHEDULER_TEST"
        self.stub_sha = hashlib.sha256(self.stub_bytes).hexdigest()
        self.source_art = self.artifacts.create_from_chunks(
            project_id="proj_sched",
            operation="test.seed",
            filename="seed.mp4",
            media_type="video/mp4",
            chunks=[self.stub_bytes],
        )
        # Seed a succeeded 24fps 124-frame H3 record so _inspect_regenerate_source_media resolves without ffprobe
        seed_rec = self.tasks.create(
            project_id="proj_sched",
            idempotency_key="seed-rec",
            input_digest="sha256:" + "0" * 64,
            request={"project_id": "proj_sched", "prompt": "Cinema courtyard"},
            runtime_task_id="seed_rt",
            status="succeeded",
        )
        self.tasks.update(
            seed_rec,
            artifact_id=self.source_art.artifact_id,
            media={"width": 832, "height": 480, "fps": 24, "frames": 124, "audio": True},
        )

    def _make_executor(self, *, cutover_downgrade_enabled: bool = True, rife_gpu: str = "GPU-0000"):
        import os

        self._saved_env = {
            k: os.environ.get(k)
            for k in (
                "VIDEO_PROJECT_SERVICE_URL",
                "VIDEO_PROJECT_SERVICE_TOKEN",
                "VIDEO_REGENERATE_2K_RUNTIME_URL",
                "VIDEO_REGENERATE_2K_RUNTIME_TOKEN",
                "VIDEO_INTERPOLATION_RUNTIME_URL",
                "VIDEO_SR_RUNTIME_URL",
            )
        }

        def _restore():
            for k, v in self._saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        self.addCleanup(_restore)
        os.environ["VIDEO_PROJECT_SERVICE_URL"] = "http://station-core:8080"
        os.environ["VIDEO_PROJECT_SERVICE_TOKEN"] = "proj-secret-token"
        os.environ["VIDEO_REGENERATE_2K_RUNTIME_URL"] = "http://regen2k:8000"
        os.environ["VIDEO_REGENERATE_2K_RUNTIME_TOKEN"] = "regen2k-token"
        os.environ["VIDEO_INTERPOLATION_RUNTIME_URL"] = "http://rife:8000"
        os.environ["VIDEO_SR_RUNTIME_URL"] = "http://sr:8000"

        sched_transport = FakeSchedulerTransport(cutover_downgrade_enabled=cutover_downgrade_enabled)
        sched_http = httpx.Client(transport=httpx.MockTransport(sched_transport.handle))
        self.addCleanup(sched_http.close)

        observed_headers: dict[str, dict[str, str]] = {}

        def runtime_handler(request: httpx.Request) -> httpx.Response:
            host = request.url.host
            path = request.url.path
            if host == "station-core" and path == "/internal/v1/project/open":
                return httpx.Response(200, json={"ok": True})
            if host == "regen2k":
                ident = {
                    "pod_uid": "pod-regen2k",
                    "container_id": "cid-regen2k",
                    "boot_id": "boot-node-1",
                    "supervisor_pid": 2100,
                    "supervisor_starttime_ticks": 91000,
                    "visible_gpu_uuids": ["GPU-0000"],
                }
                if path == "/health":
                    return httpx.Response(
                        200,
                        json={
                            "ready": True,
                            "runtime_version": "video-minimax-h3-regenerate-2k-v0.1.0",
                            "capabilities": {"regenerate_2k_profiles": ["open_h3_2k_ic"]},
                            "gpu_epoch": 3,
                            "runtime_identity": ident,
                        },
                    )
                if request.method == "POST" and path == "/v1/regenerate-2k":
                    observed_headers["regenerate_2k"] = dict(request.headers)
                    import json

                    b = json.loads(request.content.decode("utf-8"))
                    tid = b["idempotency_key"]
                    return httpx.Response(
                        200,
                        json={
                            "video_task_id": tid,
                            "status": "running",
                            "started_at": "2026-11-15T06:00:01+00:00",
                            "worker_settled": False,
                            "gpu_epoch": 4,
                        },
                    )
                if request.method == "GET" and path.startswith("/v1/regenerate-2k/"):
                    parts = path.strip("/").split("/")
                    tid = parts[2]
                    if len(parts) == 4 and parts[3] in {"content", "preview"}:
                        return httpx.Response(200, content=self.stub_bytes)
                    hdrs = observed_headers.get("regenerate_2k", {})
                    prompt_sha = hashlib.sha256(b"Cinema courtyard").hexdigest()
                    geo = resolve_2k_target_geometry(832, 480, "budget_1440p")
                    pw, ph = preview_geometry(geo["target_w"], geo["target_h"])
                    return httpx.Response(
                        200,
                        json={
                            "video_task_id": tid,
                            "status": "succeeded",
                            "started_at": "2026-11-15T06:00:01+00:00",
                            "finished_at": "2026-11-15T06:00:10+00:00",
                            "worker_settled": True,
                            "gpu_epoch": 4,
                            "reclaim_proof": {
                                "reclaimed": True,
                                "proof_source": "receiver_guard",
                                "token_sealed": True,
                                "sealed_gpu_epoch": 4,
                                "supervisor_pid": 2100,
                                "supervisor_starttime_ticks": 91000,
                                "sealed_pod_uid": "pod-regen2k",
                                "sealed_container_id": "cid-regen2k",
                                "sealed_boot_id": "boot-node-1",
                                "revoked_lease_id": hdrs.get("x-vf-gpu-lease-id", ""),
                                "consumed_fencing_token": int(hdrs.get("x-vf-gpu-fencing-token", "0") or 0),
                            },
                            "result": {
                                "input_sha256": self.source_art.sha256,
                                "output_sha256": self.stub_sha,
                                "preview_sha256": self.stub_sha,
                                "input_media": {
                                    "width": 832, "height": 480, "frames": 124,
                                    "frame_rate": "24/1", "duration_seconds": 124 / 24,
                                    "video_codec": "h264", "audio": True,
                                },
                                "media": {
                                    "width": geo["target_w"], "height": geo["target_h"], "frames": 124,
                                    "frame_rate": "24/1", "duration_seconds": 124 / 24,
                                    "video_codec": "h264", "audio": True,
                                    "audio_mode": "copy", "audio_sha256": self.stub_sha,
                                },
                                "preview_media": {
                                    "width": pw, "height": ph, "frames": 124,
                                    "frame_rate": "24/1", "duration_seconds": 124 / 24,
                                    "video_codec": "h264", "audio": True,
                                },
                                "geometry": geo,
                                "windows": [[0, 124]],
                                "model": {
                                    "backend": "open_h3_2k_ic",
                                    "profile": "open_h3_2k_ic",
                                    "code_revision": LOCKED_P2_CODE_REVISION,
                                    "weights_revision": "a" * 40,
                                    "manifest_sha256": "b" * 64,
                                    "weights_sha256": dict(LOCKED_P2_WEIGHTS_SHA256),
                                },
                                "parameters": {
                                    "profile": "open_h3_2k_ic",
                                    "grid_mode": " native_2k".strip() if False else "budget_1440p",
                                    "seed": 7,
                                    "prompt_sha256": prompt_sha,
                                    "audio": "copy",
                                    "video_codec": "h264",
                                    "pixel_format": "yuv420p",
                                    "crf": 16,
                                    "tiles": 2,
                                },
                            },
                        },
                    )
            if host == "rife":
                ident = {
                    "pod_uid": "pod-rife",
                    "container_id": "cid-rife",
                    "boot_id": "boot-node-1",
                    "supervisor_pid": 3100,
                    "supervisor_starttime_ticks": 92000,
                    "visible_gpu_uuids": [rife_gpu],
                }
                if path == "/health":
                    return httpx.Response(200, json={"status": "ok", "service": "interpolate", "runtime_identity": ident})
                if request.method == "POST" and path == "/v1/interpolate":
                    observed_headers["interpolate"] = dict(request.headers)
                    import json

                    b = json.loads(request.content.decode("utf-8"))
                    tid = b["idempotency_key"]
                    self._rife_source_sha = b["source_sha256"]
                    return httpx.Response(
                        200,
                        json={
                            "video_task_id": tid,
                            "status": "running",
                            "started_at": "2026-11-15T06:00:12+00:00",
                            "worker_settled": False,
                        },
                    )
                if request.method == "GET" and path.startswith("/v1/interpolate/"):
                    parts = path.strip("/").split("/")
                    tid = parts[2]
                    if len(parts) == 4 and parts[3] in {"content", "preview"}:
                        return httpx.Response(200, content=self.stub_bytes)
                    hdrs = observed_headers.get("interpolate", {})
                    geo = resolve_2k_target_geometry(832, 480, "budget_1440p")
                    pw, ph = preview_geometry(geo["target_w"], geo["target_h"])
                    return httpx.Response(
                        200,
                        json={
                            "video_task_id": tid,
                            "status": "succeeded",
                            "started_at": "2026-11-15T06:00:12+00:00",
                            "finished_at": "2026-11-15T06:00:18+00:00",
                            "worker_settled": True,
                            "reclaim_proof": {
                                "reclaimed": True,
                                "active_tasks": 0,
                                "proof_source": "receiver_guard",
                                "token_sealed": True,
                                "supervisor_pid": 3100,
                                "supervisor_starttime_ticks": 92000,
                                "sealed_pod_uid": "pod-rife",
                                "sealed_container_id": "cid-rife",
                                "sealed_boot_id": "boot-node-1",
                                "revoked_lease_id": hdrs.get("x-vf-gpu-lease-id", ""),
                                "consumed_fencing_token": int(hdrs.get("x-vf-gpu-fencing-token", "0") or 0),
                            },
                            "result": {
                                "input_sha256": getattr(self, "_rife_source_sha", self.stub_sha),
                                "output_sha256": self.stub_sha,
                                "preview_sha256": self.stub_sha,
                                "input_media": {
                                    "width": geo["target_w"], "height": geo["target_h"], "frames": 124,
                                    "frame_rate": "24/1", "duration_seconds": 124 / 24,
                                    "video_codec": "h264", "audio": True,
                                },
                                "media": {
                                    "width": geo["target_w"], "height": geo["target_h"], "frames": 248,
                                    "frame_rate": "48/1", "duration_seconds": 248 / 48,
                                    "video_codec": "h264", "audio": True, "audio_mode": "copy",
                                },
                                "preview_media": {
                                    "width": pw, "height": ph, "frames": 248,
                                    "frame_rate": "48/1", "duration_seconds": 248 / 48,
                                    "video_codec": "h264", "audio": True,
                                },
                                "model": {
                                    "backend": "rife",
                                    "code_revision": "c" * 40,
                                    "weights_revision": "d" * 40,
                                    "manifest_sha256": "e" * 64,
                                    "weights_sha256": {"flownet.pkl": "f" * 64},
                                },
                                "parameters": {
                                    "backend": "rife",
                                    "target_fps_num": 48,
                                    "target_fps_den": 1,
                                    "audio": "aac_if_present",
                                    "video_codec": "h264",
                                    "pixel_format": "yuv420p",
                                    "crf": 18,
                                    "scene_cut_threshold": 0.30,
                                    "last_frame": "hold",
                                },
                            },
                        },
                    )
            return httpx.Response(404)

        rt_http = httpx.Client(transport=httpx.MockTransport(runtime_handler))
        self.addCleanup(rt_http.close)
        executor = VideoExecutor(self.artifacts, self.tasks, rt_http)
        executor.scheduler = SchedulerLeaseClient(
            base_url="http://station-runtime:5056",
            token="sched-token",
            holder_app_id="video-mcp",
            enabled=True,
            client=sched_http,
        )
        return executor, sched_transport, observed_headers

    def test_same_card_p2_to_p3_downgrades_29696_to_4096_and_releases_warm_proof(self) -> None:
        executor, sched, observed = self._make_executor(cutover_downgrade_enabled=True, rife_gpu="GPU-0000")
        tok = CALLER.set({"user_id": "u1", "organization_id": "org1"})
        self.addCleanup(CALLER.reset, tok)

        rec = executor.processing["regenerate_2k"].generate(
            project_id="proj_sched",
            idempotency_key="p2-p3-same-card",
            source_artifact_id=self.source_art.artifact_id,
            grid_mode="budget_1440p",
            postprocess={"interpolate": {"profile": "rife_2x"}},
        )
        self.assertIn("x-vf-gpu-lease-id", observed["regenerate_2k"])
        lease_id = observed["regenerate_2k"]["x-vf-gpu-lease-id"]

        # Tick finishes P2, downgrades 29696 -> 4096 MiB on same card, dispatches P3, and releases warm_idle_settled proof
        rec = executor.delivery.tick(rec.video_task_id)
        self.assertEqual(rec.delivery["stage"], "completed")
        self.assertEqual(rec.delivery["status"], "succeeded")
        self.assertEqual(observed["interpolate"]["x-vf-gpu-lease-id"], lease_id)
        self.assertEqual(observed["interpolate"]["x-vf-gpu-stage-id"], "postprocess_rife")

        downgrade_calls = [c for c in sched.calls if c[1].endswith("/downgrade")]
        self.assertEqual(len(downgrade_calls), 1)
        self.assertEqual(downgrade_calls[0][2]["downgrade_to_mib"], 4096)
        self.assertEqual(downgrade_calls[0][2]["new_stage_id"], "postprocess_rife")
        self.assertEqual(downgrade_calls[0][2]["reclaim_mode"], "epoch_unload")
        self.assertEqual(downgrade_calls[0][2]["next_pod_uid"], "pod-rife")

        release_calls = [c for c in sched.calls if c[1].endswith("/release")]
        self.assertEqual(len(release_calls), 1)
        self.assertEqual(release_calls[0][2]["reclaim_mode"], "warm_idle_settled")
        self.assertEqual(release_calls[0][2]["pod_uid"], "pod-rife")
        self.assertEqual(sched.leases[lease_id]["state"], "released")

    def test_pre_cutover_409_downgrade_falls_back_to_stage_handoff(self) -> None:
        executor, sched, observed = self._make_executor(cutover_downgrade_enabled=False, rife_gpu="GPU-0000")
        tok = CALLER.set({"user_id": "u1", "organization_id": "org1"})
        self.addCleanup(CALLER.reset, tok)

        rec = executor.processing["regenerate_2k"].generate(
            project_id="proj_sched",
            idempotency_key="p2-p3-fallback",
            source_artifact_id=self.source_art.artifact_id,
            grid_mode="budget_1440p",
            postprocess={"interpolate": {"profile": "rife_2x"}},
        )
        rec = executor.delivery.tick(rec.video_task_id)
        self.assertEqual(rec.delivery["stage"], "completed")
        self.assertEqual(rec.delivery["status"], "succeeded")
        # Both /downgrade (409) and /heartbeat stage handoff were called
        self.assertEqual(len([c for c in sched.calls if c[1].endswith("/downgrade")]), 1)
        hb_calls = [c for c in sched.calls if c[1].endswith("/heartbeat") and c[2].get("new_stage_id") == "postprocess_rife"]
        self.assertEqual(len(hb_calls), 1)

    def test_cross_card_p2_to_p3_releases_source_card_before_acquiring_target_card(self) -> None:
        executor, sched, observed = self._make_executor(cutover_downgrade_enabled=True, rife_gpu="GPU-1111")
        tok = CALLER.set({"user_id": "u1", "organization_id": "org1"})
        self.addCleanup(CALLER.reset, tok)

        rec = executor.processing["regenerate_2k"].generate(
            project_id="proj_sched",
            idempotency_key="p2-p3-cross-card",
            source_artifact_id=self.source_art.artifact_id,
            grid_mode="budget_1440p",
            postprocess={"interpolate": {"profile": "rife_2x"}},
        )
        p2_lease_id = observed["regenerate_2k"]["x-vf-gpu-lease-id"]
        rec = executor.delivery.tick(rec.video_task_id)
        p3_lease_id = observed["interpolate"]["x-vf-gpu-lease-id"]
        self.assertNotEqual(p2_lease_id, p3_lease_id)

        # Verify strict order: release(p2_lease_id) occurred BEFORE second acquire(rife)!
        paths = [c[1] for c in sched.calls]
        first_rel_idx = paths.index(f"/internal/v1/gpu-scheduler/leases/{p2_lease_id}/release")
        acquire_indices = [i for i, p in enumerate(paths) if p.endswith("/acquire")]
        self.assertEqual(len(acquire_indices), 2)
        self.assertLess(first_rel_idx, acquire_indices[1])

        rec = executor.delivery.tick(rec.video_task_id)
        self.assertEqual(rec.delivery["status"], "succeeded")
        self.assertEqual(sched.leases[p3_lease_id]["state"], "released")

    def test_queue_deferral_and_local_badge_projection(self) -> None:
        executor, sched, _ = self._make_executor(cutover_downgrade_enabled=True)
        sched.force_queued = True
        executor.stage_slot.queue_seconds = 600
        tok = CALLER.set({"user_id": "u1", "organization_id": "org1"})
        self.addCleanup(CALLER.reset, tok)

        rec = executor.processing["regenerate_2k"].generate(
            project_id="proj_sched",
            idempotency_key="queued-task",
            source_artifact_id=self.source_art.artifact_id,
        )
        self.assertEqual(rec.status, "queued")
        pub = public_task(rec)
        self.assertEqual(pub["gpu_queue"]["lease_state"], "queued")
        self.assertEqual(pub["gpu_queue"]["queue_position"], 2)
        self.assertEqual(pub["gpu_queue"]["blocking_reason"], "insufficient_vram")
        self.assertEqual(pub["gpu_queue"]["blocking_holder_id"], "blender/inst-1")

        # DrainingPeer state projects lease_state="draining" and waiting_reason="draining_peer"
        sched.force_queued = False
        sched.force_draining = True
        rec_drain = executor.processing["regenerate_2k"].generate(
            project_id="proj_sched",
            idempotency_key="draining-task",
            source_artifact_id=self.source_art.artifact_id,
        )
        self.assertEqual(rec_drain.status, "queued")
        pub_drain = public_task(rec_drain)
        self.assertEqual(pub_drain["gpu_queue"]["lease_state"], "draining")
        self.assertEqual(pub_drain["gpu_queue"]["waiting_reason"], "draining_peer")
        self.assertEqual(pub_drain["gpu_queue"]["queue_position"], 1)
        self.assertEqual(pub_drain["gpu_queue"]["blocking_reason"], "draining_evictable_peer")
        self.assertEqual(pub_drain["gpu_queue"]["blocking_holder_id"], "comfyui/default")

        # Non-queued tasks never leak stale gpu_queue metadata
        done_rec = self.tasks.update(rec_drain, status="succeeded")
        self.assertIsNone(public_task(done_rec)["gpu_queue"])

        for local_svc in ("sr", "interpolate", "depth", "h3-latent-upscale"):
            pres = task_presentation(SimpleNamespace(service=local_svc, runtime_route=None, request={}))
            self.assertFalse(pres["is_generation"])
            self.assertEqual(pres["channel_label"], "本地")
            self.assertEqual(pres["channel_icon"], "local")

    def test_h1_missing_inline_proof_fetches_receiver_reclaim_and_fails_closed_if_unavailable(self) -> None:
        executor, sched, _ = self._make_executor(cutover_downgrade_enabled=True)
        reclaim_state = {"allow": False, "calls": 0}

        def sr_handler(request: httpx.Request) -> httpx.Response:
            import json

            ident = {
                "pod_uid": "pod-sr-x",
                "container_id": "cid-sr-x",
                "boot_id": "boot-sr-1",
                "supervisor_pid": 6100,
                "supervisor_starttime_ticks": 71000,
                "visible_gpu_uuids": ["GPU-0000"],
            }
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok", "service": "sr", "runtime_identity": ident})
            if request.url.path == "/internal/v1/gpu-guard/reclaim":
                reclaim_state["calls"] += 1
                if not reclaim_state["allow"]:
                    return httpx.Response(409, json={"code": "worker_not_settled", "active_tasks": 1})
                b = json.loads(request.content.decode("utf-8"))
                return httpx.Response(
                    200,
                    json={
                        "reclaim_proof": {
                            "reclaimed": True,
                            "active_tasks": 0,
                            "proof_source": "receiver_guard",
                            "token_sealed": True,
                            "supervisor_pid": 6100,
                            "supervisor_starttime_ticks": 71000,
                            "sealed_pod_uid": "pod-sr-x",
                            "sealed_container_id": "cid-sr-x",
                            "sealed_boot_id": "boot-sr-1",
                            "revoked_lease_id": b["lease_id"],
                            "consumed_fencing_token": int(b["fencing_token"]),
                        }
                    },
                )
            return httpx.Response(404)

        sr_http = httpx.Client(transport=httpx.MockTransport(sr_handler))
        self.addCleanup(sr_http.close)
        executor.client = sr_http
        executor.processing["sr"].client = sr_http

        rec = self.tasks.create(
            project_id="proj_sched",
            idempotency_key="sr-h1",
            input_digest="sha256:" + "1" * 64,
            request={"project_id": "proj_sched", "service": "sr"},
            runtime_task_id="sr-h1",
            status="succeeded",
        )
        ok, rec, _ = executor.acquire_scheduler_grant(rec, "sr", runtime_url="http://sr:8000")
        self.assertTrue(ok)
        pstate = dict(rec.provider_state or {})
        pstate["gpu_dispatched"] = True
        pstate["worker_settled"] = True
        rec = self.tasks.update(rec, status="succeeded", provider_state=pstate)

        # Receiver refuses reclaim -> must NOT fabricate proof or release lease!
        rec = executor.maybe_release_scheduler_lease(rec)
        self.assertFalse((rec.provider_state or {}).get("_gpu_lease_released", False))
        self.assertEqual(len([c for c in sched.calls if c[1].endswith("/release")]), 0)
        self.assertEqual(reclaim_state["calls"], 1)

        # Once receiver seals and returns receiver_guard proof, release succeeds
        reclaim_state["allow"] = True
        rec = executor.maybe_release_scheduler_lease(rec)
        self.assertTrue((rec.provider_state or {}).get("_gpu_lease_released", False))
        rel_calls = [c for c in sched.calls if c[1].endswith("/release")]
        self.assertEqual(len(rel_calls), 1)
        self.assertEqual(rel_calls[0][2]["reclaim_proof"]["proof_source"], "receiver_guard")
        self.assertTrue(rel_calls[0][2]["reclaim_proof"]["token_sealed"])

    def test_h4_cold_start_binds_target_identity_before_dispatch(self) -> None:
        executor, sched, _ = self._make_executor(cutover_downgrade_enabled=True)
        phase = {"cold": True}

        def cold_sr_handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                if phase["cold"]:
                    return httpx.Response(503, json={"code": "starting"})
                return httpx.Response(
                    200,
                    json={
                        "status": "ok",
                        "runtime_identity": {
                            "pod_uid": "pod-cold-sr",
                            "container_id": "cid-cold-sr",
                            "boot_id": "boot-cold-1",
                            "supervisor_pid": 7100,
                            "supervisor_starttime_ticks": 61000,
                            "visible_gpu_uuids": ["GPU-0000"],
                        },
                    },
                )
            return httpx.Response(404)

        cold_http = httpx.Client(transport=httpx.MockTransport(cold_sr_handler))
        self.addCleanup(cold_http.close)
        executor.client = cold_http

        rec = self.tasks.create(
            project_id="proj_sched",
            idempotency_key="sr-cold",
            input_digest="sha256:" + "2" * 64,
            request={"project_id": "proj_sched", "service": "sr"},
            runtime_task_id="sr-cold",
            status="queued",
        )
        # Acquire while target is still cold (bind_immediately=False)
        ok, rec, _ = executor.acquire_scheduler_grant(
            rec, "sr", runtime_url="http://sr:8000", bind_immediately=False
        )
        self.assertTrue(ok)
        self.assertEqual(rec.provider_state["_gpu_lease"]["pod_uid"], "")

        # Attempting to bind while still cold fails closed
        from app.executor import ExecutionError

        with self.assertRaises(ExecutionError):
            executor.ensure_scheduler_target_bound(rec, runtime_url="http://sr:8000")

        # Pod finishes cold start -> bind-target persists identity on :5056 and emits full 6-tuple headers
        phase["cold"] = False
        rec, hdrs = executor.ensure_scheduler_target_bound(rec, runtime_url="http://sr:8000")
        bind_calls = [c for c in sched.calls if c[1].endswith("/bind-target")]
        self.assertEqual(len(bind_calls), 1)
        self.assertEqual(bind_calls[0][2]["pod_uid"], "pod-cold-sr")
        self.assertEqual(bind_calls[0][2]["actual_gpu_uuids"], ["GPU-0000"])
        self.assertEqual(hdrs["X-VF-GPU-Pod-UID"], "pod-cold-sr")
        self.assertEqual(hdrs["X-VF-GPU-Container-ID"], "cid-cold-sr")

    def test_h5_delivery_stage_transition_blocks_when_proof_unavailable(self) -> None:
        from app.executor import ExecutionError

        executor, sched, _ = self._make_executor(cutover_downgrade_enabled=True, rife_gpu="GPU-0000")
        parent = self.tasks.create(
            project_id="proj_sched",
            idempotency_key="p1-no-proof",
            input_digest="sha256:" + "3" * 64,
            request={"project_id": "proj_sched", "prompt": "x"},
            runtime_task_id="rt-no-proof",
            status="succeeded",
        )
        ok, parent, _ = executor.acquire_scheduler_grant(
            parent, "regenerate_2k", runtime_url="http://regen2k:8000", runtime_headers={"Authorization": "Bearer regen2k-token"}
        )
        self.assertTrue(ok)
        # No inline reclaim_proof and regen2k mock returns 404 on /internal/v1/gpu-guard/reclaim -> must raise ExecutionError!
        with self.assertRaises(ExecutionError):
            executor.delivery._transition_scheduler_lease(
                parent,
                source_stage_record=parent,
                next_stage_id="postprocess_rife",
                next_adapter=executor.processing["interpolate"],
                downgrade_to_mib=4096,
            )

    def test_h6_pending_release_queue_retries_on_reconcile(self) -> None:
        executor, sched, _ = self._make_executor(cutover_downgrade_enabled=True, rife_gpu="GPU-0000")
        rec = self.tasks.create(
            project_id="proj_sched",
            idempotency_key="rife-h6",
            input_digest="sha256:" + "4" * 64,
            request={"project_id": "proj_sched", "service": "interpolate"},
            runtime_task_id="rife-h6",
            status="succeeded",
        )
        ok, rec, hdrs = executor.acquire_scheduler_grant(rec, "interpolate", runtime_url="http://rife:8000")
        self.assertTrue(ok)
        lease_id = hdrs["X-VF-GPU-Lease-ID"]
        token = int(hdrs["X-VF-GPU-Fencing-Token"])
        pstate = dict(rec.provider_state or {})
        pstate.update(
            {
                "gpu_dispatched": True,
                "worker_settled": True,
                "reclaim_proof": {
                    "reclaimed": True,
                    "active_tasks": 0,
                    "proof_source": "receiver_guard",
                    "token_sealed": True,
                    "supervisor_pid": 3100,
                    "supervisor_starttime_ticks": 92000,
                    "sealed_pod_uid": "pod-rife",
                    "sealed_container_id": "cid-rife",
                    "sealed_boot_id": "boot-node-1",
                    "revoked_lease_id": lease_id,
                    "consumed_fencing_token": token,
                },
            }
        )
        rec = self.tasks.update(rec, status="succeeded", provider_state=pstate)

        sched.fail_next_releases = 1
        rec = executor.maybe_release_scheduler_lease(rec)
        self.assertFalse((rec.provider_state or {}).get("_gpu_lease_released", False))
        self.assertIn(lease_id, executor.scheduler._pending_releases)

        executor.reconcile_gpu()
        self.assertNotIn(lease_id, executor.scheduler._pending_releases)
        self.assertEqual(sched.leases[lease_id]["state"], "released")

    def test_h7_canonical_lock_order_scheduler_first_and_unstarted_rollback(self) -> None:
        executor, sched, _ = self._make_executor(cutover_downgrade_enabled=True)
        executor.stage_slot.enabled = True
        executor.stage_slot.queue_seconds = 600
        tok = CALLER.set({"user_id": "u1", "organization_id": "org1"})
        self.addCleanup(CALLER.reset, tok)

        # 1. When :5056 queues the task, local stage_slot MUST NOT be acquired
        sched.force_queued = True
        rec_q = executor.processing["regenerate_2k"].generate(
            project_id="proj_sched",
            idempotency_key="h7-queued",
            source_artifact_id=self.source_art.artifact_id,
        )
        self.assertEqual(rec_q.status, "queued")
        self.assertIsNone(executor.stage_slot.current_holder())

        # 2. When :5056 grants while local stage_slot is busy, unbound :5056 grant is immediately aborted (abort_unstarted)
        sched.force_queued = False
        holder = self.tasks.create(
            project_id="proj_sched",
            idempotency_key="slot-holder",
            input_digest="sha256:" + "5" * 64,
            request={"project_id": "proj_sched", "service": "sr"},
            runtime_task_id="slot-holder",
            status="running",
        )
        holder = self.tasks.update(holder, service="sr", provider_state={"worker_settled": False})
        self.assertTrue(executor.stage_slot.try_acquire(holder.video_task_id, "sr"))

        rec_slot_wait = executor.processing["regenerate_2k"].generate(
            project_id="proj_sched",
            idempotency_key="h7-slot-wait",
            source_artifact_id=self.source_art.artifact_id,
        )
        self.assertEqual(rec_slot_wait.status, "queued")
        abort_calls = [
            c for c in sched.calls
            if c[1].endswith("/release") and c[2].get("reclaim_mode") == "abort_unstarted"
        ]
        self.assertEqual(len(abort_calls), 1)
        self.assertFalse(abort_calls[0][2]["worker_settled"])

    def test_r2_m1_durable_pending_release_journal_strict_receipt_and_restart_recovery(self) -> None:
        from app.gpu_lease import SchedulerError

        journal_path = Path(self.tmp.name) / "gpu_pending_releases.json"
        mode = {"resp": "disconnect"}

        def handler(request: httpx.Request) -> httpx.Response:
            m = mode["resp"]
            if m == "disconnect":
                raise RuntimeError("socket_reset_by_peer")
            if m == "404":
                return httpx.Response(404, json={"error_code": "LEASE_NOT_FOUND"})
            if m == "wrong_lease":
                return httpx.Response(200, json={"lease_id": "wrong-lease-id", "state": "released"})
            if m == "reclaiming":
                return httpx.Response(200, json={"lease_id": "lease-m1", "state": "reclaiming"})
            return httpx.Response(200, json={"lease_id": "lease-m1", "state": "released"})

        http_client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(http_client.close)
        c1 = SchedulerLeaseClient(
            base_url="http://station-runtime:5056",
            token="tok",
            holder_app_id="video-mcp",
            enabled=True,
            client=http_client,
            pending_releases_path=journal_path,
        )

        # 1. Non-HTTPError transport exception still preserves write-ahead durable journal entry
        with self.assertRaises(SchedulerError):
            c1.release("lease-m1", fencing_token=77, worker_settled=True, expected_version=3)
        self.assertIn("lease-m1", c1._pending_releases)
        self.assertNotIn("expected_version", c1._pending_releases["lease-m1"])
        self.assertTrue(journal_path.is_file())

        # 2. Simulate process restart: new client reloads durable journal; 404, wrong lease_id, and reclaiming MUST NOT delete entry!
        c2 = SchedulerLeaseClient(
            base_url="http://station-runtime:5056",
            token="tok",
            holder_app_id="video-mcp",
            enabled=True,
            client=http_client,
            pending_releases_path=journal_path,
        )
        self.assertIn("lease-m1", c2._pending_releases)

        for bad_mode in ("404", "wrong_lease", "reclaiming"):
            mode["resp"] = bad_mode
            self.assertEqual(c2.flush_pending_releases(), 0)
            self.assertIn("lease-m1", c2._pending_releases)

        # 3. Confirmed matching lease_id + terminal state clears journal on disk
        mode["resp"] = "ok"
        self.assertEqual(c2.flush_pending_releases(), 1)
        self.assertNotIn("lease-m1", c2._pending_releases)
        c3 = SchedulerLeaseClient(
            base_url="http://station-runtime:5056",
            token="tok",
            holder_app_id="video-mcp",
            enabled=True,
            client=http_client,
            pending_releases_path=journal_path,
        )
        self.assertEqual(c3._pending_releases, {})

    def test_h1_multi_heartbeat_version_bump_then_p1_p2_p3_handoff_and_downgrade(self) -> None:
        executor, sched, observed = self._make_executor(cutover_downgrade_enabled=True, rife_gpu="GPU-0000")
        tok = CALLER.set({"user_id": "u1", "organization_id": "org1"})
        self.addCleanup(CALLER.reset, tok)

        parent = self.tasks.create(
            project_id="proj_sched",
            idempotency_key="p1-h1-full-chain",
            input_digest="sha256:" + "8" * 64,
            request={
                "project_id": "proj_sched",
                "route": "h3-fasth3",
                "prompt": "Cinema courtyard",
                "postprocess": {
                    "regenerate_2k": {"profile": "open_h3_2k_ic", "grid_mode": "budget_1440p"},
                    "interpolate": {"profile": "rife_2x"},
                },
            },
            runtime_task_id="rt-h3-p1",
            status="running",
            provider_state={
                "worker_settled": False,
                "gpu_dispatched": True,
                "_gpu_target_identity": {
                    "pod_uid": "pod-h3",
                    "container_id": "cid-h3",
                    "boot_id": "boot-node-1",
                    "supervisor_pid": 1100,
                    "supervisor_starttime_ticks": 81000,
                    "visible_gpu_uuids": ["GPU-0000"],
                },
            },
        )
        ok, parent, hdrs = executor.acquire_scheduler_grant(
            parent,
            "h3-fasth3",
            runtime_url="http://regen2k:8000",
            runtime_headers={"Authorization": "Bearer regen2k-token"},
        )
        self.assertTrue(ok)
        lease_id = hdrs["X-VF-GPU-Lease-ID"]
        token = int(hdrs["X-VF-GPU-Fencing-Token"])
        self.assertEqual(sched.leases[lease_id]["version"], 1)

        # Simulate two background reconcile_gpu() heartbeats during P1 (version 1 -> 2 -> 3)
        pstate = dict(parent.provider_state or {})
        pstate["worker_settled"] = False
        parent = self.tasks.update(parent, status="running", provider_state=pstate)
        executor.reconcile_gpu()
        executor.reconcile_gpu()
        self.assertEqual(sched.leases[lease_id]["version"], 3)
        parent = self.tasks.get(parent.video_task_id)
        self.assertEqual(parent.provider_state["_gpu_lease"]["version"], 3)

        # P1 finishes with authoritative epoch_unload reclaim_proof
        pstate = dict(parent.provider_state or {})
        pstate.update(
            {
                "worker_settled": True,
                "gpu_epoch": 2,
                "reclaim_proof": {
                    "reclaimed": True,
                    "proof_source": "receiver_guard",
                    "token_sealed": True,
                    "sealed_gpu_epoch": 2,
                    "supervisor_pid": 2100,
                    "supervisor_starttime_ticks": 91000,
                    "sealed_pod_uid": "pod-regen2k",
                    "sealed_container_id": "cid-regen2k",
                    "sealed_boot_id": "boot-node-1",
                    "revoked_lease_id": lease_id,
                    "consumed_fencing_token": token,
                },
            }
        )
        parent = self.tasks.update(
            parent,
            status="succeeded",
            artifact_id=self.source_art.artifact_id,
            media={"width": 832, "height": 480, "fps": 24, "frames": 124, "audio": True},
            provider_state=pstate,
            delivery=executor.delivery.initial(parent),
        )
        executor.status = lambda tid: self.tasks.get(tid)  # type: ignore[method-assign]
        executor.result = lambda tid: self.tasks.get(tid)  # type: ignore[method-assign]

        # Tick runs P1 -> P2 stage_handoff (expected_version=3 -> 4), P2 -> P3 downgrade (expected_version=4 -> 5), and release (version=6)
        done = executor.delivery.tick(parent.video_task_id)
        self.assertEqual(done.delivery["status"], "succeeded")
        self.assertEqual(done.delivery["stage"], "completed")
        self.assertEqual(sched.leases[lease_id]["state"], "released")
        self.assertEqual(sched.leases[lease_id]["version"], 6)

        handoff_calls = [
            c for c in sched.calls
            if c[1].endswith("/heartbeat") and c[2].get("new_stage_id") == "regenerate_2k"
        ]
        self.assertEqual(len(handoff_calls), 1)
        self.assertEqual(handoff_calls[0][2]["expected_version"], 3)

        downgrade_calls = [c for c in sched.calls if c[1].endswith("/downgrade")]
        self.assertEqual(len(downgrade_calls), 1)
        self.assertEqual(downgrade_calls[0][2]["expected_version"], 4)
        self.assertEqual(downgrade_calls[0][2]["downgrade_to_mib"], 4096)

    def test_h1_cas_reconcile_strict_owner_fence_check_and_immediate_release_on_handoff_failure(self) -> None:
        from app.executor import ExecutionError

        executor, sched, observed = self._make_executor(cutover_downgrade_enabled=True, rife_gpu="GPU-0000")
        tok = CALLER.set({"user_id": "u1", "organization_id": "org1"})
        self.addCleanup(CALLER.reset, tok)

        # Part A: Restarted client (cleared _observed_versions) with matching owner/task/fence reconciles once
        rec_a = executor.processing["regenerate_2k"].generate(
            project_id="proj_sched",
            idempotency_key="h1-restart-same-owner",
            source_artifact_id=self.source_art.artifact_id,
            grid_mode="budget_1440p",
            postprocess={"interpolate": {"profile": "rife_2x"}},
        )
        lid_a = observed["regenerate_2k"]["x-vf-gpu-lease-id"]
        sched.leases[lid_a]["version"] = 4
        executor.scheduler._observed_versions.clear()

        done_a = executor.delivery.tick(rec_a.video_task_id)
        self.assertEqual(done_a.delivery["status"], "succeeded")
        self.assertEqual(sched.leases[lid_a]["state"], "released")
        dg_a = [c for c in sched.calls if c[1] == f"/internal/v1/gpu-scheduler/leases/{lid_a}/downgrade"]
        self.assertEqual(len(dg_a), 2)
        self.assertEqual(dg_a[0][2]["expected_version"], 1)
        self.assertEqual(dg_a[1][2]["expected_version"], 4)
        self.assertTrue(str(dg_a[1][2]["idempotency_key"]).endswith(":v4"))

        # Part B: Mismatched holder_task_id on server MUST NOT reconcile version, and tick MUST fail delivery & release lease when owner matches a terminal error!
        rec_b = executor.processing["regenerate_2k"].generate(
            project_id="proj_sched",
            idempotency_key="h1-foreign-owner-reject",
            source_artifact_id=self.source_art.artifact_id,
            grid_mode="budget_1440p",
            postprocess={"interpolate": {"profile": "rife_2x"}},
        )
        lid_b = observed["regenerate_2k"]["x-vf-gpu-lease-id"]
        sched.leases[lid_b]["version"] = 5
        sched.leases[lid_b]["holder_task_id"] = "foreign_video_task"
        executor.scheduler._observed_versions.clear()

        with self.assertRaises(ExecutionError):
            p2_done = executor.processing["regenerate_2k"].result(rec_b.video_task_id)
            executor.delivery._transition_scheduler_lease(
                p2_done,
                source_stage_record=p2_done,
                next_stage_id="postprocess_rife",
                next_adapter=executor.processing["interpolate"],
                downgrade_to_mib=4096,
            )
        dg_b = [c for c in sched.calls if c[1] == f"/internal/v1/gpu-scheduler/leases/{lid_b}/downgrade"]
        # Rejected without blind retry!
        self.assertEqual(len(dg_b), 1)

        # Part C: Terminal downgrade rejection on valid owner immediately fails delivery and releases the GPU lease (<1s, no 90s dangling lease)
        rec_c = executor.processing["regenerate_2k"].generate(
            project_id="proj_sched",
            idempotency_key="h1-fail-releases-immediately",
            source_artifact_id=self.source_art.artifact_id,
            grid_mode="budget_1440p",
            postprocess={"interpolate": {"profile": "rife_2x"}},
        )
        lid_c = observed["regenerate_2k"]["x-vf-gpu-lease-id"]
        orig_handle = sched.handle

        def reject_c_downgrade(request: httpx.Request) -> httpx.Response:
            if request.url.path == f"/internal/v1/gpu-scheduler/leases/{lid_c}/downgrade":
                return httpx.Response(
                    409,
                    json={"error_code": "INVALID_STAGE_TRANSITION", "message": "synthetic rejection"},
                )
            return orig_handle(request)

        sched.handle = reject_c_downgrade  # type: ignore[method-assign]
        # Re-bind MockTransport to updated handler
        executor.scheduler._client = httpx.Client(transport=httpx.MockTransport(reject_c_downgrade))
        self.addCleanup(executor.scheduler._client.close)

        failed_c = executor.delivery.tick(rec_c.video_task_id)
        self.assertEqual(failed_c.delivery["status"], "failed")
        self.assertEqual(failed_c.delivery["error"]["code"], "gpu_stage_transition_failed")
        self.assertEqual(sched.leases[lid_c]["state"], "released")
        self.assertTrue((failed_c.provider_state or {}).get("_gpu_lease_released"))

    def test_p2_unsettled_on_first_succeeded_poll_waits_for_worker_settled_before_downgrade(self) -> None:
        executor, sched, observed = self._make_executor(cutover_downgrade_enabled=True, rife_gpu="GPU-0000")
        tok = CALLER.set({"user_id": "u1", "organization_id": "org1"})
        self.addCleanup(CALLER.reset, tok)

        rec = executor.processing["regenerate_2k"].generate(
            project_id="proj_sched",
            idempotency_key="p2-unsettled-first-poll",
            source_artifact_id=self.source_art.artifact_id,
            grid_mode="budget_1440p",
            postprocess={"interpolate": {"profile": "rife_2x"}},
        )
        lid = observed["regenerate_2k"]["x-vf-gpu-lease-id"]

        # Simulate P2 first returning status='succeeded' with worker_settled=False & token_sealed=False (epoch_unload in progress)
        orig_status = executor.processing["regenerate_2k"].status
        polls = {"n": 0}

        def staged_p2_status(tid: str):
            polls["n"] += 1
            r = orig_status(tid)
            if polls["n"] == 1:
                ps = dict(r.provider_state or {})
                ps["worker_settled"] = False
                ps["reclaim_proof"] = {**(ps.get("reclaim_proof") or {}), "reclaimed": False, "token_sealed": False}
                return self.tasks.update(r, provider_state=ps)
            return r

        executor.processing["regenerate_2k"].status = staged_p2_status  # type: ignore[method-assign]

        # Tick #1 MUST keep delivery 'running' (NOT fail!) and renew lease while P2 finishes unloading weights
        tick1 = executor.delivery.tick(rec.video_task_id)
        self.assertEqual(tick1.delivery["status"], "running")
        self.assertIn(sched.leases[lid]["state"], {"leased", "running"})

        # Tick #2 sees worker_settled=True, downgrades 29696 -> 4096 MiB, runs P3, and releases cleanly!
        tick2 = executor.delivery.tick(rec.video_task_id)
        self.assertEqual(tick2.delivery["status"], "succeeded")
        self.assertEqual(tick2.delivery["stage"], "completed")
        self.assertEqual(sched.leases[lid]["state"], "released")


if __name__ == "__main__":
    unittest.main()
