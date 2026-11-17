"""Regression suite for reconciling identity-bound expired GPU scheduler leases."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import httpx

from app import server
from app.artifacts import ArtifactStore
from app.dashboard import Dashboard, public_task
from app.executor import VideoExecutor
from app.gpu_lease import SchedulerLeaseClient
from app.tasks import TaskStore


LIVE_IDENT = {
    "pod_uid": "31c1ce79-9cf3-46f4-adb3-3417b03766d4",
    "container_id": "pod-31c1ce79-9cf3-46f4-adb3-3417b03766d4-runtime",
    "boot_id": "52e4d33e-7007-4d4d-993a-0d6781405599",
    "supervisor_pid": 4369,
    "supervisor_starttime_ticks": 19594740,
    "visible_gpu_uuids": ["GPU-29a56c24-484e-d51d-fac9-b141be9e24c5"],
}

REPLACED_IDENT = {
    "pod_uid": "99999999-aaaa-4bbb-8ccc-dddddddddddd",
    "container_id": "pod-99999999-aaaa-4bbb-8ccc-dddddddddddd-runtime",
    "boot_id": "52e4d33e-7007-4d4d-993a-0d6781405599",
    "supervisor_pid": 8120,
    "supervisor_starttime_ticks": 22990000,
    "visible_gpu_uuids": ["GPU-29a56c24-484e-d51d-fac9-b141be9e24c5"],
}


def _seal_proof(lease_id: str, fencing_token: int, ident: dict, *, settled: bool = True, active_tasks: int = 0) -> dict:
    return {
        "reclaimed": bool(settled and active_tasks == 0),
        "active_tasks": active_tasks,
        "proof_source": "receiver_guard",
        "token_sealed": True,
        "sealed_gpu_epoch": 19,
        "supervisor_pid": ident["supervisor_pid"],
        "supervisor_starttime_ticks": ident["supervisor_starttime_ticks"],
        "sealed_pod_uid": ident["pod_uid"],
        "sealed_container_id": ident["container_id"],
        "sealed_boot_id": ident["boot_id"],
        "revoked_lease_id": lease_id,
        "consumed_fencing_token": fencing_token,
    }


class BoundLeaseRecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.artifacts = ArtifactStore(self.root / "artifacts")
        self.tasks = TaskStore(self.root / "tasks")
        self.runtime_calls: list[tuple[str, str]] = []
        self.sched_calls: list[tuple[str, str]] = []
        self.leases: dict[str, dict] = {}

        self.runtime_mode = {
            "by_idem": "404",
            "post_seal_by_idem": None,
            "task_status": "in_progress",
            "health": "ok_same",
            "reclaim": "200",
            "delete": "200_settled",
            "sealed": False,
        }
        self.accepted_id = "fasth3_" + "a" * 32

        def runtime_handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            self.runtime_calls.append((request.method, path))
            if request.method == "POST" and path in {"/v1/videos", "/v1/regenerate-2k"}:
                raise AssertionError(f"Unexpected duplicate runtime submission POST to {path}")

            if path.startswith("/v1/videos/by-idempotency/"):
                mode = (
                    self.runtime_mode["post_seal_by_idem"]
                    if (self.runtime_mode["sealed"] and self.runtime_mode["post_seal_by_idem"] is not None)
                    else self.runtime_mode["by_idem"]
                )
                if mode == "404":
                    return httpx.Response(404, json={"detail": "not_found"})
                if mode == "503":
                    return httpx.Response(503, json={"detail": "unavailable"})
                if mode == "200":
                    return httpx.Response(
                        200,
                        json={"id": self.accepted_id, "status": self.runtime_mode["task_status"], "stage": "generating"},
                    )

            if request.method == "GET" and path == f"/v1/videos/{self.accepted_id}":
                st = self.runtime_mode["task_status"]
                if st == "404":
                    return httpx.Response(404, json={"detail": "not_found"})
                if st == "503":
                    return httpx.Response(503, json={"detail": "unavailable"})
                payload = {"id": self.accepted_id, "status": st, "stage": "completed" if st == "completed" else "generating"}
                if st in {"completed", "failed", "cancelled"}:
                    payload["worker_settled"] = True
                return httpx.Response(200, json=payload)

            if request.method == "DELETE" and path == f"/v1/videos/{self.accepted_id}":
                if self.runtime_mode["delete"] == "200_settled":
                    return httpx.Response(
                        200,
                        json={"id": self.accepted_id, "status": "cancelled", "worker_settled": True, "cancel_requested": True},
                    )
                return httpx.Response(503, json={"detail": "busy"})

            if request.method == "GET" and path.startswith("/v1/regenerate-2k/"):
                tid = path.rsplit("/", 1)[-1]
                mode = self.runtime_mode["by_idem"]
                if mode == "404":
                    return httpx.Response(404, json={"detail": "not_found"})
                if mode == "200":
                    return httpx.Response(
                        200,
                        json={"video_task_id": tid, "status": "running", "worker_settled": False},
                    )
                return httpx.Response(503, json={"detail": "unavailable"})

            if path == "/health":
                hm = self.runtime_mode["health"]
                if hm == "503":
                    return httpx.Response(503, json={"detail": {"code": "not_ready"}})
                if hm == "missing_ident":
                    return httpx.Response(200, json={"ready": True, "runtime_identity": {"pod_uid": "unknown-pod"}})
                ident = REPLACED_IDENT if hm == "ok_replaced" else LIVE_IDENT
                return httpx.Response(
                    200,
                    json={
                        "ready": True,
                        "gpu_worker_active": self.runtime_mode["reclaim"] == "409_peer_busy",
                        "active_task_ids": [],
                        "gpu_epoch": 19,
                        "runtime_identity": ident,
                    },
                )

            if request.method == "POST" and path == "/internal/v1/gpu-guard/reclaim":
                body = json.loads(request.content.decode("utf-8"))
                lid = body.get("lease_id", "")
                tok = int(body.get("fencing_token") or 0)
                rm = self.runtime_mode["reclaim"]
                if rm == "503":
                    return httpx.Response(503, json={"code": "fence_state_corrupted"})
                ident = REPLACED_IDENT if self.runtime_mode["health"] == "ok_replaced" else LIVE_IDENT
                self.runtime_mode["sealed"] = True
                if rm == "409_peer_busy":
                    proof = _seal_proof(lid, tok, ident, settled=False, active_tasks=1)
                    return httpx.Response(
                        409,
                        json={"detail": {"code": "worker_not_settled", "settled": False, "active_tasks": 1, "reclaim_proof": proof}},
                    )
                proof = _seal_proof(lid, tok, ident, settled=True, active_tasks=0)
                return httpx.Response(
                    200,
                    json={"ok": True, "settled": True, "active_tasks": 0, "gpu_epoch": 19, "reclaim_proof": proof},
                )

            return httpx.Response(404)

        def sched_handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            self.sched_calls.append((request.method, path))
            if path == "/internal/v1/gpu-scheduler/leases/acquire":
                raise AssertionError("Expired lease reconciliation must never call scheduler acquire")
            if path.startswith("/internal/v1/gpu-scheduler/leases/"):
                rest = path[len("/internal/v1/gpu-scheduler/leases/"):]
                lid = rest.split("/", 1)[0]
                if "/" not in rest and lid in self.leases:
                    return httpx.Response(200, json=self.leases[lid])
            return httpx.Response(404, json={"error_code": "LEASE_NOT_FOUND"})

        rt_client = httpx.Client(transport=httpx.MockTransport(runtime_handler))
        sched_client = httpx.Client(transport=httpx.MockTransport(sched_handler))
        self.addCleanup(rt_client.close)
        self.addCleanup(sched_client.close)

        self.executor = VideoExecutor(self.artifacts, self.tasks, rt_client)
        self.executor.fasth3_token = "fasth3-secret"
        self.executor.runtime_route = "h3-fasth3"
        self.executor.runtime_url = "http://fasth3:8000"
        self.executor.gpu.enabled = False
        self.executor.stage_slot.enabled = False
        self.executor.scheduler = SchedulerLeaseClient(
            base_url="http://station-runtime:5056",
            token="sched-secret",
            holder_app_id="video-mcp",
            enabled=True,
            client=sched_client,
        )

    def _create_bound_expired_task(
        self,
        *,
        idem: str = "xe-e01-reference-guidance-20261116-3",
        lease_overrides: dict | None = None,
        pstate_overrides: dict | None = None,
        error_code: str | None = "gpu_admission_reconciliation_required",
        delivery: dict | None = None,
    ):
        lease_id = f"01a1e4b3-2130-7bfb-8d48-{abs(hash(idem)) % (10 ** 12):012d}"
        lease = {
            "lease_id": lease_id,
            "state": "expired",
            "holder_app_id": "video-mcp",
            "attempt_id": "att-d6e729bdb504",
            "stage_id": "generation",
            "node_uid": "node-5090",
            "pod_uid": LIVE_IDENT["pod_uid"],
            "container_id": LIVE_IDENT["container_id"],
            "boot_id": LIVE_IDENT["boot_id"],
            "gpu_uuids": [],
            "peak_budget_mib": 29696,
            "startup_free_min_mib": 28672,
            "fencing_token": 1157,
            "priority": "exclusive_batch",
            "blocking_reason": "queue_ttl_or_pod_expired",
            "expires_at": "2026-11-16T14:13:17Z",
            "version": 2,
            **(lease_overrides or {}),
        }
        rec = self.tasks.create(
            project_id="proj_xe_e01",
            idempotency_key=idem,
            input_digest="sha256:" + "1" * 64,
            request={"project_id": "proj_xe_e01", "route": "h3-fasth3", "prompt": "protective separation"},
            runtime_task_id="",
            status="queued",
            runtime_endpoint="http://fasth3:8000",
            runtime_route="h3-fasth3",
        )
        lease["holder_task_id"] = rec.video_task_id
        self.leases[lease_id] = dict(lease)
        pstate = {
            "gpu_dispatched": False,
            "_gpu_attempt_id": "att-d6e729bdb504",
            "_gpu_lease": dict(lease),
            "_gpu_lease_owned": True,
            "_gpu_active_service": "h3-fasth3",
            "_gpu_target_identity": dict(LIVE_IDENT),
            "lease_state": str(lease["state"]),
            "blocking_reason": str(lease["blocking_reason"]),
            **(pstate_overrides or {}),
        }
        err = (
            {
                "code": error_code,
                "message": "GPU admission lease expired; reconcile the original task before retrying",
                "retryable": False,
                "stage": "gpu_admission",
                "reason": str(lease["blocking_reason"]),
            }
            if error_code
            else None
        )
        return self.tasks.update(
            rec,
            service="h3-fasth3",
            provider_state=pstate,
            error=err,
            delivery=delivery,
        )

    def _call_mcp_status(self, video_task_id: str) -> dict:
        orig_exec, orig_tasks = server.executor, server.tasks
        orig_state_ratings = getattr(server.app.state, "ratings", None)
        server.app.state.ratings = server.ContentRatingStore(self.tasks.db_path, domain="video")
        orig_state_tasks = getattr(server.app.state, "tasks", None)
        server.executor, server.tasks = self.executor, self.tasks
        server.app.state.tasks = self.tasks
        try:
            call_res = server.video_status(video_task_id=video_task_id)
        finally:
            server.executor, server.tasks = orig_exec, orig_tasks
            server.app.state.tasks = orig_state_tasks
            server.app.state.ratings = orig_state_ratings
        self.assertFalse(call_res.is_error, call_res.content[0].text if call_res.content else call_res)
        return json.loads(call_res.content[0].text)

    def _call_mcp_cancel(self, video_task_id: str) -> dict:
        orig_exec, orig_tasks = server.executor, server.tasks
        orig_state_ratings = getattr(server.app.state, "ratings", None)
        server.app.state.ratings = server.ContentRatingStore(self.tasks.db_path, domain="video")
        orig_state_tasks = getattr(server.app.state, "tasks", None)
        server.executor, server.tasks = self.executor, self.tasks
        server.app.state.tasks = self.tasks
        try:
            call_res = server.video_cancel(video_task_id=video_task_id)
        finally:
            server.executor, server.tasks = orig_exec, orig_tasks
            server.app.state.tasks = orig_state_tasks
            server.app.state.ratings = orig_state_ratings
        self.assertFalse(call_res.is_error, call_res.content[0].text if call_res.content else call_res)
        return json.loads(call_res.content[0].text)

    def test_bound_expired_unstarted_same_container_seals_fence_and_converges_failed(self) -> None:
        for case_name, reclaim_mode, lease_extra in (
            ("idle_without_never_granted", "200", {}),
            ("peer_busy_409_still_seals_token", "409_peer_busy", {"never_granted": False, "gpu_uuids": ["GPU-0"]}),
        ):
            with self.subTest(case=case_name):
                self.runtime_calls.clear()
                self.sched_calls.clear()
                self.runtime_mode.update({"by_idem": "404", "post_seal_by_idem": None, "health": "ok_same", "reclaim": reclaim_mode, "sealed": False})
                rec = self._create_bound_expired_task(
                    idem=f"unstarted-{case_name}",
                    lease_overrides=lease_extra,
                    delivery={"status": "queued", "stage": "generation", "chain_id": "chain-keep"},
                )

                mcp_res = self._call_mcp_status(rec.video_task_id)

                self.assertEqual(mcp_res["status"], "failed")
                self.assertEqual(mcp_res["error"]["code"], "gpu_admission_expired")
                self.assertTrue(mcp_res["error"]["retryable"])
                self.assertEqual(mcp_res["admission_reconciliation"]["outcome"], "unstarted_fence_sealed")
                self.assertEqual(mcp_res["admission_reconciliation"]["container_check"], "matched_and_sealed")
                self.assertFalse(mcp_res["admission_reconciliation"]["runtime_task_found"])
                self.assertEqual(mcp_res["delivery"]["status"], "failed")
                self.assertEqual(mcp_res["delivery"]["chain_id"], "chain-keep")

                stored = self.tasks.get(rec.video_task_id)
                self.assertTrue(stored.provider_state["worker_settled"])
                self.assertTrue(stored.provider_state["reclaim_proof"]["token_sealed"])
                self.assertIsNone(public_task(stored)["gpu_queue"])

                calls_after_first = len(self.runtime_calls)
                again = self.executor.status(rec.video_task_id)
                self.assertEqual(again.status, "failed")
                self.assertEqual(len(self.runtime_calls), calls_after_first)

    def test_post_seal_race_recovers_accepted_task_without_duplicate_dispatch(self) -> None:
        self.runtime_mode.update({
            "by_idem": "404",
            "post_seal_by_idem": "200",
            "task_status": "in_progress",
            "health": "ok_same",
            "reclaim": "200",
            "sealed": False,
        })
        rec = self._create_bound_expired_task(idem="post-seal-race")
        recovered = self.executor.status(rec.video_task_id)
        self.assertEqual(recovered.status, "running")
        self.assertIsNone(recovered.error)
        self.assertEqual(recovered.runtime_task_id, self.accepted_id)
        self.assertEqual(recovered.provider_state["admission_reconciliation"]["outcome"], "recovered_accepted")

    def test_already_accepted_bound_expired_lease_recovers_running_then_succeeded(self) -> None:
        self.runtime_mode.update({"by_idem": "200", "task_status": "in_progress", "sealed": False})
        rec = self._create_bound_expired_task(
            idem="already-accepted",
            lease_overrides={"never_granted": False, "gpu_uuids": ["GPU-0"]},
            pstate_overrides={"gpu_dispatched": True},
        )
        running = self.executor.status(rec.video_task_id)
        self.assertEqual(running.status, "running")
        self.assertIsNone(running.error)
        self.assertEqual(running.runtime_task_id, self.accepted_id)
        self.assertEqual(running.provider_state["admission_reconciliation"]["outcome"], "recovered_accepted")
        reclaim_calls = [c for c in self.runtime_calls if c[1] == "/internal/v1/gpu-guard/reclaim"]
        self.assertEqual(reclaim_calls, [])

        self.runtime_mode["task_status"] = "completed"
        done = self.executor.status(rec.video_task_id)
        self.assertEqual(done.status, "succeeded")
        self.assertTrue(done.provider_state["worker_settled"])

    def test_execution_unknown_surfaces_waiting_reason_and_recovers_later(self) -> None:
        scenarios = [
            ("by_idem_503", {"by_idem": "503", "health": "ok_same", "reclaim": "200"}, "runtime_unreachable"),
            ("health_503", {"by_idem": "404", "health": "503", "reclaim": "200"}, "runtime_health_unreachable"),
            ("missing_ident", {"by_idem": "404", "health": "missing_ident", "reclaim": "200"}, "runtime_identity_unavailable"),
            ("reclaim_503", {"by_idem": "404", "health": "ok_same", "reclaim": "503"}, "fence_seal_unavailable"),
        ]
        for label, modes, expected_pending in scenarios:
            with self.subTest(scenario=label):
                self.runtime_mode.update({**modes, "post_seal_by_idem": None, "sealed": False})
                rec = self._create_bound_expired_task(
                    idem=f"unknown-{label}",
                    lease_overrides={"never_granted": False, "gpu_uuids": ["GPU-0"]},
                )
                mcp_res = self._call_mcp_status(rec.video_task_id)

                self.assertEqual(mcp_res["status"], "queued")
                self.assertTrue(mcp_res["needs_attention"])
                self.assertEqual(mcp_res["error"]["code"], "gpu_admission_reconciliation_required")
                self.assertFalse(mcp_res["error"]["retryable"])
                self.assertEqual(mcp_res["error"]["reconciliation_state"], expected_pending)
                self.assertEqual(mcp_res["gpu_queue"]["waiting_reason"], "runtime_reconciliation_pending")
                self.assertEqual(mcp_res["admission_reconciliation"]["outcome"], "reconciliation_pending")
                self.assertEqual(mcp_res["admission_reconciliation"]["pending_reason"], expected_pending)

                # Once runtime recovers, next video.status converges cleanly
                self.runtime_mode.update({"by_idem": "404", "health": "ok_same", "reclaim": "200", "sealed": False})
                resolved = self.executor.status(rec.video_task_id)
                self.assertEqual(resolved.status, "failed")
                self.assertEqual(resolved.error["code"], "gpu_admission_expired")
                self.assertTrue(resolved.error["retryable"])

    def test_bound_container_replaced_converges_or_holds_when_previous_worker_unsettled(self) -> None:
        self.runtime_mode.update({"by_idem": "404", "health": "ok_replaced", "reclaim": "200", "sealed": False})
        rec = self._create_bound_expired_task(
            idem="replaced-unstarted",
            lease_overrides={"never_granted": False, "gpu_uuids": ["GPU-0"]},
        )
        settled = self.executor.status(rec.video_task_id)
        self.assertEqual(settled.status, "failed")
        self.assertEqual(settled.error["code"], "gpu_admission_expired")
        self.assertTrue(settled.error["retryable"])
        self.assertEqual(settled.provider_state["admission_reconciliation"]["outcome"], "bound_container_replaced_unstarted")

        # If worker_settled was explicitly False and scheduler has NOT confirmed reconciled_terminated, hold conservatively
        unsettled = self._create_bound_expired_task(
            idem="replaced-unsettled",
            lease_overrides={"never_granted": False, "gpu_uuids": ["GPU-0"], "blocking_reason": "lease_expired"},
            pstate_overrides={"worker_settled": False},
        )
        held = self.executor.status(unsettled.video_task_id)
        self.assertEqual(held.status, "queued")
        self.assertEqual(held.error["reconciliation_state"], "previous_worker_unsettled")

        # Once scheduler confirms reconciled_terminated on host NVML/CRI, it converges safely
        lid = unsettled.provider_state["_gpu_lease"]["lease_id"]
        self.leases[lid]["blocking_reason"] = "reconciled_terminated"
        unsettled = self.tasks.update(
            held,
            provider_state={**held.provider_state, "_gpu_lease": dict(self.leases[lid])},
        )
        converged = self.executor.status(unsettled.video_task_id)
        self.assertEqual(converged.status, "failed")
        self.assertEqual(converged.error["code"], "gpu_admission_expired")

    def test_cancel_bound_expired_task_cancels_accepted_or_seals_unstarted_without_stale_release(self) -> None:
        # Sub-case A: Runtime accepted task while runtime_task_id was still empty -> DELETE is delivered
        self.runtime_mode.update({"by_idem": "200", "task_status": "in_progress", "delete": "200_settled", "sealed": False})
        rec_acc = self._create_bound_expired_task(
            idem="cancel-accepted",
            lease_overrides={"never_granted": False, "gpu_uuids": ["GPU-0"]},
        )
        cancelled_acc = self.executor.cancel_generation(rec_acc)
        self.assertEqual(cancelled_acc.status, "cancelled")
        self.assertTrue(cancelled_acc.provider_state["worker_settled"])
        self.assertIn(("DELETE", f"/v1/videos/{self.accepted_id}"), self.runtime_calls)

        # Sub-case B: Runtime never accepted -> seals token and cancels without calling scheduler /release on expired lease
        self.runtime_calls.clear()
        self.sched_calls.clear()
        self.runtime_mode.update({"by_idem": "404", "health": "ok_same", "reclaim": "200", "sealed": False})
        rec_unstarted = self._create_bound_expired_task(
            idem="cancel-unstarted",
            lease_overrides={"never_granted": False, "gpu_uuids": ["GPU-0"]},
        )
        cancelled_unstarted_res = self._call_mcp_cancel(rec_unstarted.video_task_id)
        self.assertEqual(cancelled_unstarted_res["status"], "cancelled")
        cancelled_unstarted = self.tasks.get(rec_unstarted.video_task_id)
        self.assertEqual(cancelled_unstarted.status, "cancelled")
        self.assertTrue(cancelled_unstarted.provider_state["worker_settled"])
        self.assertIn(("POST", "/internal/v1/gpu-guard/reclaim"), self.runtime_calls)
        release_calls = [c for c in self.sched_calls if c[1].endswith("/release")]
        self.assertEqual(release_calls, [])

    def test_restart_recovery_and_processing_adapter_reconcile_cleanly(self) -> None:
        self.runtime_mode.update({"by_idem": "404", "health": "ok_same", "reclaim": "200", "sealed": False})
        h3_rec = self._create_bound_expired_task(
            idem="dashboard-sync-bound",
            lease_overrides={"never_granted": False, "gpu_uuids": ["GPU-0"]},
            error_code=None,
            pstate_overrides={"lease_state": "queued", "waiting_reason": "insufficient_vram"},
        )
        dashboard = Dashboard(self.executor)
        dashboard.recover_incomplete_submissions()
        after_recover = self.tasks.get(h3_rec.video_task_id)
        self.assertIsNone(after_recover.error)

        # Simulate scheduler lease now returning expired on sync
        dashboard.sync()
        synced = self.tasks.get(h3_rec.video_task_id)
        self.assertEqual(synced.status, "failed")
        self.assertEqual(synced.error["code"], "gpu_admission_expired")
        self.assertEqual(synced.provider_state["admission_reconciliation"]["outcome"], "unstarted_fence_sealed")


if __name__ == "__main__":
    unittest.main()
