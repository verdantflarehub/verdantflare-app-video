"""Cross-card acceptance, legacy recovery, and durable failure boundaries."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from h3_fasth3.fence import EpochLeaseFenceGuard, FencePersistenceError, StaleFenceError, _read_proc_starttime_ticks
from h3_fasth3.migrate_fence import migrate


class ScopedFenceTests(unittest.TestCase):
    def setUp(self):
        self.policy = patch.dict(os.environ, {"VIDEO_GPU_GRANT_POLICY": "require"})
        self.policy.start()
        self.addCleanup(self.policy.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "guard.json"
        self.guard = EpochLeaseFenceGuard(state_path=self.path)

    def identity(self, card="A"):
        return {"pod_uid": "pod-" + card, "container_id": "container-" + card,
                "boot_id": "boot", "visible_gpu_uuids": ["GPU-" + card],
                "supervisor_pid": os.getpid(), "supervisor_starttime_ticks": _read_proc_starttime_ticks()}

    def headers(self, card, token, lease):
        ident = self.identity(card)
        return {"x-vf-gpu-lease-id": lease, "x-vf-gpu-fencing-token": str(token),
                "x-vf-gpu-attempt-id": "attempt-" + lease, "x-vf-gpu-stage-id": "generation",
                "x-vf-gpu-parent-task-id": "parent-" + lease,
                "x-vf-gpu-pod-uid": ident["pod_uid"], "x-vf-gpu-container-id": ident["container_id"],
                "x-vf-gpu-boot-id": ident["boot_id"], "x-vf-gpu-uuid": "GPU-" + card,
                "x-vf-gpu-expires-at": "2099-01-01T00:00:00Z"}

    def accept(self, card, token, lease, *, guard=None):
        g = guard or self.guard
        result = g.extract_and_validate(self.headers(card, token, lease), key_id=lease,
                                       request_digest="digest-" + lease, identity=self.identity(card))
        g.bind_task_id(lease, "task-" + lease)
        return result

    def legacy(self):
        data = json.loads(self.path.read_text())
        data.pop("schema_version")
        data.pop("scope_state")
        self.path.write_text(json.dumps(data))
        return data

    def test_cross_card_restart_and_same_card_stale_rejected(self):
        self.accept("A", 10972, "a")
        self.guard.allocate_worker_epoch("task-a")
        self.guard.seal_task_epoch("task-a")
        restarted = EpochLeaseFenceGuard(state_path=self.path)
        self.accept("B", 10900, "b", guard=restarted)
        restarted.allocate_worker_epoch("task-b")
        with self.assertRaises(StaleFenceError):
            self.accept("A", 10971, "old", guard=restarted)
        self.accept("A", 10973, "new", guard=restarted)
        self.assertFalse(restarted.is_key_revoked_or_sealed("b"))

    def test_same_number_on_other_card_not_consumed(self):
        self.accept("A", 90, "a")
        self.guard.seal_task_epoch("task-a")
        self.accept("B", 90, "b")
        self.assertFalse(self.guard.is_key_revoked_or_sealed("b"))
        self.guard.allocate_worker_epoch("task-b")
        self.guard.seal_task_epoch("task-b")
        with self.assertRaises(StaleFenceError):
            self.accept("B", 90, "b-again")

    def test_revoke_old_card_does_not_cancel_new_card(self):
        self.accept("A", 10972, "a")
        self.guard.seal_task_epoch("task-a")
        self.accept("B", 10900, "b")
        affected = self.guard.revoke_lease("a", 10972, identity=self.identity("B"))
        self.assertIn("task-a", affected)
        self.assertNotIn("task-b", affected)
        self.guard.allocate_worker_epoch("task-b")
        proof = self.guard.reclaim_and_seal(lease_id="a", fencing_token=10972, identity=self.identity("B"))
        self.assertFalse(proof["reclaimed"])
        self.assertFalse(self.guard.is_key_revoked_or_sealed("b"))
        data = json.loads(self.path.read_text())
        self.assertEqual(data["scope_state"]["cards"]["GPU-B"]["minimum"], 10900)

    def test_pre_dispatch_reclaim_requires_target_and_blocks_late_start(self):
        with self.assertRaisesRegex(StaleFenceError, "scope_unresolved"):
            self.guard.reclaim_and_seal(lease_id="lost", fencing_token=90, identity=self.identity("A"))
        with self.assertRaisesRegex(StaleFenceError, "scope_unresolved"):
            self.guard.reclaim_and_seal(lease_id="lost", fencing_token=90, identity=self.identity("A"), expected_pod_uid="pod-B")
        proof = self.guard.reclaim_and_seal(lease_id="lost", fencing_token=90, identity=self.identity("A"), expected_pod_uid="pod-A", expected_container_id="container-A", expected_boot_id="boot")
        self.assertTrue(proof["reclaimed"])
        for token, lease in ((90, "lost"), (89, "older"), (90, "different")):
            with self.assertRaises(StaleFenceError):
                self.accept("A", token, lease)
        self.accept("B", 89, "b")
        self.assertFalse(self.guard.is_key_revoked_or_sealed("b"))
        EpochLeaseFenceGuard(state_path=self.path).current_epoch

    def test_same_pod_replaced_container_cannot_infer_old_lease_scope(self):
        current = {**self.identity("B"), "pod_uid": "same-pod"}
        with self.assertRaisesRegex(StaleFenceError, "scope_unresolved"):
            self.guard.revoke_lease("old-a", 10972, identity=current, expected_pod_uid="same-pod", expected_container_id="container-A", expected_boot_id="boot")
        self.assertFalse(self.path.exists())

    def test_task_lease_token_conflicts_rejected_without_mutation(self):
        self.accept("A", 90, "a")
        original = self.path.read_bytes()
        for kwargs in ({"task_id": "task-a", "lease_id": "not-a", "fencing_token": 90},
                       {"task_id": "task-a", "lease_id": "a", "fencing_token": 91},
                       {"lease_id": "a", "fencing_token": 91},
                       {"task_id": "missing", "lease_id": "a", "fencing_token": 90}):
            for method in (self.guard.reclaim_and_seal, self.guard.build_epoch_reclaim_proof):
                with self.assertRaises(StaleFenceError):
                    method(**kwargs, identity=self.identity())
        with self.assertRaises(StaleFenceError):
            self.guard.revoke_lease("a", 91)
        self.assertEqual(original, self.path.read_bytes())

    def test_legacy_complete_migrates_and_preserves_revocation(self):
        self.accept("A", 10972, "a")
        self.guard.seal_task_epoch("task-a")
        self.legacy()
        restarted = EpochLeaseFenceGuard(state_path=self.path)
        self.accept("B", 10900, "b", guard=restarted)
        with self.assertRaises(StaleFenceError):
            self.accept("A", 10971, "old", guard=restarted)
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved["schema_version"], 2)
        self.assertEqual(saved["min_fencing_token"], 10972)  # old binary remains conservative
        self.assertIn("a", saved["revoked_leases"])

    def test_legacy_unknown_revocation_needs_ledger_even_on_token_collision(self):
        self.accept("A", 90, "a")
        self.guard.seal_task_epoch("task-a")
        data = self.legacy()
        data["revoked_leases"].append("unseen-b")
        self.path.write_text(json.dumps(data))
        blocked = EpochLeaseFenceGuard(state_path=self.path)
        with self.assertRaisesRegex(FencePersistenceError, "legacy_fence_scope_unresolved"):
            blocked.current_epoch
        records = [{"lease_id": "unseen-b", "fencing_token": 90, "gpu_uuid": "GPU-B"}]
        migrated = EpochLeaseFenceGuard(state_path=self.path, legacy_lease_records=records)
        with self.assertRaises(StaleFenceError):
            self.accept("B", 89, "older-b", guard=migrated)
        self.accept("B", 91, "new-b", guard=migrated)
        self.accept("C", 89, "new-c", guard=migrated)

    def test_legacy_unknown_gpu_retains_conservative_floor(self):
        self.accept("A", 10972, "a")
        self.guard.seal_task_epoch("task-a")
        data = self.legacy()
        data["revoked_leases"].append("historical")
        data["consumed_tokens"].append(1488)
        self.path.write_text(json.dumps(data))
        records = [{"lease_id": "historical", "fencing_token": 1488, "gpu_uuid": None}]
        migrated = EpochLeaseFenceGuard(state_path=self.path, legacy_lease_records=records)
        with self.assertRaises(StaleFenceError):
            self.accept("B", 1487, "old-b", guard=migrated)
        self.accept("B", 10900, "b", guard=migrated)

    def test_restarted_orphan_seals_original_card_only(self):
        self.accept("A", 10972, "a")
        data = json.loads(self.path.read_text())
        data["inflight_tasks"]["task-a"]["owner_pid"] = -1
        self.path.write_text(json.dumps(data))
        restarted = EpochLeaseFenceGuard(state_path=self.path)
        self.accept("B", 10900, "b", guard=restarted)
        self.assertTrue(restarted.is_key_revoked_or_sealed("a"))
        with self.assertRaises(StaleFenceError):
            self.accept("A", 10971, "old", guard=restarted)

    def test_persistence_failure_never_publishes_scoped_memory(self):
        for hook in ("_write_fn", "_fsync_fn", "_replace_fn", "_fsync_dir_fn"):
            with self.subTest(hook=hook):
                path = Path(self.tmp.name) / (hook + ".json")
                guard = EpochLeaseFenceGuard(state_path=path)
                self.accept("A", 90, "a", guard=guard)
                guard.seal_task_epoch("task-a")
                before = guard._scope_state.payload()
                def fail(*args):
                    raise OSError("injected durability failure")
                setattr(guard, hook, fail)
                with self.assertRaises(FencePersistenceError):
                    self.accept("B", 80, "b", guard=guard)
                self.assertEqual(guard._scope_state.payload(), before)
                with self.assertRaises(FencePersistenceError):
                    guard.current_epoch
                # Disk is either the complete previous or complete new state.
                EpochLeaseFenceGuard(state_path=path).current_epoch

    def test_scope_corruption_and_unknown_versions_fail_closed(self):
        self.accept("A", 90, "a")
        self.guard.seal_task_epoch("task-a")
        original = json.loads(self.path.read_text())
        for kind in ("version", "float_version", "missing", "minimum", "consumed", "lease"):
            data = json.loads(json.dumps(original))
            if kind == "version": data["schema_version"] = 3
            if kind == "float_version": data["schema_version"] = 2.0
            if kind == "missing": data.pop("scope_state")
            if kind == "minimum": data["scope_state"]["cards"]["GPU-A"]["minimum"] = 89
            if kind == "consumed": data["scope_state"]["cards"]["GPU-A"]["consumed_tokens"] = []
            if kind == "lease": data["scope_state"]["leases"]["a"]["gpu_uuid"] = "GPU-B"
            with self.subTest(kind=kind):
                self.path.write_text(json.dumps(data))
                with self.assertRaises(FencePersistenceError):
                    EpochLeaseFenceGuard(state_path=self.path).current_epoch

    def test_identity_expiry_replay_and_multi_card_rejected(self):
        headers = self.headers("A", 90, "a")
        for field, value in (("x-vf-gpu-pod-uid", "pod-B"), ("x-vf-gpu-container-id", "container-B"),
                             ("x-vf-gpu-boot-id", "other-boot"), ("x-vf-gpu-uuid", "GPU-B"),
                             ("x-vf-gpu-uuid", "GPU-A,GPU-B"), ("x-vf-gpu-uuid", ","),
                             ("x-vf-gpu-expires-at", "2000-01-01T00:00:00Z")):
            with self.subTest(field=field, value=value):
                with self.assertRaises((PermissionError, StaleFenceError)):
                    self.guard.extract_and_validate({**headers, field: value}, key_id="a", request_digest="digest-a", identity=self.identity())
        self.accept("A", 90, "a")
        _, replay = self.guard.extract_and_validate(headers, key_id="unused", request_digest="digest-a", identity=self.identity())
        self.assertEqual(replay, "a")
        with self.assertRaises(ValueError):
            self.guard.extract_and_validate(headers, key_id="unused", request_digest="changed", identity=self.identity())

    def test_hash_bound_migration_never_overwrites_source_or_output(self):
        self.accept("A", 90, "a")
        self.guard.seal_task_epoch("task-a")
        self.legacy()
        original = self.path.read_bytes()
        records = Path(self.tmp.name) / "ledger.json"
        records.write_text("[]")
        output = Path(self.tmp.name) / "migrated.json"
        with self.assertRaisesRegex(ValueError, "digest_mismatch"):
            migrate(self.path, records, output, "wrong")
        self.assertFalse(output.exists())
        receipt = migrate(self.path, records, output, hashlib.sha256(original).hexdigest())
        self.assertTrue(receipt["source_unchanged"])
        self.assertEqual(self.path.read_bytes(), original)
        with self.assertRaises(ValueError):
            migrate(self.path, records, output, hashlib.sha256(original).hexdigest())

    def test_offline_migration_refuses_active_state(self):
        self.accept("A", 90, "a")
        records = Path(self.tmp.name) / "ledger.json"
        records.write_text("[]")
        output = Path(self.tmp.name) / "migrated.json"
        with self.assertRaisesRegex(ValueError, "drained_state"):
            migrate(self.path, records, output, hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
