"""Reclaim must forward the lease's binding, never the current Pod as a substitute."""

import json
import unittest
from unittest.mock import Mock, patch

import httpx
import test_bound_lease_recovery as recovery


class FenceScopeForwardingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = recovery.BoundLeaseRecoveryTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.executor = self.fixture.executor

    def test_reconciliation_forwards_original_bound_identity(self):
        record = self.fixture._create_bound_expired_task()
        with patch.object(self.executor, "_seal_expired_lease_on_runtime", return_value=("fence_seal_unavailable", None)) as seal:
            self.executor.reconcile_expired_scheduler_admission(record)
        kwargs = seal.call_args.kwargs
        self.assertEqual(kwargs["expected_pod_uid"], recovery.LIVE_IDENT["pod_uid"])
        self.assertEqual(kwargs["expected_container_id"], recovery.LIVE_IDENT["container_id"])
        self.assertEqual(kwargs["expected_boot_id"], recovery.LIVE_IDENT["boot_id"])

    def test_unbound_lease_is_not_filled_from_current_runtime(self):
        record = self.fixture._create_bound_expired_task(lease_overrides={"pod_uid": "", "container_id": "", "boot_id": ""})
        with patch.object(self.executor, "_seal_expired_lease_on_runtime", return_value=("fence_seal_unavailable", None)) as seal:
            self.executor.reconcile_expired_scheduler_admission(record)
        seal.assert_called_once()
        for key in ("expected_pod_uid", "expected_container_id", "expected_boot_id"):
            self.assertEqual(seal.call_args.kwargs[key], "")

    def test_reclaim_http_payload_carries_only_lease_identity(self):
        sent = []
        def transport(request):
            sent.append(json.loads(request.content))
            return httpx.Response(409, json={"detail": {"code": "fence_scope_unresolved"}})
        client = httpx.Client(transport=httpx.MockTransport(transport))
        self.addCleanup(client.close)
        self.executor.client = client
        self.executor._resolve_service_runtime_target = Mock(return_value=("http://runtime", {}))
        lease = {"lease_id": "lease-a", "fencing_token": 90,
                 "pod_uid": "bound-pod", "container_id": "bound-container", "boot_id": "bound-boot"}
        self.executor.fetch_receiver_reclaim_proof("h3-fasth3", None, lease)
        self.assertEqual(sent.pop(), lease)
        self.executor._seal_expired_lease_on_runtime("http://runtime", {}, lease_id="lease-a", fencing_token=90,
                                                    expected_pod_uid="bound-pod", expected_container_id="bound-container", expected_boot_id="bound-boot")
        self.assertEqual(sent.pop(), lease)
