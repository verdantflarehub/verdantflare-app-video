import base64
from datetime import UTC, datetime
import json
import unittest

import httpx

from app.channel_registry import ChannelRegistry, RegistryUnavailable


class RegistryTests(unittest.TestCase):
    def setup_registry(self, *, state='enabled', capability_state='enabled', ttl=200,
                       lease=10, origin=None, extra=None, extra_lease=12, configured=True, validation=''):
        self.requests = []
        expected = 'http://grok-channel.verdantflare-app.svc.cluster.local:8001'
        prefix = '/verdantflare/channels/v1/dev/grok/'
        documents = [(prefix + 'status', {'state': state, 'observed_at': datetime.now(UTC).isoformat()}, 10),
            (prefix + 'instances/instance', {'schema_version': 1, 'service_identity': 'grok',
             'generation': 2, 'execution_url': origin or expected,
             'capabilities': [{'capability_id': 'video.actual', 'execution_state': capability_state}]}, lease)]
        if extra:
            documents.append((prefix + 'instances/newer', extra, extra_lease))
        def handle(request):
            self.requests.append(request.url.path)
            if request.url.path.endswith('authenticate'):
                return httpx.Response(200, json={'token': 'test-token'})
            if request.url.path.endswith('timetolive'):
                return httpx.Response(200, json={'TTL': ttl})
            encode = lambda value: base64.b64encode(value.encode()).decode()
            return httpx.Response(200, json={'kvs': [{'key': encode(key), 'value': encode(json.dumps(value)), 'lease': str(item_lease)}
                                                    for key, value, item_lease in documents]})
        env = {'CHANNEL_DISCOVERY_ETCD_URL': 'http://etcd.platform.svc.cluster.local:2379',
               'CHANNEL_DISCOVERY_SCOPE_ID': 'dev', 'ETCD_USERNAME': 'reader', 'ETCD_PASSWORD': 'test',
               'VIDEO_CHANNEL_ALLOW_UNVERIFIED': validation} if configured else {}
        return ChannelRegistry({'grok': expected}, env=env, transport=httpx.MockTransport(handle))

    def test_fresh_leased_exact_registration_allows_only_published_capability(self):
        registry = self.setup_registry()
        data = registry.snapshot()
        self.assertTrue(registry.permits('grok', 'video.actual', data))
        self.assertFalse(registry.permits('grok', 'video.other', data))
        self.assertIn('/v3/lease/timetolive', self.requests)
        self.assertFalse(any(path.endswith(('put', 'txn', 'grant')) for path in self.requests))

    def test_expired_disabled_wrong_origin_or_mixed_lease_never_authorize(self):
        for args in ({'ttl': 0}, {'state': 'disabled'}, {'lease': 9}, {'origin': 'http://unexpected.internal'},
                     {'extra': {'generation': 3}},
                     {'extra_lease': 10, 'extra': {'generation': 2, 'schema_version': 1,
                        'service_identity': 'grok', 'execution_url': 'http://grok-channel.verdantflare-app.svc.cluster.local:8001',
                        'capabilities': [{'capability_id': 'video.actual', 'execution_state': 'enabled'}]}}):
            with self.subTest(args=args):
                registry = self.setup_registry(**args)
                self.assertFalse(registry.permits('grok', 'video.actual', registry.snapshot()))

    def test_unverified_is_explicit_per_channel_and_missing_config_fails_closed(self):
        registry = self.setup_registry(capability_state='adapter_connected_unverified')
        self.assertFalse(registry.permits('grok', 'video.actual', registry.snapshot()))
        registry = self.setup_registry(capability_state='adapter_connected_unverified', validation='grok')
        self.assertTrue(registry.permits('grok', 'video.actual', registry.snapshot()))
        with self.assertRaises(RegistryUnavailable):
            self.setup_registry(configured=False).snapshot()
