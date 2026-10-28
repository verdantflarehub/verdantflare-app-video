"""Fresh, bounded, read-only channel registration checks for new work."""
from __future__ import annotations

import base64
from datetime import UTC, datetime
import json
import os
import re
from urllib.parse import urlsplit

import httpx


class RegistryUnavailable(ValueError):
    pass


def encoded(value):
    return base64.b64encode(value.encode()).decode()


class ChannelRegistry:
    def __init__(self, origins, *, env=None, transport=None):
        env = os.environ if env is None else env
        self.origins = origins
        self.endpoint = env.get('CHANNEL_DISCOVERY_ETCD_URL', '').rstrip('/')
        self.scope = env.get('CHANNEL_DISCOVERY_SCOPE_ID', '')
        self.username = env.get('ETCD_USERNAME', '')
        self.password = env.get('ETCD_PASSWORD', '')
        self.transport = transport
        self.allow_unverified = frozenset(filter(None, env.get('VIDEO_CHANNEL_ALLOW_UNVERIFIED', '').split(',')))
        if not self.allow_unverified <= {'higgsfield', 'fal', 'grok', 'google'}:
            raise ValueError('invalid_channel_validation_configuration')

    def snapshot(self):
        parsed = urlsplit(self.endpoint)
        if (parsed.scheme not in {'http', 'https'} or not parsed.hostname
                or not parsed.hostname.endswith('.svc.cluster.local')
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path or parsed.port not in {None, 2379}
                or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', self.scope)
                or not self.username or not self.password):
            raise RegistryUnavailable('channel_registry_not_configured')
        with httpx.Client(base_url=self.endpoint, timeout=3, trust_env=False,
                          follow_redirects=False, transport=self.transport) as client:
            def post(path, body, token=None):
                with client.stream('POST', '/v3/' + path, json=body,
                                   headers={'Authorization': token} if token else {}) as response:
                    content = bytearray()
                    for chunk in response.iter_bytes():
                        content.extend(chunk)
                        if len(content) > 2 * 1024 * 1024:
                            raise RegistryUnavailable('channel_registry_response_too_large')
                    if response.status_code != 200:
                        raise RegistryUnavailable('channel_registry_unavailable')
                value = json.loads(content)
                if not isinstance(value, dict) or 'error' in value or value.get('code'):
                    raise RegistryUnavailable('channel_registry_unavailable')
                return value
            token = post('auth/authenticate', {'name': self.username, 'password': self.password}).get('token')
            if not isinstance(token, str) or not token or len(token) > 16384 or any(c.isspace() for c in token):
                raise RegistryUnavailable('channel_registry_unavailable')
            prefix = f'/verdantflare/channels/v1/{self.scope}/'
            result = post('kv/range', {'key': encoded(prefix),
                'range_end': encoded(prefix[:-1] + chr(ord(prefix[-1]) + 1)), 'limit': 512}, token)
            if result.get('more'):
                raise RegistryUnavailable('channel_registry_response_too_large')
            values = {}
            for raw in result.get('kvs', []):
                key = base64.b64decode(raw['key'], validate=True).decode()
                value = json.loads(base64.b64decode(raw['value'], validate=True))
                if not key.startswith(prefix) or key in values or not isinstance(value, dict):
                    raise RegistryUnavailable('channel_registry_invalid')
                values[key] = (value, int(raw.get('lease', 0)))
            live_leases = {}
            channels = {}
            for channel, origin in self.origins.items():
                base = prefix + channel + '/'
                status, lease = values.get(base + 'status', ({}, 0))
                entries = [(value, instance_lease) for key, (value, instance_lease) in values.items()
                           if key.startswith(base + 'instances/') and '/' not in key[len(base + 'instances/'):]]
                generations = [e.get('generation', 0) for e, _ in entries]
                if any(type(generation) is not int or generation < 1 for generation in generations):
                    continue
                if not lease or status.get('state') != 'enabled' or not entries:
                    continue
                observed = datetime.fromisoformat(status.get('observed_at', '').replace('Z', '+00:00'))
                if not 0 <= (datetime.now(UTC) - observed).total_seconds() < 300:
                    continue
                if lease not in live_leases:
                    live_leases[lease] = int(post('lease/timetolive', {'ID': str(lease)}, token).get('TTL', 0)) > 0
                if not live_leases[lease]:
                    continue
                matches = [entry for entry, entry_lease in entries
                           if entry_lease == lease and entry.get('schema_version') == 1
                           and entry.get('service_identity') == channel
                           and entry.get('generation') == max(generations)
                           and entry.get('execution_url', '').rstrip('/') == origin
                           and isinstance(entry.get('capabilities'), list)]
                if len(matches) == 1:
                    channels[channel] = {'state': 'enabled', 'capabilities': matches[0]['capabilities'],
                                         'version': matches[0].get('version'), 'lease_valid': True}
            return channels

    def permits(self, channel, capability, registrations):
        entry = registrations.get(channel, {})
        states = {'enabled'} | ({'adapter_connected_unverified'} if channel in self.allow_unverified else set())
        return entry.get('lease_valid') is True and any(
            isinstance(item, dict) and item.get('capability_id') == capability
            and item.get('execution_state') in states for item in entry.get('capabilities', []))
