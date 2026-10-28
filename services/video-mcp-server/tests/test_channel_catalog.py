from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

from app.channel_catalog import project, read_directory, validate_directory


class ChannelCatalogTests(unittest.TestCase):
    def directory(self, channel='grok', name='Grok Imagine Video'):
        return {'schema_version': 1, 'channel_id': channel, 'channel_name': 'Grok', 'version': 'actual-sha',
            'state': 'available', 'capabilities': [{'capability_id': 'video.actual', 'type': 'video',
            'model_id': 'grok-imagine-video', 'display_name': name, 'operation': 'video.generate',
            'execution_state': 'adapter_connected_unverified', 'parameter_schema': {}}]}

    def executor(self, snapshot=None):
        return SimpleNamespace(catalog=SimpleNamespace(snapshot=lambda: deepcopy(snapshot or {
            'state': 'error', 'channels': [], 'groups': [], 'conflicts': []})),
            fal=SimpleNamespace(gateway_url='', gateway_token=''),
            grok=SimpleNamespace(channel_url='http://grok-channel:8001', channel_token='t' * 32,
                                 capability=lambda: {'configured': True, 'enabled': True}),
            google=SimpleNamespace(channel_url='', channel_token=''),
            higgsfield=SimpleNamespace(url='', token=''))

    def test_new_channel_comes_from_response_without_reviewed_static_entry(self):
        def read(channel, *_):
            return self.directory(channel) if channel == 'grok' else {
                'schema_version': 1, 'channel_id': channel, 'state': 'unavailable', 'capabilities': []}
        registration = {'grok': {'lease_valid': True, 'capabilities': [
            {'capability_id': 'video.actual', 'execution_state': 'adapter_connected_unverified'}]}}
        with patch('app.channel_catalog.read_directory', side_effect=read), patch(
                'app.channel_registry.ChannelRegistry.snapshot', return_value=registration), patch.dict(
                'os.environ', {'VIDEO_CHANNEL_ALLOW_UNVERIFIED': 'grok'}):
            value = project(self.executor())
        self.assertEqual(value['channels'][0]['models'][0]['display_name'], 'Grok Imagine Video')
        self.assertEqual(value['channels'][0]['models'][0]['declaration_source'], 'installed_channel')
        self.assertTrue(value['execution']['grok']['video.actual']['can_submit'])

    def test_unavailable_keeps_reviewed_specs_without_enabling_and_does_not_mutate_snapshot(self):
        old = {'channels': [{'id': 'grok', 'display_name': 'Grok', 'models': [
            {'id': 'old', 'model_id': 'old', 'identity_review': 'pending', 'display_name': 'Reviewed name',
             'entry_kind': 'model', 'business_type': 'video_generation'}]}]}
        with patch('app.channel_catalog.read_directory', side_effect=lambda c, *_: {
                'schema_version': 1, 'channel_id': c, 'state': 'unavailable', 'capabilities': []}), patch(
                'app.channel_registry.ChannelRegistry.snapshot', return_value={}):
            value = project(self.executor(old))
        self.assertEqual(value['execution']['grok']['old']['state'], 'unavailable')
        self.assertNotIn('declaration_source', old['channels'][0]['models'][0])

    def test_invalid_schema_duplicate_and_corrupt_name_rejected(self):
        for alter in (lambda d: d.update(schema_version=2), lambda d: d['capabilities'].append(d['capabilities'][0]),
                      lambda d: d['capabilities'][0].update(display_name='Broken\ufffd')):
            value = self.directory()
            alter(value)
            with self.assertRaises(ValueError):
                validate_directory(value, 'grok')
        value = read_directory('grok', 'http://channel:8001', 't' * 32,
            transport=httpx.MockTransport(lambda r: httpx.Response(503, json={'secret': 'must not escape'})))
        self.assertEqual(value['state'], 'unavailable')
        self.assertNotIn('secret', str(value))
