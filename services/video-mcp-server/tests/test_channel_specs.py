import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.artifacts import ArtifactStore
from app.executor import VideoExecutor
from app.providers.fal import FAL_PUBLIC_MODEL, FAL_SD2_PUBLIC_MODEL, FAL_LIP_PUBLIC_MODEL
from app.tasks import TaskStore


class ChannelSpecificationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        env = patch.dict(os.environ, {'H3_RUNTIME_ROUTE': 'h3', 'H3_RUNTIME_ROUTES': '', 'H3_SOL_RUNTIME_URL': ''})
        env.start()
        self.addCleanup(env.stop)
        self.executor = VideoExecutor(ArtifactStore(Path(temp.name)), TaskStore(Path(temp.name)))
        self.addCleanup(self.executor.client.close)
        self.addCleanup(self.executor.higgsfield.client.close)
        self.addCleanup(self.executor.grok.download_client.close)
        self.catalog = {'execution': {}, 'channel_capabilities': {}}
        for channel, models in {'higgsfield': ['seedance_2_5', 'seedance_2_5_draft', 'wan3_0', 'grok_video_v15_lite'],
                'fal': [FAL_PUBLIC_MODEL, FAL_SD2_PUBLIC_MODEL, FAL_LIP_PUBLIC_MODEL],
                'grok': ['grok-imagine-video', 'grok-imagine-video-1.5']}.items():
            self.catalog['execution'][channel] = {}
            self.catalog['channel_capabilities'][channel] = {'capabilities': []}
            for model in models:
                self.catalog['execution'][channel][model] = {'model': model, 'capability_id': model,
                    'state': 'adapter_connected_unverified', 'can_submit': False}
                self.catalog['channel_capabilities'][channel]['capabilities'].append({
                    'capability_id': model, 'display_name': 'Registered human name', 'parameter_schema': {}})

    def routes(self):
        with patch.object(self.executor, 'catalog_view', return_value=self.catalog):
            return {(item['route'], item['model']): item for item in self.executor.capabilities()['routes']}

    def test_higgsfield_keeps_each_models_own_modes_resolutions_and_defaults(self):
        routes = self.routes()
        for model, resolutions, modes in [
            ('seedance_2_5', ['720p'], ['text_to_video']),
            ('seedance_2_5_draft', ['480p'], ['reference_to_video']),
            ('wan3_0', ['480p', '720p'], ['text_to_video', 'reference_to_video']),
            ('grok_video_v15_lite', ['480p', '720p'], ['text_to_video'])]:
            with self.subTest(model=model):
                route = routes['higgsfield', model]
                self.assertEqual(route['resolutions'], resolutions)
                self.assertEqual(route['modes'], modes)
                self.assertNotIn('models', route)
        self.assertEqual(routes['higgsfield', 'seedance_2_5_draft']['defaults']['resolution'], '480p')
        self.assertEqual(routes['higgsfield', 'wan3_0']['defaults']['resolution'], '720p')
        self.assertEqual(routes['higgsfield', 'seedance_2_5']['reference_limits']['total'], 0)
        self.assertEqual(routes['higgsfield', 'wan3_0']['reference_limits']['total'], 5)

    def test_fal_h3_and_seedance_have_distinct_constraints_without_borrowing(self):
        routes = self.routes()
        h3, sd2 = routes['fal', FAL_PUBLIC_MODEL], routes['fal', FAL_SD2_PUBLIC_MODEL]
        self.assertEqual(h3['duration_seconds']['minimum'], 5)
        self.assertEqual(sd2['duration_seconds']['minimum'], 4)
        self.assertNotIn('resolution', h3['defaults'])
        self.assertEqual(sd2['defaults']['resolution'], '720p')
        self.assertEqual(h3['reference_limits']['max_bytes']['images'], 20 * 1024 * 1024)
        self.assertEqual(sd2['reference_limits']['max_bytes']['images'], 30 * 1024 * 1024)
        self.assertEqual(h3['reference_limits']['request_body_max_bytes'], 96 * 1024 * 1024)
        self.assertIsNone(h3['reference_limits']['max_total_bytes'])
        self.assertEqual(sd2['reference_limits']['total'], 12)

    def test_declared_bounds_intersect_adapter_and_no_template_is_not_invented(self):
        cap = next(item for item in self.catalog['channel_capabilities']['higgsfield']['capabilities']
                   if item['capability_id'] == 'wan3_0')
        cap['parameter_schema'] = {'properties': {'duration_seconds': {'minimum': 6, 'maximum': 9},
                                                'resolution': {'enum': ['720p']}}}
        self.catalog['execution']['higgsfield']['future'] = {'model': 'future-model', 'capability_id': 'future',
                                                           'state': 'enabled', 'can_submit': False}
        routes = self.routes()
        self.assertEqual(routes['higgsfield', 'wan3_0']['duration_seconds'], {'minimum': 6, 'maximum': 9})
        self.assertEqual(routes['higgsfield', 'wan3_0']['resolutions'], ['720p'])
        self.assertNotIn(('higgsfield', 'future-model'), routes)

    def test_lip_sync_keeps_upstream_resolution_mapping_and_exact_media_contract(self):
        cap = next(item for item in self.catalog['channel_capabilities']['fal']['capabilities']
                   if item['capability_id'] == FAL_LIP_PUBLIC_MODEL)
        cap['parameter_schema'] = {'properties': {'resolution': {'enum': ['480P', '768P']}}}
        route = self.routes()['fal', FAL_LIP_PUBLIC_MODEL]
        self.assertEqual(route['resolutions'], ['480p', '768p'])
        self.assertEqual(route['defaults'], {'mode': 'lip_sync', 'resolution': '768p', 'aspect_ratio': 'adaptive'})
        self.assertEqual(route['reference_limits']['minimum_total'], 2)
        self.assertEqual(route['reference_limits']['audios'], 1)
        self.assertEqual(route['reference_limits']['videos'], 0)
        self.assertEqual(route['reference_limits']['formats']['audios'], ['WAV/PCM'])

    def test_each_grok_row_has_one_execution_identity_and_one_image_limit(self):
        self.executor.grok.channel_mode = True
        routes = self.routes()
        for model in ('grok-imagine-video', 'grok-imagine-video-1.5'):
            route = routes['grok', model]
            self.assertNotIn('models', route)
            self.assertEqual(route['model'], model)
            self.assertEqual(route['reference_limits']['total'], 1)
            self.assertEqual(route['reference_limits']['max_bytes']['images'], 10 * 1024 * 1024)
