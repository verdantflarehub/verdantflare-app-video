"""Capacity evidence must not transfer to a different conditioning bundle."""
import unittest
from h3_latent_upscaler.engine import Engine
from h3_latent_upscaler.resources import ResourceError


class CapacityProfileTests(unittest.TestCase):
    def setUp(self):
        self.engine = object.__new__(Engine)
        self.engine.profile = {
            'id': 'measured', 'source_manifest_sha256': 'a' * 64,
            'source_routes': ['h3-vdn'], 'source_width': 768,
            'source_height': 1344, 'source_frames': 124,
            'target_width': 1152, 'target_height': 2016,
        }
        self.resources = {
            'manifest_sha256': 'a' * 64,
            'manifest': {'source_route': 'h3-vdn',
                         'media': {'width': 768, 'height': 1344, 'frames': 124}},
        }

    def test_calibrated_source_is_accepted(self):
        self.assertEqual(self.engine.validate_request(self.resources, {}), (1152, 2016))

    def test_same_geometry_different_conditioning_is_rejected(self):
        self.resources['manifest_sha256'] = 'b' * 64
        with self.assertRaisesRegex(ResourceError, 'capacity_profile_mismatch'):
            self.engine.validate_request(self.resources, {})

    def test_missing_verified_manifest_hash_is_rejected(self):
        del self.resources['manifest_sha256']
        with self.assertRaisesRegex(ResourceError, 'capacity_profile_mismatch'):
            self.engine.validate_request(self.resources, {})

    def test_changed_geometry_is_still_rejected(self):
        self.resources['manifest']['media']['frames'] = 345
        with self.assertRaisesRegex(ResourceError, 'capacity_profile_mismatch'):
            self.engine.validate_request(self.resources, {})
