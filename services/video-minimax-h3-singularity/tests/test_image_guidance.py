import unittest
from types import SimpleNamespace
from h3_singularity.image_guidance import anchor_first_image, validate_first_image
from h3_singularity.audio_guidance import anchor_encoded_audio
from h3_singularity.contract import validate_request
from h3_singularity.errors import RuntimeErrorCode
from test_audio_guidance import request


def tensor(*shape):
    return SimpleNamespace(shape=shape)


class FirstFrameTests(unittest.TestCase):
    def test_exact_canvas_only(self):
        validate_first_image([tensor(1, 1344, 768, 3)], 768, 1344)
        for images in ([], [tensor(1, 1671, 941, 3)], [tensor(2, 1344, 768, 3)], [tensor(1, 1344, 768, 4)]):
            with self.subTest(images=images), self.assertRaises(RuntimeErrorCode):
                validate_first_image(images, 768, 1344)

    def test_preserve_all_references_audio_and_original_metadata(self):
        image = tensor(1, 24, 1, 84, 48)
        audio = tensor(1, 32, 2, 320)
        refs = [{'kind': 'image', 'latent': image}, {'kind': 'image', 'latent': object()},
                {'kind': 'audio', 'audio_latent': audio}]
        original = [[object(), {'minimax_refs': refs}]]
        for audio_first in (False, True):
            start = anchor_encoded_audio(original, 320) if audio_first else original
            result = anchor_first_image(start, 768, 1344)
            if not audio_first:
                result = anchor_encoded_audio(result, 320)
            self.assertIs(result[0][1]['minimax_refs'], refs)
            self.assertNotIn('minimax_keyframes', original[0][1])
            keys = result[0][1]['minimax_keyframes']
            self.assertEqual(len(keys), 2)
            self.assertIs(next(k['latent'] for k in keys if 'latent' in k), image)
            self.assertIs(next(k['audio_latent'] for k in keys if 'audio_latent' in k), audio)
            self.assertTrue(all(k['resolved_frame_index'] == 0 for k in keys))

    def test_missing_wrong_shape_and_visual_conflict_rejected(self):
        for shape in ((1, 24, 1, 83, 48), (1, 24, 2, 84, 48), (1, 32, 1, 84, 48)):
            with self.assertRaisesRegex(RuntimeErrorCode, 'shape_mismatch'):
                anchor_first_image([[None, {'minimax_refs': [{'kind': 'image', 'latent': tensor(*shape)}]}]], 768, 1344)
        with self.assertRaises(RuntimeErrorCode):
            anchor_first_image([[None, {}]], 768, 1344)
        for field in ('image', 'latent'):
            with self.assertRaisesRegex(RuntimeErrorCode, 'conflicting'):
                anchor_first_image([[None, {'minimax_refs': [{'kind': 'image', 'latent': tensor(1, 24, 1, 84, 48)}],
                                           'minimax_keyframes': [{field: object()}]}]], 768, 1344)

    def test_contract_modes_and_image_required(self):
        p = request()
        validate_request(p, 'http://video')
        with self.assertRaisesRegex(RuntimeErrorCode, 'requires_image'):
            validate_request({**p, 'image_guidance': 'first_frame'}, 'http://video')
        p['conditions'][0]['type'] = 'image'
        for mode in ('reference', 'first_frame'):
            validate_request({**p, 'image_guidance': mode}, 'http://video')
        for mode in (None, True, {}, 'last_frame'):
            with self.assertRaises(RuntimeErrorCode):
                validate_request({**p, 'image_guidance': mode}, 'http://video')
