import copy
from types import SimpleNamespace
import unittest
from h3_fasth3.image_guides import anchor_images, validate_guides, validate_images
from h3_fasth3.contract import validate_request
from h3_fasth3.errors import RuntimeErrorCode


def tensor(*shape):
    return SimpleNamespace(shape=shape)


class ImageGuidesTests(unittest.TestCase):
    def setUp(self):
        self.images = [tensor(1, 941, 1672, 3), tensor(1, 1024, 768, 3), tensor(1, 1024, 768, 3)]
        self.guides = [{'image_index': 1, 'frame_index': 0}, {'image_index': 2, 'frame_index': 110},
                       {'image_index': 2, 'frame_index': 123}]

    def test_exact_image_reuse_preserves_references_audio_and_all_conditioning(self):
        refs = [{'kind': 'image', 'latent': object()}, {'kind': 'video', 'latent': object()}]
        audio = {'resolved_frame_index': 0, 'audio_latent': object()}
        original = [[object(), {'minimax_refs': refs, 'minimax_keyframes': [audio]}],
                    [object(), {'other': 'unchanged'}]]
        calls = []
        def encode(image):
            calls.append(image)
            return tensor(1, 24, 1, 64, 48)
        result = anchor_images(original, self.images, self.guides, 768, 1024, 124, encode)
        self.assertEqual(calls, self.images[1:])
        self.assertIs(result[0][1]['minimax_refs'], refs)
        self.assertIs(result[0][1]['minimax_keyframes'][0], audio)
        keys = result[0][1]['minimax_keyframes'][1:]
        self.assertEqual([k['resolved_frame_index'] for k in keys], [0, 110, 123])
        self.assertIs(keys[1]['latent'], keys[2]['latent'])
        self.assertIsNot(keys[0]['latent'], keys[1]['latent'])
        self.assertEqual(len(original[0][1]['minimax_keyframes']), 1)
        self.assertNotIn('minimax_keyframes', original[1][1])
        self.assertIs(result[1][0], original[1][0])
        self.assertEqual(len(result[1][1]['minimax_keyframes']), 3)

    def test_conflicts_in_later_conditioning_block_before_encoding(self):
        calls = []
        for field in ('latent', 'image'):
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeErrorCode, 'conflicting'):
                anchor_images([[None, {}], [None, {'minimax_keyframes': [{field: object()}]}]],
                    self.images, self.guides, 768, 1024, 124, lambda image: calls.append(image))
        self.assertFalse(calls)

    def test_bad_canvas_or_encoded_shape_is_rejected(self):
        for shape in ((2, 1024, 768, 3), (1, 941, 1672, 3), (1, 1024, 768, 4)):
            with self.subTest(shape=shape), self.assertRaisesRegex(RuntimeErrorCode, 'canvas'):
                validate_images([self.images[0], tensor(*shape), self.images[2]], self.guides, 768, 1024, 124)
        for shape in ((1, 24, 2, 64, 48), (1, 24, 1, 63, 48), (1, 32, 1, 64, 48)):
            with self.subTest(shape=shape), self.assertRaisesRegex(RuntimeErrorCode, 'latent_shape'):
                anchor_images([[None, {}]], self.images, self.guides, 768, 1024, 124, lambda image: tensor(*shape))

    def test_strict_indices_bounds_order_and_no_clip_fields(self):
        invalid = [[], {}, self.guides * 3,
                   [{'image_index': 0, 'frame_index': 124}], [{'image_index': 3, 'frame_index': 0}],
                   [{'image_index': 0, 'frame_index': -1}], [{'image_index': 0, 'frame_index': 0, 'frames': 14}],
                   self.guides[::-1], [self.guides[0], self.guides[0]]]
        for field in ('image_index', 'frame_index'):
            for value in (True, '0', 0.0, None):
                invalid.append([{**self.guides[0], field: value}])
        for guides in invalid:
            with self.subTest(guides=guides), self.assertRaises(RuntimeErrorCode):
                validate_guides(guides, 3, 124)
        validate_guides(self.guides, 3, 124, 'reference')
        validate_guides(None, 0, 124, 'first_frame')
        with self.assertRaisesRegex(RuntimeErrorCode, 'conflicting'):
            validate_guides(self.guides, 3, 124, 'first_frame')

    def test_runtime_contract_rejects_invalid_guides_without_changing_old_modes(self):
        payload = {'idempotency_key': 'image-guides-test', 'model': 'MiniMaxAI/MiniMax-H3', 'task': 'ref2va',
                   'prompt': 'Fashion camera', 'seconds': 5, 'target': {'aspect_ratio': '3:4'},
                   'conditions': [{'type': 'image', 'role': 'reference', 'uri': 'http://video/runtime-artifacts/art_'+'a'*32+'/content',
                                   'size': 10, 'sha256': 'b'*64}]}
        for extra in ({}, {'image_guides': None}, {'image_guidance': 'first_frame'},
                      {'image_guides': [{'image_index': 0, 'frame_index': 123}]}):
            validate_request({**payload, **extra}, 'http://video')
        for extra in ({'image_guides': [{'image_index': 0, 'frame_index': 124}]},
                      {'image_guides': []}, {'image_guides': [{'image_index': 0, 'frame_index': 0}], 'image_guidance': 'first_frame'}):
            with self.subTest(extra=extra), self.assertRaises(RuntimeErrorCode):
                validate_request({**payload, **extra}, 'http://video')
