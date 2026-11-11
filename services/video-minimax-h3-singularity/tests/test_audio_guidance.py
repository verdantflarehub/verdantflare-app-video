import copy
import unittest
from types import SimpleNamespace

from h3_singularity.audio_guidance import anchor_encoded_audio, validate_timeline_pcm
from h3_singularity.contract import validate_request
from h3_singularity.errors import RuntimeErrorCode


def tensor(*shape):
    return SimpleNamespace(shape=shape, ndim=len(shape))


def request():
    return {'idempotency_key': 'test', 'model': 'MiniMaxAI/MiniMax-H3',
            'task': 'ref2va', 'prompt': 'fixed scene', 'seconds': 8,
            'target': {'aspect_ratio': '9:16'}, 'conditions': [
                {'type': 'audio', 'role': 'reference',
                 'uri': 'http://video/runtime-artifacts/art_' + 'a' * 32 + '/content',
                 'size': 100, 'sha256': 'b' * 64}]}


class TimelineAudioTests(unittest.TestCase):
    def test_full_track_and_single_sample_rounding(self):
        for samples in (384000, 383999, 384001):
            result = validate_timeline_pcm({'waveform': tensor(1, 1, samples), 'sample_rate': 48000}, 192)
            self.assertEqual(result['start_frame'], 0)
        # 107 frames is 4.45833s, not the requested 4s: use the resolved bucket.
        validate_timeline_pcm({'waveform': tensor(1, 2, 214000), 'sample_rate': 48000}, 107)

    def test_short_long_empty_and_requested_not_resolved_duration_rejected(self):
        for samples in (0, 100000, 383998, 384002, 480000):
            with self.subTest(samples=samples), self.assertRaisesRegex(RuntimeErrorCode, 'duration_mismatch'):
                validate_timeline_pcm({'waveform': tensor(1, 1, samples), 'sample_rate': 48000}, 192)
        with self.assertRaisesRegex(RuntimeErrorCode, 'duration_mismatch'):
            validate_timeline_pcm({'waveform': tensor(1, 1, 192000), 'sample_rate': 48000}, 107)

    def test_invalid_pcm_shape_rate_rejected(self):
        for shape, rate in [((2, 1, 384000), 48000), ((1, 3, 384000), 48000),
                            ((1, 384000), 48000), ((1, 1, 384000), True),
                            ((1, 1, 384000), 0)]:
            with self.subTest(shape=shape, rate=rate), self.assertRaises(RuntimeErrorCode):
                validate_timeline_pcm({'waveform': tensor(*shape), 'sample_rate': rate}, 192)

    def test_anchor_reuses_tensor_preserves_image_reference_and_input(self):
        encoded = tensor(1, 32, 2, 320)
        refs = [{'kind': 'image', 'latent': object()}, {'kind': 'audio', 'audio_latent': encoded}]
        visual_keyframe = {'resolved_frame_index': 0, 'latent': object()}
        original = [[object(), {'minimax_refs': refs, 'minimax_keyframes': [visual_keyframe]}]]
        result = anchor_encoded_audio(original, 320)
        self.assertIs(result[0][0], original[0][0])
        self.assertIs(result[0][1]['minimax_refs'], refs)
        self.assertEqual(len(original[0][1]['minimax_keyframes']), 1)
        self.assertIs(result[0][1]['minimax_keyframes'][0], visual_keyframe)
        self.assertIs(result[0][1]['minimax_keyframes'][1]['audio_latent'], encoded)
        self.assertEqual(result[0][1]['minimax_keyframes'][1]['resolved_frame_index'], 0)

    def test_encoded_length_is_not_silently_cropped_or_padded(self):
        for shape in [(1, 32, 2, 319), (1, 32, 2, 321), (1, 32, 1, 320)]:
            with self.subTest(shape=shape), self.assertRaisesRegex(RuntimeErrorCode, 'latent_length_mismatch'):
                anchor_encoded_audio([[None, {'minimax_refs': [{'kind': 'audio', 'audio_latent': tensor(*shape)}]}]], 320)

    def test_missing_multiple_embedded_and_conflicting_audio_rejected(self):
        audio = {'kind': 'audio', 'audio_latent': tensor(1, 32, 2, 320)}
        for refs in ([], [audio, audio], [audio, {'kind': 'video_audio'}]):
            with self.assertRaises(RuntimeErrorCode):
                anchor_encoded_audio([[None, {'minimax_refs': refs}]], 320)
        with self.assertRaisesRegex(RuntimeErrorCode, 'conflicting'):
            anchor_encoded_audio([[None, {'minimax_refs': [audio], 'minimax_keyframes': [{'audio_latent': object()}]}]], 320)

    def test_default_contract_unchanged_opt_in_is_explicit(self):
        payload = request()
        before = copy.deepcopy(payload)
        validate_request(payload, 'http://video')
        self.assertEqual(payload, before)
        for mode in ('reference', 'timeline'):
            validate_request({**payload, 'audio_guidance': mode}, 'http://video')
        for mode in (None, True, {}, 'lip_sync'):
            with self.subTest(mode=mode), self.assertRaises(RuntimeErrorCode):
                validate_request({**payload, 'audio_guidance': mode}, 'http://video')

    def test_timeline_contract_rejects_video_or_multiple_tracks(self):
        payload = request()
        payload['audio_guidance'] = 'timeline'
        for kind in ('audio', 'video'):
            candidate = copy.deepcopy(payload)
            candidate['conditions'].append({**candidate['conditions'][0], 'type': kind})
            with self.subTest(kind=kind), self.assertRaisesRegex(RuntimeErrorCode, 'one_audio_and_no_video'):
                validate_request(candidate, 'http://video')
        payload['conditions'][0]['type'] = 'image'
        with self.assertRaises(RuntimeErrorCode):
            validate_request(payload, 'http://video')


if __name__ == '__main__':
    unittest.main()
