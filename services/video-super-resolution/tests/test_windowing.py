import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from video_sr.capacity import CapacityError, CapacityProfile, search_capacity
from video_sr.media import is_scene_cut
from video_sr.schemas import SRRequest
from video_sr.windowing import restore_windows


class IdentityEngine:
    def __init__(self):
        self.calls = []

    def restore_window(self, frames, req):
        self.calls.append((len(frames), req.seed))
        return frames.copy()


class WindowTest(unittest.TestCase):
    def setUp(self):
        self.req = SRRequest(project_id='p', idempotency_key='video_task_'+'a'*32,
            source_artifact_id='art_'+'b'*32, source_sha256='c'*64, target_width=16, target_height=16)
        self.policy = dict(window_frames=9, overlap_frames=4)

    def restore(self, engine, frames, cut=lambda a, b: False):
        report = dict(chunks=0, max_chunk_frames=0)
        result = list(restore_windows(engine, frames, self.req, self.policy, cut, report))
        return result, report

    def test_no_missing_or_duplicate_frames_for_short_tail_and_long_video(self):
        for count in (1, 4, 8, 9, 10, 13, 14, 100, 101, 241, 1001):
            with self.subTest(count=count):
                source = [np.full((16, 16, 3), i % 256, dtype=np.uint8) for i in range(count)]
                engine = IdentityEngine()
                actual, report = self.restore(engine, iter(source))
                np.testing.assert_array_equal(actual, source)
                self.assertEqual(report['output_frames'], count)
                self.assertLessEqual(report['max_chunk_frames'], 9)
                self.assertEqual([seed for _, seed in engine.calls], [666 + 5 * i for i in range(len(engine.calls))])

    def test_bounded_read_ahead_before_first_output(self):
        pulled = []
        def source():
            for i in range(10000):
                pulled.append(i)
                yield np.zeros((16, 16, 3), dtype=np.uint8)
        output = restore_windows(IdentityEngine(), source(), self.req, self.policy,
                                 lambda a, b: False, dict(chunks=0, max_chunk_frames=0))
        next(output)
        self.assertEqual(len(pulled), 10)  # One window plus one frame to detect its end/cut.
        output.close()

    def test_overlap_blends_same_timestamp_estimates(self):
        class BiasedEngine(IdentityEngine):
            def restore_window(self, frames, req):
                return np.full_like(frames, req.seed - 666)
        source = iter([np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(14)])
        result, _ = self.restore(BiasedEngine(), source)
        self.assertEqual([int(f[0, 0, 0]) for f in result], [0]*5 + [1, 2, 3, 4] + [5]*5)

    def test_hard_cuts_do_not_mix_scenes_or_reset_global_seed(self):
        class SingleSceneEngine(IdentityEngine):
            def restore_window(self, frames, req):
                self_outer.assertEqual(len(np.unique(frames)), 1)
                return super().restore_window(frames, req)
        self_outer = self
        for cut_at in (4, 9, 10, 13, 20):
            source = [np.full((16, 16, 3), 0 if i < cut_at else 255, dtype=np.uint8) for i in range(40)]
            engine = SingleSceneEngine()
            actual, _ = self.restore(engine, iter(source), is_scene_cut)
            np.testing.assert_array_equal(actual, source)
            self.assertIn(666 + cut_at, [seed for _, seed in engine.calls])

    def test_later_window_failure_is_not_a_truncated_success(self):
        class BrokenEngine(IdentityEngine):
            def restore_window(self, frames, req):
                if req.seed != 666:
                    raise RuntimeError('inference failure')
                return frames.copy()
        with self.assertRaisesRegex(RuntimeError, 'inference failure'):
            self.restore(BrokenEngine(), (np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(20)))


class CapacityTest(unittest.TestCase):
    def test_exponential_and_binary_search_finds_aligned_boundary(self):
        result = search_capacity(lambda n: 'ok' if n <= 33 else 'oom', 257)
        self.assertEqual(result['window_frames'], 33)
        self.assertEqual(result['next_rejected_frames'], 37)
        self.assertEqual(result['bound'], 'measured_budget_boundary')
        self.assertTrue(all(len(t['statuses']) == (3 if t['statuses'][0] == 'ok' else 1) for t in result['trials']))

    def test_search_ceiling_is_not_claimed_as_maximum(self):
        result = search_capacity(lambda n: 'ok', 100)
        self.assertEqual(result['window_frames'], 97)
        self.assertIsNone(result['next_rejected_frames'])
        self.assertEqual(result['bound'], 'lower_bound_only')

    def test_memory_reserve_boundary(self):
        result = search_capacity(lambda n: 'ok' if n <= 17 else 'headroom_exceeded', 129)
        self.assertEqual(result['window_frames'], 17)
        self.assertEqual(result['next_rejected_frames'], 21)

    def test_dependency_failure_is_not_an_oom_result(self):
        for status in ('initialization_failed', 'inference_failed', 'trial_timeout'):
            calls = []
            def trial(n):
                calls.append(n)
                return status
            with self.assertRaisesRegex(CapacityError, 'non_capacity'):
                search_capacity(trial, 129)
            self.assertEqual(calls, [9])
        with self.assertRaisesRegex(CapacityError, 'minimum_window'):
            search_capacity(lambda n: 'oom', 129)

    def test_profile_rejects_unmeasured_geometry_and_runtime_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'capacity.json'
            data = dict(schema_version=1, status='measured', identity={'fixture': True}, source_sha256='a'*64,
                        reserve_bytes=1024**3, baseline_used_bytes=1024**3,
                        geometry=dict(input_width=16, input_height=16, target_width=32, target_height=32),
                        result=search_capacity(lambda n: 'ok', 17))
            path.write_text(json.dumps(data))
            profile = CapacityProfile(path)
            from types import SimpleNamespace
            req = SimpleNamespace(target_width=32, target_height=32)
            media = dict(width=16, height=16)
            self.assertEqual(profile.select({'fixture': True}, media, req)['window_frames'], 17)
            with self.assertRaises(CapacityError):
                profile.select({'fixture': 'different'}, media, req)
            with self.assertRaises(CapacityError):
                profile.select({'fixture': True}, dict(width=32, height=32), req)
            data['result']['trials'][1]['statuses'] = ['ok']
            path.write_text(json.dumps(data))
            with self.assertRaises(CapacityError):
                CapacityProfile(path).select({'fixture': True}, media, req)
