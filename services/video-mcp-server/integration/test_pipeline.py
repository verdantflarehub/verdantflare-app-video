"""CPU fixtures exercise real HTTP, persistence, FFmpeg and Artifact contracts, not model quality."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cv2
import httpx
import numpy as np
from fastapi import HTTPException
from fastapi.testclient import TestClient

from video_sr.runtime import Queue, create_app
from video_interpolation.runtime import Queue as InterpolationQueue, create_app as create_interpolation_app
from video_sr.schemas import SRRequest
from video_interpolation.schemas import InterpolationRequest
from video_sr.media import probe, validate_request, MediaError
from video_interpolation.media import is_scene_cut, interpolate_frames, validate_request as validate_interpolation, MediaError as InterpolationMediaError
from video_interpolation.integrity import verify_bundle
from app.artifacts import ArtifactStore
from app.executor import VideoExecutor
from app.tasks import TaskStore


class FixtureEngine:
    def __init__(self, service):
        files = ['seedvr2_ema_3b.pth', 'ema_vae.pth', 'pos_emb.pt', 'neg_emb.pt'] if service == 'sr' else ['flownet.pkl']
        self.metadata = dict(backend='seedvr2' if service == 'sr' else 'rife', code_revision='a' * 40,
                             weights_revision='b' * 40, manifest_sha256='c' * 64,
                             weights_sha256={name: 'd' * 64 for name in files})
        self.calls = 0

    def select_window(self, media, req):
        return dict(window_frames=9, overlap_frames=4, capacity_profile_sha256='e'*64,
                    reserve_bytes=1024**3, seam_method='linear_overlap', window_seed='source_offset', scene_cut_threshold=.30)

    def restore_window(self, frames, req):
        self.calls += 1
        return np.stack([cv2.resize(frame, (req.target_width, req.target_height)) for frame in frames])

    def midpoint(self, left, right):
        self.calls += 1
        return ((left.astype(np.uint16) + right.astype(np.uint16)) // 2).astype(np.uint8)


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        path = self.root / 'input.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=64x48:rate=24',
                        '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000', '-t', '0.25',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(path)], check=True)
        self.artifacts = ArtifactStore(self.root / 'shared')
        self.source = self.artifacts.create_from_chunks(project_id='p', operation='import', filename='input.mp4',
            media_type='video/mp4', chunks=[path.read_bytes()])
        self.queues, self.clients = {}, {}
        for service, request_type in [('sr', SRRequest), ('interpolate', InterpolationRequest)]:
            queue = (Queue if service == 'sr' else InterpolationQueue)(self.root / service, self.root / 'shared', FixtureEngine(service))
            app = (create_app if service == 'sr' else create_interpolation_app)(lambda: None)
            app.state.queue = queue
            self.queues[service], self.clients[service] = queue, TestClient(app)
        def transport(request):
            service = request.url.path.split('/')[2]
            response = self.clients[service].request(request.method, request.url.path, content=request.content,
                                                    headers={'content-type': 'application/json'})
            return httpx.Response(response.status_code, content=response.content, headers=response.headers)
        self.http = httpx.Client(transport=httpx.MockTransport(transport))
        self.executor = VideoExecutor(self.artifacts, TaskStore(self.root / 'shared'), self.http)

    def tearDown(self):
        self.http.close()
        for client in self.clients.values():
            client.close()
        for queue in self.queues.values():
            queue.close()
        self.temp.cleanup()

    def request(self, service='sr', **values):
        common = dict(project_id='p', idempotency_key='video_task_' + 'a'*32,
                      source_artifact_id=self.source.artifact_id, source_sha256=self.source.sha256)
        common.update(dict(target_width=128, target_height=96) if service == 'sr' else {})
        common.update(values)
        return common

    def generate_sr(self):
        return self.executor.processing['sr'].generate(project_id='p', idempotency_key='sr-1',
            source_artifact_id=self.source.artifact_id, target_width=128, target_height=96)

    def test_chain_real_media_audio_preview_and_retry_only_failed_stage(self):
        sr = self.generate_sr()
        self.assertEqual(sr.status, 'queued')
        self.queues['sr'].run_one()
        sr = self.executor.result(sr.video_task_id)
        self.assertEqual((sr.media['width'], sr.media['height'], sr.media['frames'], sr.media['frame_rate']), (128, 96, 6, '24'))
        self.assertTrue(sr.media['audio'])
        adapter = self.executor.processing['interpolate']
        def submit(key):
            return adapter.generate(project_id='p', idempotency_key=key, source_artifact_id=sr.artifact_id)
        failed = submit('interpolate-failed')
        with patch.object(self.queues['interpolate'].engine, 'midpoint', side_effect=RuntimeError('GPU failure')):
            self.queues['interpolate'].run_one()
        self.assertEqual(self.executor.status(failed.video_task_id).status, 'failed')
        self.assertEqual(submit('interpolate-failed').video_task_id, failed.video_task_id)
        retry = submit('interpolate-retry')
        self.queues['interpolate'].run_one()
        result = self.executor.result(retry.video_task_id)
        self.assertEqual((result.media['frames'], result.media['frame_rate'], result.media['duration_seconds']), (12, '48', .25))
        self.assertTrue(result.media['audio'])
        self.assertEqual(self.queues['sr'].engine.calls, 1)
        self.assertEqual(result.request['source_artifact_id'], sr.artifact_id)
        for preview in (False, True):
            public = adapter.public_result(result, preview=preview)
            artifact = self.artifacts.get(public['artifact_id'], 'p')
            path = self.artifacts.content_path(artifact)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), public['sha256'])
            self.assertEqual(probe(path)['width'], 256 if preview else 128)
        repeat = self.executor.result(retry.video_task_id)
        self.assertEqual(result.artifact_id, repeat.artifact_id)

    def test_internal_schema_and_project_isolation(self):
        for service in self.clients:
            client = self.clients[service]
            self.assertEqual(client.get('/health').status_code, 503)
            for changes in ({'source_artifact_id': '/tmp/input.mp4'}, {'source_artifact_id': 'https://example.com/video'},
                            {'source_sha256': 'x'*64}, {'extra': 1}, {'backend': 'unavailable'}, {'project_id': '../p'}):
                self.assertEqual(client.post(f'/v1/{service}', json=self.request(service, **changes)).status_code, 422)
            self.assertEqual(client.post(f'/v1/{service}', json=self.request(service, project_id='other')).status_code, 404)
            self.assertEqual(client.post(f'/v1/{service}', json=self.request(service, source_sha256='0'*64)).status_code, 422)
        for changes in ({'target_width': 127}, {'target_width': True}, {'target_height': 4096}):
            self.assertEqual(self.clients['sr'].post('/v1/sr', json=self.request(**changes)).status_code, 422)

    def test_queue_idempotency_restart_and_interruption(self):
        queue = self.queues['sr']
        req = SRRequest(**self.request())
        first = queue.submit(req)
        self.assertEqual(queue.submit(req), first)
        with self.assertRaises(HTTPException) as error:
            queue.submit(req.model_copy(update={'seed': 667}))
        self.assertEqual(error.exception.status_code, 409)
        queue.close()
        queue = self.queues['sr'] = Queue(self.root / 'sr', self.root / 'shared', FixtureEngine('sr'))
        self.assertEqual(queue.get(req.idempotency_key)['status'], 'queued')
        queue.db.execute("UPDATE tasks SET state='running'")
        queue.db.commit()
        queue.close()
        queue = self.queues['sr'] = Queue(self.root / 'sr', self.root / 'shared', FixtureEngine('sr'))
        self.assertEqual(queue.get(req.idempotency_key)['error']['code'], 'interrupted')
        self.assertFalse(queue.run_one())

    def test_source_modified_after_enqueue_fails_without_inference(self):
        task = self.generate_sr()
        self.artifacts.content_path(self.source).write_bytes(b'tampered')
        self.queues['sr'].run_one()
        status = self.executor.status(task.video_task_id)
        self.assertEqual(status.error['code'], 'source_integrity_failed')
        self.assertEqual(self.queues['sr'].engine.calls, 0)

    def test_scene_cut_holds_left_frame_and_duration(self):
        black = np.zeros((48, 64, 3), dtype=np.uint8)
        white = np.full_like(black, 255)
        self.assertTrue(is_scene_cut(black, white))
        engine = FixtureEngine('interpolate')
        with patch('video_interpolation.media.read_frames', return_value=iter([black, white])):
            frames = list(interpolate_frames(engine, None, {}))
        self.assertEqual(len(frames), 4)
        np.testing.assert_array_equal(frames[1], black)
        np.testing.assert_array_equal(frames[-1], white)
        self.assertEqual(engine.calls, 0)

    def test_fractional_rate_and_explicit_limits(self):
        media = dict(width=64, height=48, frames=6, frame_rate='30000/1001', duration_seconds=.2002)
        req = InterpolationRequest(**self.request('interpolate', target_fps_num=60000, target_fps_den=1001))
        self.assertEqual(validate_interpolation(req, media)['frame_rate'], '60000/1001')
        with self.assertRaises(InterpolationMediaError):
            validate_interpolation(req.model_copy(update={'target_fps_num': 60000, 'target_fps_den': 1000}), media)
        req = SRRequest(**self.request())
        self.assertEqual(validate_request(req, {**media, 'frames': 100001})['frames'], 100001)
        for changed in ({**media, 'width': 62},):
            with self.assertRaises(MediaError):
                validate_request(req, changed)

    def test_fractional_rate_real_video_without_audio(self):
        path = self.root / 'fractional.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=64x48:rate=30000/1001',
                        '-frames:v', '6', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(path)], check=True)
        source = self.artifacts.create_from_chunks(project_id='p', operation='import', filename='fractional.mp4',
            media_type='video/mp4', chunks=[path.read_bytes()])
        task = self.executor.processing['interpolate'].generate(project_id='p', idempotency_key='fractional',
            source_artifact_id=source.artifact_id, target_fps_num=60000, target_fps_den=1001)
        self.queues['interpolate'].run_one()
        result = self.executor.result(task.video_task_id)
        self.assertEqual(result.media['frames'], 12)
        self.assertEqual(result.media['frame_rate'], '60000/1001')
        self.assertAlmostEqual(result.media['duration_seconds'], .2002)
        self.assertFalse(result.media['audio'])

    def test_whole_long_video_exceeds_old_frame_and_duration_limits(self):
        path = self.root / 'long.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=gray:size=64x48:rate=2',
                        '-frames:v', '302', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(path)], check=True)
        source = self.artifacts.create_from_chunks(project_id='p', operation='import', filename='long.mp4',
            media_type='video/mp4', chunks=[path.read_bytes()])
        sr = self.executor.processing['sr'].generate(project_id='p', idempotency_key='long-sr',
            source_artifact_id=source.artifact_id, target_width=128, target_height=96)
        self.queues['sr'].run_one()
        result = self.executor.result(sr.video_task_id)
        self.assertEqual((result.media['frames'], result.media['duration_seconds']), (302, 151))
        windows = result.media['parameters']['windowing']
        self.assertGreater(windows['chunks'], 1)
        self.assertEqual(windows['max_chunk_frames'], 9)
        task = self.executor.processing['interpolate'].generate(project_id='p', idempotency_key='long-rife',
            source_artifact_id=result.artifact_id, target_fps_num=4)
        self.queues['interpolate'].run_one()
        final = self.executor.result(task.video_task_id)
        self.assertEqual((final.media['frames'], final.media['frame_rate'], final.media['duration_seconds']), (604, '4', 151))

    def test_later_window_failure_never_publishes_partial_artifact(self):
        path = self.root / 'chunks.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=gray:size=64x48:rate=24',
                        '-frames:v', '20', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(path)], check=True)
        self.source = self.artifacts.create_from_chunks(project_id='p', operation='import', filename='chunks.mp4',
            media_type='video/mp4', chunks=[path.read_bytes()])
        task = self.generate_sr()
        engine = self.queues['sr'].engine
        restore = engine.restore_window
        def fail_after_first(frames, req):
            if engine.calls:
                raise RuntimeError('later window failed')
            return restore(frames, req)
        with patch.object(engine, 'restore_window', side_effect=fail_after_first):
            self.queues['sr'].run_one()
        self.assertEqual(self.executor.status(task.video_task_id).status, 'failed')
        with self.assertRaises(HTTPException):
            self.queues['sr'].content(task.video_task_id, 'output.mp4')
        self.assertIsNone(self.executor.tasks.get(task.video_task_id).artifact_id)

    def test_queue_capacity_and_unknown_task(self):
        queue = self.queues['sr']
        for i in range(8):
            queue.submit(SRRequest(**self.request(idempotency_key=f'video_task_{i:032x}')))
        with self.assertRaises(HTTPException) as error:
            queue.submit(SRRequest(**self.request()))
        self.assertEqual(error.exception.status_code, 429)
        self.assertEqual(queue.submit(SRRequest(**self.request(idempotency_key='video_task_'+'0'*32)))['status'], 'queued')
        for task_id in ('../etc/passwd', 'video_task_'+'f'*32):
            with self.assertRaises(HTTPException) as error:
                queue.get(task_id)
            self.assertEqual(error.exception.status_code, 404)

    def test_model_bundle_hashes_and_no_overwrite(self):
        from video_interpolation.prepare_model import prepare
        source, bundle = self.root / 'weights', self.root / 'bundle'
        source.mkdir()
        (source / 'flownet.pkl').write_bytes(b'fixture, not model weights')
        lock = dict(backend='rife', code_revision='a'*40, weight_files=['flownet.pkl'], repository='fixture')
        lock_path = self.root / 'lock.json'
        lock_path.write_text(json.dumps(lock))
        prepare(lock_path, source, bundle, 'b'*40)
        with self.assertRaises(FileExistsError):
            prepare(lock_path, source, bundle, 'b'*40)
        with patch('video_interpolation.integrity.subprocess.run') as run:
            run.return_value.stdout = 'a'*40
            self.assertEqual(verify_bundle(bundle, lock, self.root)['backend'], 'rife')
            (bundle / 'flownet.pkl').write_bytes(b'tampered')
            with self.assertRaisesRegex(RuntimeError, 'model_integrity_failed'):
                verify_bundle(bundle, lock, self.root)

    def test_model_bundle_cannot_replace_pinned_weights_and_manifest_together(self):
        from video_interpolation.prepare_model import prepare
        source, bundle = self.root / 'locked-weights', self.root / 'locked-bundle'
        source.mkdir()
        (source / 'flownet.pkl').write_bytes(b'pinned fixture')
        digest = hashlib.sha256(b'pinned fixture').hexdigest()
        lock = dict(backend='rife', code_revision='a'*40, weight_files=['flownet.pkl'], repository='fixture',
                    weights_revision='b'*40, weight_sha256={'flownet.pkl': digest})
        lock_path = self.root / 'pinned-lock.json'
        lock_path.write_text(json.dumps(lock))
        with self.assertRaisesRegex(ValueError, 'revision does not match'):
            prepare(lock_path, source, bundle, 'c'*40)
        prepare(lock_path, source, bundle, 'b'*40)
        manifest = json.loads((bundle / 'manifest.json').read_text())
        manifest['files']['flownet.pkl'] = hashlib.sha256(b'replaced').hexdigest()
        (bundle / 'flownet.pkl').write_bytes(b'replaced')
        (bundle / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(RuntimeError, 'model_integrity_failed'):
            verify_bundle(bundle, lock, self.root)

    def test_content_addressed_weights_require_pinned_file_hashes(self):
        from video_interpolation.prepare_model import prepare
        source, bundle = self.root / 'content-weights', self.root / 'content-bundle'
        source.mkdir()
        (source / 'flownet.pkl').write_bytes(b'content fixture')
        revision = 'sha256:' + 'b' * 64
        lock = dict(backend='rife', repository='fixture', code_revision='a'*40,
                    weight_files=['flownet.pkl'])
        lock_path = self.root / 'content-lock.json'
        lock_path.write_text(json.dumps(lock))
        with self.assertRaisesRegex(ValueError, 'require pinned'):
            prepare(lock_path, source, bundle, revision)
        lock.update(weights_revision=revision,
                    weight_sha256={'flownet.pkl': hashlib.sha256(b'content fixture').hexdigest()})
        lock_path.write_text(json.dumps(lock))
        prepare(lock_path, source, bundle, revision)
        with patch('video_interpolation.integrity.subprocess.run') as run:
            run.return_value.stdout = 'a'*40
            self.assertEqual(verify_bundle(bundle, lock, self.root)['weights_revision'], revision)
            unpinned = {key: value for key, value in lock.items() if key != 'weight_sha256'}
            with self.assertRaisesRegex(RuntimeError, 'model_integrity_failed'):
                verify_bundle(bundle, unpinned, self.root)

    def test_missing_model_bundle_cannot_claim_ready(self):
        with self.assertRaisesRegex(RuntimeError, 'model_integrity_failed'):
            verify_bundle(self.root / 'missing', {}, self.root)


if __name__ == '__main__':
    unittest.main()
