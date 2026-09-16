import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import httpx
from verdantflare_video_mcp.artifacts import ArtifactNotFound, ArtifactStore
from verdantflare_video_mcp.executor import VideoExecutor, ExecutionError
from verdantflare_video_mcp.tasks import TaskConflict, TaskStore


class SRMCPTest(unittest.TestCase):
    service = 'sr'

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.artifacts = ArtifactStore(Path(self.temp.name))
        self.tasks = TaskStore(Path(self.temp.name))
        self.source = self.artifacts.create_from_chunks(project_id='p', operation='import', filename='input.mp4',
                                                        media_type='video/mp4', chunks=[b'rgb fixture'])
        self.requests = []
        self.state = 'queued'
        self.timeout = False
        self.lost = False
        self.payload = b'processed artifact fixture'
        self.digest = hashlib.sha256(self.payload).hexdigest()
        self.client = httpx.Client(transport=httpx.MockTransport(self.respond))
        self.executor = VideoExecutor(self.artifacts, self.tasks, self.client)
        self.depth = self.executor.processing[self.service]

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def respond(self, request):
        self.requests.append(request)
        if request.method == 'POST':
            self.task_id = json.loads(request.content)['idempotency_key']
            if self.timeout:
                raise httpx.ReadTimeout('internal hostname secret', request=request)
            return httpx.Response(202, json={'video_task_id': self.task_id, 'status': self.state})
        if request.url.path.endswith(('/content', '/preview')):
            return httpx.Response(200, content=self.payload)
        if self.lost:
            return httpx.Response(404)
        source = dict(width=64, height=48, frames=6, frame_rate='24', duration_seconds=.25, video_codec='h264', audio=False)
        media = {**source, **({'width': 128, 'height': 96} if self.service == 'sr' else {'frames': 12, 'frame_rate': '48'})}
        parameters = dict(self.parameters())
        parameters.update(audio='aac_if_present', video_codec='h264', pixel_format='yuv420p', crf=18)
        parameters.update({'sample_steps': 1, 'cfg_scale': 1.0, 'normalization': 'torch_rms_layer', 'color_fix': False} if self.service == 'sr' else {'scene_cut_threshold': .30, 'last_frame': 'hold'})
        if self.service == 'sr':
            parameters['windowing'] = dict(window_frames=9, overlap_frames=4, capacity_profile_sha256='e'*64,
                reserve_bytes=1024**3, seam_method='linear_overlap', window_seed='source_offset',
                scene_cut_threshold=.30, chunks=1, max_chunk_frames=6, output_frames=6)
        files = ['seedvr2_ema_3b.pth', 'ema_vae.pth', 'pos_emb.pt', 'neg_emb.pt'] if self.service == 'sr' else ['flownet.pkl']
        result = dict(input_sha256=self.source.sha256, output_sha256=self.digest, preview_sha256=self.digest,
                      media=media, input_media=source, preview_media={**media, 'width': media['width'] * 2},
                      model=dict(backend='seedvr2' if self.service == 'sr' else 'rife', code_revision='a'*40,
                                 weights_revision=('b'*40 if self.service == 'sr' else 'sha256:'+'b'*64), manifest_sha256='c'*64, weights_sha256={f:'d'*64 for f in files}),
                      parameters=parameters, output_path='/private/path', runtime_task_id='secret')
        if getattr(self, 'bad_metadata', False):
            result['media']['frames'] += 1

        return httpx.Response(200, json={'video_task_id': self.task_id, 'status': self.state, 'result': result})

    def parameters(self):
        return (dict(target_width=128, target_height=96, quality_mode='standard', backend='seedvr2', seed=666)
                if self.service == 'sr' else dict(target_fps_num=48, target_fps_den=1, backend='rife'))

    def generate(self, **kwargs):
        values = dict(project_id='p', idempotency_key='key', source_artifact_id=self.source.artifact_id, **self.parameters())
        values.update(kwargs)
        return self.depth.generate(**values)

    def test_submit_and_duplicate_no_second_post(self):
        first = self.generate()
        second = self.generate()
        self.assertEqual(first.video_task_id, second.video_task_id)
        self.assertEqual(len(self.requests), 1)
        body = json.loads(self.requests[0].content)
        self.assertEqual(body['source_artifact_id'], self.source.artifact_id)
        self.assertNotIn(str(self.artifacts.root), json.dumps(body))
        self.assertEqual(first.service, self.service)

    def test_different_source_conflicts_and_wrong_project_rejected(self):
        self.generate()
        source2 = self.artifacts.create_from_chunks(project_id='p', operation='import', filename='b.mp4', media_type='video/mp4', chunks=[b'b'])
        with self.assertRaises(TaskConflict):
            self.generate(source_artifact_id=source2.artifact_id)
        with self.assertRaises(ArtifactNotFound):
            self.generate(project_id='other')

    def test_reject_nonvideo_and_unsupported_output(self):
        image = self.artifacts.create_from_chunks(project_id='p', operation='import', filename='a.png', media_type='image/png', chunks=[b'img'])
        with self.assertRaises(ValueError):
            self.generate(source_artifact_id=image.artifact_id)
        with self.assertRaises(ValueError):
            self.generate(output_format='png')
        self.assertEqual(self.requests, [])

    def test_timeout_recovers_same_id_and_persisted_key(self):
        self.timeout = True
        task = self.generate()
        self.assertEqual(task.error['code'], 'submission_unconfirmed')
        self.assertNotIn('hostname', json.dumps(self.depth.public_status(task)))
        self.timeout = False
        self.lost = True
        recovered = self.depth.status(task.video_task_id)
        self.assertEqual(recovered.video_task_id, task.video_task_id)
        posts = [r for r in self.requests if r.method == 'POST']
        self.assertEqual(posts[0].content, posts[1].content)
        self.assertEqual(recovered.status, 'queued')

    def test_result_registers_once_and_hides_internal_details(self):
        task = self.generate()
        self.state = 'succeeded'
        result = self.executor.result(task.video_task_id)
        public = self.depth.public_result(result)
        preview = self.depth.public_result(result, preview=True)
        self.assertNotEqual(public['artifact_id'], preview['artifact_id'])
        self.assertEqual(public['sha256'], self.digest)
        self.assertEqual(len(list(self.artifacts.artifacts_root.glob('art_*'))), 3)
        self.depth.result(task.video_task_id)
        self.assertEqual(len(list(self.artifacts.artifacts_root.glob('art_*'))), 3)
        self.assertNotIn('/private', json.dumps(public))
        self.assertNotIn('runtime_task_id', json.dumps(public))

    def test_status_timeout_is_retryable(self):
        task = self.generate()
        self.client.close()
        self.depth.client = httpx.Client(transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(httpx.ReadTimeout('timeout'))))
        with self.assertRaises(ExecutionError):
            self.depth.status(task.video_task_id)
        self.assertEqual(self.tasks.get(task.video_task_id).status, 'queued')
        self.depth.client.close()

    def test_lost_confirmed_task_fails_without_regeneration(self):
        task = self.generate()
        self.lost = True
        self.assertEqual(self.depth.status(task.video_task_id).error['code'], 'runtime_task_lost')
        self.assertEqual(len([r for r in self.requests if r.method == 'POST']), 1)

    def test_all_processing_tools_registered(self):
        from verdantflare_video_mcp import server
        import asyncio
        names = {t.name for t in asyncio.run(server.mcp.list_tools())}
        self.assertTrue({f'video.{self.service}.{a}' for a in ('generate', 'status', 'result', 'preview')} <= names)


    def test_wrong_service_and_cross_service_idempotency(self):
        task = self.generate()
        other = self.executor.processing['interpolate' if self.service == 'sr' else 'sr']
        with self.assertRaises(ValueError):
            other.status(task.video_task_id)
        with self.assertRaises(TaskConflict):
            self.executor.depth.generate(project_id='p', idempotency_key='key', source_artifact_id=self.source.artifact_id)

    def test_bad_metadata_never_registers_output(self):
        task = self.generate()
        self.state = 'succeeded'
        self.bad_metadata = True
        with self.assertRaises(ExecutionError):
            self.executor.result(task.video_task_id)
        self.assertEqual(len(list(self.artifacts.artifacts_root.glob('art_*'))), 1)

    def test_crash_before_submit_keeps_processing_identity(self):
        from unittest.mock import patch
        with patch.object(self.depth, '_submit', side_effect=RuntimeError('process crashed')):
            with self.assertRaises(RuntimeError):
                self.generate()
        task = self.tasks.find_idempotency('p', 'key')
        self.assertEqual(task.service, self.service)
        self.lost = True
        recovered = self.executor.status(task.video_task_id)
        self.assertEqual(recovered.video_task_id, task.video_task_id)
        self.assertEqual(recovered.status, 'queued')


class InterpolationMCPTest(SRMCPTest):
    service = 'interpolate'

    def test_equivalent_rational_rates_deduplicate(self):
        first = self.generate(target_fps_num=48000, target_fps_den=1000)
        second = self.generate(target_fps_num=48, target_fps_den=1)
        self.assertEqual(first.video_task_id, second.video_task_id)
        self.assertEqual(len(self.requests), 1)
