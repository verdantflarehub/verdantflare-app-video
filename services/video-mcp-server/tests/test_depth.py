import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import httpx
from verdantflare_video_mcp.artifacts import ArtifactNotFound, ArtifactStore
from verdantflare_video_mcp.executor import VideoExecutor, ExecutionError
from verdantflare_video_mcp.tasks import TaskConflict, TaskStore


class DepthMCPTest(unittest.TestCase):
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
        self.payload = b'depth artifact fixture'
        self.digest = hashlib.sha256(self.payload).hexdigest()
        self.client = httpx.Client(transport=httpx.MockTransport(self.respond))
        self.executor = VideoExecutor(self.artifacts, self.tasks, self.client)
        self.depth = self.executor.depth

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
        media = dict(width=720, height=1280, frames=300, frame_rate='30', duration_seconds=10, video_codec='h264')
        result = dict(input_sha256=self.source.sha256, output_sha256=self.digest, preview_sha256=self.digest,
                      media=media, input_media=media,
                      model=dict(model='video-depth-anything', encoder='vits', code_revision='a'*40,
                                 repository='depth-anything/Video-Depth-Anything-Small', revision='b'*40, sha256='c'*64),
                      parameters=dict(input_size=518, fp32=False, normalization='whole_video_minmax', depth_min=0,
                                      depth_max=1, audio=False, crf=18), output_path='/private/path', runtime_task_id='secret')
        return httpx.Response(200, json={'video_task_id': self.task_id, 'status': self.state, 'result': result})

    def generate(self, **kwargs):
        return self.depth.generate(**dict(project_id='p', idempotency_key='key', source_artifact_id=self.source.artifact_id, **kwargs))

    def test_submit_and_duplicate_no_second_post(self):
        first = self.generate()
        second = self.generate()
        self.assertEqual(first.video_task_id, second.video_task_id)
        self.assertEqual(len(self.requests), 1)
        body = json.loads(self.requests[0].content)
        self.assertEqual(body['source_artifact_id'], self.source.artifact_id)
        self.assertNotIn(str(self.artifacts.root), json.dumps(body))
        self.assertEqual(first.service, 'depth')

    def test_different_source_conflicts_and_wrong_project_rejected(self):
        self.generate()
        source2 = self.artifacts.create_from_chunks(project_id='p', operation='import', filename='b.mp4', media_type='video/mp4', chunks=[b'b'])
        with self.assertRaises(TaskConflict):
            self.depth.generate(project_id='p', idempotency_key='key', source_artifact_id=source2.artifact_id)
        with self.assertRaises(ArtifactNotFound):
            self.depth.generate(project_id='other', idempotency_key='key', source_artifact_id=self.source.artifact_id)

    def test_reject_nonvideo_and_unsupported_output(self):
        image = self.artifacts.create_from_chunks(project_id='p', operation='import', filename='a.png', media_type='image/png', chunks=[b'img'])
        with self.assertRaises(ValueError):
            self.depth.generate(project_id='p', idempotency_key='key', source_artifact_id=image.artifact_id)
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

    def test_all_depth_tools_registered(self):
        from verdantflare_video_mcp import server
        import asyncio
        names = {t.name for t in asyncio.run(server.mcp.list_tools())}
        self.assertTrue({'video.depth.generate','video.depth.status','video.depth.result','video.depth.preview','video.generate'} <= names)


if __name__ == '__main__':
    unittest.main()

class AssetRewriteTest(unittest.TestCase):
    def test_only_allowlisted_public_prefix_rewrites(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp, patch.dict('os.environ', {
            'VIDEO_ASSET_IMPORT_ORIGINS': 'https://storage.example.com:9090',
            'VIDEO_ASSET_IMPORT_REWRITES': '{"https://storage.example.com:9090/bucket/":"http://storage.internal:8082/tenant:bucket/"}'
        }):
            requests = []
            def respond(request):
                requests.append(request)
                return httpx.Response(200, content=b'video')
            client = httpx.Client(transport=httpx.MockTransport(respond))
            executor = VideoExecutor(ArtifactStore(Path(temp)), TaskStore(Path(temp)), client)
            digest = hashlib.sha256(b'video').hexdigest()
            executor.import_asset(project_id='p', source_url='https://storage.example.com:9090/bucket/project/source.mp4',
                                  filename='source.mp4', expected_sha256=digest)
            self.assertEqual(str(requests[0].url), 'http://storage.internal:8082/tenant:bucket/project/source.mp4')
            with self.assertRaises(ValueError):
                executor.import_asset(project_id='p', source_url='http://storage.internal:8082/tenant:bucket/project/source.mp4',
                                      filename='source.mp4', expected_sha256=digest)
            with self.assertRaises(ValueError):
                executor.import_asset(project_id='p', source_url='https://storage.example.com.evil.test:9090/bucket/source.mp4',
                                      filename='source.mp4', expected_sha256=digest)
            self.assertEqual(len(requests), 1)
            client.close()
