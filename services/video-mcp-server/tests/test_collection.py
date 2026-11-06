import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import httpx

from app.artifacts import ArtifactStore
from app.collection import ResultCollector
from app.executor import VideoExecutor
from app.providers.fal import FalError
from app.tasks import TaskStore


class CollectionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tasks = TaskStore(Path(self.tmp.name))
        self.client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
        self.addCleanup(self.client.close)
        self.executor = VideoExecutor(ArtifactStore(Path(self.tmp.name)), self.tasks, self.client)
        self.collector = self.executor.collector
        self.addCleanup(self.collector.close)

    def task(self, service='fal'):
        record = self.tasks.create(project_id='demo', idempotency_key=str(time.time_ns()),
            input_digest='digest', request={'route': service}, runtime_task_id='remote', status='succeeded')
        return self.tasks.update(record, service=service)

    def collected(self, record):
        record = self.tasks.get(record.video_task_id)
        if record.artifact_id:
            return record
        artifact = self.executor.artifacts.create_from_chunks(project_id='demo', operation='video.result',
            filename='video.mp4', media_type='video/mp4', chunks=[b'validated-video'])
        return self.tasks.update(record, artifact_id=artifact.artifact_id)

    def test_scan_observation_reports_scale_without_task_identity(self):
        with self.assertLogs('uvicorn.error', level='INFO') as logs:
            self.collector.observe_scan(491, 0.025)
        self.assertIn('files=491', logs.output[0])
        self.assertIn('p95_ms=25.0', logs.output[0])
        self.assertNotIn('video_task_', logs.output[0])
        self.collector.last_scan_log = 0
        with self.assertLogs('uvicorn.error', level='WARNING') as logs:
            self.collector.observe_scan(5000, 0.201)
        self.assertIn('files=5000', logs.output[0])

    def test_retry_is_durable_and_does_not_regenerate(self):
        record = self.task()
        with patch.object(self.executor.fal, 'result', side_effect=FalError('secret-signed-url')) as result, patch.object(self.executor, 'generate') as generate:
            self.collector.collect(record.video_task_id)
            failed = self.tasks.get(record.video_task_id)
            self.assertEqual(failed.status, 'succeeded')
            self.assertEqual(failed.collection['state'], 'retry_wait')
            self.assertNotIn('secret', str(failed.collection))
            replacement = ResultCollector(self.executor)
            self.addCleanup(replacement.close)
            replacement.collect(record.video_task_id)
            self.assertEqual(result.call_count, 1)
            generate.assert_not_called()
        self.tasks.update(failed, collection={**failed.collection, 'next_attempt_at': 0})
        with patch.object(self.executor.fal, 'result', side_effect=self.collected):
            replacement.collect(record.video_task_id)
        ready = self.tasks.get(record.video_task_id)
        self.assertEqual(ready.collection['state'], 'ready')
        self.assertEqual(ready.collection['attempts'], 2)
        self.assertTrue(ready.artifact_id)

    def test_slow_download_does_not_block_status_other_result_or_submission_lock(self):
        slow, other = self.task(), self.task()
        started, release = threading.Event(), threading.Event()
        def download(record):
            if record.video_task_id == slow.video_task_id:
                started.set()
                self.assertTrue(release.wait(5))
            return self.collected(record)
        with patch.object(self.executor.fal, 'result', side_effect=download), ThreadPoolExecutor(3) as pool:
            future = pool.submit(self.collector.collect, slow.video_task_id)
            try:
                self.assertTrue(started.wait(2))
                self.assertEqual(pool.submit(self.executor.status, slow.video_task_id).result(1).status, 'succeeded')
                self.assertTrue(pool.submit(self.executor.result, other.video_task_id).result(1).artifact_id)
                def identity_lock():
                    with self.executor._lock, self.tasks.exclusive():
                        return True
                self.assertTrue(pool.submit(identity_lock).result(1))
            finally:
                release.set()
            future.result(2)

    def test_automatic_and_manual_result_share_one_download(self):
        record = self.task()
        started, release = threading.Event(), threading.Event()
        downloads = []
        def download(rec):
            if rec.artifact_id:
                return rec
            downloads.append(rec.video_task_id)
            started.set()
            self.assertTrue(release.wait(5))
            return self.collected(rec)
        with patch.object(self.executor.fal, 'result', side_effect=download), ThreadPoolExecutor(2) as pool:
            auto = pool.submit(self.collector.collect, record.video_task_id)
            try:
                self.assertTrue(started.wait(2))
                manual = pool.submit(self.executor.result, record.video_task_id)
            finally:
                release.set()
            auto.result(2)
            self.assertEqual(manual.result(2).artifact_id, self.tasks.get(record.video_task_id).artifact_id)
        self.assertEqual(len(downloads), 1)

    def test_interrupted_download_recovers_and_ready_is_not_recollected(self):
        rec = self.task()
        self.tasks.update(rec, collection={'state': 'downloading', 'attempts': 1})
        with patch.object(self.executor.fal, 'result', side_effect=self.collected) as result:
            self.collector.collect(rec.video_task_id)
            self.collector.collect(rec.video_task_id)
            self.assertEqual(result.call_count, 1)

    def test_stale_update_keeps_artifact_and_retry_state(self):
        stale = self.task()
        self.tasks.update(stale, artifact_id='art_kept', collection={'state': 'ready'})
        updated = self.tasks.update(stale, runtime_stage='completed')
        self.assertEqual(updated.artifact_id, 'art_kept')
        self.assertEqual(updated.collection, {'state': 'ready'})

    def test_separate_store_obeys_task_lock_without_global_lock(self):
        rec = self.task()
        other = TaskStore(Path(self.tmp.name))
        with self.tasks.task_lock(rec.video_task_id):
            with other.task_lock(rec.video_task_id, blocking=False) as acquired:
                self.assertFalse(acquired)
            with self.tasks.task_lock(rec.video_task_id) as nested:
                self.assertTrue(nested)

    def test_pool_is_bounded_and_does_not_queue_entire_backlog(self):
        for _ in range(8):
            self.task()
        release = threading.Event()
        with patch.object(self.collector, 'collect', side_effect=lambda _: release.wait(5)):
            try:
                self.collector.sync()
                self.collector.sync()
                self.assertEqual(len(self.collector.active), self.collector.workers)
            finally:
                release.set()
                for future in self.collector.active.values():
                    future.result(2)

    def test_existing_managed_collectors_and_preview_recovery(self):
        for service in ('grok', 'higgsfield', 'h3-latent-upscale'):
            self.assertFalse(self.collector.eligible(self.task(service)))
        for service in ('depth', 'sr', 'interpolate'):
            rec = self.tasks.update(self.task(service), artifact_id='already-present')
            self.assertTrue(self.collector.eligible(rec))
            rec = self.tasks.update(rec, media={'preview_artifact_id': 'preview'})
            self.assertFalse(self.collector.eligible(rec))

    def test_real_fal_adapter_retries_transport_and_commits_artifact(self):
        rec = self.task()
        self.executor.fal.key = "unit-test-key"
        attempts = []
        def transport(request):
            self.assertEqual(request.method, 'GET')
            if request.url.host.endswith('fal.media'):
                attempts.append(1)
                if len(attempts) == 1:
                    raise httpx.ReadTimeout('temporary CDN stall', request=request)
                return httpx.Response(200, content=b'validated-video', headers={'content-type': 'video/mp4'})
            return httpx.Response(200, json={'video': {'url': 'https://v3.fal.media/files/result.mp4'}})
        with httpx.Client(transport=httpx.MockTransport(transport)) as client, patch.object(self.executor, 'client', client), patch.object(self.executor.fal, '_probe', return_value={'duration_ms': 10000}):
            self.collector.collect(rec.video_task_id)
            rec = self.tasks.get(rec.video_task_id)
            self.assertEqual(rec.collection['state'], 'retry_wait')
            self.tasks.update(rec, collection={**rec.collection, 'next_attempt_at': 0})
            self.collector.collect(rec.video_task_id)
        rec = self.tasks.get(rec.video_task_id)
        self.assertEqual(rec.collection['state'], 'ready')
        artifact = self.executor.artifacts.get(rec.artifact_id, 'demo')
        self.assertEqual(self.executor.artifacts.content_path(artifact).read_bytes(), b'validated-video')
        self.assertEqual(len(attempts), 2)

    def test_status_during_dispatch_reads_snapshot_without_polling_provider(self):
        rec = self.tasks.update(self.task(), status='queued', runtime_task_id='')
        with patch.object(self.executor.fal, 'status') as status, ThreadPoolExecutor(1) as pool:
            with self.executor._lock, self.tasks.exclusive():
                snapshot = pool.submit(self.executor.status, rec.video_task_id).result(1)
                self.assertEqual(snapshot.status, 'queued')
            status.assert_not_called()

    def test_slow_but_progressing_cdn_stream_can_finish_after_ninety_seconds(self):
        clock = iter((0, 100, 100))
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(
                200, content=b'video', headers={'content-type': 'video/mp4'}))) as client:
            with patch.object(self.executor, 'client', client), patch('app.providers.fal.time.monotonic', side_effect=lambda: next(clock, 100)):
                self.assertEqual(self.executor.fal._download('https://v3.fal.media/result.mp4'), b'video')

    def stale_collection(self, record):
        return self.tasks.update(record, collection={
            'state': 'retry_wait', 'attempts': 11,
            'started_at': '2026-10-28T09:07:28+00:00',
            'completed_at': None, 'next_attempt_at': time.time() + 300,
            'last_error': 'result_collection_failed'})

    def test_manual_recovery_clears_retry_without_changing_generation_or_identity(self):
        record = self.stale_collection(self.task())
        before = record.model_dump()
        with patch.object(self.executor.fal, 'result', side_effect=self.collected) as result:
            ready = self.executor.result(record.video_task_id)
            again = self.executor.result(record.video_task_id)
        self.assertEqual(result.call_count, 2)
        self.assertEqual(ready.collection['state'], 'ready')
        self.assertEqual(ready.collection['attempts'], 11)
        self.assertIsNone(ready.collection['last_error'])
        self.assertEqual(ready.collection['next_attempt_at'], 0)
        artifact = self.executor.artifacts.get(ready.artifact_id, 'demo')
        self.assertEqual(ready.collection['completed_at'], artifact.created_at)
        for key in ('video_task_id', 'input_digest', 'runtime_task_id', 'created_at', 'completed_at', 'request'):
            self.assertEqual(ready.model_dump()[key], before[key])
        self.assertEqual(again.artifact_id, ready.artifact_id)
        self.assertEqual(again.collection, ready.collection)
        self.assertEqual(again.updated_at, ready.updated_at)

    def test_background_reconciles_cached_result_without_provider_or_download(self):
        record = self.stale_collection(self.collected(self.task()))
        record = self.tasks.update(record, collection={**record.collection, 'next_attempt_at': 0})
        artifact = self.executor.artifacts.get(record.artifact_id, 'demo')
        with patch.object(self.executor, 'result') as result, patch.object(self.executor, 'generate') as generate:
            self.collector.sync()
            for future in list(self.collector.active.values()):
                future.result(2)
        recovered = TaskStore(Path(self.tmp.name)).get(record.video_task_id)
        self.assertEqual(recovered.collection['state'], 'ready')
        self.assertEqual(recovered.collection['completed_at'], artifact.created_at)
        self.assertEqual(recovered.collection['attempts'], 11)
        self.assertEqual(recovered.artifact_id, record.artifact_id)
        result.assert_not_called()
        generate.assert_not_called()

    def test_cached_artifact_must_be_same_project_present_and_hash_verified(self):
        for failure in ('missing', 'corrupt', 'other_project'):
            with self.subTest(failure=failure):
                record = self.stale_collection(self.collected(self.task()))
                artifact = self.executor.artifacts.get(record.artifact_id, 'demo')
                path = self.executor.artifacts.content_path(artifact)
                if failure == 'missing':
                    path.unlink()
                elif failure == 'corrupt':
                    path.write_bytes(b'corrupted-video')  # same size, different digest
                else:
                    record = self.tasks.update(record, project_id='other-project')
                record = self.tasks.update(record, collection={**record.collection, 'next_attempt_at': 0})
                with patch.object(self.executor, 'result') as result:
                    self.collector.sync()
                    for future in list(self.collector.active.values()):
                        future.result(2)
                current = self.tasks.get(record.video_task_id)
                self.assertEqual(current.collection['state'], 'retry_wait')
                self.assertEqual(current.collection['attempts'], 11)
                self.assertGreater(current.collection['next_attempt_at'], time.time())
                result.assert_not_called()

    def test_manual_incomplete_preview_does_not_become_ready(self):
        record = self.stale_collection(self.collected(self.task('sr')))
        with patch.object(self.executor.processing['sr'], 'result', return_value=record):
            returned = self.executor.result(record.video_task_id)
        self.assertEqual(returned.collection['state'], 'retry_wait')

    def test_late_collection_error_does_not_regress_persisted_result(self):
        record = self.stale_collection(self.task())
        self.tasks.update(record, collection={**record.collection, 'next_attempt_at': 0})
        def committed_then_lost_response(rec):
            self.collected(rec)
            raise FalError('response_lost_after_save')
        with patch.object(self.executor.fal, 'result', side_effect=committed_then_lost_response):
            self.collector.collect(record.video_task_id)
        current = self.tasks.get(record.video_task_id)
        self.assertEqual(current.collection['state'], 'ready')
        self.assertIsNone(current.collection['last_error'])

    def test_preview_bytes_required_and_completion_uses_last_artifact(self):
        record = self.stale_collection(self.collected(self.task('sr')))
        preview = self.executor.artifacts.create_from_chunks(project_id='demo', operation='video.preview',
            filename='preview.mp4', media_type='video/mp4', chunks=[b'preview'])
        record = self.tasks.update(record, media={'preview_artifact_id': preview.artifact_id})
        path = self.executor.artifacts.content_path(preview)
        path.write_bytes(b'corrupt')
        with patch.object(self.executor.processing['sr'], 'result', return_value=record):
            self.assertEqual(self.executor.result(record.video_task_id).collection['state'], 'retry_wait')
            path.write_bytes(b'preview')
            ready = self.executor.result(record.video_task_id)
        self.assertEqual(ready.collection['state'], 'ready')
        self.assertEqual(ready.collection['completed_at'], preview.created_at)

    def test_unknown_artifact_time_stays_unknown_after_reconciliation(self):
        record = self.stale_collection(self.collected(self.task()))
        artifact = self.executor.artifacts.get(record.artifact_id, 'demo')
        metadata = self.executor.artifacts.artifacts_root / artifact.artifact_id / 'metadata.json'
        metadata.write_text(artifact.model_copy(update={'created_at': 'not-a-time'}).model_dump_json())
        with patch.object(self.executor.fal, 'result', return_value=record):
            ready = self.executor.result(record.video_task_id)
        self.assertEqual(ready.collection['state'], 'ready')
        self.assertIsNone(ready.collection['completed_at'])

    def test_reconciling_legacy_record_does_not_invent_generation_time(self):
        record = self.stale_collection(self.collected(self.task()))
        record = record.model_copy(update={'completed_at': None, 'dispatched_at': None})
        self.tasks._write(record)
        with patch.object(self.executor.fal, 'result', return_value=record):
            ready = self.executor.result(record.video_task_id)
        self.assertEqual(ready.collection['state'], 'ready')
        self.assertIsNone(ready.completed_at)
        self.assertIsNone(ready.dispatched_at)
        self.assertIsNone(ready.timing['total_seconds'])
