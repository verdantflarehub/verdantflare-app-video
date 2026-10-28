import asyncio
import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, PropertyMock

from app import server
from app.artifacts import ArtifactStore
from app.imports import ImportStore


class ImportToolErrorTests(unittest.TestCase):
    def test_rejected_commit_retains_upload_without_publishing(self):
        # Exercise the real import transaction; only the decoder is substituted.
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(Path(directory))
            def reject(path, media_type):
                raise ValueError('reference_duration_must_be_2_to_15_seconds')
            imports = ImportStore(artifacts, reject)
            payload = b'invalid-duration-fixture'
            digest = hashlib.sha256(payload).hexdigest()
            prepared = imports.prepare(project_id='project-test', idempotency_key='short',
                filename='short.mp4', size=len(payload), sha256=digest, purpose='motion')
            args = dict(project_id='project-test', import_id=prepared['import_id'])
            imports.chunk(**args, offset=0, content_base64=base64.b64encode(payload).decode(), sha256=digest)
            before = imports.status(**args)
            with patch.object(type(server.executor), 'imports', new_callable=PropertyMock, return_value=imports):
                result = server.video_import_commit(**args)
            self.assertTrue(result.is_error)
            self.assertEqual(result.structured_content['error']['code'],
                'reference_duration_must_be_2_to_15_seconds')
            self.assertEqual(imports.status(**args), before)
            self.assertEqual((imports.root / (prepared['import_id'] + '.part')).read_bytes(), payload)
            self.assertFalse(list(artifacts.artifacts_root.glob('art_*')))

    def test_duration_failures_explain_media_and_video_stream_limits(self):
        for code in ('reference_duration_must_be_2_to_15_seconds',
                     'reference_video_duration_must_be_2_to_15_seconds'):
            with self.subTest(code=code), patch.object(server.executor.imports, 'commit', side_effect=ValueError(code)):
                result = server.video_import_commit('project-test', 'imp_' + 'a' * 32)
                self.assertTrue(result.is_error)
                error = result.structured_content['error']
                self.assertEqual(error['code'], code)
                self.assertIn('2 to 15 seconds', error['message'])
                self.assertFalse(error['retryable'])
                self.assertEqual(json.loads(result.content[0].text), result.structured_content)

    def test_unknown_exception_text_is_not_disclosed(self):
        for raw in ('/private/media/key-secret', 'reference_media_decode_failed: token=secret'):
            with self.subTest(raw=raw), patch.object(server.executor.imports, 'commit', side_effect=ValueError(raw)):
                result = server.video_import_commit('project-test', 'imp_' + 'a' * 32)
                self.assertEqual(result.structured_content['error']['code'], 'import_validation_failed')
                self.assertNotIn('secret', result.model_dump_json())
                self.assertNotIn('/private', result.model_dump_json())

    def test_incomplete_upload_remains_recoverable_and_success_is_unchanged(self):
        with patch.object(server.executor.imports, 'commit', side_effect=ValueError('import_incomplete')):
            result = server.video_import_commit('project-test', 'imp_' + 'a' * 32)
            self.assertTrue(result.structured_content['error']['retryable'])
            self.assertIn('original import', result.structured_content['error']['message'])
        payload = {'status': 'committed', 'artifact': {'artifact_id': 'art-test'}}
        with patch.object(server.executor.imports, 'commit', return_value=payload):
            result = server.video_import_commit('project-test', 'imp_' + 'a' * 32)
            self.assertFalse(result.is_error)
            self.assertEqual(result.structured_content, payload)

    def test_prepare_rejects_exact_size_and_format_boundaries_before_allocating(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(Path(directory))
            imports = ImportStore(artifacts, lambda *args: {})
            base = dict(project_id='project-test', idempotency_key='bounds', filename='input.png',
                        size=1, sha256='a' * 64, purpose='reference')
            cases = [({'size': 0}, 'import_size_must_be_1_byte_to_1_GiB'),
                     ({'size': 1073741825}, 'import_size_must_be_1_byte_to_1_GiB'),
                     ({'filename': 'input.gif'}, 'unsupported_import_format')]
            with patch.object(type(server.executor), 'imports', new_callable=PropertyMock, return_value=imports):
                for changes, expected in cases:
                    with self.subTest(changes=changes):
                        result = server.video_import_prepare(**{**base, **changes})
                        self.assertTrue(result.is_error)
                        self.assertEqual(result.structured_content['error']['code'], expected)
                        self.assertEqual(json.loads(result.content[0].text), result.structured_content)
            with imports._transaction(write=False) as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM imports').fetchone()[0], 0)

    def test_chunk_rejection_preserves_original_offset_and_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(Path(directory))
            imports = ImportStore(artifacts, lambda *args: {})
            payload = b'x' * (524288 + 1)
            digest = hashlib.sha256(payload).hexdigest()
            prepared = imports.prepare(project_id='project-test', idempotency_key='chunk',
                filename='input.mp4', size=len(payload), sha256=digest, purpose='motion')
            args = dict(project_id='project-test', import_id=prepared['import_id'])
            with patch.object(type(server.executor), 'imports', new_callable=PropertyMock, return_value=imports):
                result = server.video_import_chunk(**args, offset=0,
                    content_base64=base64.b64encode(payload).decode(), sha256=digest)
            self.assertEqual(result.structured_content['error']['code'], 'import_chunk_size_or_hash_mismatch')
            self.assertEqual(imports.status(**args), prepared)
            self.assertFalse(list(imports.root.glob('*.part')))

    def test_media_format_and_audio_limit_errors_are_specific(self):
        for code in ('reference_audio_requires_mono_or_stereo', 'reference_audio_sample_rate_exceeded',
                     'reference_image_must_be_still', 'reference_video_requires_mp4_h264_or_hevc',
                     'import_final_size_or_hash_mismatch'):
            with self.subTest(code=code), patch.object(server.executor.imports, 'commit', side_effect=ValueError(code)):
                result = server.video_import_commit('project-test', 'imp_' + 'a' * 32)
                self.assertEqual(result.structured_content['error']['code'], code)
                self.assertEqual(result.structured_content['error']['source'], 'video')
                self.assertEqual(json.loads(result.content[0].text), result.structured_content)

    def test_mcp_schema_retains_original_arguments(self):
        tools = asyncio.run(server.mcp.list_tools())
        tool = next(t for t in tools if t.name == 'video.import_commit')
        self.assertEqual(set(tool.input_schema['required']), {'project_id', 'import_id'})
        self.assertEqual(set(tool.input_schema['properties']), {'project_id', 'import_id'})
