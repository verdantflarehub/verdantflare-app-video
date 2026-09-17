import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import boto3
from botocore.stub import Stubber, ANY
from botocore.response import StreamingBody
import httpx
from verdantflare_video_mcp.archive import S3Archive, ArchiveError
from verdantflare_video_mcp.artifacts import ArtifactStore


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = ArtifactStore(Path(self.tmp.name))
        self.data = b'immutable-test-output'
        self.artifact = self.store.create_from_chunks(project_id='project-a', operation='test',
            filename='content.mp4', media_type='video/mp4', chunks=[self.data])
        self.client = boto3.client('s3', endpoint_url='https://archive.invalid',
                                  aws_access_key_id='test', aws_secret_access_key='test', region_name='us-east-1')
        self.stub = Stubber(self.client)
        self.stub.activate()
        self.addCleanup(self.stub.deactivate)
        self.archive = S3Archive(self.client, 'test-bucket', 'https://public.invalid')
        self.task = 'video_task_' + 'a' * 32
        self.key = f'video/projects/project-a/h3-latent/{self.task}/{self.artifact.sha256}/content.mp4'
        self.params = {'Bucket': 'test-bucket', 'Key': self.key}

    def get_response(self, data):
        return {'Body': StreamingBody(io.BytesIO(data), len(data)), 'ContentLength': len(data)}

    def run_store(self):
        return self.archive.store(self.artifact, self.store.content_path(self.artifact), self.task, 'content')

    def test_conditional_create_and_both_downloads_verified(self):
        self.stub.add_client_error('head_object', 'NoSuchKey', http_status_code=404, expected_params=self.params)
        self.stub.add_response('put_object', {}, {**self.params, 'Body': ANY, 'ContentLength': len(self.data),
            'ContentType': 'video/mp4', 'Metadata': {'sha256': self.artifact.sha256}, 'IfNoneMatch': '*'})
        self.stub.add_response('get_object', self.get_response(self.data), self.params)
        response = httpx.Response(200, content=self.data, request=httpx.Request('GET', 'https://public.invalid'))
        with patch('verdantflare_video_mcp.archive.httpx.stream') as stream:
            stream.return_value.__enter__.return_value = response
            result = self.run_store()
        self.assertEqual(result['sha256'], hashlib.sha256(self.data).hexdigest())
        self.assertEqual(result['key'], self.key)
        self.stub.assert_no_pending_responses()

    def test_existing_corrupt_object_is_never_overwritten(self):
        self.stub.add_response('head_object', {}, self.params)
        self.stub.add_response('get_object', self.get_response(b'corrupt'), self.params)
        with self.assertRaisesRegex(ArchiveError, 'archive_content_mismatch'):
            self.run_store()
        self.stub.assert_no_pending_responses()

    def test_racing_create_verifies_winner_and_public_corruption_fails(self):
        self.stub.add_client_error('head_object', 'NoSuchKey', http_status_code=404, expected_params=self.params)
        self.stub.add_client_error('put_object', 'PreconditionFailed', http_status_code=412)
        self.stub.add_response('get_object', self.get_response(self.data), self.params)
        response = httpx.Response(200, content=b'wrong-public-bytes', request=httpx.Request('GET', 'https://public.invalid'))
        with patch('verdantflare_video_mcp.archive.httpx.stream') as stream:
            stream.return_value.__enter__.return_value = response
            with self.assertRaisesRegex(ArchiveError, 'archive_content_mismatch'):
                self.run_store()
        self.stub.assert_no_pending_responses()
