"""Immutable Chengdu S3 delivery for H3 post-processing artifacts."""
import hashlib
import os
from urllib.parse import quote, urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
import httpx

from .artifacts import require_project_id
from .tasks import TASK_ID_PATTERN


class ArchiveError(RuntimeError):
    pass


class S3Archive:
    def __init__(self, client, bucket, public_base_url):
        self.client = client
        self.bucket = bucket
        self.public_base_url = public_base_url.rstrip('/')

    @classmethod
    def from_environment(cls):
        values = [os.environ.get(k, '').strip() for k in
                  ('S3_ENDPOINT', 'S3_BUCKET', 'S3_ACCESS_KEY', 'S3_SECRET_KEY', 'S3_PUBLIC_BASE_URL')]
        if not all(values):
            raise ArchiveError('archive_not_configured')
        endpoint, bucket, access, secret, public = values
        for url in (endpoint, public):
            parsed = urlparse(url)
            if parsed.scheme not in {'http', 'https'} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ArchiveError('invalid_archive_configuration')
        if urlparse(public).scheme != 'https':
            raise ArchiveError('invalid_archive_configuration')
        client = boto3.client('s3', endpoint_url=endpoint, aws_access_key_id=access,
            aws_secret_access_key=secret, region_name=os.environ.get('S3_REGION', 'us-east-1'),
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'},
                          connect_timeout=10, read_timeout=120, retries={'max_attempts': 2},
                          request_checksum_calculation='when_required', response_checksum_validation='when_required'))
        return cls(client, bucket, public)

    def preflight(self):
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except (BotoCoreError, ClientError) as exc:
            raise ArchiveError('archive_unavailable') from exc

    def store(self, artifact, path, task_id, name):
        if not TASK_ID_PATTERN.fullmatch(task_id) or name not in {'content', 'preview'}:
            raise ArchiveError('invalid_archive_identity')
        project = require_project_id(artifact.project_id)
        key = f'video/projects/{project}/h3-latent/{task_id}/{artifact.sha256}/{name}.mp4'
        try:
            # A retry never overwrites an existing object, even when metadata is wrong.
            exists = True
            try:
                self.client.head_object(Bucket=self.bucket, Key=key)
            except ClientError as exc:
                if exc.response['ResponseMetadata']['HTTPStatusCode'] != 404:
                    raise
                exists = False
            if not exists:
                with path.open('rb') as body:
                    try:
                        self.client.put_object(Bucket=self.bucket, Key=key, Body=body,
                            ContentLength=artifact.size, ContentType=artifact.media_type,
                            Metadata={'sha256': artifact.sha256}, IfNoneMatch='*')
                    except ClientError as exc:
                        if exc.response['ResponseMetadata']['HTTPStatusCode'] != 412:
                            raise
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            body = response['Body']
            try:
                self._verify(iter(lambda: body.read(1024 * 1024), b''), artifact)
            finally:
                body.close()
            url = f'{self.public_base_url}/{quote(self.bucket, safe="")}/{quote(key, safe="/")}'
            with httpx.stream('GET', url, timeout=120, follow_redirects=False) as public:
                public.raise_for_status()
                self._verify(public.iter_bytes(), artifact)
            return {'bucket': self.bucket, 'key': key, 'sha256': artifact.sha256,
                    'size': artifact.size, 'download_url': url}
        except (BotoCoreError, ClientError, httpx.HTTPError, OSError) as exc:
            raise ArchiveError('archive_unavailable') from exc

    @staticmethod
    def _verify(chunks, artifact):
        digest, size = hashlib.sha256(), 0
        for chunk in chunks:
            size += len(chunk)
            if size > artifact.size:
                raise ArchiveError('archive_content_mismatch')
            digest.update(chunk)
        if size != artifact.size or digest.hexdigest() != artifact.sha256:
            raise ArchiveError('archive_content_mismatch')
