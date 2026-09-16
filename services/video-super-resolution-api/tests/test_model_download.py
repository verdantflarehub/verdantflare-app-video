import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('sr_download', Path(__file__).resolve().parents[2] / 'video-super-resolution-api/download_model.py')
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


class Response(io.BytesIO):
    def __init__(self, data, status, content_range):
        super().__init__(data)
        self.status = status
        self.headers = {'Content-Range': content_range}


class DownloadTest(unittest.TestCase):
    def test_resumes_with_explicit_client_header(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'part'
            path.write_bytes(b'abc')
            def respond(request, **kwargs):
                self.assertEqual(request.get_header('Range'), 'bytes=3-7')
                self.assertEqual(request.get_header('User-agent'), download.USER_AGENT)
                return Response(b'defgh', 206, 'bytes 3-7/8')
            with patch.object(download.urllib.request, 'urlopen', side_effect=respond):
                download.fetch_part('https://mirror.example.test/model', path, 0, 7, 8)
            self.assertEqual(path.read_bytes(), b'abcdefgh')

    def test_server_ignoring_resume_cannot_corrupt_partial_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'part'
            path.write_bytes(b'abc')
            with patch.object(download.urllib.request, 'urlopen', return_value=Response(b'abcdefgh', 200, None)):
                with self.assertRaisesRegex(ValueError, 'honor'):
                    download.fetch_part('https://mirror.example.test/model', path, 0, 7, 8)
            self.assertEqual(path.read_bytes(), b'abc')

    def test_short_response_retries_only_missing_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'part'
            requests = []
            def respond(request, **kwargs):
                requests.append(request.get_header('Range'))
                return Response(b'abc', 206, 'bytes 0-7/8') if len(requests) == 1 else Response(b'defgh', 206, 'bytes 3-7/8')
            with patch.object(download.urllib.request, 'urlopen', side_effect=respond), patch.object(download.time, 'sleep'):
                download.fetch_part('https://mirror.example.test/model', path, 0, 7, 8)
            self.assertEqual(requests, ['bytes=0-7', 'bytes=3-7'])
            self.assertEqual(path.read_bytes(), b'abcdefgh')

    def test_oversized_range_response_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'part'
            with patch.object(download.urllib.request, 'urlopen', return_value=Response(b'abcdef', 206, 'bytes 0-3/4')):
                with self.assertRaisesRegex(ValueError, 'exceeds'):
                    download.fetch_part('https://mirror.example.test/model', path, 0, 3, 4)
            self.assertEqual(path.read_bytes(), b'')

    def test_verified_model_is_reused_and_mismatched_model_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'model.pth').write_bytes(b'fixture')
            lock = dict(backend='fixture', code_revision='a'*40, weights_repository='fixture/model',
                        weights_revision='b'*40, weight_files=['model.pth'], weight_size_bytes={'model.pth': 7},
                        weight_sha256={'model.pth': hashlib.sha256(b'fixture').hexdigest()})
            (root / 'backend-lock.json').write_text(json.dumps(lock))
            with patch.object(download, '__file__', str(root / 'download_model.py')), patch.object(download.urllib.request, 'urlopen') as fetch:
                download.download(root, 'https://mirror.example.test', 1)
                fetch.assert_not_called()
                self.assertEqual(json.loads((root / 'manifest.json').read_text())['weights_revision'], 'b'*40)
                (root / 'model.pth').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'refusing overwrite'):
                    download.download(root, 'https://mirror.example.test', 1)
                self.assertEqual((root / 'model.pth').read_bytes(), b'changed')

    def test_endpoint_does_not_accept_credentials_or_plaintext(self):
        for endpoint in ('http://example.test', 'https://user:secret@example.test', 'https://example.test/?token=secret'):
            with self.assertRaises(ValueError):
                download.download('/unused', endpoint, 1)
