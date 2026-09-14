import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from ref2va_io import download


class DownloadTests(unittest.TestCase):
    def setUp(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/redirect':
                    self.send_response(302)
                    self.send_header('Location', '/asset')
                else:
                    self.send_response(200)
                self.end_headers()
                self.wfile.write(b'reference bytes')
            def log_message(self, *_):
                pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ref = dict(type='image', uri=f'http://127.0.0.1:{self.server.server_port}/asset',
                        size=15, sha256=hashlib.sha256(b'reference bytes').hexdigest())

    def test_download_preserves_order_and_bytes(self):
        paths = download({'conditions':[dict(self.ref, type=k) for k in ('image','audio','video')]}, self.temp.name)
        self.assertEqual([k for k, _ in paths], ['image','audio','video'])
        for _, path in paths:
            self.assertEqual(Path(path).read_bytes(), b'reference bytes')

    def test_invalid_size_and_hash_are_rejected(self):
        for index, bad in enumerate(({'size':14}, {'size':16}, {'sha256':'a'*64})):
            directory = Path(self.temp.name)/str(index)
            directory.mkdir()
            with self.assertRaises(ValueError):
                download({'conditions':[dict(self.ref, **bad)]}, directory)

    def test_redirect_is_not_followed(self):
        with self.assertRaises(ValueError):
            download({'conditions':[dict(self.ref, uri=self.ref['uri'].replace('/asset','/redirect'))]}, self.temp.name)
