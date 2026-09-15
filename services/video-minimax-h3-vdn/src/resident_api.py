"""Authenticated HTTP transport for the durable VDN queue (no GPU imports)."""
import hmac
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from task_store import Conflict, QueueFull
from ref2va_io import validate

VERSION = 'video-minimax-h3-vdn-v0.3.6'



class State:
    def __init__(self, instance):
        self.lock = threading.RLock()
        self.data = dict(ready=False, stage='starting', execution_instance_id=instance,
                         runtime_version=VERSION, started_at=time.time(), model_load_count=0)

    def update(self, **values):
        with self.lock:
            self.data.update(values)

    def snapshot(self):
        with self.lock:
            return dict(self.data)


def serve(state, store, token, source, host='0.0.0.0', port=8000):
    if not token or not source:
        raise ValueError('runtime token and artifact source required')

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def log_message(self, *_):
            pass

        def reply(self, code, value):
            body = json.dumps(value).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            if hmac.compare_digest(self.headers.get('Authorization', '').encode(), ('Bearer ' + token).encode()):
                return True
            self.reply(401, {'error': 'unauthorized'})
            return False

        def do_GET(self):
            if self.path in {'/health', '/live'}:
                data = state.snapshot()
                healthy = data['ready'] if self.path == '/health' else data['stage'] not in {'failed', 'stopping'}
                return self.reply(200 if healthy else 503, data)
            if not self.authorized():
                return
            parts = self.path.strip('/').split('/')
            try:
                if len(parts) not in {3, 4} or parts[:2] != ['v1', 'videos']:
                    raise KeyError()
                if len(parts) == 3:
                    return self.reply(200, dict(store.get(parts[2]), runtime_version=VERSION))
                if parts[3] != 'content':
                    raise KeyError()
                path = store.output(parts[2])
                with path.open('rb') as stream:
                    self.send_response(200)
                    self.send_header('Content-Type', 'video/mp4')
                    self.send_header('Content-Length', str(path.stat().st_size))
                    self.end_headers()
                    for chunk in iter(lambda: stream.read(1024**2), b''):
                        self.wfile.write(chunk)
            except KeyError:
                self.reply(404, {'error': 'not_found'})

        def do_POST(self):
            if not self.authorized():
                return
            if self.path != '/v1/videos':
                return self.reply(404, {'error': 'not_found'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 131072 or self.headers.get('Transfer-Encoding'):
                    return self.reply(413, {'error': 'invalid_body_size'})
                payload = validate(json.loads(self.rfile.read(size)), source)
                if not state.snapshot()['ready']:
                    return self.reply(503, {'error': 'model_not_ready'})
                key = payload.pop('idempotency_key')
                return self.reply(200, store.submit(key, payload))
            except Conflict:
                self.reply(409, {'error': 'idempotency_conflict'})
            except QueueFull:
                self.reply(429, {'error': 'queue_full'})
            except (ValueError, TypeError, KeyError):
                self.reply(400, {'error': 'invalid_request'})

        def do_DELETE(self):
            if not self.authorized():
                return
            parts = self.path.strip('/').split('/')
            try:
                if len(parts) != 3 or parts[:2] != ['v1', 'videos']:
                    raise KeyError()
                return self.reply(200, store.cancel(parts[2]))
            except KeyError:
                self.reply(404, {'error': 'not_found'})
            except Conflict:
                self.reply(409, {'error': 'only_queued_tasks_can_be_cancelled'})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
