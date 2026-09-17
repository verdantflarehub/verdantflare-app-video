"""Durable single-worker post-processing queue with immutable idempotent inputs."""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import threading
import time


class QueueError(ValueError):
    pass


class Queue:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.owner = (self.root / 'owner.lock').open('a')
        try:
            fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.owner.close()
            raise QueueError('worker_already_running') from None
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.root / 'queue.sqlite', check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('''CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY, digest TEXT NOT NULL, request TEXT NOT NULL,
            status TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL,
            result TEXT, error TEXT)''')
        with self.db:
            self.db.execute("UPDATE tasks SET status='failed',error='interrupted',updated=? WHERE status='running'", (time.time(),))

    def close(self):
        self.db.close()
        self.owner.close()

    def get(self, task_id):
        if not isinstance(task_id, str) or not re.fullmatch('video_task_[0-9a-f]{32}', task_id):
            raise QueueError('invalid_task_id')
        with self.lock:
            row = self.db.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        return {'video_task_id': row['id'], 'status': row['status'],
                'request': json.loads(row['request']), 'created_at': row['created'], 'updated_at': row['updated'],
                'result': json.loads(row['result']) if row['result'] else None,
                'error': {'code': row['error']} if row['error'] else None}

    def submit(self, request):
        request = dict(request)
        task_id = request.pop('idempotency_key')
        if not isinstance(task_id, str) or not re.fullmatch('video_task_[0-9a-f]{32}', task_id):
            raise QueueError('invalid_task_id')
        encoded = json.dumps(request, sort_keys=True, allow_nan=False)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        with self.lock, self.db:
            row = self.db.execute('SELECT digest FROM tasks WHERE id=?', (task_id,)).fetchone()
            if row:
                if row['digest'] != digest:
                    raise QueueError('idempotency_conflict')
                return self.get(task_id)
            if self.db.execute("SELECT COUNT(*) FROM tasks WHERE status IN ('queued','running')").fetchone()[0] >= 4:
                raise QueueError('queue_full')
            now = time.time()
            self.db.execute('INSERT INTO tasks VALUES (?,?,?,?,?,?,NULL,NULL)',
                            (task_id, digest, encoded, 'queued', now, now))
        return self.get(task_id)

    def take(self):
        with self.lock, self.db:
            if self.db.execute("SELECT 1 FROM tasks WHERE status='running'").fetchone():
                return None
            row = self.db.execute("SELECT id FROM tasks WHERE status='queued' ORDER BY created,id LIMIT 1").fetchone()
            if row is None:
                return None
            self.db.execute("UPDATE tasks SET status='running',updated=? WHERE id=?", (time.time(), row['id']))
        return self.get(row['id'])

    def finish(self, task_id, *, result=None, error=None):
        if (result is None) == (error is None):
            raise QueueError('one_outcome_required')
        with self.lock, self.db:
            if self.get(task_id)['status'] != 'running':
                raise QueueError('task_not_running')
            self.db.execute('UPDATE tasks SET status=?,updated=?,result=?,error=? WHERE id=?',
                ('succeeded' if result is not None else 'failed', time.time(),
                 json.dumps(result, allow_nan=False) if result is not None else None, error, task_id))
