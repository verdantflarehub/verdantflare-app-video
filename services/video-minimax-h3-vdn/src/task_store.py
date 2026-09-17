"""Durable, single-worker task queue for the VDN resident process."""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import threading
import time
import uuid

TASK_ID = re.compile(r'vdn_[0-9a-f]{32}\Z')


class Conflict(ValueError):
    pass


class QueueFull(RuntimeError):
    pass


class TaskStore:
    def __init__(self, root, instance_id, capacity=4):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.owner = (self.root / 'worker.lock').open('a')
        try:
            fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.owner.close()
            raise RuntimeError('another runtime owns this task directory') from None
        self.instance_id = instance_id
        self.capacity = capacity
        self.mutex = threading.RLock()
        self.db = sqlite3.connect(self.root / 'tasks.sqlite3', check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('''CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE NOT NULL,
            digest TEXT NOT NULL, request TEXT NOT NULL, status TEXT NOT NULL,
            stage TEXT NOT NULL, instance_id TEXT, created REAL, updated REAL,
            error TEXT, result TEXT)''')
        with self.db:
            self.db.execute("UPDATE tasks SET status='failed', stage='interrupted', error='runtime_restarted', updated=? WHERE status IN ('queued','in_progress')", (time.time(),))

    def close(self):
        with self.mutex:
            self.db.close()
            self.owner.close()

    def submit(self, key, request):
        if not isinstance(key, str) or not key.strip() or len(key) > 128:
            raise ValueError('invalid idempotency key')
        encoded = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        with self.mutex, self.db:
            row = self.db.execute('SELECT id,digest FROM tasks WHERE idempotency_key=?', (key,)).fetchone()
            if row:
                if row['digest'] != digest:
                    raise Conflict('idempotency key has different input')
                return self.get(row['id'])
            active = self.db.execute("SELECT count(*) FROM tasks WHERE status IN ('queued','in_progress')").fetchone()[0]
            if active >= self.capacity:
                raise QueueFull('queue capacity reached')
            task_id = 'vdn_' + uuid.uuid4().hex
            now = time.time()
            self.db.execute('INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                            (task_id, key, digest, encoded, 'queued', 'queued', None, now, now, None, None))
            return self.get(task_id)

    def get(self, task_id, include_request=False):
        if not isinstance(task_id, str) or not TASK_ID.fullmatch(task_id):
            raise KeyError('unknown task')
        with self.mutex:
            row = self.db.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()
        if row is None:
            raise KeyError('unknown task')
        result = {'id':row['id'], 'status':row['status'], 'stage':row['stage'],
                  'execution_instance_id':row['instance_id'], 'created_at':row['created'],
                  'updated_at':row['updated'], 'error':row['error'],
                  'result':json.loads(row['result']) if row['result'] else None}
        if include_request:
            result['request'] = json.loads(row['request'])
            result['idempotency_key'] = row['idempotency_key']
        return result

    def take(self):
        with self.mutex, self.db:
            if self.db.execute("SELECT 1 FROM tasks WHERE status='in_progress' LIMIT 1").fetchone():
                return None
            row = self.db.execute("SELECT id FROM tasks WHERE status='queued' ORDER BY created,rowid LIMIT 1").fetchone()
            if row is None:
                return None
            self.db.execute("UPDATE tasks SET status='in_progress',stage='downloading',instance_id=?,updated=? WHERE id=?", (self.instance_id, time.time(), row['id']))
            return self.get(row['id'], include_request=True)

    def progress(self, task_id, stage):
        if stage not in {'downloading','generating','saving'}:
            raise ValueError('invalid stage')
        with self.mutex, self.db:
            if self.get(task_id)['status'] != 'in_progress':
                raise Conflict('task is not running')
            self.db.execute('UPDATE tasks SET stage=?,updated=? WHERE id=?', (stage, time.time(), task_id))

    def finish(self, task_id, *, result=None, error=None):
        if (result is None) == (error is None):
            raise ValueError('provide either result or error')
        with self.mutex, self.db:
            if self.get(task_id)['status'] != 'in_progress':
                raise Conflict('task is not running')
            status = 'failed' if error else 'completed'
            self.db.execute('UPDATE tasks SET status=?,stage=?,updated=?,error=?,result=? WHERE id=?',
                            (status, 'engine_failed' if error else 'completed', time.time(), error,
                             json.dumps(result, allow_nan=False) if result is not None else None, task_id))
            return self.get(task_id)

    def cancel(self, task_id):
        with self.mutex, self.db:
            row = self.get(task_id)
            if row['status'] == 'cancelled':
                return row
            if row['status'] != 'queued':
                raise Conflict('only queued tasks can be cancelled')
            self.db.execute("UPDATE tasks SET status='cancelled',stage='cancelled',updated=? WHERE id=?", (time.time(), task_id))
            return self.get(task_id)

    def output(self, task_id):
        if self.get(task_id)['status'] != 'completed':
            raise KeyError('task has no completed output')
        path = self.root / task_id / 'video.mp4'
        if not path.is_file() or not path.resolve().is_relative_to(self.root.resolve()):
            raise KeyError('output unavailable')
        return path
