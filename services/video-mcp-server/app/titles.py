"""Mutable naming metadata, separate from immutable generation requests."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
import re
import sqlite3
import unicodedata

PROVIDED_TITLE = ContextVar('video_provided_title', default=None)
OPERATIONS = {'interpolate': '视频补帧', 'sr': '视频超分', 'depth': '深度视频',
              'h3-latent-upscale': '视频超分'}


class TitleConflict(RuntimeError):
    pass


def validate_title(value):
    if not isinstance(value, str) or any(unicodedata.category(c).startswith('C') or c in '\r\n\u2028\u2029' for c in value):
        raise ValueError('title must be single-line plain text')
    value = value.strip()
    if not 1 <= len(value) <= 64 or '<' in value or '>' in value:
        raise ValueError('title must contain 1–64 plain text characters')
    return value


def fallback(record):
    if record.service in OPERATIONS:
        return OPERATIONS[record.service]
    text = str(record.request.get('prompt') or '')
    text = re.sub(r'<[^>]*>|https?://\S+|[A-Za-z]:[\\/]\S+|(?:[/\\][\w.-]+){2,}', ' ', text)
    text = re.sub(r'\b(?:[\w.-]*_[\w.-]+|[0-9a-fA-F-]{24,}|[A-Za-z0-9]{24,})\b', ' ', text)
    text = ''.join(c if not unicodedata.category(c).startswith('C') else ' ' for c in text)
    text = re.sub(r'\s+', ' ', text).strip(' ,，。:：;；-')
    text = re.split(r'[。！？!?\n]', text, maxsplit=1)[0].strip()
    return text[:32].strip() or '视频生成'


@contextmanager
def connection(store):
    store.ensure_ready()
    existing = getattr(store._local, 'connection', None)
    db = existing or sqlite3.connect(store.root / 'coordination.sqlite3', timeout=15)
    try:
        db.execute('CREATE TABLE IF NOT EXISTS task_titles (task TEXT PRIMARY KEY, title TEXT NOT NULL, source TEXT NOT NULL, revision INTEGER NOT NULL, updated_at TEXT NOT NULL)')
        yield db
        if existing is None:
            db.commit()
    finally:
        if existing is None:
            db.close()


def project(store, record):
    with connection(store) as db:
        row = db.execute('SELECT title,source,revision,updated_at FROM task_titles WHERE task=?', (record.video_task_id,)).fetchone()
    return dict(zip(('title', 'title_source', 'title_revision', 'title_updated_at'), row)) if row else {
        'title': fallback(record), 'title_source': 'fallback', 'title_revision': 0, 'title_updated_at': None}


def initialize(store, record):
    value = PROVIDED_TITLE.get()
    source = 'provided'
    if value is None and record.service in OPERATIONS:
        artifact = record.request.get('source_artifact_id')
        parent_id = record.request.get('source_video_task_id')
        if artifact or parent_id:
            for path in store.root.glob('video_task_*.json'):
                # Same-project source only; never use an unrelated task's name.
                from .tasks import TaskRecord
                try:
                    parent = TaskRecord.model_validate_json(path.read_text(encoding='utf-8'))
                except (OSError, ValueError):
                    continue
                if parent.project_id == record.project_id and (bool(artifact) and parent.artifact_id == artifact or bool(parent_id) and parent.video_task_id == parent_id):
                    info = project(store, parent)
                    if info['title_source'] != 'fallback':
                        suffix = {'interpolate': ' · 补帧', 'depth': ' · 深度'}.get(record.service, ' · 超分')
                        value = info['title'][:64-len(suffix)] + suffix
                        source = 'fallback'
                    break
    if value is not None:
        with connection(store) as db:
            db.execute('INSERT OR IGNORE INTO task_titles VALUES(?,?,?,?,?)',
                       (record.video_task_id, value, source, 1, datetime.now(UTC).isoformat()))


def update(store, record, title, expected_title_revision):
    title = validate_title(title)
    if type(expected_title_revision) is not int or expected_title_revision < 0:
        raise ValueError('expected_title_revision must be a nonnegative integer')
    with connection(store) as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT revision FROM task_titles WHERE task=?', (record.video_task_id,)).fetchone()
        if (row[0] if row else 0) != expected_title_revision:
            raise TitleConflict('title changed; read the latest title before editing')
        db.execute('INSERT INTO task_titles VALUES(?,?,?,?,?) ON CONFLICT(task) DO UPDATE SET title=excluded.title, source=excluded.source, revision=excluded.revision, updated_at=excluded.updated_at',
                   (record.video_task_id, title, 'manual', expected_title_revision + 1, datetime.now(UTC).isoformat()))
    return project(store, record)


def create(executor, title=None, **kwargs):
    warning = None
    if title is not None:
        try:
            title = validate_title(title)
        except ValueError:
            title, warning = None, 'invalid_optional_title; readable fallback used'
    token = PROVIDED_TITLE.set(title)
    try:
        record = executor.generate(**kwargs)
    finally:
        PROVIDED_TITLE.reset(token)
    info = project(executor.tasks, record)
    if warning:
        info['title_warning'] = warning
    return record, info


def rename(executor, task_id, title, expected_title_revision):
    from .delivery import authorize, CALLER
    actor = CALLER.get()
    if not actor or not all(actor.get(k) for k in ('user_id', 'organization_id')):
        raise PermissionError('verified Studio identity required')
    record = executor.tasks.get(task_id)
    if record.service == 'higgsfield':
        executor.higgsfield.authorize_record(record)
    authorize(executor.client, record.project_id)
    return update(executor.tasks, record, title, expected_title_revision)
