"""Resolve immutable generation resources without accepting user file paths.

The index is populated by the trusted generation/MCP integration and mounted
read-only in the post-processing runtime. It never searches arbitrary folders.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sqlite3


class ResourceError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def confined_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ResourceError('invalid_resource_path')
    if '..' in Path(relative).parts:
        raise ResourceError('invalid_resource_path')
    root = root.resolve(strict=True)
    try:
        path = (root / relative).resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ResourceError('resource_unavailable') from exc
    if not path.is_file():
        raise ResourceError('resource_unavailable')
    return path


class SourceIndex:
    """Read-only source lookup scoped to an authenticated project and node."""

    def __init__(self, database: Path, root: Path, node: str):
        self.database, self.root, self.node = database, root, node

    def resolve(self, project_id: str, task_id: str) -> dict:
        if not isinstance(project_id, str) or not project_id:
            raise ResourceError('invalid_project')
        if not isinstance(task_id, str) or not re.fullmatch(r'video_task_[0-9a-f]{32}', task_id):
            raise ResourceError('invalid_source_task_id')
        try:
            with sqlite3.connect(self.database.resolve().as_uri() + '?mode=ro', uri=True) as db:
                db.row_factory = sqlite3.Row
                row = db.execute(
                    'SELECT * FROM h3_sources WHERE project_id=? AND task_id=?',
                    (project_id, task_id),
                ).fetchone()
        except sqlite3.Error as exc:
            raise ResourceError('source_index_unavailable') from exc
        if row is None:
            raise ResourceError('source_not_found')
        if row['node'] != self.node:
            raise ResourceError('source_node_mismatch')
        if row['status'] != 'succeeded':
            raise ResourceError('source_not_ready')
        if not row['manifest_path'] or not row['manifest_sha256']:
            raise ResourceError('missing_latent_bundle')
        manifest_path = confined_file(self.root, row['manifest_path'])
        if manifest_path.stat().st_size > 4 * 1024 * 1024:
            raise ResourceError('invalid_manifest')
        if sha256(manifest_path) != row['manifest_sha256']:
            raise ResourceError('source_integrity_failed')
        try:
            manifest = json.loads(manifest_path.read_text())
        except (ValueError, UnicodeError) as exc:
            raise ResourceError('invalid_manifest') from exc
        if not isinstance(manifest, dict) or (
            manifest.get('schema') != 'h3-latent-bundle/v1'
            or manifest.get('project_id') != project_id
            or manifest.get('source_video_task_id') != task_id
            or manifest.get('node') != self.node
            or manifest.get('latent_state') != 'clean'
        ):
            raise ResourceError('incompatible_latent_bundle')
        files = manifest.get('files')
        if not isinstance(files, dict) or not {'video_latent', 'audio_latent', 'conditions', 'source_video'} <= files.keys():
            raise ResourceError('missing_latent_bundle')
        resolved = {}
        for name, item in files.items():
            if not isinstance(item, dict) or type(item.get('size')) is not int or item['size'] <= 0:
                raise ResourceError('invalid_manifest')
            if not isinstance(item.get('sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', item['sha256']):
                raise ResourceError('invalid_manifest')
            path = confined_file(self.root, item.get('path'))
            if path.stat().st_size != item['size'] or sha256(path) != item['sha256']:
                raise ResourceError('source_integrity_failed')
            if name in {'video_latent', 'audio_latent'} and path.suffix != '.safetensors':
                raise ResourceError('incompatible_latent_bundle')
            resolved[name] = path
        return {'manifest': manifest, 'files': resolved, 'manifest_sha256': row['manifest_sha256']}


SCHEMA = '''CREATE TABLE IF NOT EXISTS h3_sources (
    project_id TEXT NOT NULL, task_id TEXT NOT NULL, node TEXT NOT NULL,
    status TEXT NOT NULL, manifest_path TEXT, manifest_sha256 TEXT,
    PRIMARY KEY(project_id, task_id)
)'''
