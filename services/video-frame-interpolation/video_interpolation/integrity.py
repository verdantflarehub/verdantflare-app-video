"""Verify operator-provisioned, immutable model bundles before importing weights."""
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(root, lock, upstream):
    root = Path(root).resolve()
    # The bundle manifest is provisioned by the operator, never by a tool request.
    manifest_path = root / 'manifest.json'
    try:
        manifest = json.loads(manifest_path.read_text())
        if manifest['backend'] != lock['backend'] or manifest['code_revision'] != lock['code_revision']:
            raise ValueError('bundle does not match backend')
        if not re.fullmatch(r'(?:[0-9a-f]{40}|sha256:[0-9a-f]{64})', manifest['weights_revision']):
            raise ValueError('weights must use a source commit or SHA-256 content identifier')
        if lock.get('weights_revision') and manifest['weights_revision'] != lock['weights_revision']:
            raise ValueError('weight revision does not match lock')
        if manifest['weights_revision'].startswith('sha256:') and (lock.get('weights_revision') != manifest['weights_revision'] or not lock.get('weight_sha256')):
            raise ValueError('content-addressed weights require pinned revision and file hashes')
        if lock.get('weight_sha256') and manifest['files'] != lock['weight_sha256']:
            raise ValueError('weight digests do not match lock')
        if set(manifest['files']) != set(lock['weight_files']):
            raise ValueError('weight file set mismatch')
        for name, digest in manifest['files'].items():
            path = root / name
            if path.is_symlink() or path.resolve().parent != root or not re.fullmatch(r'[0-9a-f]{64}', digest):
                raise ValueError('invalid weight file')
            if sha256(path) != digest:
                raise ValueError('weight checksum mismatch')
        revision = subprocess.run(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], check=True,
                                  capture_output=True, text=True, timeout=10).stdout.strip()
        if revision != lock['code_revision']:
            raise ValueError('upstream code mismatch')
        subprocess.run(['git', '-C', str(upstream), 'diff', '--exit-code', 'HEAD'], check=True,
                       capture_output=True, timeout=10)
    except (OSError, KeyError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        raise RuntimeError('model_integrity_failed') from exc
    return {'backend': lock['backend'], 'repository': lock['repository'], 'code_revision': lock['code_revision'],
            'weights_revision': manifest['weights_revision'], 'weights_sha256': manifest['files'],
            'manifest_sha256': sha256(manifest_path)}
