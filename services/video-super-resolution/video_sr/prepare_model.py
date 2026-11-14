"""Package operator-reviewed weights without overwriting an existing model bundle."""
import argparse
import json
from pathlib import Path
import re
import shutil

from video_sr.integrity import sha256


def prepare(lock_path, source, destination, revision):
    if not re.fullmatch(r'(?:[0-9a-f]{40}|sha256:[0-9a-f]{64})', revision):
        raise ValueError('weights_revision must be a source commit or SHA-256 content identifier')
    lock = json.loads(Path(lock_path).read_text())
    if revision.startswith('sha256:') and (lock.get('weights_revision') != revision or not lock.get('weight_sha256')):
        raise ValueError('content-addressed weights require pinned revision and file hashes')
    if lock.get('weights_revision') and revision != lock['weights_revision']:
        raise ValueError('weight revision does not match lock')
    source, destination = Path(source).resolve(), Path(destination).resolve()
    for name in lock['weight_files']:
        if Path(name).name != name or not (source / name).is_file():
            raise ValueError('required weight file missing')
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in lock['weight_files']:
        digest = sha256(source / name)
        if lock.get('weight_sha256') and digest != lock['weight_sha256'][name]:
            raise ValueError('weight digest does not match lock')
        shutil.copyfile(source / name, destination / name)
        if sha256(destination / name) != digest:
            raise ValueError('weight changed during copy; discard the incomplete bundle')
        hashes[name] = digest
    manifest = {'backend': lock['backend'], 'code_revision': lock['code_revision'],
                'weights_revision': revision, 'files': hashes}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--weights-revision', required=True)
    args = parser.parse_args()
    prepare(args.lock, args.source, args.destination, args.weights_revision)
