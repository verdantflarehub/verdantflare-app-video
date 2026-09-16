"""Download the pinned public SeedVR2 bundle with resumable ranges and checksum verification."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import time
import urllib.request
from urllib.parse import urlsplit

USER_AGENT = 'verdantflare-video-sr-model-downloader/0.1'
CHUNK_SIZE = 256 * 1024 * 1024


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def fetch_part(url, path, start, end, total):
    expected = end - start + 1
    for attempt in range(4):
        have = path.stat().st_size if path.exists() else 0
        if have == expected:
            return path
        if have > expected:
            raise ValueError('partial file exceeds its range')
        request = urllib.request.Request(url, headers={
            'User-Agent': USER_AGENT, 'Accept-Encoding': 'identity', 'Range': f'bytes={start + have}-{end}'})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                full_file = start + have == 0 and end == total - 1 and response.status == 200
                if not full_file and (response.status != 206 or response.headers.get('Content-Range') != f'bytes {start + have}-{end}/{total}'):
                    raise ValueError('server did not honor the requested range')
                with path.open('ab') as stream:
                    while data := response.read(min(8 * 1024 * 1024, expected - have + 1)):
                        if have + len(data) > expected:
                            raise ValueError('response exceeds requested range')
                        stream.write(data)
                        have += len(data)
                if have != expected:
                    raise OSError('incomplete response')
            return path
        except (OSError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def download(destination, endpoint, workers):
    parsed = urlsplit(endpoint)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ('', '/'):
        raise ValueError('endpoint must be an HTTPS origin without credentials')
    if not 1 <= workers <= 16:
        raise ValueError('workers must be within 1..16')
    lock = json.loads(Path(__file__).with_name('backend-lock.json').read_text())
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.download.lock').open('w') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache = root / '.download'
        cache.mkdir(exist_ok=True)
        for name in lock['weight_files']:
            target = root / name
            expected_hash = lock['weight_sha256'][name]
            if target.exists():
                if digest(target) != expected_hash:
                    raise ValueError(f'existing {name} does not match the lock; refusing overwrite')
                print(f'{name}: already verified', flush=True)
                continue
            url = f"{endpoint.rstrip('/')}/{lock['weights_repository']}/resolve/{lock['weights_revision']}/{name}"
            total = lock['weight_size_bytes'][name]
            if total <= 0:
                raise ValueError('invalid download size')
            ranges = [(start, min(start + CHUNK_SIZE, total) - 1) for start in range(0, total, CHUNK_SIZE)]
            def fetch(bounds):
                start, end = bounds
                path = fetch_part(url, cache / f'{name}.{start}-{end}.part', start, end, total)
                print(f'{name}: range {start}-{end} complete', flush=True)
                return path
            with ThreadPoolExecutor(max_workers=workers) as pool:
                parts = list(pool.map(fetch, ranges))
            assembled = cache / f'{name}.assembled'
            h = hashlib.sha256()
            with assembled.open('wb') as output:
                for part in parts:
                    with part.open('rb') as source:
                        for data in iter(lambda: source.read(8 * 1024 * 1024), b''):
                            output.write(data)
                            h.update(data)
            if assembled.stat().st_size != total or h.hexdigest() != expected_hash:
                raise ValueError(f'{name}: checksum mismatch; no model file published')
            assembled.rename(target)
            for part in parts:
                part.unlink()
            print(f'{name}: SHA-256 verified', flush=True)
        manifest = dict(backend=lock['backend'], code_revision=lock['code_revision'],
                        weights_revision=lock['weights_revision'], files=lock['weight_sha256'])
        path = root / 'manifest.json'
        if path.exists():
            if json.loads(path.read_text()) != manifest:
                raise ValueError('existing manifest does not match the lock')
        else:
            temporary = cache / 'manifest.json'
            temporary.write_text(json.dumps(manifest, indent=2) + '\n')
            temporary.rename(path)
    print('Model bundle verified', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--endpoint', default='https://hf-mirror.com')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    download(args.destination, args.endpoint, args.workers)
