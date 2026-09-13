"""Download only the locked checkpoint; never overwrite an existing checkpoint."""
import json
import os
from pathlib import Path
import tempfile
import urllib.request
from model import LOCK, checkpoint, sha256, verify


def main():
    destination = checkpoint()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify()
    else:
        endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')
        url = f"{endpoint}/{LOCK['repository']}/resolve/{LOCK['revision']}/{LOCK['filename']}"
        fd, name = tempfile.mkstemp(prefix='.download-', dir=destination.parent)
        try:
            with os.fdopen(fd, 'wb') as output, urllib.request.urlopen(url, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            pending = Path(name)
            if pending.stat().st_size != LOCK['size'] or sha256(pending) != LOCK['sha256']:
                raise RuntimeError('download_integrity_failed')
            os.link(pending, destination)
        finally:
            Path(name).unlink(missing_ok=True)
        verify()
    print(json.dumps({'event': 'model_verified', **LOCK}), flush=True)


if __name__ == '__main__':
    main()
