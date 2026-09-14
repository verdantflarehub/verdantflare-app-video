#!/usr/bin/env python3
"""Run a locked local VDN model; no model downloads or implicit retries."""
import argparse
import json
import os
from pathlib import Path
import time
import tomllib
from vdn_io import validate_request, validate_model, run_upstream, inspect_media, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--model-lock', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new run directory; existing directory is rejected')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    request = json.loads(args.manifest.read_text())
    paths = validate_request(request, args.inputs.resolve())
    lock = json.loads(args.model_lock.read_text())
    validate_model(args.models.resolve(), lock, request['steps'])
    if args.dry_run:
        print('Input and model checksums validated; no GPU inference performed')
        return
    args.output.mkdir(parents=True, exist_ok=False)
    record = {'status': 'in_progress', 'request': request, 'model': lock, 'started_at': time.time(),
              'runtime_version': '0.2.0', 'upstream': tomllib.loads((Path(__file__).resolve().parents[1] / 'pyproject.toml').read_text())['tool']['vdn']['upstream']}
    def save():
        temp = args.output / 'record.tmp'
        temp.write_text(json.dumps(record, indent=2) + '\n')
        temp.replace(args.output / 'record.json')
    save()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    try:
        start = time.monotonic()
        partial = args.output / 'partial.mp4'
        run_upstream(args.models.resolve(), request, paths, partial)
        record['load_and_generate_seconds'] = time.monotonic() - start
        record['media'] = inspect_media(partial, request['frames'])
        digest = sha256(partial)
        partial.replace(args.output / 'video.mp4')
        record.update(status='completed', artifact={'path': 'video.mp4', 'sha256': digest})
    except Exception as error:
        record.update(status='failed', error_type=type(error).__name__)
        raise
    finally:
        record['finished_at'] = time.time()
        save()


if __name__ == '__main__':
    main()
