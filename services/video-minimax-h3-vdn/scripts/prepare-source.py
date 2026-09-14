#!/usr/bin/env python3
"""Fetch pinned upstream sources and apply the two reviewed Diffusers patches."""
import argparse
import tomllib
from pathlib import Path
import subprocess


def git(*args):
    subprocess.run(['git', *map(str, args)], check=True)


def fetch(url, revision, path):
    path.mkdir(parents=True, exist_ok=False)
    git('init', path)
    git('-C', path, 'remote', 'add', 'origin', url)
    git('-C', path, 'fetch', '--depth=1', 'origin', revision)
    git('-C', path, 'checkout', '--detach', 'FETCH_HEAD')
    actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != revision:
        raise RuntimeError('source revision mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    lock = tomllib.loads((Path(__file__).resolve().parents[1] / 'pyproject.toml').read_text())['tool']['vdn']['upstream']
    root = args.destination.resolve()
    fetch(lock['repository'], lock['commit'], root / 'vdn')
    fetch('https://github.com/huggingface/diffusers', lock['diffusers_commit'], root / 'diffusers')
    for patch in sorted((root / 'vdn/diffusers_patches').glob('*.patch')):
        git('-C', root / 'diffusers', 'apply', '--check', patch)
        git('-C', root / 'diffusers', 'apply', patch)


if __name__ == '__main__':
    main()
