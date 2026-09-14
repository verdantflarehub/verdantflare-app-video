#!/usr/bin/env python3
"""Inventory a reviewed local HF snapshot; writes the lock outside the snapshot."""
import argparse
import json
from pathlib import Path
from vdn_io import sha256, validate_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--revision', required=True, help='immutable model repository commit')
    parser.add_argument('--steps', type=int, choices=[8, 50], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.models.resolve()
    if args.output.resolve().is_relative_to(root):
        parser.error('lock must be outside the model directory')
    data = dict(repository='OpenVDN/vdn-minimax-h3', revision=args.revision, steps=args.steps,
                files={str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*'))
                       if p.is_file() and '.cache' not in p.relative_to(root).parts})
    validate_model(root, data, args.steps)
    with args.output.open('x') as stream:
        stream.write(json.dumps(data, indent=2) + '\n')


if __name__ == '__main__':
    main()
