#!/usr/bin/env python3
"""Verify vendored code equals pinned upstream plus the recorded local patch."""
import argparse
import tomllib
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--upstream', required=True, type=Path)
args = parser.parse_args()
lock = tomllib.loads((root / 'pyproject.toml').read_text())['tool']['vdn']['upstream']
name = 'src/inference/infer_diffusers.py'
actual = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
if actual != lock['commit']:
    raise RuntimeError('upstream commit mismatch')
original = subprocess.check_output(['git', '-C', str(args.upstream), 'show', f"{lock['commit']}:{name}"])
with tempfile.TemporaryDirectory() as temp:
    target = Path(temp) / 'infer_diffusers.py'
    target.write_bytes(original)
    subprocess.run(['git', 'apply', str(root / 'patches/0001-local-model-paths.patch')], cwd=temp, check=True)
    assert target.read_bytes() == (root / 'src/vendor/infer_diffusers.py').read_bytes(), 'unrecorded vendor modification'
print('Vendored inference verified: pinned upstream + local-path patch')
