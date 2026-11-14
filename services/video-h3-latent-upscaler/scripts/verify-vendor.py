"""Verify copied upstream sources against their pinned checkout and patches."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('checkout', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
lock = json.loads((root / 'backend-lock.json').read_text())
revision = lock['upstream']['revision']
pairs = {'nodes/minimax_h3_latent_upscaler_3d.py': 'upscaler_3d.py',
         'nodes/MMH3_Split_Upscale.py': 'split_upscale.py'}
with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    for source in pairs:
        content = subprocess.check_output(['git', '-C', str(args.checkout), 'show', f'{revision}:{source}'])
        target = work / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    for patch in sorted((root / 'patches').glob('*.patch')):
        subprocess.run(['git', 'apply', str(patch.resolve())], cwd=work, check=True)
    for source, filename in pairs.items():
        if (work / source).read_bytes() != (root / 'src/h3_latent_upscaler/vendor' / filename).read_bytes():
            raise SystemExit(f'vendor mismatch: {filename}')
print('Vendor files match pinned source plus recorded patches.')
