"""Fetch exact source commits for image builds; no dynamic branches."""
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
lock = json.loads((root / 'backend-lock.json').read_text())
for name, destination in [('upstream', '/opt/h3-upscale-upstream'), ('comfyui', '/opt/comfy')]:
    source = lock[name]
    subprocess.run(['git', 'init', destination], check=True)
    subprocess.run(['git', '-C', destination, 'remote', 'add', 'origin', source['repo']], check=True)
    subprocess.run(['git', '-C', destination, 'fetch', '--depth', '1', 'origin', source['revision']], check=True)
    subprocess.run(['git', '-C', destination, 'checkout', '--detach', 'FETCH_HEAD'], check=True)
    actual = subprocess.check_output(['git', '-C', destination, 'rev-parse', 'HEAD'], text=True).strip()
    if actual != source['revision']:
        raise RuntimeError('source revision mismatch')
