"""Local, offline adapter for OpenVDN's published Diffusers pipeline."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess

TRANSFORMERS = {8: 'stage-dmd-step-250/diffusers', 50: 'stage-b-step-2000/diffusers'}

MODES = {'t2va': (), 'i2va': ('first',), 'l2va': ('last',), 'fl2va': ('first', 'last')}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_path(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('expected a relative file path')
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f'file missing or outside root: {relative}')
    return path


def validate_request(request, inputs):
    allowed = {'schema_version', 'task', 'prompt', 'seed', 'frames', 'steps', 'input_artifacts'}
    if set(request) != allowed or request['schema_version'] != 1:
        raise ValueError('request fields must match schema_version 1')
    if request['task'] not in MODES:
        raise ValueError('unsupported task (Ref2VA is not a VDN keyframe mode)')
    if not isinstance(request['prompt'], str) or not request['prompt'].strip():
        raise ValueError('prompt must be nonempty')
    for key, low, high in [('seed', 0, 2**63 - 1), ('frames', 124, 362), ('steps', 8, 50)]:
        if type(request[key]) is not int or not low <= request[key] <= high:
            raise ValueError(f'invalid {key}')
    if request['frames'] % 17 != 5 or request['steps'] not in (8, 50):
        raise ValueError('frames must be 17n+5; steps must be 8 or 50')
    assets = request['input_artifacts']
    if not isinstance(assets, list):
        raise ValueError('input_artifacts must be a list')
    paths = {}
    for asset in assets:
        if set(asset) != {'role', 'path', 'sha256'} or asset['role'] in paths:
            raise ValueError('invalid or duplicate keyframe')
        path = checked_path(inputs, asset['path'])
        if sha256(path) != asset['sha256']:
            raise ValueError('keyframe hash mismatch')
        paths[asset['role']] = path
    if set(paths) != set(MODES[request['task']]):
        raise ValueError('keyframes do not match task')
    return paths


def validate_model(root, lock, steps):
    if lock.get('repository') != 'OpenVDN/vdn-minimax-h3':
        raise ValueError('unexpected model repository')
    revision = lock.get('revision', '')
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('model revision must be an immutable commit')
    if lock.get('steps') != steps:
        raise ValueError('checkpoint and requested NFE differ')
    files = lock.get('files', {})
    transformer = TRANSFORMERS[steps]
    config_path = transformer + '/config.json'
    if config_path not in files:
        raise ValueError('model tree lacks the requested stage configuration')
    if not {'model_index.json', 'modular_model_index.json'} & files.keys() or not any(p.endswith('.safetensors') for p in files):
        raise ValueError('model lock lacks index or weights')
    for relative, digest in files.items():
        if sha256(checked_path(root, relative)) != digest:
            raise ValueError(f'model checksum mismatch: {relative}')
    # The published Diffusers component assembles sibling branch/adapter files
    # onto a sharded base; there are no weights in its diffusers/ subdirectory.
    spec = json.loads(checked_path(root, config_path).read_text())['vdn']
    stage = Path(transformer).parent
    for relative in [spec['branch'], *spec['adapters']]:
        dependency = str(stage / relative)
        if dependency not in files:
            raise ValueError('missing branch or adapter in model inventory')
        checked_path(root, dependency)
    base = spec['base']
    if Path(base['source']).resolve() != root.resolve():
        raise ValueError('base transformer must load from the verified local tree')
    index_path = str(Path(base['subfolder']) / 'diffusion_pytorch_model.safetensors.index.json')
    if index_path not in files:
        raise ValueError('base transformer shard index missing')
    index = json.loads(checked_path(root, index_path).read_text())
    shards = set(index['weight_map'].values())
    if not shards:
        raise ValueError('base transformer shard index empty')
    for shard in shards:
        relative = str(Path(base['subfolder']) / shard)
        if relative not in files:
            raise ValueError('base transformer shard missing')
        checked_path(root, relative)
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and '.cache' not in p.relative_to(root).parts}
    if actual != set(files):
        raise ValueError('model tree contains unlisted or missing files')


def run_upstream(models, request, paths, destination):
    """Delegate all inference/offload/encoding to the copied upstream script."""
    import os
    import sys
    command = [sys.executable, str(Path(__file__).parent / 'vendor/infer_diffusers.py'),
               '--models', str(models), '--out', str(destination),
               '--steps', str(request['steps']), '--frames', str(request['frames']),
               '--seed', str(request['seed']), '--device', 'cuda:0', '--offload_dit',
               '--transformer', TRANSFORMERS[request['steps']]]
    for role, path in paths.items():
        command.extend(['--' + role, str(path)])
    command.extend(['--', request['prompt']])
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    subprocess.run(command, env=env, check=True)


def inspect_media(path, frames):
    data = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-show_streams', '-show_format', '-of', 'json', str(path)]))
    video = [s for s in data['streams'] if s['codec_type'] == 'video']
    audio = [s for s in data['streams'] if s['codec_type'] == 'audio']
    if len(video) != 1 or len(audio) != 1 or int(video[0].get('nb_read_frames', 0)) != frames:
        raise RuntimeError('media streams or frame count invalid')
    if Fraction(video[0].get('avg_frame_rate', '0/1')) != 24:
        raise RuntimeError('media frame rate mismatch')
    for stream in (video[0], audio[0]):
        stream_duration = float(stream.get('duration', 'nan'))
        if not math.isfinite(stream_duration) or abs(stream_duration - frames / 24) > 0.25:
            raise RuntimeError('audio/video stream duration mismatch')
    duration = float(data['format']['duration'])
    if not math.isfinite(duration) or abs(duration - frames / 24) > 0.25:
        raise RuntimeError('media duration mismatch')
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path), '-f', 'null', '-'], check=True, capture_output=True)
    return data
