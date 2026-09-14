"""One resident upstream pipeline, one durable queue, no automatic generation retry."""
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
import time
import tomllib
from types import SimpleNamespace
from urllib.request import build_opener, HTTPRedirectHandler, ProxyHandler, Request
import uuid

from resident_api import State, VERSION, serve
from task_store import TaskStore
from vdn_io import TRANSFORMERS, inspect_media, sha256, validate_model, validate_request


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        raise ValueError('artifact redirects forbidden')


def download(request, directory):
    opener = build_opener(ProxyHandler({}), NoRedirect())
    local = dict(request, input_artifacts=[])
    for asset in request['input_artifacts']:
        path = directory / (asset['role'] + '.image')
        size, digest = 0, hashlib.sha256()
        with opener.open(Request(asset['uri']), timeout=60) as response, path.open('xb') as stream:
            if response.status != 200:
                raise ValueError('artifact unavailable')
            while chunk := response.read(1024**2):
                size += len(chunk)
                if size > asset['size']:
                    raise ValueError('artifact too large')
                digest.update(chunk)
                stream.write(chunk)
        if size != asset['size'] or digest.hexdigest() != asset['sha256']:
            raise ValueError('artifact integrity failed')
        from PIL import Image
        with Image.open(path) as image:
            if image.format not in {'PNG', 'JPEG', 'WEBP'} or image.width * image.height > 16_000_000:
                raise ValueError('unsupported keyframe image')
            image.verify()
        local['input_artifacts'].append(dict(role=asset['role'], path=path.name, sha256=asset['sha256']))
    return validate_request(local, directory)


class Engine:
    def __init__(self, models):
        import torch
        from diffusers import ModularPipeline
        from vendor.infer_diffusers import load_pipeline
        self.args = SimpleNamespace(models=str(models), transformer=TRANSFORMERS[8], fp8=False,
                                    device='cuda:0', offload_dit=True)
        # Workflow graphs share exactly the same already-loaded components. The
        # upstream workflow selector still chooses distinct keyframe/text blocks.
        with torch.inference_mode():
            text = load_pipeline(self.args, 't2va')
            keyframes = ModularPipeline.from_pretrained(str(models), workflow='fl2va', local_files_only=True)
            for name in keyframes.pretrained_component_names:
                component = getattr(text, name, None)
                if component is None:
                    raise ValueError('keyframe workflow requires an unloaded component')
                keyframes.update_components(**{name: component})
        self.pipelines = {'t2va': text, 'fl2va': keyframes}

    def generate(self, request, paths, output):
        import torch
        from vendor.infer_diffusers import render
        for index in range(2):
            torch.cuda.reset_peak_memory_stats(index)
        args = SimpleNamespace(**vars(self.args), first=paths.get('first'), last=paths.get('last'),
                               out=str(output), frames=request['frames'], steps=request['steps'], seed=request['seed'])
        render(self.pipelines['fl2va' if paths else 't2va'], args, request['prompt'])
        for index in range(2):
            torch.cuda.synchronize(index)
        return {'peak_allocated_bytes': [torch.cuda.max_memory_allocated(i) for i in range(2)],
                'peak_reserved_bytes': [torch.cuda.max_memory_reserved(i) for i in range(2)]}


def execute(store, task, engine, provenance):
    """Publish output only after full media validation; failures retain their attempt."""
    directory = store.root / task['id']
    directory.mkdir(exist_ok=False)
    started = time.monotonic()
    try:
        paths = download(task['request'], directory)
    except Exception:
        store.finish(task['id'], error='artifact_download_failed')
        return
    store.progress(task['id'], 'generating')
    partial = directory / 'partial.mp4'
    gpu = engine.generate(task['request'], paths, partial)
    store.progress(task['id'], 'saving')
    media = inspect_media(partial, task['request']['frames'])
    result = dict(provenance, sha256=sha256(partial), size=partial.stat().st_size,
                  media=media, gpu=gpu, generate_seconds=time.monotonic()-started)
    (directory / 'record.json').write_text(json.dumps(dict(request=task['request'], result=result), indent=2) + '\n')
    partial.replace(directory / 'video.mp4')
    store.finish(task['id'], result=result)


def verify_gpus():
    import torch
    expected = set(os.environ['VDN_GPU_UUIDS'].split(','))
    if len(expected) != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError('exactly two dedicated GPUs required')
    observed = []
    for i in range(2):
        prop = torch.cuda.get_device_properties(i)
        observed.append('GPU-' + str(prop.uuid).removeprefix('GPU-'))
        if '4090' not in prop.name or prop.total_memory < 23 * 1024**3:
            raise RuntimeError('full RTX4090 required')
        a = torch.ones((64, 64), dtype=torch.bfloat16, device=f'cuda:{i}')
        if not torch.all(a @ a == 64).item():
            raise RuntimeError('BF16 check failed')
        torch.cuda.synchronize(i)
        del a
    if set(observed) != expected:
        raise RuntimeError('GPU allocation differs from manifest')
    return observed


def main():
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    instance = str(uuid.uuid4())
    state = State(instance)
    store = TaskStore(os.environ.get('VDN_TASK_ROOT', '/projects/h3-vdn/tasks'), instance)
    server = serve(state, store, os.environ['VDN_RUNTIME_TOKEN'], os.environ['VDN_ARTIFACT_SOURCE'])
    stop = threading.Event()
    current = None

    def terminate(*_):
        state.update(ready=False, stage='stopping')
        stop.set()
        # CUDA cannot be safely cancelled in-process. Persist the active outcome
        # before process termination; queued tasks are invalidated on restart.
        if current and store.get(current)['status'] == 'in_progress':
            store.finish(current, error='instance_stopped')
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        state.update(stage='gpu_check')
        state.update(gpu_uuids=verify_gpus(), stage='model_integrity')
        models = Path(os.environ.get('VDN_MODELS', '/models/VDN-H3')).resolve()
        lock_path = Path(os.environ.get('VDN_MODEL_LOCK', '/models/VDN-H3.lock.json'))
        lock = json.loads(lock_path.read_text())
        validate_model(models, lock, 8)
        provenance = dict(runtime_version=VERSION, execution_instance_id=instance, model_revision=lock['revision'],
                          model_lock_sha256=sha256(lock_path),
                          upstream=tomllib.loads((Path(__file__).resolve().parents[1]/'pyproject.toml').read_text())['tool']['vdn']['upstream'])
        state.update(stage='loading')
        started = time.monotonic()
        engine = Engine(models)
        state.update(ready=True, stage='ready', model_load_count=1, load_seconds=time.monotonic()-started)
        print(json.dumps(state.snapshot()), flush=True)
        while not stop.is_set():
            task = store.take()
            if task is None:
                stop.wait(0.5)
                continue
            current = task['id']
            execute(store, task, engine, provenance)
            print(json.dumps({'task':current, 'status':store.get(current)['status']}), flush=True)
            current = None
    except Exception as error:
        state.update(ready=False, stage='failed')
        if current and store.get(current)['status'] == 'in_progress':
            store.finish(current, error=type(error).__name__)
        # No prompt, credentials or signed input URLs in logs.
        print(json.dumps({'stage':'failed', 'error_type':type(error).__name__}), flush=True)
        raise RuntimeError('VDN runtime failed; consult task/model validation evidence') from None
    finally:
        server.shutdown()
        server.server_close()
        store.close()


if __name__ == '__main__':
    main()
