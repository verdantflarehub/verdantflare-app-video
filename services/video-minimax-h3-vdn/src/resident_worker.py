"""One resident upstream pipeline, one durable queue, no automatic generation retry."""
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import tomllib
from types import SimpleNamespace
import uuid

from resident_api import State, VERSION, serve
from task_store import TaskStore
from vdn_io import TRANSFORMERS, inspect_media, sha256, validate_model

from ref2va_io import FRAMES, download
from vdn_layout import attach_layout_bridge

class Engine:
    def __init__(self, models):
        import torch
        from vendor.infer_diffusers import load_pipeline
        self.args = SimpleNamespace(models=str(models), transformer=TRANSFORMERS[8], fp8=False,
                                    device='cuda:0', offload_dit=True)
        with torch.inference_mode():
            self.pipeline = load_pipeline(self.args, 'ref2va')
            self.hybrid_blocks = attach_layout_bridge(self.pipeline.transformer_ref)
            upstream = sys.modules[type(self.pipeline.transformer_ref).__module__]
            self._install_chunked_decomposed_attention(upstream)
            from vdn_memory import install_frame_statistics
            install_frame_statistics(upstream)
            self.softmax_backend = upstream.set_softmax_backend(self.pipeline.transformer_ref, "decomposed")

    @staticmethod
    def _install_chunked_decomposed_attention(upstream):
        """Keep the upstream decomposed mask exact while bounding KV gather memory.

        The upstream implementation concatenates every window's gathered K/V rows
        before calling FA4.  At 345 frames that temporary can exceed a 24 GiB card.
        Execute the same varlen calls one window group at a time instead; the plan,
        row sets and kernel are unchanged, only the lifetime of each gather is shorter.
        """
        import torch
        from flash_attn.cute.interface import flash_attn_varlen_func

        def chunked(query, key, value, layout, bounds, scale, anchor_frames="none"):
            plan = upstream._plan(layout, bounds, anchor_frames, query.device)
            if not key.is_contiguous():
                key = key.contiguous()
            if not value.is_contiguous():
                value = value.contiguous()
            out = torch.empty(query.shape, dtype=query.dtype, device=query.device)
            if len(plan.dense_q):
                qd = query[plan.dense_q]
                with upstream.sdpa_kernel(upstream.SDPBackend.CUDNN_ATTENTION):
                    od = upstream.scaled_dot_product_attention(
                        qd.transpose(0, 1).unsqueeze(0), key.transpose(0, 1).unsqueeze(0),
                        value.transpose(0, 1).unsqueeze(0), scale=scale)
                out[plan.dense_q] = od[0].transpose(0, 1)
            if plan.has_windows:
                for group in range(len(plan.cu_q) - 1):
                    q0, q1 = (int(x) for x in plan.cu_q[group:group + 2].tolist())
                    k0, k1 = (int(x) for x in plan.cu_k[group:group + 2].tolist())
                    kw = key[plan.kv_gather[k0:k1]]
                    vw = value[plan.kv_gather[k0:k1]]
                    ow = flash_attn_varlen_func(
                        query[plan.win_q[q0:q1]], kw, vw,
                        cu_seqlens_q=torch.tensor([0, q1 - q0], device=query.device, dtype=torch.int32),
                        cu_seqlens_k=torch.tensor([0, k1 - k0], device=query.device, dtype=torch.int32),
                        max_seqlen_q=q1 - q0, max_seqlen_k=k1 - k0, softmax_scale=scale)
                    ow = ow[0] if isinstance(ow, tuple) else ow
                    out[plan.win_q[q0:q1]] = ow
            return out

        upstream.window_softmax_decomposed = chunked

    def generate(self, request, paths, output):
        import torch
        from vendor.infer_diffusers import render_ref2va
        for index in range(2):
            torch.cuda.reset_peak_memory_stats(index)
        transformer = self.pipeline.transformer_ref
        transformer._vdn_layout_calls = 0
        transformer._vdn_linear_calls = 0
        args = SimpleNamespace(**vars(self.args), out=str(output), frames=FRAMES[request['seconds']],
                               steps=request['num_inference_steps'], seed=request['seed'])
        render_ref2va(self.pipeline, args, request['prompt'], paths)
        for index in range(2):
            torch.cuda.synchronize(index)
        if transformer._vdn_layout_calls != args.steps or transformer._vdn_linear_calls < args.steps*self.hybrid_blocks:
            raise RuntimeError('VDN branch execution was not verified')
        return {'vdn_softmax_backend': self.softmax_backend, 'peak_allocated_bytes': [torch.cuda.max_memory_allocated(i) for i in range(2)],
                'peak_reserved_bytes': [torch.cuda.max_memory_reserved(i) for i in range(2)],
                'vdn_layout_calls':transformer._vdn_layout_calls, 'vdn_linear_calls':transformer._vdn_linear_calls,
                'vdn_hybrid_blocks':self.hybrid_blocks}


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
    media = inspect_media(partial, FRAMES[task['request']['seconds']])
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
        # cuSOLVER initializes lazily. Do this on the inference thread before
        # long-sequence activations occupy the card, using the delta-rule ops.
        eye = torch.eye(128, dtype=torch.float32, device=f'cuda:{i}').expand(2, -1, -1).contiguous()
        chol = torch.linalg.cholesky(eye)
        inverse_factor = torch.linalg.solve_triangular(chol, eye, upper=False, left=True)
        if not torch.equal(inverse_factor, eye):
            raise RuntimeError('FP32 delta-rule solver check failed')
        torch.cuda.synchronize(i)
        del a, eye, chol, inverse_factor
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
        models = Path(os.environ.get('VDN_MODELS', '/models/VDN-H3-Ref2VA')).resolve()
        lock_path = Path(os.environ.get('VDN_MODEL_LOCK', '/models/VDN-H3-Ref2VA.lock.json'))
        lock = json.loads(lock_path.read_text())
        if lock.get("task") != "ref2va" or lock.get("profile") != "vdn-ref2va-8step":
            raise ValueError("Ref2VA VDN model package required")
        validate_model(models, lock, 8)
        provenance = dict(profile='vdn-ref2va-8step', runtime_version=VERSION, execution_instance_id=instance, model_revision=lock['revision'],
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
