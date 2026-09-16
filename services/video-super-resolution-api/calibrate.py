"""Measure temporal-window capacity with the actual model in isolated CUDA processes."""
import argparse
import hashlib
from contextlib import closing
from datetime import datetime, timezone
from itertools import islice
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from video_sr.capacity import search_capacity, CapacityError
from video_sr.integrity import sha256
from video_sr.media import probe, read_frames, validate_request


def worker(job_path, result_path):
    # Never classify a dependency, driver, input or model failure as a capacity boundary.
    result = {'status': 'initialization_failed'}
    try:
        import numpy as np
        from engine import Engine
        job = json.loads(Path(job_path).read_text())
        engine = Engine(calibration=True)
        torch = engine.torch
        result['identity'] = engine.capacity_identity()
        if sha256(job['source']) != job['source_sha256']:
            raise ValueError('source changed')
        with closing(read_frames(job['source'], job['media'])) as decoded:
            frames = np.stack(list(islice(decoded, job['frames'])))
        if len(frames) != job['frames']:
            raise ValueError('source shorter than requested window')
        req = SimpleNamespace(**job['parameters'])
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        baseline = total - free
        reserved_before = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        result.update(status='inference_failed', baseline_used_bytes=baseline)
        try:
            output = engine.restore_window(frames, req)
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            result['status'] = 'oom'
            import traceback
            traceback.print_exc()
        else:
            if output.shape != (len(frames), req.target_height, req.target_width, 3) or output.dtype != np.uint8:
                raise ValueError('invalid output')
            peak = baseline + max(0, torch.cuda.max_memory_reserved() - reserved_before)
            result.update(status='ok' if peak + job['reserve_bytes'] <= total else 'headroom_exceeded',
                          estimated_peak_used_bytes=peak, output_sha256=hashlib.sha256(output.tobytes()).hexdigest())
        result.update(seconds=time.monotonic() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as exc:
        result['error_type'] = type(exc).__name__
        if str(exc) in {'model_integrity_failed', 'one_bf16_cuda_device_required'}:
            result['error_code'] = str(exc)
        # Full traceback is operator-only output, never a public task error.
        import traceback
        traceback.print_exc()
    finally:
        Path(result_path).write_text(json.dumps(result, indent=2) + '\n')


def calibrate(args):
    source, directory = args.source.resolve(), args.output.resolve()
    if args.max_frames < 9 or args.repeats < 2 or args.reserve_mib < 0 or args.trial_timeout <= 0:
        raise ValueError('invalid search budget')
    if any(n < 16 or n > 2048 or n % 16 for n in (args.target_width, args.target_height)):
        raise ValueError('target dimensions must be divisible by 16 and within 16..2048')
    media = probe(source)
    params = dict(target_width=args.target_width, target_height=args.target_height, seed=args.seed)
    if not 0 <= args.seed <= 4294967295:
        raise ValueError('invalid seed')
    validate_request(SimpleNamespace(**params), media)
    ceiling = min(args.max_frames, media['frames'])
    if ceiling < 9:
        raise ValueError('at least nine real source frames required; never synthesize extra calibration frames')
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, status='testing', created_at=datetime.now(timezone.utc).isoformat(),
                  source_sha256=sha256(source), reserve_bytes=args.reserve_mib * 1024 * 1024,
                  geometry=dict(input_width=media['width'], input_height=media['height'],
                                target_width=args.target_width, target_height=args.target_height),
                  input_media=media, parameters=params, measurements=[])
    (directory / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    script = str(Path(__file__).resolve())

    def trial(frames):
        number = len(report['measurements'])
        job = dict(source=str(source), source_sha256=report['source_sha256'], media=media, frames=frames,
                   reserve_bytes=report['reserve_bytes'], parameters={**params, 'seed': (args.seed + number) % 4294967296})
        job_path, result_path = directory / f'{number:04d}-input.json', directory / f'{number:04d}-result.json'
        job_path.write_text(json.dumps(job, indent=2) + '\n')
        report['active_trial'] = {'number': number, 'frames': frames, 'seed': job['parameters']['seed']}
        (directory / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
        with (directory / f'{number:04d}.log').open('wb') as log:
            try:
                process = subprocess.run([sys.executable, script, '--worker', str(job_path), str(result_path)],
                                         stdout=log, stderr=log, timeout=args.trial_timeout)
                if process.returncode or not result_path.exists():
                    result = {'status': 'trial_process_failed', 'returncode': process.returncode}
                else:
                    result = json.loads(result_path.read_text())
            except subprocess.TimeoutExpired:
                result = {'status': 'trial_timeout'}
        report['measurements'].append({'frames': frames, 'seed': job['parameters']['seed'], **result})
        report.pop('active_trial', None)
        if 'identity' in result:
            if 'identity' in report and report['identity'] != result['identity']:
                raise CapacityError('hardware_model_or_runtime_changed_during_search')
            report['identity'] = result['identity']
        if 'baseline_used_bytes' in result:
            report['baseline_used_bytes'] = min(report.get('baseline_used_bytes', result['baseline_used_bytes']), result['baseline_used_bytes'])
        (directory / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'frames': frames, 'status': result['status'], 'seconds': result.get('seconds')}), flush=True)
        return result['status']

    try:
        report['result'] = search_capacity(trial, ceiling, args.repeats)
        if sha256(source) != report['source_sha256']:
            raise CapacityError('source_changed_during_search')
        report['status'] = 'measured'
        (directory / 'capacity.json').write_text(json.dumps(report, indent=2) + '\n')
    except Exception:
        report['status'] = 'failed'
        raise
    finally:
        (directory / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['result'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', nargs=2, metavar=('JOB', 'RESULT'), help=argparse.SUPPRESS)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--target-width', type=int)
    parser.add_argument('--target-height', type=int)
    parser.add_argument('--max-frames', type=int, help='Explicit test budget, not a video length limit')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--reserve-mib', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--trial-timeout', type=int, default=1800)
    args = parser.parse_args()
    if args.worker:
        worker(*args.worker)
    else:
        if any(getattr(args, key) is None for key in ('source', 'output', 'target_width', 'target_height', 'max_frames')):
            parser.error('--source, --output, --target-width, --target-height and --max-frames are required')
        calibrate(args)
