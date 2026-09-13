#!/usr/bin/env python3
"""Launch the sealed official Sol-H3 CLI; never route to another engine."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
try:
    import resource
except ModuleNotFoundError:  # Windows development hosts do not provide POSIX resource.
    class _ResourceFallback:
        RUSAGE_CHILDREN = 0

        @staticmethod
        def getrusage(_who):
            class _Usage:
                ru_maxrss = 0
            return _Usage()

    resource = _ResourceFallback()
import signal
import subprocess
import sys
import time

from sol_common import (ROOT, FRAMES, load_profile, read_json, sha256,
                        validate_request, verify_models, verify_source, write_json)


def gpu_probe(profile, model_root):
    import torch

    if torch.cuda.device_count() != profile["gpu_count"]:
        raise ValueError("visible GPU count differs from the selected profile")
    receipt = read_json(Path(model_root) / "models.manifest.json")
    weight_bytes = sum(item["bytes"] for name, item in receipt["files"].items()
                       if name.endswith(".safetensors"))
    inventory = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        capability = properties.major * 10 + properties.minor
        free, total = torch.cuda.mem_get_info(index)
        if capability != profile["compute_capability"]:
            raise ValueError("GPU architecture has no validation for this profile")
        if free < weight_bytes:
            raise ValueError("insufficient free GPU memory for resident-profile screening")
        uuid = str(getattr(properties, "uuid", ""))
        if not uuid:
            raise ValueError("CUDA GPU UUID is unavailable")
        inventory.append({"uuid": uuid, "name": properties.name, "capability": capability,
                          "free_bytes": free, "total_bytes": total})
    if len({entry["uuid"] for entry in inventory}) != len(inventory):
        raise ValueError("duplicate CUDA GPU UUID")
    return inventory


def execute_process(command, output, environment, timeout):
    started = time.monotonic()
    with (output / "inference.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            if os.name == "nt":
                process.terminate()
            else:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    elapsed = time.monotonic() - started
    if code:
        raise RuntimeError(f"upstream inference exited with code {code}; see inference.log")
    return elapsed


def validate_media(path, duration):
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format",
        "-of", "json", str(path)], timeout=300))
    videos = [s for s in data["streams"] if s["codec_type"] == "video"]
    audios = [s for s in data["streams"] if s["codec_type"] == "audio"]
    if len(videos) != 1 or len(audios) != 1:
        raise ValueError("expected one video stream and one native audio stream")
    v, a = videos[0], audios[0]
    from fractions import Fraction
    if (v["width"], v["height"], Fraction(v["avg_frame_rate"]), int(v["nb_read_frames"])) != (
            1344, 768, 24, FRAMES[duration]):
        raise ValueError("output video does not match the frozen profile")
    if a.get("channels") != 2:
        raise ValueError("expected stereo native audio")
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=300)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("/opt/sol-h3"))
    parser.add_argument("--request", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true", help="default is CPU-only preflight")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    record = None
    try:
        profile = load_profile(args.profile)
        source_lock = verify_source(args.source)
        if args.request is not None and args.inputs is None:
            raise ValueError("--request requires --inputs")
        request, refs = validate_request(args.request, args.inputs) if args.request else (None, [])
        if args.models:
            base, adapter = verify_models(args.models)
        if profile["blocked_reason"]:
            print(json.dumps({"status": "blocked", "reason": profile["blocked_reason"]}))
            return 2
        if not args.execute:
            print(json.dumps({"status": "preflight_passed", "gpu_validated": False,
                              "profile": profile["id"], "revision": source_lock["revision"]}))
            return 0
        if any(x is None for x in (args.request, args.inputs, args.models, args.output)):
            raise ValueError("execution requires --request, --inputs, --models and --output")
        if args.timeout_seconds <= 0:
            raise ValueError("timeout must be positive")
        args.output = args.output.resolve()
        # New directory per attempt; a retry cannot overwrite provenance or media.
        args.output.mkdir(parents=True, exist_ok=False)
        record = {"status": "preparing", "profile": profile, "upstream": source_lock["revision"],
                  "request_sha256": sha256(args.request), "request": request,
                  "models_manifest_sha256": sha256(args.models / "models.manifest.json"),
                  "requirements_sha256": sha256(ROOT / "requirements.lock"),
                  "upstream_lock_sha256": sha256(ROOT / "upstream.lock.json"),
                  "quality_review": "pending", "stage_timings": None,
                  "warmups": 1, "measured_runs": 1, "timeout_seconds": args.timeout_seconds}
        write_json(args.output / "manifest.json", record)
        record["gpus"] = gpu_probe(profile, args.models)
        prompt_file = args.output / "prompt.txt"
        prompt_file.write_text(request["prompt"], encoding="utf-8")
        video = args.output / "output.mp4"
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   f"--nproc_per_node={profile['gpu_count']}", str(args.source.resolve() / "infer.py"),
                   "--model", str(base.resolve()), "--adapter", str(adapter.resolve()),
                   "--task", "ref2va", "--attention-backend", profile["attention_backend"],
                   "--reference-image-resize-mode", profile["reference_image_resize_mode"],
                   "--duration", str(request["duration"]), "--seed", str(request["seed"]),
                   "--prompt-file", str(prompt_file), "--output", str(video), "--warmup"]
        for kind, path in refs:
            command += ["--reference", f"{kind}:{path}"]
        env = os.environ.copy()
        # Local inference must not fetch models or forward download credentials.
        for key in list(env):
            if (any(part in key.upper() for part in ("TOKEN", "SECRET", "PASSWORD", "API_KEY"))
                    or key.startswith(("H3_", "SOL_"))
                    or key in {"RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}):
                env.pop(key)
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        record["status"] = "running"
        write_json(args.output / "manifest.json", record)
        with (args.output / "gpu.csv").open("w") as gpu_log:
            monitor = subprocess.Popen([
                "nvidia-smi", "--query-gpu=timestamp,uuid,driver_version,memory.used,utilization.gpu",
                "--format=csv", "--loop-ms=1000"], stdout=gpu_log, stderr=subprocess.DEVNULL)
            try:
                record["process_wall_s"] = execute_process(command, args.output, env, args.timeout_seconds)
            finally:
                monitor.terminate()
                monitor.wait(timeout=10)
            if monitor.returncode not in (0, -signal.SIGTERM):
                raise RuntimeError("GPU resource monitoring failed")
        record["child_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        record["status"] = "validating"
        write_json(args.output / "manifest.json", record)
        write_json(args.output / "media.json", validate_media(video, request["duration"]))
        metrics = []
        for line in (args.output / "inference.log").read_text().splitlines():
            try:
                value = json.loads(line)
                if isinstance(value, dict) and "inference_s" in value:
                    metrics.append(value)
            except json.JSONDecodeError:
                pass
        if len(metrics) != 1:
            raise ValueError("missing or ambiguous upstream timing record")
        record.update(status="technical_success", upstream_metrics=metrics[0],
                      output_sha256=sha256(video), output_bytes=video.stat().st_size)
        write_json(args.output / "manifest.json", record)
        print("Technical media checks passed; creative review and detailed profiling remain pending.")
        return 0
    except Exception as error:
        if record is not None:
            record.update(status="failed", error_type=type(error).__name__)
            if isinstance(error, (ValueError, RuntimeError)):
                record["error"] = str(error)
            write_json(args.output / "manifest.json", record)
        # Do not print subprocess/network exception payloads that could include credentials.
        print(f"Sol-H3 operation failed ({type(error).__name__}).")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
