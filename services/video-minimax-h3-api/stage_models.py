#!/usr/bin/env python3
"""Stage and verify MiniMax-H3 Ref2VA model weights and metadata.

Packaged inside the video-minimax-h3-api container image to provide
a reliable, declarative staging CLI for Kubernetes jobs and init hooks.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

MODEL_REPOSITORY = "MiniMaxAI/MiniMax-H3"
DEFAULT_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"

# minimax_h3_ref2va_int8_convrot.safetensors
EXPECTED_SIZE = 34038894550
EXPECTED_SHA256 = "9eef934046a0671bc8a5daf87100705e1478419c574cfde70c50fbe6885f76a9"

DEFAULT_URL_MODELSCOPE = (
    "https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3/resolve/master/"
    "diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors"
)
DEFAULT_URL_HF = (
    "https://hf-mirror.com/Comfy-Org/MiniMax-H3/resolve/main/"
    "diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors"
)

CHUNK_SIZE = 16 * 1024 * 1024  # 16 MiB


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def fetch_piece(
    urls: list[str],
    target_path: Path,
    start_byte: int,
    end_byte: int,
    max_attempts: int = 15,
) -> None:
    expected_bytes = end_byte - start_byte + 1
    if target_path.exists() and target_path.stat().st_size == expected_bytes:
        return

    part_path = target_path.with_suffix(".part")
    retained = part_path.stat().st_size if part_path.exists() else 0

    last_error: Exception | None = None
    for attempt in range(max_attempts):
        url = urls[attempt % len(urls)]
        try:
            current_offset = start_byte + retained
            req = urllib.request.Request(
                url,
                headers={
                    "Range": f"bytes={current_offset}-{end_byte}",
                    "Accept-Encoding": "identity",
                    "User-Agent": "VerdantFlare-H3-Staging/1.0",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                if resp.status not in (200, 206):
                    raise RuntimeError(f"HTTP status {resp.status} from {url}")
                with part_path.open("ab" if retained else "wb") as f:
                    while block := resp.read(1024 * 1024):
                        f.write(block)
                        retained += len(block)

            if retained == expected_bytes:
                part_path.replace(target_path)
                return
        except Exception as err:
            last_error = err
            backoff = min(15, 2**attempt)
            time.sleep(backoff)

    raise RuntimeError(
        f"Failed to fetch chunk {start_byte}-{end_byte} after {max_attempts} attempts: {last_error}"
    )


def stage_metadata(model_root: Path, revision: str) -> None:
    revision_file = model_root / ".verdantflare-revision"
    if revision_file.exists() and revision_file.read_text().strip() == revision:
        print("=== [H3-STAGE] Base metadata already up to date. ===")
        return

    print("=== [H3-STAGE] Downloading Base Metadata from HF Mirror ===")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=MODEL_REPOSITORY,
            revision=revision,
            local_dir=str(model_root),
            allow_patterns=["LICENSE", "README.md", "model_index.json", "Ref2VA/*"],
        )
        print("=== [H3-STAGE] Metadata snapshot downloaded successfully. ===")
    except Exception as e:
        print(f"Warning: snapshot_download encountered: {e}, continuing...")


def stage_weights(
    model_root: Path,
    revision: str,
    workers: int = 16,
    urls: list[str] | None = None,
) -> None:
    if urls is None:
        urls = [DEFAULT_URL_MODELSCOPE, DEFAULT_URL_HF]

    target_dir = model_root / "serialized-int8" / "diffusion_models"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "minimax_h3_ref2va_int8_convrot.safetensors"
    revision_file = model_root / ".verdantflare-revision"

    if target_file.exists() and target_file.stat().st_size == EXPECTED_SIZE:
        print(
            f"=== [H3-STAGE] Found existing weights ({EXPECTED_SIZE} bytes), checking SHA256 ==="
        )
        existing_sha = compute_sha256(target_file)
        if existing_sha == EXPECTED_SHA256:
            print(f"SUCCESS: Integrity verified ({existing_sha})")
            revision_file.write_text(revision)
            return
        print(
            f"Mismatch: found {existing_sha}, expected {EXPECTED_SHA256}. Re-downloading..."
        )
        target_file.unlink()

    piece_root = model_root / ".pieces" / EXPECTED_SHA256
    piece_root.mkdir(parents=True, exist_ok=True)

    ranges = [
        (s, min(s + CHUNK_SIZE, EXPECTED_SIZE) - 1)
        for s in range(0, EXPECTED_SIZE, CHUNK_SIZE)
    ]
    total_chunks = len(ranges)
    print(
        f"=== [H3-STAGE] Staging {EXPECTED_SIZE} bytes in {total_chunks} chunks "
        f"({CHUNK_SIZE // 1024 // 1024} MB each, {workers} workers) ==="
    )

    completed = sum(
        1 for s, e in ranges if (piece_root / f"{s:012d}-{e:012d}").exists()
    )
    print(f"Pre-existing verified chunks: {completed}/{total_chunks}")

    t_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                fetch_piece, urls, piece_root / f"{s:012d}-{e:012d}", s, e
            ): (s, e)
            for s, e in ranges
        }
        last_log = time.time()
        for f in concurrent.futures.as_completed(futures):
            f.result()
            completed += 1
            now = time.time()
            if now - last_log >= 10 or completed == total_chunks:
                elapsed = max(0.1, now - t_start)
                pct = (completed / total_chunks) * 100
                speed = ((completed * CHUNK_SIZE) / 1024 / 1024) / elapsed
                print(
                    f"Progress: {completed}/{total_chunks} ({pct:.1f}%) speed ~{speed:.1f} MB/s",
                    flush=True,
                )
                last_log = now

    print("=== [H3-STAGE] All chunks downloaded. Assembling final weights file... ===")
    assembling = target_file.with_suffix(".assembling")
    if assembling.exists():
        assembling.unlink()

    with assembling.open("wb") as out:
        for s, e in ranges:
            chunk_path = piece_root / f"{s:012d}-{e:012d}"
            with chunk_path.open("rb") as src:
                shutil.copyfileobj(src, out, length=CHUNK_SIZE)

    print("=== [H3-STAGE] Validating final file integrity... ===")
    final_sha = compute_sha256(assembling)
    if final_sha != EXPECTED_SHA256:
        assembling.unlink()
        raise RuntimeError(
            f"Integrity check failed: got {final_sha}, expected {EXPECTED_SHA256}"
        )

    assembling.replace(target_file)
    revision_file.write_text(revision)

    # Clean up chunk fragments after successful verification
    try:
        shutil.rmtree(piece_root)
    except OSError:
        pass

    print(f"SUCCESS: MiniMax H3 Weights Staged & Verified ({final_sha})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage and verify MiniMax-H3 Ref2VA model weights."
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH", "/models/MiniMax-H3"),
        help="Target model root directory (default: /models/MiniMax-H3)",
    )
    parser.add_argument(
        "--revision",
        default=os.environ.get("MODEL_REVISION", DEFAULT_REVISION),
        help=f"Target model revision (default: {DEFAULT_REVISION})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("STAGING_WORKERS", "16")),
        help="Parallel download workers (default: 16)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing files, do not download.",
    )
    args = parser.parse_args()

    model_root = Path(args.model_path)
    model_root.mkdir(parents=True, exist_ok=True)

    print(f"=== [H3-STAGE] Target Model Root: {model_root} ===")

    if args.verify_only:
        target_file = (
            model_root
            / "serialized-int8"
            / "diffusion_models"
            / "minimax_h3_ref2va_int8_convrot.safetensors"
        )
        if not target_file.exists():
            print(f"ERROR: Target file not found: {target_file}", file=sys.stderr)
            return 1
        sha = compute_sha256(target_file)
        if sha != EXPECTED_SHA256:
            print(
                f"ERROR: SHA256 mismatch: got {sha}, expected {EXPECTED_SHA256}",
                file=sys.stderr,
            )
            return 1
        print(f"SUCCESS: File verified: {target_file} ({sha})")
        return 0

    stage_metadata(model_root, args.revision)
    stage_weights(model_root, args.revision, workers=args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
