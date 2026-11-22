"""Offline, hash-bound legacy migration. Never modifies the source state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .fence import EpochLeaseFenceGuard


def migrate(source: Path, records: Path, output: Path, expected_sha256: str) -> dict:
    if source.is_symlink() or not source.is_file() or os.path.lexists(output):
        raise ValueError("migration_requires_regular_source_and_new_output")
    original = source.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    if digest != expected_sha256:
        raise ValueError("migration_source_digest_mismatch")
    if json.loads(original).get("inflight_tasks"):
        raise ValueError("migration_requires_drained_state")
    ledger_bytes = records.read_bytes()
    ledger = json.loads(ledger_bytes)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fence-migration-", dir=output.parent) as tmp:
        candidate = Path(tmp) / "gpu_guard_state.json"
        candidate.write_bytes(original)
        guard = EpochLeaseFenceGuard(state_path=candidate, legacy_lease_records=ledger)
        epoch = guard.current_epoch  # Raises on any ambiguity/corruption/write failure.
        migrated = candidate.read_bytes()
        # Verify restart before producing an operator-installable output.
        if EpochLeaseFenceGuard(state_path=candidate).current_epoch != epoch:
            raise ValueError("migration_restart_epoch_mismatch")
        if source.read_bytes() != original:
            raise ValueError("migration_source_changed")
        with output.open("xb") as stream:
            stream.write(migrated)
            stream.flush()
            os.fsync(stream.fileno())
    return {
        "source_sha256": digest,
        "ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        "output_sha256": hashlib.sha256(migrated).hexdigest(),
        "gpu_epoch": epoch,
        "source_unchanged": True,
        "remote_state_modified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ledger-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(migrate(args.source, args.ledger_records, args.output, args.source_sha256), sort_keys=True))


if __name__ == "__main__":
    main()
