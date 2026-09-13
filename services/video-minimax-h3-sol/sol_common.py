"""CPU-only validation and provenance for the isolated Sol-H3 runtime."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRAMES = {5: 124, 10: 243, 15: 362}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def relative_file(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("expected a nonempty relative file path")
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("file is missing or escapes its declared root")
    return path


def verify_source(source):
    lock = read_json(ROOT / "upstream.lock.json")
    for name, expected in lock["source_sha256"].items():
        if sha256(relative_file(source, name)) != expected:
            raise ValueError(f"upstream source mismatch: {name}")
    actual = {str(p.relative_to(source)) for p in Path(source).rglob("*.py")}
    expected = {n for n in lock["source_sha256"] if n.endswith(".py")}
    if actual != expected:
        raise ValueError("unexpected Python modules in upstream source")
    return lock


def load_profile(path):
    profile = read_json(path)
    # Profiles are reviewed code, not an unchecked way to bypass the 4090 gate.
    canonical = ROOT / "profiles" / (profile.get("id", "") + ".json")
    if canonical.parent != ROOT / "profiles" or not canonical.is_file():
        raise ValueError("unknown profile")
    if profile != read_json(canonical):
        raise ValueError("profile differs from its registered definition")
    return profile


def validate_request(path, input_root):
    request = read_json(path)
    allowed = {"schema_version", "task", "status", "approval_id", "generation_unit_id",
               "prompt", "duration", "seed", "references"}
    if (set(request) != allowed or type(request["schema_version"]) is not int
            or request["schema_version"] != 1):
        raise ValueError("request fields do not match schema v1")
    if request["task"] != "ref2va" or request["status"] != "frozen":
        raise ValueError("only frozen Ref2VA research inputs are accepted")
    for key in ("approval_id", "generation_unit_id", "prompt"):
        if not isinstance(request[key], str) or not request[key].strip():
            raise ValueError(f"missing {key}")
    if request["prompt"] != request["prompt"].strip():
        raise ValueError("frozen prompt has outer whitespace that upstream would strip")
    if type(request["duration"]) is not int or request["duration"] not in FRAMES:
        raise ValueError("Sol-H3 accepts only 5/10/15 second profiles")
    if type(request["seed"]) is not int or not 0 <= request["seed"] < 2**63:
        raise ValueError("seed must be a nonnegative signed 64-bit integer")
    refs = request["references"]
    if not isinstance(refs, list) or not refs:
        raise ValueError("Ref2VA requires references")
    resolved = []
    for ref in refs:
        if not isinstance(ref, dict) or set(ref) != {"type", "path", "sha256"}:
            raise ValueError("reference requires type, path, sha256")
        if ref["type"] not in {"image", "video", "audio"}:
            raise ValueError("unknown reference type")
        media = relative_file(input_root, ref["path"])
        if sha256(media) != ref["sha256"]:
            raise ValueError("reference SHA-256 mismatch")
        resolved.append((ref["type"], media))
    if not any(kind in {"image", "video"} for kind, _ in resolved):
        raise ValueError("audio-only Ref2VA input is forbidden")
    return request, resolved


def verify_models(root):
    root = Path(root)
    lock = read_json(ROOT / "models.lock.json")
    receipt = read_json(root / "models.manifest.json")
    if receipt.get("lock_sha256") != sha256(ROOT / "models.lock.json"):
        raise ValueError("model receipt belongs to a different lock")
    names = {"base/" + name for name in lock["base"]["files"]}
    adapter_name = "adapter/" + lock["adapter"]["file"]
    names.add(adapter_name)
    if set(receipt.get("files", {})) != names:
        raise ValueError("model receipt is incomplete or has an unexpected partition")
    for name in sorted(names):
        path = relative_file(root, name)
        record = receipt["files"][name]
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"model integrity mismatch: {name}")
    if receipt["files"][adapter_name]["sha256"] != lock["adapter"]["sha256"]:
        raise ValueError("wrong Ref2VA adapter")
    # Check every index reference, not just the presence of an index file.
    for name in lock["base"]["files"]:
        if name.endswith(".safetensors.index.json"):
            index_path = relative_file(root, "base/" + name)
            index = read_json(index_path)
            for shard in set(index["weight_map"].values()):
                indexed_name = "base/" + (Path(name).parent / shard).as_posix()
                if indexed_name not in names:
                    raise ValueError("weight index refers to an unsealed shard")
    return root / "base", root / adapter_name
