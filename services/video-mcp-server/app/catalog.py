"""Read-only reviewed channel catalogs; never enables generation routes."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from datetime import date, datetime
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator, FormatChecker


class CatalogError(ValueError):
    pass


class UniqueLoader(yaml.SafeLoader):
    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            raise CatalogError("aliases are not allowed")
        return super().compose_node(parent, index)

    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise CatalogError("duplicate or non-string key")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def read_yaml(path):
    raw = path.read_bytes()
    if len(raw) > 1024 * 1024:
        raise CatalogError("file too large")
    return raw, yaml.load(raw.decode("utf-8"), Loader=UniqueLoader)


def fields(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected.split()):
        raise CatalogError("invalid fields")


def unique(values):
    if len(values) != len(set(values)):
        raise CatalogError("duplicate identity")


def validate_semantics(data):
    sources = [s["id"] for s in data["sources"]]
    unique(sources)
    unique([m["id"] for m in data["models"]])
    def refs(item):
        if not set(item["source_refs"]) <= set(sources):
            raise CatalogError("unknown evidence reference")
    for model in data["models"]:
        refs(model)
        for key in ("supplier_profiles", "team_profiles"):
            unique([p["id"] for p in model[key]])
            for profile in model[key]:
                refs(profile)
                duration = profile["duration_seconds"]
                if duration and (duration["maximum"] < duration["minimum"] or profile["duration_choices_seconds"]):
                    raise CatalogError("invalid duration")
                maximum = profile["max_resolution"]
                resolutions = profile["resolutions"]
                if maximum is not None and (maximum not in resolutions or
                        int(maximum[:-1]) != max(int(r[:-1]) for r in resolutions)):
                    raise CatalogError("invalid resolution")


def project_channel(data):
    models = []
    for model in sorted(data["models"], key=lambda m: (m["sort_order"], m["id"])):
        if model["catalog_state"] != "listed":
            continue
        public = {k: copy.deepcopy(v) for k, v in model.items() if k not in {"source_refs", "review_notes"}}
        for key in ("supplier_profiles", "team_profiles"):
            for profile in public[key]:
                profile.pop("source_refs")
        models.append(public)
    return {**data["channel"], "version": data["channel_version"], "models": models}


def group_entries(channels):
    candidates = {}
    for channel in channels:
        for model in channel["models"]:
            candidates.setdefault(model["model_id"], []).append((channel["id"], model))
    groups, conflicts = [], []
    for model_id, entries in candidates.items():
        approved = [(c, m) for c, m in entries if m["identity_review"] == "approved"]
        signatures = {(m["display_name"], m["entry_kind"], m["business_type"]) for _, m in approved}
        conflicting = len(signatures) > 1
        if conflicting:
            conflicts.append(model_id)
        if approved and not conflicting:
            groups.append({"id": "model:" + model_id,
                           "entries": [{"channel_id": c, "entry_id": m["id"]} for c, m in approved]})
        for channel_id, model in entries:
            if model["identity_review"] != "approved" or conflicting:
                groups.append({"id": f"entry:{channel_id}:{model['id']}",
                               "entries": [{"channel_id": channel_id, "entry_id": model["id"]}]})
    return groups, conflicts


class ModelCatalog:
    def __init__(self, directory=None):
        self._value = {"schema_version": 1, "state": "error", "snapshot_id": None,
                       "channels": [], "groups": [], "conflicts": [], "error_code": "catalog_not_configured"}
        directory = directory if directory is not None else os.environ.get("VIDEO_MODEL_CATALOG_DIR")
        if not directory:
            return
        try:
            self._value = self._load(Path(directory))
        except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError, RecursionError):
            self._value["error_code"] = "catalog_invalid_configuration"

    def snapshot(self):
        return copy.deepcopy(self._value)

    def _load(self, root):
        root = root.resolve()
        def file(name):
            if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
                raise CatalogError("invalid path")
            resolved = (root / name).resolve()
            if root not in resolved.parents:
                raise CatalogError("path outside root")
            return resolved
        _, selection = read_yaml(file("channels.yaml"))
        fields(selection, "schema_version channels")
        if type(selection["schema_version"]) is not int or selection["schema_version"] != 1 or not isinstance(selection["channels"], list):
            raise CatalogError("invalid selection")
        schema = json.loads(Path(__file__).with_name("channel-catalog.schema.json").read_text("utf-8"))
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        channels, identity = [], []
        for item in selection["channels"]:
            fields(item, "channel_id channel_version sha256 catalog_file review_file")
            if not all(isinstance(v, str) and v for v in item.values()) or not re.fullmatch("[0-9a-f]{64}", item["sha256"]):
                raise CatalogError("invalid selection entry")
            raw, data = read_yaml(file(item["catalog_file"]))
            if hashlib.sha256(raw).hexdigest() != item["sha256"] or list(validator.iter_errors(data)):
                raise CatalogError("invalid channel data")
            if (data["channel"]["id"], data["channel_version"]) != (item["channel_id"], item["channel_version"]):
                raise CatalogError("channel mismatch")
            validate_semantics(data)
            _, review = read_yaml(file(item["review_file"]))
            fields(review, "schema_version channel_id channel_version candidate_path candidate_key candidate_sha256 decision scope approved_by approved_at decision_evidence_ref review_notes_ref")
            if (type(review["schema_version"]) is not int or review["schema_version"] != 1 or
                    review["decision"] != "approved" or review["scope"] != "complete_channel" or
                    (review["channel_id"], review["channel_version"], review["candidate_sha256"]) !=
                    (item["channel_id"], item["channel_version"], item["sha256"]) or
                    review["candidate_key"] not in (None, "channel-catalog.yaml") or
                    not all(isinstance(review[k], str) and review[k].strip() for k in
                            ("candidate_path", "approved_by", "approved_at", "decision_evidence_ref", "review_notes_ref")) or
                    not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:T.*)?", review["approved_at"])):
                raise CatalogError("unapproved channel")
            if "T" in review["approved_at"]:
                datetime.fromisoformat(review["approved_at"].replace("Z", "+00:00"))
            else:
                date.fromisoformat(review["approved_at"])
            channels.append(project_channel(data))
            identity.append([item["channel_id"], item["channel_version"], item["sha256"]])
        unique([c["id"] for c in channels])
        channels.sort(key=lambda c: (c["sort_order"], c["id"]))
        groups, conflicts = group_entries(channels)
        digest = hashlib.sha256(json.dumps(sorted(identity), separators=(",", ":")).encode()).hexdigest()
        return {"schema_version": 1, "state": "ready" if groups else "empty", "snapshot_id": digest,
                "channels": channels, "groups": groups, "conflicts": conflicts, "error_code": None}
