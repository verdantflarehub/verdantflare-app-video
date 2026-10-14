"""Durable single-account journal. Ambiguous create responses are never retried."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
import uuid

from PIL import Image

from .cli import ProviderError
from .models import Generation, JobRequest, MODELS


class GatewayError(RuntimeError):
    def __init__(self, code, status=409):
        self.code, self.status = code, status
        super().__init__(code)


def now():
    return datetime.now(UTC).isoformat()


class Gateway:
    def __init__(self, root: Path, cli, *, binding: str, organization: str,
                 workspace: str, credit_limit: int = 0, submission_limit: int = 0,
                 max_concurrent: int = 1, max_job_credits: int = 70, media=None):
        if min(credit_limit, submission_limit, max_job_credits) < 0 or max_concurrent < 1:
            raise ValueError("invalid_limits")
        if not all((binding, organization, workspace)):
            raise ValueError("account_binding_required")
        self.root, self.cli = root, cli
        self.binding, self.organization, self.workspace = binding, organization, workspace
        self.credit_limit, self.submission_limit = credit_limit, submission_limit
        self.max_concurrent, self.max_job_credits = max_concurrent, max_job_credits
        self.media = media
        self._lock = threading.RLock()
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, digest TEXT NOT NULL, record TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS binding (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL)")
            identity = json.dumps([binding, organization, workspace])
            old = db.execute("SELECT value FROM binding WHERE id=1").fetchone()
            if old and old[0] != identity:
                raise GatewayError("account_binding_changed")
            db.execute("INSERT OR IGNORE INTO binding VALUES(1, ?)", (identity,))
            # Any process death after durable intent may already have spent credits.
            for row in db.execute("SELECT record FROM jobs").fetchall():
                record = json.loads(row[0])
                if record["stage"] == "submitting":
                    record.update(stage="submission_unknown", error="submission_unknown", updated_at=now())
                    self.save(db, record)
        os.chmod(root / "jobs.sqlite3", 0o600)

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.root / "jobs.sqlite3", timeout=120)
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                db.execute("BEGIN IMMEDIATE")
                yield db
        finally:
            db.close()

    @staticmethod
    def save(db, record):
        db.execute("UPDATE jobs SET record=? WHERE id=?", (json.dumps(record), record["request_id"]))

    def load(self, request_id):
        with self.db() as db:
            row = db.execute("SELECT record FROM jobs WHERE id=?", (request_id,)).fetchone()
        if not row:
            raise GatewayError("job_not_found", 404)
        return json.loads(row[0])

    def check_binding(self):
        path = self.cli.state / "binding.json"
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
            if actual.get("workspace_id") != self.workspace:
                raise ValueError
        except (OSError, ValueError, AttributeError):
            raise GatewayError("authorization_required", 503) from None
        # Verify the CLI actually selected the bound workspace, not just our metadata.
        value = self.cli.run(["workspace", "status"])
        def matches(item):
            if not isinstance(item, dict):
                return False
            return any((key in {"id", "workspace_id", "active_workspace_id"} and val == self.workspace)
                       or (isinstance(val, dict) and matches(val)) for key, val in item.items())
        if not matches(value):
            raise GatewayError("workspace_binding_mismatch", 503)

    def account(self):
        self.check_binding()
        data = self.cli.run(["account", "status"])
        if (not isinstance(data, dict) or type(data.get("credits")) not in (int, float) or
                not math.isfinite(data["credits"]) or data["credits"] < 0):
            raise ProviderError("provider_protocol_error")
        return {"credits": data["credits"], "plan": str(data.get("subscription_plan_type", "unknown")),
                "binding_id": self.binding, "authorized": True}

    @contextmanager
    def source(self, generation):
        if generation.source is None:
            yield None
            return
        data = generation.source.content()
        try:
            with Image.open(io.BytesIO(data)) as image:
                if (image.format.lower() != generation.source.format or
                        getattr(image, "n_frames", 1) != 1 or image.width * image.height > 16 * 1024 * 1024):
                    raise ValueError
                width, height = image.size
                a, b = map(int, generation.aspect_ratio.split(":"))
                if abs(width * b - height * a) > max(a, b):
                    raise ValueError
                image.verify()
        except (OSError, ValueError, Image.DecompressionBombError):
            raise GatewayError("source_integrity_failed", 400) from None
        import tempfile
        with tempfile.TemporaryDirectory(prefix="input-", dir=self.root) as directory:
            path = Path(directory) / ("frame." + generation.source.format)
            path.write_bytes(data)
            os.chmod(path, 0o600)
            yield path

    def quote(self, generation: Generation, path=None):
        data = self.cli.run(generation.arguments("cost", path))
        if not isinstance(data, dict) or type(data.get("credits")) not in (int, float) or not 0 < data["credits"] <= self.max_job_credits:
            raise GatewayError("cost_not_allowed")
        return data["credits"]

    def cost(self, generation):
        with self._lock:
            self.check_binding()
            with self.source(generation) as path:
                return {"credits": self.quote(generation, path)}

    def submit(self, request: JobRequest):
        if request.binding_id != self.binding or request.organization_id != self.organization:
            raise GatewayError("account_access_denied", 403)
        # Image bytes never enter the durable journal, only their immutable digest.
        body = request.model_dump()
        if request.generation.source:
            body["generation"]["source"].pop("content_base64")
        digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        with self._lock:
            with self.db() as db:
                old = db.execute("SELECT digest,record FROM jobs WHERE id=?", (request.request_id,)).fetchone()
                if old:
                    if old[0] != digest:
                        raise GatewayError("idempotency_conflict")
                    return self.public(json.loads(old[1]))
            account = self.account()
            with self.source(request.generation) as path:
                cost = self.quote(request.generation, path)
                with self.db() as db:
                    jobs = [json.loads(row[0]) for row in db.execute("SELECT record FROM jobs")]
                    if any(job["stage"] == "submission_unknown" for job in jobs):
                        raise GatewayError("account_reconciliation_required")
                    if (len(jobs) >= self.submission_limit or
                            sum(job["quoted_credits"] for job in jobs) + cost > self.credit_limit):
                        raise GatewayError("account_budget_exceeded")
                    active = [job for job in jobs if job["status"] not in {"completed", "failed"}]
                    if len(active) >= self.max_concurrent:
                        raise GatewayError("account_concurrency_exceeded", 429)
                    # Balance may already include active deductions; conservative double
                    # reservation is deliberate when provider accounting is not final.
                    if sum(job["quoted_credits"] for job in active) + cost > account["credits"]:
                        raise GatewayError("insufficient_credits")
                    record = {**body, "status": "running", "stage": "submitting", "remote_id": None,
                              "quoted_credits": cost, "created_at": now(), "updated_at": now(), "error": None}
                    db.execute("INSERT INTO jobs VALUES(?,?,?)", (request.request_id, digest, json.dumps(record)))
                try:
                    response = self.cli.run(request.generation.arguments("create", path), timeout=90)
                    # Fixed CLI 1.1.26 returns a list of job UUIDs for create.
                    if not isinstance(response, list) or len(response) != 1 or not isinstance(response[0], str):
                        raise ValueError
                    identity = str(uuid.UUID(response[0]))
                    if identity != response[0]:
                        raise ValueError
                    record.update(remote_id=identity, stage="remote_pending")
                except (ProviderError, ValueError, TypeError):
                    record.update(stage="submission_unknown", error="submission_unknown")
                record["updated_at"] = now()
                with self.db() as db:
                    self.save(db, record)
                return self.public(record)

    def status(self, request_id):
        with self._lock:
            record = self.load(request_id)
            if record["stage"] == "submission_unknown" or record["status"] in {"completed", "failed"}:
                return self.public(record)
            self.check_binding()
            data = self.cli.run(["generate", "get", record["remote_id"]])
            if not isinstance(data, dict) or data.get("id") != record["remote_id"] or data.get("job_type") != record["generation"]["model"]:
                raise ProviderError("provider_protocol_error")
            status = data.get("status")
            if status == "completed":
                if self.media is None:
                    raise GatewayError("media_collector_unavailable", 503)
                record.update(stage="collecting", updated_at=now())
                with self.db() as db:
                    self.save(db, record)
                result = self.media.collect(request_id, data.get("result_url"), record["generation"])
                record.update(status="completed", stage="completed", media=result, error=None)
            elif status in {"failed", "cancelled", "canceled", "error"}:
                record.update(status="failed", stage="failed", error="provider_generation_failed")
            elif status in {"pending", "queued", "running", "processing", "in_progress", "created"}:
                record.update(stage="remote_pending")
            else:
                raise ProviderError("provider_protocol_error")
            record["updated_at"] = now()
            with self.db() as db:
                self.save(db, record)
            return self.public(record)

    @staticmethod
    def public(record):
        keys = ("request_id", "binding_id", "status", "stage", "remote_id", "quoted_credits",
                "created_at", "updated_at", "error", "media")
        return {key: record[key] for key in keys if key in record}

    def capabilities(self):
        return {"binding_id": self.binding, "models": MODELS,
                "generation_enabled": self.credit_limit > 0 and self.submission_limit > 0,
                "readiness": "not_checked"}
