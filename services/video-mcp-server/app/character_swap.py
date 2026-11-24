"""Pure character-swap validation shared by preflight and submission."""
from contextlib import closing
import os
import re

from .artifacts import ArtifactNotFound

TURBO_VARIANTS = ("fl2v_turbo_8step", "pika_h3_flex_turbo_8step_hd_ref2va")
IDENTIFIERS = {
    "project_id": r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}",
    "idempotency_key": r"video_task_[0-9a-f]{32}",
    "source_artifact_id": r"art_[0-9a-f]{32}",
    "character_image_artifact_id": r"art_[0-9a-f]{32}",
    "source_sha256": r"[0-9a-f]{64}",
    "character_image_sha256": r"[0-9a-f]{64}",
}
OPTIONS = {
    "character_lora_version": ("v1.5", "v1"),
    "turbo_variant": TURBO_VARIANTS,
    "resolution_preset": ("768p", "480p"),
    "seed_mode": ("fixed",),
}


def validate_swap_request(payload: dict) -> dict:
    required = set(IDENTIFIERS) | {"prompt"}
    allowed = required | set(OPTIONS) | {"seed", "lora_strength"}
    if not isinstance(payload, dict) or required - payload.keys() or payload.keys() - allowed:
        raise ValueError("invalid_request_fields")
    request = payload.copy()
    for key, pattern in IDENTIFIERS.items():
        value = request[key]
        if not isinstance(value, str) or not re.fullmatch(pattern, value):
            raise ValueError(f"invalid_{key}")
    prompt = request["prompt"]
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16384:
        raise ValueError("invalid_prompt")
    if any(int(n) != 1 for n in re.findall(r"<(?:Picture|Image|Video)\s+(\d+)>", prompt, re.I)):
        raise ValueError("multi_character_single_pass_unsupported")
    for key, choices in OPTIONS.items():
        if request.setdefault(key, choices[0]) not in choices:
            raise ValueError(f"unsupported_{key}")
    seed = request.setdefault("seed", 7)
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("invalid_seed")
    strength = request.setdefault("lora_strength", 1.0)
    if type(strength) not in (int, float) or not 0.5 <= strength <= 1.2:
        raise ValueError("invalid_lora_strength")
    return request



def prepare_request(store, project_id, source_artifact_id, parameters):
    source = store.get(source_artifact_id, project_id)
    image = store.get(parameters["character_image_artifact_id"], project_id)
    if not source.media_type.startswith("video/"):
        raise ValueError("invalid_source_media")
    if image.media_type not in {"image/png", "image/jpeg", "image/webp"}:
        raise ValueError("invalid_character_image_media")
    for role, artifact in (("source", source), ("character_image", image)):
        try:
            with store.content_path(artifact).open("rb") as stream:
                size = os.fstat(stream.fileno()).st_size
        except (OSError, ArtifactNotFound) as exc:
            raise ValueError(role + "_not_found") from exc
        if size != artifact.size:
            raise ValueError(role + "_integrity_failed")
    # The receiver's idempotency key is the generated MCP task ID.
    validated = validate_swap_request({**parameters, "project_id": project_id,
        "source_artifact_id": source_artifact_id, "source_sha256": source.sha256,
        "character_image_sha256": image.sha256, "idempotency_key": "video_task_" + "0" * 32})
    del validated["idempotency_key"]
    return {**validated, "service": "character_swap", "model": "h3-character-swap-lora"}

class EngineJournal:
    """Persist individual Station acquire requests and engine replica generations."""

    def __init__(self, path):
        self.path = path

    def connect(self):
        import sqlite3
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA busy_timeout=30000")
        db.row_factory = sqlite3.Row
        db.execute("""CREATE TABLE IF NOT EXISTS engine_scale_state (
            service TEXT PRIMARY KEY, generation INTEGER NOT NULL, last_active REAL NOT NULL,
            woken_pod_uid TEXT, bound_pod_uid TEXT, dispatched_pod_uid TEXT)""")
        db.execute("""CREATE TABLE IF NOT EXISTS pending_lease_acquires (
            task_id TEXT, attempt_id TEXT, stage_id TEXT, service TEXT NOT NULL,
            request_json TEXT NOT NULL, created_at REAL NOT NULL,
            PRIMARY KEY(task_id, attempt_id, stage_id))""")
        db.execute("""CREATE TABLE IF NOT EXISTS recovered_engine_generations (
            service TEXT PRIMARY KEY, generation INTEGER NOT NULL)""")
        return db

    def stage_acquire(self, task_id, attempt_id, service, request):
        import json
        import time
        with closing(self.connect()) as db, db:
            cursor = db.execute("INSERT OR IGNORE INTO pending_lease_acquires VALUES(?,?,?,?,?,?)",
                (task_id, attempt_id, service, service, json.dumps(request, sort_keys=True), time.time()))
            row = db.execute("SELECT request_json FROM pending_lease_acquires WHERE task_id=? AND attempt_id=? AND stage_id=?",
                             (task_id, attempt_id, service)).fetchone()
            return json.loads(row[0]), cursor.rowcount == 1

    def confirm_acquire(self, task_id, attempt_id, service):
        with closing(self.connect()) as db, db:
            db.execute("DELETE FROM pending_lease_acquires WHERE task_id=? AND attempt_id=? AND stage_id=?",
                       (task_id, attempt_id, service))

    def pending(self, service):
        with closing(self.connect()) as db:
            return [dict(row) for row in db.execute(
                "SELECT task_id, attempt_id FROM pending_lease_acquires WHERE service=?", (service,))]

    def recover_ready_generation(self, service, uid, now):
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT generation, woken_pod_uid FROM engine_scale_state WHERE service=?",
                (service,),
            ).fetchone()
            if row is None:
                db.execute("INSERT INTO engine_scale_state VALUES(?,1,?,?,NULL,NULL)", (service, now, uid))
                db.execute("INSERT OR REPLACE INTO recovered_engine_generations VALUES(?,1)", (service,))
            elif row["woken_pod_uid"] != uid:
                next_gen = int(row["generation"]) + 1
                db.execute(
                    "UPDATE engine_scale_state SET generation=?, last_active=?, woken_pod_uid=?, "
                    "bound_pod_uid=NULL, dispatched_pod_uid=NULL WHERE service=?",
                    (next_gen, now, uid, service),
                )
                db.execute("INSERT OR REPLACE INTO recovered_engine_generations VALUES(?,?)", (service, next_gen))

    def generation(self, service):
        with closing(self.connect()) as db:
            row = db.execute("SELECT * FROM engine_scale_state WHERE service=?", (service,)).fetchone()
            return dict(row) if row else None

    def wake(self, gpu, service, now):
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            replicas = gpu.engine_replicas(service)
            if replicas == 0:
                db.execute("""INSERT INTO engine_scale_state VALUES(?,1,?,NULL,NULL,NULL)
                    ON CONFLICT(service) DO UPDATE SET generation=generation+1,last_active=excluded.last_active,
                    woken_pod_uid=NULL,bound_pod_uid=NULL,dispatched_pod_uid=NULL""", (service, now))
                gpu.scale_engine(service, 1)
            elif replicas == 1:
                db.execute("UPDATE engine_scale_state SET last_active=? WHERE service=?", (now,service))
            else:
                raise ValueError("unexpected_engine_replica_count")

    def observed(self, service, phase, uid):
        if phase not in {"woken_pod_uid", "bound_pod_uid", "dispatched_pod_uid"} or not uid:
            raise ValueError("invalid_engine_identity")
        with closing(self.connect()) as db, db:
            predecessor = {"woken_pod_uid": None, "bound_pod_uid": "woken_pod_uid",
                           "dispatched_pod_uid": "bound_pod_uid"}[phase]
            sql = (f"UPDATE engine_scale_state SET {phase}=? WHERE service=? AND "
                   f"({phase} IS NULL OR {phase}=?)")
            params = [uid, service, uid]
            if predecessor:
                sql += f" AND {predecessor}=?"
                params.append(uid)
            cur = db.execute(sql, params)
            if not cur.rowcount:
                raise ValueError("engine_generation_requires_reconciliation")
