"""Durable sequential chunk imports; metadata is not a grant to central content."""

import base64
import binascii
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import uuid

from .artifacts import MAX_ARTIFACT_BYTES, ArtifactStore, require_filename, require_project_id


MAX_CHUNK_BYTES = 512 * 1024
IMPORT_ID = re.compile(r"imp_[0-9a-f]{32}")
SHA256 = re.compile(r"[0-9a-f]{64}")
UUID7 = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
FORMATS = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
           ".mp4": "video/mp4", ".wav": "audio/wav", ".mp3": "audio/mpeg"}


class ImportStore:
    def __init__(self, artifacts: ArtifactStore, validate_media):
        self.artifacts = artifacts
        self.validate_media = validate_media
        self.root = artifacts.root / "imports"
        self.root.mkdir(parents=True, exist_ok=True)
        with self._transaction() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS imports (
                id TEXT PRIMARY KEY, project TEXT NOT NULL, idem TEXT NOT NULL,
                metadata TEXT NOT NULL, offset INTEGER NOT NULL DEFAULT 0,
                artifact_id TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'uploading',
                media TEXT, UNIQUE(project, idem))""")

    @contextmanager
    def _transaction(self, *, write=True):
        db = sqlite3.connect(self.root / "imports.sqlite", timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _read(self, db, project_id, import_id):
        project_id = require_project_id(project_id)
        if not isinstance(import_id, str) or not IMPORT_ID.fullmatch(import_id):
            raise ValueError("invalid_import_id")
        row = db.execute("SELECT * FROM imports WHERE id=? AND project=?", (import_id, project_id)).fetchone()
        if row is None:
            raise ValueError("import_not_found")
        return row

    def _public(self, row):
        metadata = json.loads(row["metadata"])
        result = {"import_id": row["id"], "project_id": row["project"], "status": row["status"],
                  "offset": row["offset"], "chunk_max_bytes": MAX_CHUNK_BYTES, **metadata}
        if row["status"] == "committed":
            result["artifact"] = self.artifacts.get(row["artifact_id"], row["project"]).model_dump()
            result["media"] = json.loads(row["media"])
        return result

    def prepare(self, *, project_id, idempotency_key, filename, size, sha256, purpose, source_content_ref=None):
        project_id = require_project_id(project_id)
        filename = require_filename(filename)
        if not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key.strip()) <= 128:
            raise ValueError("invalid_import_idempotency_key")
        if type(size) is not int or not 0 < size <= MAX_ARTIFACT_BYTES:
            raise ValueError("import_size_must_be_1_byte_to_1_GiB")
        if not isinstance(sha256, str) or not SHA256.fullmatch(sha256):
            raise ValueError("invalid_import_sha256")
        if not isinstance(purpose, str) or not 1 <= len(purpose.strip()) <= 1000:
            raise ValueError("import_purpose_required")
        media_type = FORMATS.get(Path(filename).suffix.lower())
        if not media_type:
            raise ValueError("unsupported_import_format")
        if source_content_ref is not None:
            if (not isinstance(source_content_ref, dict)
                or set(source_content_ref) != {"store_id", "artifact_id", "version_id"}
                or any(not isinstance(v, str) or not UUID7.fullmatch(v) for v in source_content_ref.values())):
                raise ValueError("fixed_content_ref_required")
        metadata = {"filename": filename, "size": size, "sha256": sha256, "media_type": media_type,
                    "purpose": purpose.strip(), "source_content_ref": source_content_ref,
                    "source_verification": "caller_supplied" if source_content_ref else "local_bytes",
                    "project_archive_status": "not_archived"}
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        with self._transaction() as db:
            old = db.execute("SELECT * FROM imports WHERE project=? AND idem=?", (project_id, idempotency_key)).fetchone()
            if old:
                if old["metadata"] != encoded:
                    raise ValueError("import_idempotency_conflict")
                return self._public(old)
            identity = "imp_" + uuid.uuid4().hex
            db.execute("INSERT INTO imports(id,project,idem,metadata,artifact_id) VALUES (?,?,?,?,?)",
                       (identity, project_id, idempotency_key, encoded, "art_" + uuid.uuid4().hex))
            return self._public(self._read(db, project_id, identity))

    def status(self, *, project_id, import_id):
        with self._transaction(write=False) as db:
            return self._public(self._read(db, project_id, import_id))

    def chunk(self, *, project_id, import_id, offset, content_base64, sha256):
        if type(offset) is not int or offset < 0:
            raise ValueError("invalid_import_offset")
        if not isinstance(content_base64, str) or not 0 < len(content_base64) <= 4 * ((MAX_CHUNK_BYTES + 2) // 3):
            raise ValueError("import_chunk_too_large_or_empty")
        try:
            content = base64.b64decode(content_base64, validate=True)
        except (ValueError, binascii.Error):
            raise ValueError("invalid_import_base64") from None
        if not content or len(content) > MAX_CHUNK_BYTES or hashlib.sha256(content).hexdigest() != sha256:
            raise ValueError("import_chunk_size_or_hash_mismatch")
        with self._transaction() as db:
            row = self._read(db, project_id, import_id)
            meta = json.loads(row["metadata"])
            if offset + len(content) > meta["size"]:
                raise ValueError("import_chunk_exceeds_size")
            path = self.root / (import_id + ".part")
            if offset < row["offset"]:
                if offset + len(content) > row["offset"]:
                    raise ValueError("import_chunk_overlap")
                with path.open("rb") as stream:
                    stream.seek(offset)
                    if stream.read(len(content)) != content:
                        raise ValueError("import_repeated_chunk_mismatch")
                return self._public(row)
            if offset != row["offset"] or row["status"] != "uploading":
                raise ValueError("import_offset_or_status_conflict")
            if offset and (not path.is_file() or path.stat().st_size < offset):
                raise ValueError("import_committed_bytes_missing")
            with path.open("r+b" if path.exists() else "w+b") as stream:
                # Uncommitted tails can survive a process crash after fsync.
                stream.truncate(offset)
                stream.seek(offset)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            db.execute("UPDATE imports SET offset=? WHERE id=?", (offset + len(content), import_id))
            return self._public(self._read(db, project_id, import_id))

    def commit(self, *, project_id, import_id):
        with self._transaction() as db:
            row = self._read(db, project_id, import_id)
            if row["status"] == "committed":
                return self._public(row)
            meta = json.loads(row["metadata"])
            if row["offset"] != meta["size"]:
                raise ValueError("import_incomplete")
            path = self.root / (import_id + ".part")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if path.stat().st_size != meta["size"] or digest.hexdigest() != meta["sha256"]:
                raise ValueError("import_final_size_or_hash_mismatch")
            media = self.validate_media(path, meta["media_type"])
            # A crash between publishing the artifact and committing SQLite
            # must recover the preassigned identity rather than duplicate it.
            final = self.artifacts.artifacts_root / row["artifact_id"]
            if final.exists():
                artifact = self.artifacts.get(row["artifact_id"], project_id)
                if (artifact.sha256, artifact.size, artifact.filename, artifact.media_type) != (
                    meta["sha256"], meta["size"], meta["filename"], meta["media_type"]):
                    raise ValueError("import_published_artifact_conflict")
            else:
                with path.open("rb") as stream:
                    self.artifacts.create_from_chunks(project_id=project_id, operation="video.import",
                        filename=meta["filename"], media_type=meta["media_type"],
                        chunks=iter(lambda: stream.read(1024 * 1024), b""), expected_sha256=meta["sha256"],
                        artifact_id=row["artifact_id"])
            db.execute("UPDATE imports SET status='committed',media=? WHERE id=?", (json.dumps(media), import_id))
            return self._public(self._read(db, project_id, import_id))
