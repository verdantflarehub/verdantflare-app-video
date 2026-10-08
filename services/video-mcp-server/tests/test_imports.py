import base64
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from app.artifacts import ArtifactStore
from app.imports import ImportStore, MAX_CHUNK_BYTES


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.artifacts = ArtifactStore(Path(self.temp.name))
        self.validator = Mock(return_value={"width": 1024, "height": 768})
        self.store = ImportStore(self.artifacts, self.validator)
        self.content = b"chunk-contract-fixture" * 200000
        self.args = dict(project_id="project-test", idempotency_key="input-1", filename="ref.png",
                         size=len(self.content), sha256=hashlib.sha256(self.content).hexdigest(), purpose="scene")

    def tearDown(self):
        self.temp.cleanup()

    def send(self, identity, offset, block):
        return self.store.chunk(project_id="project-test", import_id=identity, offset=offset,
            content_base64=base64.b64encode(block).decode(), sha256=hashlib.sha256(block).hexdigest())

    def upload(self):
        identity = self.store.prepare(**self.args)["import_id"]
        for offset in range(0, len(self.content), MAX_CHUNK_BYTES):
            self.send(identity, offset, self.content[offset:offset + MAX_CHUNK_BYTES])
        return identity

    def test_over_three_mib_resumes_and_commits_one_immutable_artifact(self):
        self.assertGreater(len(self.content), 3 * 1024 * 1024)
        identity = self.store.prepare(**self.args)["import_id"]
        first = self.content[:MAX_CHUNK_BYTES]
        self.send(identity, 0, first)
        self.store = ImportStore(self.artifacts, self.validator)
        self.assertEqual(self.store.prepare(**self.args)["offset"], len(first))
        self.assertEqual(self.send(identity, 0, first)["offset"], len(first))
        for offset in range(len(first), len(self.content), MAX_CHUNK_BYTES):
            self.send(identity, offset, self.content[offset:offset + MAX_CHUNK_BYTES])
        result = self.store.commit(project_id="project-test", import_id=identity)
        replay = self.store.commit(project_id="project-test", import_id=identity)
        self.assertEqual(result["artifact"], replay["artifact"])
        self.assertEqual(result["artifact"]["sha256"], self.args["sha256"])
        self.assertEqual(result["project_archive_status"], "not_archived")
        self.validator.assert_called_once()

    def test_conflicting_key_chunk_offset_and_project_are_rejected(self):
        identity = self.store.prepare(**self.args)["import_id"]
        with self.assertRaisesRegex(ValueError, "idempotency_conflict"):
            self.store.prepare(**{**self.args, "sha256": "f" * 64})
        with self.assertRaisesRegex(ValueError, "offset_or_status"):
            self.send(identity, 1, b"a")
        self.send(identity, 0, b"chunk")
        with self.assertRaisesRegex(ValueError, "repeated_chunk"):
            self.send(identity, 0, b"other")
        with self.assertRaisesRegex(ValueError, "not_found"):
            self.store.status(project_id="another-project", import_id=identity)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.store.commit(project_id="project-test", import_id=identity)

    def test_lost_commit_response_after_publish_reuses_identity(self):
        identity = self.upload()
        with patch.object(self.store, "_public", side_effect=RuntimeError("lost response")):
            with self.assertRaises(RuntimeError):
                self.store.commit(project_id="project-test", import_id=identity)
        published = list(self.artifacts.artifacts_root.glob("art_*"))
        self.assertEqual(len(published), 1)
        self.store = ImportStore(self.artifacts, self.validator)
        result = self.store.commit(project_id="project-test", import_id=identity)
        self.assertEqual(result["artifact"]["artifact_id"], published[0].name)
        self.assertEqual(len(list(self.artifacts.artifacts_root.glob("art_*"))), 1)

    def test_uncommitted_chunk_tail_is_removed_on_retry(self):
        identity = self.store.prepare(**self.args)["import_id"]
        (self.store.root / (identity + ".part")).write_bytes(b"uncommitted garbage")
        self.send(identity, 0, self.content[:5])
        self.assertEqual((self.store.root / (identity + ".part")).read_bytes(), self.content[:5])

    def test_corrupt_or_undecodable_content_is_not_published(self):
        identity = self.upload()
        path = self.store.root / (identity + ".part")
        path.write_bytes(b"x" * len(self.content))
        with self.assertRaisesRegex(ValueError, "final_size_or_hash"):
            self.store.commit(project_id="project-test", import_id=identity)
        path.write_bytes(self.content)
        self.validator.side_effect = ValueError("media_invalid")
        with self.assertRaisesRegex(ValueError, "media_invalid"):
            self.store.commit(project_id="project-test", import_id=identity)
        self.assertFalse(list(self.artifacts.artifacts_root.glob("art_*")))


if __name__ == "__main__":
    unittest.main()
