import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from h3_latent_upscaler.resources import SCHEMA, SourceIndex, ResourceError, confined_file, sha256


class ResourcesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.task = 'video_task_' + 'a' * 32
        files = {}
        for role in ('video_latent', 'audio_latent', 'conditions', 'source_video'):
            path = self.root / (role + ('.safetensors' if 'latent' in role else '.json'))
            path.write_bytes(b'fixture; tensor validation belongs to bundle loader')
            files[role] = {'path': path.name, 'size': path.stat().st_size, 'sha256': sha256(path)}
        manifest = {'schema': 'h3-latent-bundle/v1', 'project_id': 'project-a',
                    'source_video_task_id': self.task, 'node': 'node-a',
                    'latent_state': 'clean', 'files': files}
        self.manifest = self.root / 'manifest.json'
        self.manifest.write_text(json.dumps(manifest))
        self.db = self.root / 'index.sqlite'
        with sqlite3.connect(self.db) as db:
            db.execute(SCHEMA)
            db.execute('INSERT INTO h3_sources VALUES (?,?,?,?,?,?)',
                       ('project-a', self.task, 'node-a', 'succeeded', 'manifest.json', sha256(self.manifest)))
        self.index = SourceIndex(self.db, self.root, 'node-a')

    def test_resolve_by_task(self):
        result = self.index.resolve('project-a', self.task)
        self.assertEqual(result['files']['video_latent'], self.root / 'video_latent.safetensors')

    def test_cross_project_is_not_found(self):
        with self.assertRaisesRegex(ResourceError, 'source_not_found'):
            self.index.resolve('project-b', self.task)

    def test_other_node_not_downloaded(self):
        with self.assertRaisesRegex(ResourceError, 'source_node_mismatch'):
            SourceIndex(self.db, self.root, 'node-b').resolve('project-a', self.task)

    def test_modified_latent_rejected(self):
        (self.root / 'video_latent.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ResourceError, 'source_integrity_failed'):
            self.index.resolve('project-a', self.task)

    def test_mp4_only_does_not_trigger_reencoding(self):
        with sqlite3.connect(self.db) as db:
            db.execute('UPDATE h3_sources SET manifest_path=NULL')
        with self.assertRaisesRegex(ResourceError, 'missing_latent_bundle'):
            self.index.resolve('project-a', self.task)

    def test_no_absolute_or_parent_paths(self):
        for path in ('/etc/passwd', '../manifest.json'):
            with self.assertRaises(ResourceError):
                confined_file(self.root, path)

    def test_symlink_cannot_escape(self):
        (self.root / 'escape').symlink_to('/etc/passwd')
        with self.assertRaises(ResourceError):
            confined_file(self.root, 'escape')

    def test_untrusted_task_identifier(self):
        with self.assertRaisesRegex(ResourceError, 'invalid_source_task_id'):
            self.index.resolve('project-a', '../../etc/passwd')

    def test_manifest_identity_must_match(self):
        manifest = json.loads(self.manifest.read_text())
        manifest['project_id'] = 'project-b'
        self.manifest.write_text(json.dumps(manifest))
        with sqlite3.connect(self.db) as db:
            db.execute('UPDATE h3_sources SET manifest_sha256=?', (sha256(self.manifest),))
        with self.assertRaisesRegex(ResourceError, 'incompatible_latent_bundle'):
            self.index.resolve('project-a', self.task)


if __name__ == '__main__':
    unittest.main()
