import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(ROOT))
from vdn_io import validate_request, validate_model, sha256, inspect_media


def request():
    return dict(schema_version=1, task='t2va', prompt='A river at dawn', seed=7, frames=124, steps=8, input_artifacts=[])


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_supported_modes_require_exact_keyframes_and_hashes(self):
        p = self.root / 'image.png'
        p.write_bytes(b'image')
        for mode, roles in [('t2va', []), ('i2va', ['first']), ('l2va', ['last']), ('fl2va', ['first', 'last'])]:
            r = request()
            r.update(task=mode, input_artifacts=[dict(role=x, path=p.name, sha256=sha256(p)) for x in roles])
            self.assertEqual(set(validate_request(r, self.root)), set(roles))
        r['input_artifacts'][0]['sha256'] = 'bad'
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            validate_request(r, self.root)

    def test_rejects_incompatible_or_silently_snapped_requests(self):
        for key, value in [('task', 'ref2va'), ('frames', 125), ('seed', True), ('steps', 4), ('prompt', '')]:
            r = request(); r[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_request(r, self.root)
        r = request(); r['task'] = 'fl2va'
        with self.assertRaisesRegex(ValueError, 'keyframes'):
            validate_request(r, self.root)

    def test_escape_is_rejected(self):
        r = request()
        r.update(task='i2va', input_artifacts=[dict(role='first', path='../outside', sha256='')])
        with self.assertRaises(ValueError):
            validate_request(r, self.root)

    def model(self):
        root = self.root / 'model'; root.mkdir()
        (root / 'model_index.json').write_text('{}')
        (root / 'stage-dmd-step-250/diffusers').mkdir(parents=True)
        (root / 'stage-dmd-step-250/diffusers/weights.safetensors').write_bytes(b'fixture only')
        lock = dict(repository='OpenVDN/vdn-minimax-h3', revision='a'*40, steps=8,
                    files={str(p.relative_to(root)): sha256(p) for p in root.rglob('*') if p.is_file()})
        return root, lock

    def test_model_nfe_hash_and_unlisted_code(self):
        root, lock = self.model()
        validate_model(root, lock, 8)
        with self.assertRaisesRegex(ValueError, 'NFE'):
            validate_model(root, lock, 50)
        (root / 'injected.py').write_text('pass')
        with self.assertRaisesRegex(ValueError, 'unlisted'):
            validate_model(root, lock, 8)
        (root / 'injected.py').unlink()
        (root / 'stage-dmd-step-250/diffusers/weights.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            validate_model(root, lock, 8)

    def test_failed_inference_persists_failure_and_never_reuses_output(self):
        root, lock = self.model()
        manifest = self.root / 'request.json'; manifest.write_text(json.dumps(request()))
        model_lock = self.root / 'lock.json'; model_lock.write_text(json.dumps(lock))
        spec = importlib.util.spec_from_file_location('runner', ROOT / 'run-inference.py')
        runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
        output = self.root / 'run'
        argv = ['runner', '--manifest', str(manifest), '--inputs', str(self.root), '--models', str(root), '--model-lock', str(model_lock), '--output', str(output)]
        with patch.object(sys, 'argv', argv), patch.object(runner, 'run_upstream', side_effect=RuntimeError('load failed')):
            with self.assertRaises(RuntimeError): runner.main()
            self.assertEqual(json.loads((output / 'record.json').read_text())['status'], 'failed')
            self.assertFalse((output / 'video.mp4').exists())
            with self.assertRaises(FileExistsError): runner.main()

    def test_help_without_gpu_import(self):
        subprocess.run([sys.executable, str(ROOT / 'run-inference.py'), '--help'], check=True, capture_output=True)

    def test_real_media_validation(self):
        output = self.root / 'media.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=size=64x64:rate=24', '-f', 'lavfi', '-i', 'anullsrc=r=32000:cl=stereo', '-t', str(124/24), '-c:v', 'libx264', '-c:a', 'aac', str(output)], check=True)
        self.assertIn('streams', inspect_media(output, 124))
        with self.assertRaises(RuntimeError): inspect_media(output, 141)


if __name__ == '__main__':
    unittest.main()
