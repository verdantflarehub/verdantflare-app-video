import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import calibrate


class CalibrationFailureTest(unittest.TestCase):
    def test_killed_worker_retains_evidence_without_capacity_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.mp4'
            source.write_bytes(b'fixture')
            output = root / 'trial'
            args = SimpleNamespace(source=source, output=output, max_frames=9, repeats=3,
                reserve_mib=1024, trial_timeout=60, target_width=32, target_height=32, seed=666)
            def killed_worker(*positional, **keywords):
                pending = json.loads((output / 'run.json').read_text())
                self.assertEqual(pending['active_trial']['frames'], 9)
                return SimpleNamespace(returncode=-9)
            with patch.object(calibrate, 'probe', return_value={'width': 16, 'height': 16, 'frames': 20}), \
                 patch.object(calibrate.subprocess, 'run', side_effect=killed_worker):
                with self.assertRaisesRegex(calibrate.CapacityError, 'non_capacity_trial_failure'):
                    calibrate.calibrate(args)
            report = json.loads((output / 'run.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['measurements'][0]['status'], 'trial_process_failed')
            self.assertEqual(report['measurements'][0]['returncode'], -9)
            self.assertNotIn('active_trial', report)
            self.assertFalse((output / 'capacity.json').exists())


if __name__ == '__main__':
    unittest.main()
