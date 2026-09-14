"""Validate the process boundary; inference belongs to the copied upstream."""
from pathlib import Path
import sys
import subprocess
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from vdn_io import run_upstream

class UpstreamTests(unittest.TestCase):
    def test_original_cli_receives_explicit_inputs(self):
        for steps, stage in [(8, 'stage-dmd-step-250'), (50, 'stage-b-step-2000')]:
            with patch('subprocess.run') as run:
                run_upstream(Path('/models'), dict(prompt='literal $(text)', steps=steps, frames=124, seed=7),
                             {'first': Path('/inputs/first.png'), 'last': Path('/inputs/last.png')}, Path('/output.mp4'))
            args = run.call_args.args[0]
            self.assertEqual(args[-2:], ['--', 'literal $(text)'])
            self.assertTrue(args[1].endswith('vendor/infer_diffusers.py'))
            for flag, value in [('--steps', str(steps)), ('--transformer', stage+'/diffusers'), ('--first', '/inputs/first.png'), ('--last', '/inputs/last.png')]:
                self.assertEqual(args[args.index(flag)+1], value)
            self.assertIn('--offload_dit', args)
            self.assertTrue(run.call_args.kwargs['check'])
            self.assertEqual(run.call_args.kwargs['env']['HF_HUB_OFFLINE'], '1')

    def test_upstream_failure_propagates(self):
        with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'upstream')):
            with self.assertRaises(subprocess.CalledProcessError):
                run_upstream(Path('/models'), dict(prompt='test', steps=8, frames=124, seed=7), {}, Path('/out'))
