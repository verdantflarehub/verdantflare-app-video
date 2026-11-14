from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch
import torch
from h3_latent_upscaler.engine import configure_comfy
from h3_latent_upscaler.resources import ResourceError


class ResidencyPolicyTests(unittest.TestCase):
    def test_full_loading_required_and_oom_cannot_trigger_fallback(self):
        comfy = ModuleType('comfy')
        cli = ModuleType('comfy.cli_args')
        cli.args = SimpleNamespace()
        mm = ModuleType('comfy.model_management')
        mm.load_models_gpu = Mock()
        original = mm.load_models_gpu
        comfy.model_management = mm
        with patch.dict(sys.modules, {'comfy': comfy, 'comfy.cli_args': cli, 'comfy.model_management': mm}):
            configure_comfy()
            model = SimpleNamespace(loaded_size=lambda: 10, model_size=lambda: 20)
            with self.assertRaisesRegex(ResourceError, 'cpu_offload_forbidden'):
                mm.load_models_gpu([model], force_full_load=False)
            self.assertTrue(original.call_args.kwargs['force_full_load'])
            self.assertTrue(cli.args.disable_dynamic_vram)
            error = torch.OutOfMemoryError('fixture OOM')
            with self.assertRaises(torch.OutOfMemoryError) as caught:
                mm.raise_non_oom(error)
            self.assertIs(caught.exception, error)
