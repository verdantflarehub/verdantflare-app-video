"""Ref2VA must receive the precision option on its actual component name."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import tempfile
import unittest


class PrecisionLoaderTests(unittest.TestCase):
    def test_fp8_is_scoped_to_either_transformer_component(self):
        source = Path(__file__).parents[1] / 'src/vendor/infer_diffusers.py'
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_pipeline')
        for name in ('transformer', 'transformer_ref'):
            for enabled in (False, True):
                with self.subTest(component=name, fp8=enabled), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    (root/'dit').mkdir(); (root/'text_encoder').mkdir()
                    specs = {n: SimpleNamespace(subfolder=n, load=Mock(return_value=object()))
                             for n in (name, 'text_encoder')}
                    pipe = Mock(pretrained_component_names=list(specs))
                    pipe.get_component_spec.side_effect = specs.__getitem__
                    namespace = dict(ModularPipeline=Mock(from_pretrained=Mock(return_value=pipe)),
                                     torch=SimpleNamespace(bfloat16='bf16', device=lambda x: x), offload=Mock())
                    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
                    namespace['load_pipeline'](SimpleNamespace(models=temp, transformer='dit', fp8=enabled,
                                                               device='cuda:0', offload_dit=True), 'ref2va')
                    self.assertEqual(specs[name].load.call_args.kwargs.get('fp8', False), enabled)
                    self.assertNotIn('fp8', specs['text_encoder'].load.call_args.kwargs)
