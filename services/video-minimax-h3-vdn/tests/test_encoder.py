"""Numerical MLP equivalence and conditioner/pipeline device boundary contracts."""
import contextlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from vdn_encoder import (chunked_mlp, encoder_config, install_device_boundary, install_chunking,
                         chunked_rmsnorm, install_qk_norm_chunking, selected_qwen_prompt_embeds)
try:
    import torch
except ImportError:
    torch = None


class DeviceTests(unittest.TestCase):
    def test_config_rejects_invalid_capacity_and_device(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(encoder_config(), ('cuda:1', 512))
        for env in ({'VDN_ENCODER_DEVICE': 'cuda:2'}, {'VDN_ENCODER_MLP_CHUNK': '0'},
                    {'VDN_ENCODER_MLP_CHUNK': '4097'}, {'VDN_ENCODER_MLP_CHUNK': 'abc'}):
            with patch.dict(os.environ, env, clear=True), self.assertRaises(ValueError):
                encoder_config()

    def test_boundary_moves_conditioner_only_and_preserves_unmarked_pipelines(self):
        class Pipeline:
            def __init__(self, components): self.components = components
            @property
            def _execution_device(self): return 'original-device'
        result = Mock()
        observed = []
        def original(text_encoder, processor, token_ids, vision_inputs=None,
                     text_encoder_layer=50, device=None, dtype=None):
            observed.append((text_encoder, processor, token_ids, vision_inputs,
                             text_encoder_layer, device, dtype))
            return result
        encoders = SimpleNamespace(get_qwen3vl_prompt_embeds=original)
        cuda_context = Mock(side_effect=lambda device: contextlib.nullcontext())
        fake_torch = SimpleNamespace(device=lambda x: x, cuda=SimpleNamespace(device=cuda_context))
        fake_modules = {'torch': fake_torch, 'diffusers': SimpleNamespace(ModularPipeline=Pipeline),
                        'diffusers.modular_pipelines.minimax_h3': SimpleNamespace(encoders=encoders)}
        with patch.dict(sys.modules, fake_modules), patch("vdn_encoder.selected_qwen_prompt_embeds", original):
            install_device_boundary()
            wrapper = encoders.get_qwen3vl_prompt_embeds
            install_device_boundary()
            self.assertIs(encoders.get_qwen3vl_prompt_embeds, wrapper)
            encoder = SimpleNamespace(_vdn_encoder_device='cuda:1', _vdn_generation_device='cuda:0')
            refs = {'pixel_values': object()}
            wrapper(encoder, 'processor', [1, 2, 3], refs, device='cuda:0', dtype='bf16')
            self.assertEqual(observed[-1][5], 'cuda:1')
            self.assertIs(observed[-1][3], refs)
            result.to.assert_called_once_with(device='cuda:0')
            cuda_context.assert_called_once_with('cuda:1')
            self.assertEqual(Pipeline({'text_encoder': encoder})._execution_device, 'cuda:0')
            self.assertEqual(Pipeline({})._execution_device, 'original-device')
            self.assertIs(wrapper(object(), 'p', [3], device='cuda:0'), result)
            self.assertEqual(observed[-1][5], 'cuda:0')


@unittest.skipIf(torch is None, 'PyTorch numerical tests run in image build or CPU torch environment')
class NumericalTests(unittest.TestCase):
    def test_pinned_qwen_class_accepts_chunking_and_rejects_double_install(self):
        try:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextMLP
            from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
        except ImportError:
            self.skipTest('Pinned Transformers integration runs in image build')
        cfg = Qwen3VLTextConfig(hidden_size=32, intermediate_size=96, num_hidden_layers=2,
                               num_attention_heads=4, num_key_value_heads=2)
        modules = [Qwen3VLTextMLP(cfg), Qwen3VLTextMLP(cfg)]
        encoder = SimpleNamespace(modules=lambda: iter(modules), config=SimpleNamespace(text_config=cfg))
        x = torch.randn(1, 19, 32)
        with torch.inference_mode():
            expected = [m(x) for m in modules]
            self.assertEqual(install_chunking(encoder, 8), 2)
            for m, value in zip(modules, expected):
                torch.testing.assert_close(m(x), value)
        with self.assertRaises(RuntimeError): install_chunking(encoder, 8)

    def test_chunking_preserves_full_mlp_and_bounds_projection_inputs(self):
        torch.manual_seed(7)
        for dtype in (torch.float32, torch.bfloat16):
            gate = torch.nn.Linear(32, 96, bias=False).to(dtype)
            up = torch.nn.Linear(32, 96, bias=False).to(dtype)
            down = torch.nn.Linear(96, 32, bias=False).to(dtype)
            mlp = SimpleNamespace(gate_proj=gate, up_proj=up, down_proj=down, act_fn=torch.nn.functional.silu)
            for length in (0, 1, 7, 8, 9, 25):
                with self.subTest(dtype=dtype, length=length), torch.inference_mode():
                    x = torch.randn(2, 32, length, dtype=dtype).transpose(1, 2)
                    expected = down(mlp.act_fn(gate(x)) * up(x))
                    seen = []
                    hook = gate.register_forward_pre_hook(lambda module, args: seen.append(args[0].shape[-2]))
                    try: actual = chunked_mlp(mlp, x, 8)
                    finally: hook.remove()
                    torch.testing.assert_close(actual, expected, rtol=1e-2 if dtype==torch.bfloat16 else 1e-5,
                                               atol=2e-3 if dtype==torch.bfloat16 else 1e-6)
                    self.assertLessEqual(max(seen), 8)
                    self.assertEqual(sum(seen), length)
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertEqual(actual.dtype, expected.dtype)


@unittest.skipIf(torch is None, 'Pinned PyTorch required')
class SelectedEncoderTests(unittest.TestCase):
    def setUp(self):
        try:
            from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
        except ImportError:
            self.skipTest('Pinned Transformers integration runs in image build')
        torch.manual_seed(17)
        torch.set_num_threads(2)
        cfg = Qwen3VLConfig(
            text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=8, pad_token_id=0, rope_parameters={
                                 'rope_type': 'default', 'rope_theta': 10000.0,
                                 'mrope_section': [1, 1, 2]}),
            vision_config=dict(depth=2, hidden_size=32, intermediate_size=64,
                               num_heads=4, patch_size=2, temporal_patch_size=2,
                               spatial_merge_size=2, out_hidden_size=32,
                               num_position_embeddings=16, deepstack_visual_indexes=[0, 1]),
            image_token_id=61, video_token_id=62, vision_start_token_id=60,
            vision_end_token_id=59)
        self.encoder = Qwen3VLForConditionalGeneration(cfg).eval()
        self.encoder.config._attn_implementation = 'eager'
        self.encoder.model.language_model.config._attn_implementation = 'eager'
        self.encoder.model.visual.config._attn_implementation = 'eager'
        self.processor = SimpleNamespace(create_mm_token_type_ids=lambda batches: [
            [1 if t == 61 else 2 if t == 62 else 0 for t in batch] for batch in batches])

    def test_selected_states_match_pinned_upstream_with_images_video_and_deepstack(self):
        from diffusers.modular_pipelines.minimax_h3.encoders import get_qwen3vl_prompt_embeds
        cases = [([1, 2, 3, 4], {}),
                 ([1, 60, 61, 59, 3], {'pixel_values': torch.randn(4, 24),
                                      'image_grid_thw': torch.tensor([[1, 2, 2]])}),
                 ([1, 60, 62, 59, 3], {'pixel_values_videos': torch.randn(4, 24),
                                      'video_grid_thw': torch.tensor([[1, 2, 2]])})]
        for dtype in (torch.float32, torch.bfloat16):
            self.encoder.to(dtype=dtype)
            for tokens, vision in cases:
                for index in (0, 1, 2, 3):
                    with self.subTest(dtype=dtype, layer=index, media=list(vision)), torch.inference_mode():
                        expected = get_qwen3vl_prompt_embeds(self.encoder, self.processor, tokens,
                            vision, text_encoder_layer=index, device='cpu', dtype=dtype)
                        with patch.object(self.encoder.model, 'forward', wraps=self.encoder.model.forward) as call:
                            actual = selected_qwen_prompt_embeds(self.encoder, self.processor, tokens,
                                vision, text_encoder_layer=index, device='cpu', dtype=dtype)
                        self.assertIs(call.call_args.kwargs['output_hidden_states'], False)
                        self.assertIs(call.call_args.kwargs['use_cache'], False)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_hooks_removed_after_forward_failure_and_invalid_final_layer(self):
        layers = self.encoder.model.language_model.layers
        counts = [len(m._forward_hooks) for m in layers]
        with patch.object(layers[2], 'forward', side_effect=RuntimeError('injected failure')):
            with self.assertRaisesRegex(RuntimeError, 'injected failure'), torch.inference_mode():
                selected_qwen_prompt_embeds(self.encoder, self.processor, [1, 2, 3],
                                           text_encoder_layer=1, device='cpu', dtype=torch.float32)
        self.assertEqual([len(m._forward_hooks) for m in layers], counts)
        for bad in (-1, 4, True):
            with self.assertRaises(ValueError):
                selected_qwen_prompt_embeds(self.encoder, self.processor, [1], text_encoder_layer=bad)

    def test_qk_norm_chunking_matches_original_and_preserves_dtype(self):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm
        for dtype in (torch.float32, torch.bfloat16):
            norm = Qwen3VLTextRMSNorm(8).to(dtype=dtype)
            for length in (0, 1, 7, 8, 9, 25):
                x = torch.randn(2, 4, length, 8, dtype=dtype).transpose(1, 2)
                seen = []
                def original(part):
                    seen.append(part.shape[1])
                    return norm(part)
                with self.subTest(dtype=dtype, length=length), torch.inference_mode():
                    expected = norm(x)
                    actual = chunked_rmsnorm(original, x, 8)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(actual.dtype, expected.dtype)
                    self.assertLessEqual(max(seen), 8)
                    self.assertEqual(sum(seen), length)
        self.assertEqual(install_qk_norm_chunking(self.encoder, 2), 8)
        with self.assertRaises(RuntimeError): install_qk_norm_chunking(self.encoder, 2)

    def test_full_embedding_equivalence_with_both_chunkers_installed(self):
        from diffusers.modular_pipelines.minimax_h3.encoders import get_qwen3vl_prompt_embeds
        self.encoder.to(dtype=torch.bfloat16)
        tokens = [1, 60, 61, 59, 3, 4, 5, 6, 7]
        vision = {'pixel_values':torch.randn(4,24), 'image_grid_thw':torch.tensor([[1,2,2]])}
        with torch.inference_mode():
            expected = get_qwen3vl_prompt_embeds(self.encoder, self.processor, tokens,
                vision, text_encoder_layer=3, device='cpu', dtype=torch.bfloat16)
            install_chunking(self.encoder, 2)
            install_qk_norm_chunking(self.encoder, 2)
            actual = selected_qwen_prompt_embeds(self.encoder, self.processor, tokens,
                vision, text_encoder_layer=3, device='cpu', dtype=torch.bfloat16)
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-3)


if __name__ == '__main__': unittest.main()
