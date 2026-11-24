"""Exercise the complete worker function with explicit CPU-only dependency peers.

No torch/Comfy/CUDA implementation, model weights, encoder or GPU is executed.
This tests wiring and restoration only, never visual quality or native encoding.
"""
from contextlib import nullcontext
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
from h3_fasth3.worker import _run_fasth3_nvfp4
from h3_fasth3.errors import RuntimeErrorCode


class Tensor:
    def __init__(self, *shape): self.shape = shape; self.ndim = len(shape)
    def permute(self, *order): return Tensor(*(self.shape[i] for i in order))
    def contiguous(self): return self
    def detach(self): return self
    def to(self, *args): return self


def module(**values):
    result = ModuleType('test_peer')
    result.__dict__.update(values)
    return result


class WorkerImageGuidesTests(unittest.TestCase):
    def fixture(self, *, bad_canvas=False, bad_latent=False):
        self.images = [Tensor(1, 941, 1672, 3), Tensor(1, 1024, 768, 3), Tensor(1, 1024, 768, 3)]
        if bad_canvas: self.images[1] = Tensor(1, 512, 384, 3)
        self.guides = [{'image_index': 1, 'frame_index': 0}, {'image_index': 2, 'frame_index': 110},
                       {'image_index': 2, 'frame_index': 123}]
        self.model_load = Mock(return_value=[NS(model=NS(diffusion_model=NS(final_layer=NS(video_out=NS(weight=Tensor(96))))))])
        self.encodes = []
        def encode(image):
            self.encodes.append(image)
            return Tensor(1, 24, 2 if bad_latent else 1, 64, 48)
        self.original_encode = encode
        self.vae = NS(encode=encode, patcher=NS())
        self.audio_vae = NS(encode=Mock(), patcher=NS())
        self.clip = NS(encode_from_tokens_scheduled=Mock())
        self.clip_encode = self.clip.encode_from_tokens_scheduled
        self.audio_encode = self.audio_vae.encode
        self.refs = [{'kind': 'image', 'latent': Tensor(1, 24, 1, 32, 24)}, {'kind': 'video', 'latent': object()}]
        self.audio_key = {'resolved_frame_index': 0, 'audio_latent': object()}
        self.conditioning = [[object(), {'minimax_refs': self.refs, 'minimax_keyframes': [self.audio_key]}],
                             [object(), {'metadata': 'retained'}]]
        self.ref_execute = Mock(return_value=(self.conditioning, {'samples': 'unmodified-reference-latent'}))
        self.guider = Mock(return_value=['guider'])
        self.callback = lambda *args: None
        preview = module(prepare_callback=lambda *a, **kw: self.callback)
        self.original_callback = preview.prepare_callback
        def sample(*args):
            callback = preview.prepare_callback()
            for index in range(8): callback(index, None, None, 8)
            return ['sampled']
        custom = module(**{name: NS(execute=Mock(return_value=[name])) for name in
                           ('KSamplerSelect', 'BasicScheduler', 'RandomNoise')},
                        BasicGuider=NS(execute=self.guider), SamplerCustomAdvanced=NS(execute=Mock(side_effect=sample)))
        h3 = module(MiniMaxH3SigmaShift=NS(execute=Mock(side_effect=lambda model,*a:[model])),
                    MiniMaxH3ReferenceToVideo=NS(execute=self.ref_execute))
        management = module(unload_all_models=Mock(), soft_empty_cache=Mock(), load_models_gpu=Mock(),
                            VRAMState=NS(NORMAL_VRAM='normal'))
        cuda = NS(is_available=lambda: False, synchronize=Mock(), empty_cache=Mock(), reset_peak_memory_stats=Mock(),
                  max_memory_reserved=lambda: 0, max_memory_allocated=lambda: 0)
        torch = module(cuda=cuda, device=lambda name: name, no_grad=nullcontext,
            nn=NS(functional=NS(interpolate=lambda tensor, *, size, **kw: Tensor(tensor.shape[0], tensor.shape[1], *size))))
        nodes = module(UNETLoader=lambda:NS(load_unet=self.model_load), CLIPLoader=lambda:NS(load_clip=lambda *a,**kw:[self.clip]),
                       VAELoader=lambda:NS(load_vae=lambda name:[self.audio_vae if 'audio' in name else self.vae]),
                       VAEDecode=lambda:NS(decode=Mock(return_value=[Tensor(124, 1024, 768, 3)])))
        self.mux = Mock(return_value={'frames': 124, 'width': 768, 'height': 1024})
        media = module(load_image=lambda path:self.images[int(path.name)],
                       load_video=lambda path:(Tensor(124, 1024, 768, 3), None, 24),
                       align_video_reference=lambda frames,audio,length:(frames,audio),
                       load_audio=Mock(), mux_mp4=self.mux, sha256=lambda path:'a'*64)
        self.modules = {'torch': torch, 'comfy': module(model_management=management), 'comfy.model_management': management,
            'comfy_extras': module(nodes_audio=module(VAEDecodeAudio=NS(execute=Mock(return_value=[{}]))),
                                   nodes_custom_sampler=custom, nodes_minimax_h3=h3),
            'comfy_extras.nodes_custom_sampler': custom, 'comfy_extras.nodes_minimax_h3': h3,
            'folder_paths': module(add_model_folder_path=Mock()), 'latent_preview': preview, 'nodes': nodes,
            'h3_fasth3.media': media,
            'h3_fasth3.vsa_patch': module(FastH3VSAPatch=lambda:NS(patch=lambda model,**kw:[model]))}
        self.modules['comfy_extras.nodes_audio'] = self.modules['comfy_extras'].nodes_audio
        self.request = {'prompt': 'real photo, schematic motion', 'seconds': 5, 'target': {'aspect_ratio': '3:4'},
                        'image_guidance': 'reference', 'image_guides': self.guides}
        self.files = {'images': [Path(str(i)) for i in range(3)], 'videos': [Path('motion.mp4')]}
        self.stages = []
        return preview

    def run_worker(self):
        with patch.dict(sys.modules, self.modules), patch.dict('os.environ', {'FASTH3_REFERENCE_VIDEO_SCALE':'0.5'}):
            return _run_fasth3_nvfp4(self.request, self.files, Path('not-written.mp4'), self.stages.append,
                gpu_uuid='test-only', free_vram_mib_before=0, sm120_info={})

    def test_actual_worker_passes_exact_guide_latents_and_all_references_to_sampler(self):
        preview = self.fixture()
        result = self.run_worker()
        args = self.ref_execute.call_args.args
        self.assertEqual(args[2:5], (768, 1024, 124))
        self.assertEqual(list(args[8].values()), self.images)
        self.assertEqual(args[9]['ref_video_0'].shape, (124, 512, 384, 3))
        self.assertEqual(self.encodes, self.images[1:])
        condition = self.guider.call_args.args[1]
        self.assertIs(condition[0][1]['minimax_refs'], self.refs)
        self.assertIs(condition[0][1]['minimax_keyframes'][0], self.audio_key)
        keys = condition[0][1]['minimax_keyframes'][1:]
        self.assertEqual([k['resolved_frame_index'] for k in keys], [0, 110, 123])
        self.assertIs(keys[1]['latent'], keys[2]['latent'])
        self.assertIsNot(keys[0]['latent'], self.refs[0]['latent'])
        self.assertEqual(len(condition[1][1]['minimax_keyframes']), 3)
        self.assertEqual(len(self.conditioning[0][1]['minimax_keyframes']), 1)
        self.assertIs(self.vae.encode, self.original_encode)
        self.assertIs(self.audio_vae.encode, self.audio_encode)
        self.assertIs(self.clip.encode_from_tokens_scheduled, self.clip_encode)
        self.assertIs(preview.prepare_callback, self.original_callback)
        self.assertEqual(self.stages, ['warming', 'generating', 'saving'])
        self.assertEqual(result['runtime_metrics']['sequence_frames'], 124)
        self.mux.assert_called_once()

    def test_bad_canvas_stops_before_model_loading(self):
        self.fixture(bad_canvas=True)
        with self.assertRaisesRegex(RuntimeErrorCode, 'canvas_mismatch'): self.run_worker()
        self.model_load.assert_not_called()
        self.assertFalse(self.encodes)
        self.assertFalse(self.stages)

    def test_encoding_error_restores_original_encoders_and_never_samples(self):
        self.fixture(bad_latent=True)
        with self.assertRaisesRegex(RuntimeErrorCode, 'latent_shape_mismatch'): self.run_worker()
        self.assertIs(self.vae.encode, self.original_encode)
        self.assertIs(self.audio_vae.encode, self.audio_encode)
        self.assertIs(self.clip.encode_from_tokens_scheduled, self.clip_encode)
        self.guider.assert_not_called()
        self.mux.assert_not_called()

    def test_old_default_does_not_add_guides_or_encode_extra_images(self):
        self.fixture()
        self.request.pop('image_guides')
        self.run_worker()
        self.assertFalse(self.encodes)
        self.assertIs(self.guider.call_args.args[1], self.conditioning)

    def test_old_first_frame_path_uses_original_encoded_reference(self):
        self.fixture()
        self.request.pop('image_guides')
        self.request['image_guidance'] = 'first_frame'
        self.images[0] = Tensor(1, 1024, 768, 3)
        self.refs[0]['latent'] = Tensor(1, 24, 1, 64, 48)
        self.conditioning.pop()
        self.run_worker()
        self.assertFalse(self.encodes)
        keys = self.guider.call_args.args[1][0][1]['minimax_keyframes']
        self.assertIs(keys[0], self.audio_key)
        self.assertEqual(keys[1]['resolved_frame_index'], 0)
        self.assertIs(keys[1]['latent'], self.refs[0]['latent'])
