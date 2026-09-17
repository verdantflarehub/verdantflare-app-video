"""Dual-GPU numerical check of pinned Qwen, selective states and CPU offload.

Uses a small randomly initialized architecture for numerical equivalence only;
this is not the real-media capacity/quality acceptance test.
"""
import copy,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from test_encoder import SelectedEncoderTests
from vdn_encoder import configure_encoder
from diffusers.modular_pipelines.minimax_h3 import encoders
assert torch.cuda.device_count()==2
case=SelectedEncoderTests();case.setUp()
reference=case.encoder.to(dtype=torch.bfloat16)
optimized=copy.deepcopy(reference)
reference.to('cuda:1')
tokens=[1,60,61,59]+[3,4,5]*341
vision={'pixel_values':torch.randn(4,24),'image_grid_thw':torch.tensor([[1,2,2]])}
with torch.inference_mode():
    expected=encoders.get_qwen3vl_prompt_embeds(reference,case.processor,tokens,vision,
        text_encoder_layer=3,device='cuda:1',dtype=torch.bfloat16).cpu()
    del reference
    case.encoder=None
    torch.cuda.empty_cache()
    configure_encoder(SimpleNamespace(text_encoder=optimized),torch.device('cuda:0'))
    for repeat in range(2):
        actual=encoders.get_qwen3vl_prompt_embeds(optimized,case.processor,tokens,vision,
            text_encoder_layer=3,device='cuda:0',dtype=torch.bfloat16)
        assert actual.device==torch.device('cuda:0')
        torch.testing.assert_close(actual.cpu(),expected,rtol=1e-2,atol=2e-3)
        assert all(p.device.type=='cpu' for p in optimized.parameters())
        profile=optimized._vdn_encoder_profile
        assert profile['hidden_states']=='selected_intermediate_only' and profile['qk_norm_modules']==8
        print(json.dumps({'gpu_encoder_equivalence':'passed','repeat':repeat+1,'tokens':len(tokens),'profile':profile}),flush=True)
