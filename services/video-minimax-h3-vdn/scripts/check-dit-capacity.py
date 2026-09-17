"""Long-sequence, full-width single-block memory check; not a video acceptance."""
import os
os.environ['TORCHINDUCTOR_CACHE_DIR']='/tmp/vdn-dit-capacity-inductor'
os.environ['TRITON_CACHE_DIR']='/tmp/vdn-dit-capacity-triton'
import importlib.util,json,sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vdn_memory import install_frame_statistics,install_linear_readout
from vdn_dit_memory import install_dit_memory
from resident_worker import Engine
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock
spec=importlib.util.spec_from_file_location('dit_capacity','/models/VDN-H3-Ref2VA/stage-dmd-step-250/diffusers/modeling_vdn_h3.py')
u=importlib.util.module_from_spec(spec);sys.modules[spec.name]=u;spec.loader.exec_module(u)
torch.cuda.set_device(0);torch.manual_seed(23)
install_frame_statistics(u);install_linear_readout(u);Engine._install_chunked_decomposed_attention(u)
class Trunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        b=MiniMaxH3TransformerBlock(7168,56,128,14336,512,1e-5,1e-5)
        b.attn=u.HybridAttention(b.attn,7168,delta_rule='vdn_solve',radius=1,chunk=5,
                  short_conv=('k','v'),enable_text_state=True,anchor_frames='both',softmax_impl='decomposed')
        b.attn.layout=u.SequenceLayout(seq_len=262144,video_start=64,num_frames=65,
                  tokens_per_frame=4032,frame_height=63,frame_width=64,text_len=32)
        self.transformer_blocks=torch.nn.ModuleList([b])
    def forward(self,x,t,indices,rope):return self.transformer_blocks[0](x,t,indices,rope)
with torch.inference_mode():
    model=Trunk().to(dtype=torch.bfloat16).eval()
    u.set_inference_mode(model,True);u.convert_linear_to_fp8(model,skip_end_blocks=0)
    model.cuda()
    install_dit_memory(u,model)
    # Keep 4 GiB alive to expose overlap with surrounding pipeline tensors.
    retained=torch.empty(4*1024**3,device='cuda:0',dtype=torch.uint8)
    n=262144
    x=torch.randn(1,n,7168,device='cuda:0',dtype=torch.bfloat16)*0.1
    t=torch.randn(3,512,device='cuda:0',dtype=torch.bfloat16)*0.1
    indices=torch.arange(n,device='cuda:0')%3
    angles=torch.randn(n,128,device='cuda:0',dtype=torch.float32)
    rope=(angles.cos(),angles.sin());del angles
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(0)
    started=time.monotonic()
    print(json.dumps({'dit_capacity':'started','tokens':n,'width':7168,'retained_bytes':retained.numel()}),flush=True)
    y=model(x,t,indices,rope);torch.cuda.synchronize()
    # A chunked check avoids a large FP32 verification tensor becoming the peak.
    assert all(torch.isfinite(y[:,i:i+512]).all() for i in range(0,n,512))
    result={'dit_capacity':'passed','tokens':n,'width':7168,'retained_bytes':retained.numel(),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(0),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(0),'seconds':time.monotonic()-started}
    print(json.dumps(result),flush=True)
