"""Real pinned DiT modules: head/token grouping equivalence and block offload."""
import os
os.environ['TORCHINDUCTOR_CACHE_DIR']='/tmp/vdn-dit-probe-inductor'
os.environ['TRITON_CACHE_DIR']='/tmp/vdn-dit-probe-triton'
import copy, importlib.util, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from vdn_memory import install_frame_statistics, install_linear_readout
from vdn_dit_memory import install_dit_memory, grouped_attention, sliced_projection
from resident_worker import Engine
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention, MiniMaxH3TransformerBlock
spec=importlib.util.spec_from_file_location('dit_probe','/models/VDN-H3-Ref2VA/stage-dmd-step-250/diffusers/modeling_vdn_h3.py')
u=importlib.util.module_from_spec(spec);sys.modules[spec.name]=u;spec.loader.exec_module(u)
torch.cuda.set_device(0);torch.manual_seed(19)
install_frame_statistics(u);install_linear_readout(u);Engine._install_chunked_decomposed_attention(u)
def check(a,b,label,tolerance=0.01):
    relative=((a.float()-b.float()).norm()/a.float().norm().clamp_min(1e-12)).item()
    assert torch.isfinite(b).all() and relative<tolerance,(label,relative)
    print(json.dumps({'dit_numerical':label,'relative_l2':relative,'status':'passed'}),flush=True)
with torch.inference_mode():
    # Six heads in groups of four and 65-token chunks exercise both remainders.
    for fp8 in (False,True):
        for anchors,conv,text,full in [('none',(),False,False),('both',('k','v'),True,False),('rows',('k','v'),True,False),('columns',(),False,False),('none',(),False,True)]:
            orig=MiniMaxH3Attention(768,6,128)
            a=u.HybridAttention(orig,768,delta_rule='vdn_solve',radius=1,chunk=3,
                 short_conv=conv,enable_text_state=text,anchor_frames=anchors,softmax_impl='decomposed')
            a=a.to('cuda:0',torch.bfloat16).eval();a.inference_mode=a.hybrid_inference_mode=True
            if fp8:u.convert_linear_to_fp8(a,skip_end_blocks=0,min_width=256)
            layout=u.SequenceLayout(seq_len=32+8*16,video_start=32,num_frames=8,tokens_per_frame=16,frame_height=4,frame_width=4,text_len=7)
            a.layout=None if full else layout
            x=torch.randn(layout.seq_len,768,device='cuda:0',dtype=torch.bfloat16)*0.4
            angles=torch.randn(layout.seq_len,128,device='cuda:0',dtype=torch.float32)
            rope=(angles.cos(),angles.sin())
            expected=a._hybrid_forward(x,rope)
            actual=grouped_attention(u,a,x,rope,4,65)
            check(expected,actual,f'attention/fp8={fp8}/anchors={anchors}/conv={conv}/text={text}/full={full}')
            del a,orig,expected,actual,x
    class Trunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            b=MiniMaxH3TransformerBlock(768,6,128,1536,768,1e-5,1e-5)
            b.attn=u.HybridAttention(b.attn,768,delta_rule='vdn_solve',radius=1,chunk=3,
                    short_conv=('k','v'),enable_text_state=True,anchor_frames='both',softmax_impl='decomposed')
            b.attn.layout=layout
            self.transformer_blocks=torch.nn.ModuleList([b])
        def forward(self,x,t,indices,rope):
            return self.transformer_blocks[0](x,t,indices,rope)
    model=Trunk().to('cuda:0',torch.bfloat16).eval();u.set_inference_mode(model,True)
    u.convert_linear_to_fp8(model,skip_end_blocks=0,min_width=256)
    x=torch.randn(1,layout.seq_len,768,device='cuda:0',dtype=torch.bfloat16)*0.4
    t=torch.randn(3,768,device='cuda:0',dtype=torch.bfloat16)*0.1
    indices=torch.arange(layout.seq_len,device='cuda:0')%3
    preserved=x.clone()
    expected=model(x,t,indices,rope)
    from diffusers.hooks import apply_group_offloading
    model.cpu()
    apply_group_offloading(model,onload_device=torch.device('cuda:0'),offload_device=torch.device('cpu'),
                          offload_type='block_level',num_blocks_per_group=1,use_stream=False)
    profile=install_dit_memory(u,model,head_group=4,token_group=65)
    for repeat in range(2):
        actual=model(x,t,indices,rope)
        check(expected,actual,f'full_block_offload/repeat={repeat+1}')
        assert torch.equal(x,preserved), 'block modified its caller input'
        assert all(p.device.type=='cpu' for p in model.parameters())
        assert all(b.device.type=='cpu' for b in model.buffers())
    print(json.dumps({'dit_profile':profile,'status':'passed'}),flush=True)
