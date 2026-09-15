import os
os.environ['TORCHINDUCTOR_CACHE_DIR']='/tmp/vdn-readout-probe-inductor'
os.environ['TRITON_CACHE_DIR']='/tmp/vdn-readout-probe-triton'
import importlib.util,sys,json,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from vdn_memory import install_frame_statistics, install_linear_readout
spec=importlib.util.spec_from_file_location('readout_probe','/models/VDN-H3-Ref2VA/stage-dmd-step-250/diffusers/modeling_vdn_h3.py')
u=importlib.util.module_from_spec(spec);sys.modules[spec.name]=u;spec.loader.exec_module(u)
torch.cuda.set_device(1);torch.manual_seed(7)
install_frame_statistics(u)
original=u.BidirectionalLinearBranch._readout_inference
with torch.inference_mode():
 for conv in ((),('k','v')):
  branch=u.BidirectionalLinearBranch(512,4,128,delta_rule='vdn_solve',short_conv=conv).to(device='cuda:1',dtype=torch.bfloat16)
  frames,tokens=16,16
  x=torch.randn(frames*tokens,512,device='cuda:1',dtype=torch.bfloat16)
  qkv=tuple(torch.randn(frames*tokens,4,128,device='cuda:1',dtype=torch.bfloat16) for _ in range(3))
  tx=torch.randn(5,512,device='cuda:1',dtype=torch.bfloat16)
  tq=tuple(torch.randn(5,4,128,device='cuda:1',dtype=torch.bfloat16) for _ in range(3))
  bounds=u.window_bounds(frames,1,chunk=5)
  for anchors in (False,True):
   for text in (False,True):
    for sliced in (False,True):
     kwargs=dict(frame_size=(4,4),skip_ends=anchors,text_x=tx if text else None,text_qkv_raw=tq if text else None,inference=True)
     xx=x;qs=qkv
     if sliced:
      h=slice(1,3);xx=None;qs=tuple(t[:,h] for t in qkv)
      kwargs.update(heads=h,beta=torch.sigmoid(branch.beta_proj(x))[:,h],gate=branch.output_gate(x)[:,h],frame_mean=x.view(frames,tokens,-1).mean(1,dtype=torch.float32),text_beta=torch.sigmoid(branch.beta_proj(tx))[:,h] if text else None,text_qkv_raw=tuple(t[:,h] for t in tq) if text else None)
     u.BidirectionalLinearBranch._readout_inference=original
     a=branch(xx,frames,tokens,bounds,qs,**kwargs)
     install_linear_readout(u)
     b=branch(xx,frames,tokens,bounds,qs,**kwargs)
     relative=((a.float()-b.float()).norm()/a.float().norm().clamp_min(1e-12)).item()
     assert torch.isfinite(a).all() and torch.isfinite(b).all() and relative<1e-6,relative
     print(json.dumps({'conv':conv,'anchors':anchors,'text':text,'heads_sliced':sliced,'relative_l2':relative,'status':'passed'}),flush=True)
