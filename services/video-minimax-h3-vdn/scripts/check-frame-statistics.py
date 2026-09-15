import os
os.environ['TORCHINDUCTOR_CACHE_DIR']='/tmp/vdn-frame-probe-inductor'
os.environ['TRITON_CACHE_DIR']='/tmp/vdn-frame-probe-triton'
import importlib.util,sys,json,torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from vdn_memory import install_frame_statistics
spec=importlib.util.spec_from_file_location('frame_probe','/models/VDN-H3-Ref2VA/stage-dmd-step-250/diffusers/modeling_vdn_h3.py')
u=importlib.util.module_from_spec(spec);sys.modules[spec.name]=u;spec.loader.exec_module(u)
torch.cuda.set_device(1);torch.manual_seed(7)
original=u.frame_statistics
install_frame_statistics(u)
with torch.inference_mode():
 for frames in (1,4,5,13):
  for strided in (False,True):
   k=torch.randn(frames,16,4,128,device='cuda:1',dtype=torch.bfloat16).transpose(1,2)
   v=torch.randn_like(k);beta=torch.rand(frames,4,16,device='cuda:1',dtype=torch.bfloat16)
   if not strided:k=k.contiguous();v=v.contiguous()
   for a_fp32 in (False,True):
    a,b=original(k,v,beta,a_fp32=a_fp32,inference=True)
    ga,gb=u.frame_statistics(k,v,beta,a_fp32=a_fp32,inference=True)
    ra=((a-ga).norm()/a.norm()).item();rb=((b-gb).norm()/b.norm()).item()
    assert torch.isfinite(ga).all() and torch.isfinite(gb).all()
    assert max(ra,rb)<0.0001,(ra,rb)
    print(json.dumps({'frames':frames,'strided':strided,'a_fp32':a_fp32,'relative_a':ra,'relative_b':rb,'status':'passed'}),flush=True)
