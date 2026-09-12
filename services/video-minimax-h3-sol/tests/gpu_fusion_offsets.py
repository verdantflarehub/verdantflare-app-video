"""Opt-in GPU regression. Usage: python gpu_fusion_offsets.py FUSIONS_PY ROWS.

Run in an isolated process on a GPU with at least 8 GiB free. With FFN width
14336, 74000 rows is below the signed int32 element limit and 75000 exceeds it.
The former passes upstream; the latter must pass with the int64 offset patch.
"""
import importlib.util,sys,torch
spec=importlib.util.spec_from_file_location('fusion_probe',sys.argv[1]);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
torch.cuda.set_device(0)
rows=int(sys.argv[2]);width=14336
with torch.inference_mode():
 x=torch.ones((rows,2*width),device='cuda',dtype=torch.bfloat16)
 y=m.fused_swiglu(x);torch.cuda.synchronize()
 expected=torch.tensor(1/(1+__import__('math').exp(-1)),dtype=torch.bfloat16).item()
 for start in range(0,rows,1024):assert torch.all(y[start:start+1024]==expected).item()
 print({'rows':rows,'input_elements':x.numel(),'verified':True})
