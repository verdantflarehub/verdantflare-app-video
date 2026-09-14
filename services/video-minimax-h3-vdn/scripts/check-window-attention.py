import importlib.util,json,sys
import torch
path='/models/VDN-H3-Ref2VA/stage-dmd-step-250/diffusers/modeling_vdn_h3.py'
spec=importlib.util.spec_from_file_location('vdn_window_probe',path)
u=importlib.util.module_from_spec(spec);sys.modules[spec.name]=u;spec.loader.exec_module(u)
torch.manual_seed(7)
with torch.inference_mode():
 layout=u.SequenceLayout(seq_len=256+16*128,video_start=256,num_frames=16,tokens_per_frame=128,frame_height=8,frame_width=16,text_len=32)
 bounds=u.window_bounds(16,2,chunk=4)
 q,k,v=[torch.randn(layout.seq_len,4,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
 mask=u.build_window_block_mask(layout,bounds,v.device,anchor_frames='none')
 a=u.window_softmax_decomposed(q,k,v,layout,bounds,128**-.5)
 b=u.window_softmax_flex(q,k,v,mask,128**-.5,inference=True)
 relative=(a.float()-b.float()).norm()/a.float().norm()
 assert torch.isfinite(b).all() and relative<0.01,relative.item()
 print(json.dumps({'relative_l2':relative.item(),'status':'passed'}))
