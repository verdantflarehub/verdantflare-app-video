"""Exercise the published FP8 Linear on the assigned GPUs; not a quality gate."""
import importlib.util
import json
import sys
import torch

spec = importlib.util.spec_from_file_location('fp8_check', '/models/VDN-H3-Ref2VA/stage-dmd-step-250/diffusers/modeling_vdn_h3.py')
u = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = u
spec.loader.exec_module(u)
with torch.inference_mode():
    for index in range(2):
        torch.cuda.set_device(index)
        torch.manual_seed(7)
        for rows, width, output in ((2048, 1024, 4096), (1024, 7168, 7168)):
            linear = torch.nn.Linear(width, output, bias=True, device=f'cuda:{index}', dtype=torch.bfloat16)
            x = torch.randn(rows, width, device=f'cuda:{index}', dtype=torch.bfloat16)
            baseline = linear(x)
            quantized = u.Fp8Linear(linear)
            result = quantized(x)
            relative = ((result.float()-baseline.float()).norm()/baseline.float().norm()).item()
            assert torch.isfinite(result).all() and relative < 0.06, relative
            assert quantized.weight_fp8.dtype == torch.float8_e4m3fn
            quantized.cpu()
            assert all(b.device.type == 'cpu' for b in quantized.buffers())
            torch.cuda.synchronize(index)
            print(json.dumps(dict(gpu=index, shape=[rows,width,output], relative_l2=relative,
                                  per_tensor=u.per_tensor_gemm(), fp8='passed')), flush=True)
            del linear, x, baseline, quantized, result
