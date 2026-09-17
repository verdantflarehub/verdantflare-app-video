"""Verify NFE+1 sigma grids with the pinned scheduler on both GPUs."""
import json
import torch
from diffusers import MiniMaxH3Scheduler

with torch.inference_mode():
    for device in ('cuda:0', 'cuda:1'):
        for nfe in (4, 8):
            scheduler = MiniMaxH3Scheduler()
            scheduler.set_timesteps(nfe + 1, device=device)
            assert len(scheduler.timesteps) == nfe
            assert len(scheduler.sigmas) == nfe + 1 and scheduler.sigmas[-1].item() == 0
            sample = torch.ones((2, 4, 8), device=device)
            calls = 0
            for t in scheduler.timesteps:
                sample = scheduler.step(torch.zeros_like(sample), t, sample).prev_sample
                calls += 1
            assert calls == nfe and torch.isfinite(sample).all()
            torch.cuda.synchronize(device)
            print(json.dumps(dict(sampling_nfe=nfe, calls=calls, device=device, status='passed')), flush=True)
