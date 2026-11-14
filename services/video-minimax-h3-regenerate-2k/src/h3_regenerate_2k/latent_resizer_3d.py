"""Pure-3D H3 latent upscaler backbone (adapted from Open-H3-Regenerate-2K-IC upscaler_node.py, MIT License).

See LICENSE-upscaler-node.txt in this directory for license terms.
"""

from __future__ import annotations

import gc
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

VAE_DOWNSAMPLE = 16

LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
]

LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
]


def make_norm_tensors(device: torch.device | str, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    return mean, std


def normalization(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels)


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.detach().zero_()
    return module


class AttnBlock3D(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = normalization(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, 1)
        self.k = nn.Conv3d(in_channels, in_channels, 1)
        self.v = nn.Conv3d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        x_norm = self.norm(x)
        q = self.q(x_norm).permute(0, 2, 3, 4, 1).reshape(b, 1, t * h * w, c)
        k = self.k(x_norm).permute(0, 2, 3, 4, 1).reshape(b, 1, t * h * w, c)
        v = self.v(x_norm).permute(0, 2, 3, 4, 1).reshape(b, 1, t * h * w, c)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
        return x + self.proj_out(out)


class ResBlockEmb3D(nn.Module):
    def __init__(self, channels: int, emb_channels: int, dropout: float = 0.0, out_channels: int | None = None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        self.skip = nn.Conv3d(channels, self.out_channels, 1) if self.out_channels != channels else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return self.skip(x) + h


class TemporalConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.norm = normalization(channels)
        self.dwconv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(padding, 0, 0),
            groups=channels,
        )
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        h = F.silu(self.norm(x))
        h = self.dwconv(h)
        h = self.pwconv(h)
        return identity + h


class LatentResizer3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 24,
        in_blocks: int = 12,
        out_blocks: int = 12,
        channels: int = 512,
        dropout: float = 0.1,
        attn: bool = False,
        temporal_every: int = 2,
        temporal_kernel: int = 5,
    ):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            if (b == 1 or b == in_blocks - 1) and attn:
                self.in_blocks.append(AttnBlock3D(channels))
            self.in_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.in_blocks.append(TemporalConv(channels, temporal_kernel))

        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            if (b == 1 or b == out_blocks - 1) and attn:
                self.out_blocks.append(AttnBlock3D(channels))
            self.out_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.out_blocks.append(TemporalConv(channels, temporal_kernel))

        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        scale: float | None = None,
        target_size: tuple[int, int, int] | None = None,
        enable_chunking: bool = True,
    ) -> torch.Tensor:
        if target_size is not None:
            size = target_size
        elif scale is not None:
            size = tuple(int(round(s * scale)) for s in x.shape[-3:])
        else:
            return x

        if size == x.shape[-3:]:
            return x

        b_sz, c_sz, t_sz, _, _ = x.shape
        tk = 0
        for blk in self.in_blocks:
            if isinstance(blk, TemporalConv):
                tk = int(blk.dwconv.weight.shape[2])
                break

        overlap = tk
        chunk = 32
        if not enable_chunking or t_sz <= chunk:
            return self._forward_seg(x, scale, size)

        x_padded = F.pad(x, (0, 0, 0, 0, overlap, overlap), mode="replicate")
        out_full = torch.zeros(b_sz, c_sz, t_sz, size[-2], size[-1], device=x.device, dtype=x.dtype)
        weight_full = torch.zeros(1, 1, t_sz, 1, 1, device=x.device, dtype=x.dtype)

        start = 0
        while start < t_sz:
            seg_start = start
            seg_end = min(t_sz, start + chunk)
            out_start = max(0, seg_start - overlap)
            out_end = min(t_sz, seg_end + overlap)
            lo = max(0, out_start - overlap)
            hi = min(t_sz + 2 * overlap, out_end + overlap)

            seg = x_padded[:, :, lo:hi].contiguous()
            seg_size = (hi - lo, size[-2], size[-1])
            seg_out = self._forward_seg(seg, scale, seg_size)

            s0 = (out_start + overlap) - lo
            s1 = s0 + (out_end - out_start)
            valid_out = seg_out[:, :, s0:s1]
            n_valid = out_end - out_start

            weight = torch.ones(n_valid, device=x.device, dtype=x.dtype)
            if seg_start > out_start:
                blend_len = seg_start - out_start
                weight[:blend_len] = torch.arange(1, blend_len + 1, device=x.device, dtype=x.dtype) / (blend_len + 1)
            if out_end > seg_end:
                blend_len = out_end - seg_end
                weight[-blend_len:] = torch.arange(blend_len, 0, -1, device=x.device, dtype=x.dtype) / (blend_len + 1)

            out_full[:, :, out_start:out_end] += valid_out * weight.view(1, 1, n_valid, 1, 1)
            weight_full[:, :, out_start:out_end] += weight.view(1, 1, n_valid, 1, 1)

            start += chunk
            del seg, seg_out, valid_out
            if start % (chunk * 4) == 0:
                gc.collect()

        return out_full / weight_full.clamp(min=1e-8)

    def _forward_seg(self, x: torch.Tensor, scale: float | None, size: tuple[int, int, int]) -> torch.Tensor:
        scale_emb = torch.tensor(
            [scale - 1.0 if scale is not None else 0.0],
            dtype=x.dtype,
            device=x.device,
        ).unsqueeze(0)
        emb = self.embed(scale_emb)
        x = self.conv_in(x)
        for blk in self.in_blocks:
            if isinstance(blk, ResBlockEmb3D):
                x = blk(x, emb.expand(x.shape[0], -1))
            else:
                x = blk(x)
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        for blk in self.out_blocks:
            if isinstance(blk, ResBlockEmb3D):
                x = blk(x, emb.expand(x.shape[0], -1))
            else:
                x = blk(x)
        x = F.silu(self.norm_out(x))
        return self.conv_out(x)


def extract_upscaler_sd(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(k.startswith("upscaler.") for k in sd):
        return {k[len("upscaler.") :]: v for k, v in sd.items() if k.startswith("upscaler.")}
    return sd


def detect_arch(sd: dict[str, torch.Tensor]) -> dict:
    cfg = {
        "in_channels": 24,
        "in_blocks": 12,
        "out_blocks": 12,
        "channels": 512,
        "dropout": 0.1,
        "attn": False,
        "temporal_every": 2,
        "temporal_kernel": 5,
    }
    if "conv_in.weight" in sd:
        cfg["in_channels"] = int(sd["conv_in.weight"].shape[1])
        cfg["channels"] = int(sd["conv_in.weight"].shape[0])

    in_ids, out_ids = set(), set()
    temporal_in_indices, temporal_out_indices = set(), set()
    for k in sd:
        m = re.match(r"in_blocks\.(\d+)\.in_layers\.", k)
        if m:
            in_ids.add(int(m.group(1)))
        m = re.match(r"out_blocks\.(\d+)\.in_layers\.", k)
        if m:
            out_ids.add(int(m.group(1)))
        m = re.match(r"in_blocks\.(\d+)\.dwconv\.weight", k)
        if m:
            temporal_in_indices.add(int(m.group(1)))
        m = re.match(r"out_blocks\.(\d+)\.dwconv\.weight", k)
        if m:
            temporal_out_indices.add(int(m.group(1)))

    if in_ids:
        cfg["in_blocks"] = len(in_ids)
    if out_ids:
        cfg["out_blocks"] = len(out_ids)
    if temporal_in_indices or temporal_out_indices:
        cfg["temporal_every"] = 2
        for k in sd:
            if k.endswith("dwconv.weight"):
                cfg["temporal_kernel"] = int(sd[k].shape[2])
                break
    else:
        cfg["temporal_every"] = 0
    cfg["attn"] = False
    return cfg


def load_resizer_3d(checkpoint_path: Path, device: torch.device | str = "cuda:0", dtype: torch.dtype = torch.bfloat16):
    from safetensors.torch import load_file

    raw = load_file(str(checkpoint_path), device="cpu")
    sd = extract_upscaler_sd(raw)
    cfg = detect_arch(sd)
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        resizer = LatentResizer3D(
            in_channels=cfg["in_channels"],
            in_blocks=cfg["in_blocks"],
            out_blocks=cfg["out_blocks"],
            channels=cfg["channels"],
            dropout=cfg["dropout"],
            attn=cfg["attn"],
            temporal_every=cfg["temporal_every"],
            temporal_kernel=cfg["temporal_kernel"],
        )
    finally:
        torch.set_default_dtype(old_dtype)
    resizer.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=True)
    del sd, raw
    resizer.eval().requires_grad_(False).to(device)
    mean, std = make_norm_tensors(device, dtype)
    return resizer, mean, std
