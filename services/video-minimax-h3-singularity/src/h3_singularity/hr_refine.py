"""Pure tensor helpers for the H3 high resolution tile refinement profile."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


def frames_for_tokens(tokens: int) -> int:
    return sum(FRAME_PER_TOKEN[index % 5] for index in range(tokens))


def clip_tokens(frames: int) -> int:
    if frames < 5:
        return max(1, int(frames))
    return (int(frames) - 5) // 17 * 5 + 2


def audio_range(frame_start: int, frame_end: int, audio_tokens: int) -> tuple[int, int]:
    # H3 audio/video timing uses 5/3 audio tokens per video frame.
    start = max(0, round(frame_start * 5.0 / 3.0))
    end = min(audio_tokens, round(frame_end * 5.0 / 3.0))
    return start, max(start, end)


def temporal_windows(tokens: int, chunk_frames: int = 73,
                    overlap_frames: int = 22) -> list[tuple[int, int, int, int]]:
    """Return (token_start, token_end, frame_start, frame_end) H3 windows."""
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    chunk = min(tokens, clip_tokens(chunk_frames))
    overlap = min(max(0, chunk - 1), clip_tokens(overlap_frames))
    if chunk >= tokens:
        return [(0, tokens, 0, frames_for_tokens(tokens))]
    hop = max(1, chunk - overlap)
    result: list[tuple[int, int, int, int]] = []
    start = 0
    while start < tokens:
        end = min(tokens, start + chunk)
        if end == tokens and end - start < chunk:
            start = max(0, tokens - chunk)
            end = tokens
        result.append((start, end, frames_for_tokens(start), frames_for_tokens(end)))
        if end == tokens:
            break
        start += hop
    # De-duplicate the final window after the backward adjustment.
    return list(dict.fromkeys(result))


def _grid_1d(size: int, tile: int, overlap: int) -> list[tuple[int, int, int]]:
    if size <= tile:
        return [(0, size, 0)]
    if tile <= overlap:
        raise ValueError("tile must be larger than overlap")
    hop = tile - overlap
    starts = list(range(0, max(1, size - tile + 1), hop))
    last = size - tile
    if starts[-1] != last:
        starts.append(last)
    result = []
    previous_end = 0
    for start in starts:
        end = min(size, start + tile)
        result.append((start, end, max(0, previous_end - start)))
        previous_end = end
    return result


def spatial_tiles(height: int, width: int, tile_h: int = 42,
                  tile_w: int = 24, overlap: int = 8) -> list[tuple[int, int, int, int, int, int]]:
    """Return row/column tile bounds and their top/left overlap."""
    rows = _grid_1d(height, tile_h, overlap)
    cols = _grid_1d(width, tile_w, overlap)
    return [(r0, r1, c0, c1, rov, cov)
            for r0, r1, rov in rows for c0, c1, cov in cols]


def h3_resize_video(video: torch.Tensor, target_height: int,
                    target_width: int, mean: torch.Tensor,
                    std: torch.Tensor) -> torch.Tensor:
    """Resize only H/W in normalized H3 latent space."""
    if video.ndim != 5 or video.shape[1] != 24:
        raise ValueError(f"invalid H3 video latent shape: {tuple(video.shape)}")
    if target_height < 2 or target_width < 2:
        raise ValueError("target latent dimensions are too small")
    if video.shape[-2:] == (target_height, target_width):
        return video.contiguous()
    source = (video.to(dtype=torch.float32) - mean.to(video.device, torch.float32)) / std.to(video.device, torch.float32)
    resized = F.interpolate(source, size=(video.shape[2], target_height, target_width),
                            mode="trilinear", align_corners=False)
    return (resized * std.to(video.device, torch.float32) + mean.to(video.device, torch.float32)).to(video.dtype).contiguous()


def tile_blend_mask(height: int, width: int, top_overlap: int,
                    left_overlap: int, device: torch.device,
                    dtype: torch.dtype) -> torch.Tensor:
    """Create a [1,1,1,H,W] cross-fade mask for an output tile."""
    mask = torch.ones((height, width), device=device, dtype=torch.float32)
    if top_overlap:
        mask[:top_overlap] *= torch.linspace(0.0, 1.0, top_overlap, device=device)[:, None]
    if left_overlap:
        mask[:, :left_overlap] *= torch.linspace(0.0, 1.0, left_overlap, device=device)[None, :]
    return mask.to(dtype=dtype).view(1, 1, 1, height, width)
