"""Bounded temporal windows with overlap blending and hard-cut isolation."""
import numpy as np


def windows(frames, size, overlap, cut):
    if type(size) is not int or size < 9 or (size - 1) % 4 or overlap != 4:
        raise ValueError('window must be 4n+1 >= 9; overlap must be 4')
    buffer, start = [], 0
    for frame in frames:
        if buffer and cut(buffer[-1], frame):
            yield start, buffer, True
            start += len(buffer)
            buffer = []
        elif len(buffer) == size:
            yield start, buffer, False
            start += size - overlap
            buffer = buffer[-overlap:]
        buffer.append(frame)
    if buffer:
        yield start, buffer, True


def restore_windows(engine, frames, req, policy, cut, report):
    size, overlap = policy['window_frames'], policy['overlap_frames']
    pending = None
    count = 0
    for start, inputs, last in windows(frames, size, overlap, cut):
        # Absolute source position gives every window a stable seed across retries.
        params = req.model_copy(update={'seed': (req.seed + start) % 4294967296})
        restored = engine.restore_window(np.stack(inputs), params)
        if restored.shape != (len(inputs), req.target_height, req.target_width, 3) or restored.dtype != np.uint8:
            raise ValueError('invalid_restoration_window')
        report['chunks'] += 1
        report['max_chunk_frames'] = max(report['max_chunk_frames'], len(inputs))
        if pending is not None:
            # Blend two estimates of the SAME original frames, never different timestamps.
            for i in range(overlap):
                alpha = (i + 1) / (overlap + 1)
                restored[i] = np.rint(pending[i].astype(np.float32) * (1 - alpha)
                                      + restored[i].astype(np.float32) * alpha).astype(np.uint8)
        if last:
            pending = None
            output = restored
        else:
            pending = restored[-overlap:].copy()
            output = restored[:-overlap]
        for frame in output:
            count += 1
            yield frame
    report['output_frames'] = count
