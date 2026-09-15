"""Bound temporary storage for upstream per-frame attention statistics."""


def install_frame_statistics(upstream, frames_per_group=4):
    if frames_per_group < 1:
        raise ValueError('frames_per_group must be positive')
    original = upstream.frame_statistics

    def frame_statistics(kf, vf, beta, a_fp32=True, inference=False):
        frames = kf.shape[0]
        if not inference or frames <= frames_per_group:
            return original(kf, vf, beta, a_fp32=a_fp32, inference=inference)
        import torch
        if torch.is_grad_enabled():
            raise RuntimeError('grouped inference statistics require disabled gradients')
        result_a = result_b = None
        for start in range(0, frames, frames_per_group):
            stop = min(start + frames_per_group, frames)
            a, b = original(kf[start:stop], vf[start:stop], beta[start:stop],
                            a_fp32=a_fp32, inference=True)
            if result_a is None:
                result_a = a.new_empty((frames, *a.shape[1:]))
                result_b = b.new_empty((frames, *b.shape[1:]))
            result_a[start:stop].copy_(a)
            result_b[start:stop].copy_(b)
            del a, b
        return result_a, result_b

    upstream.frame_statistics = frame_statistics
