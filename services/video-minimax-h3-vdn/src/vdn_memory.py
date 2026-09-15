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


def install_linear_readout(upstream):
    """Upstream inference readout with intermediates released after their last use."""
    import torch

    def readout(self, xv, num_frames, tokens_per_frame, bounds, qkv_raw,
                frame_size=None, text_x=None, text_qkv_raw=None, heads=None,
                beta=None, gate=None, frame_mean=None, text_beta=None):
        head_dim = self.head_dim
        n_heads = self.num_heads if heads is None else heads.stop - heads.start
        backend = self._delta_backend('backend', tokens_per_frame)
        shape_per_frame = (num_frames, tokens_per_frame, n_heads, head_dim)
        query_by_frame, key, value = self._features(
            qkv_raw, num_frames, frame_size, inference=True,
            query_fhsd=(num_frames, tokens_per_frame), heads=heads)
        key_by_frame = key.view(shape_per_frame).permute(0, 2, 1, 3)
        value_by_frame = value.view(shape_per_frame).permute(0, 2, 1, 3)
        if beta is None:
            beta = torch.sigmoid(self.beta_proj(xv))
        beta = beta.view(num_frames, tokens_per_frame, n_heads).permute(0, 2, 1)
        a, b = upstream.frame_statistics(key_by_frame, value_by_frame, beta,
                                         a_fp32=self.a_fp32, inference=True)
        del key, value, key_by_frame, value_by_frame, beta
        if frame_mean is None:
            frame_mean = xv.view(num_frames, tokens_per_frame, -1).mean(dim=1, dtype=torch.float32)
        alpha = self.alpha(frame_mean, heads=heads)
        del frame_mean
        text_state = self._text_state(text_x, text_qkv_raw, heads=heads, text_beta=text_beta)
        prefix_states, suffix_states = upstream._run_scans_inference(
            backend, alpha, a, b, text_state=text_state)
        del a, b
        if gate is None:
            gate = self.output_gate(xv)
        linear_state = upstream.gather_linear_state(
            prefix_states, suffix_states, alpha, bounds, bridge=self.bridge,
            text_state=text_state, inference=True, out_dtype=gate.dtype)
        del prefix_states, suffix_states, alpha, text_state
        result = torch.matmul(query_by_frame, linear_state.transpose(-1, -2))
        del query_by_frame, linear_state
        return upstream.linear_epilogue(result, self.norm.weight, gate, self.norm.eps,
                                         inference=True, fhsd=True)

    upstream.BidirectionalLinearBranch._readout_inference = readout
