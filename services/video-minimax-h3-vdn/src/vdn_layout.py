"""Bind Diffusers' target-only geometry to the upstream VDN attention modules."""
import sys
import json


def attach_layout_bridge(transformer):
    upstream = sys.modules[type(transformer).__module__]
    hybrids = list(upstream.iter_hybrids(transformer))
    if not hybrids:
        raise ValueError('Ref2VA transformer has no VDN hybrid blocks')
    transformer._vdn_layout_calls = 0
    transformer._vdn_linear_calls = 0
    transformer._vdn_hybrid_blocks = len(hybrids)

    def count_linear(_module, _args):
        transformer._vdn_linear_calls += 1

    for hybrid in hybrids:
        if hybrid.teacher_mode or not hybrid.linear_attention_enabled:
            raise ValueError('VDN attention branch disabled')
        hybrid.linear_attention.register_forward_pre_hook(count_linear)

    def prepare(state, fields):
        frames = state.num_latent_frames
        height, width = state.latent_height, state.latent_width
        count = state.num_condition_video_rows
        patch_t, patch_h, patch_w = transformer.config.patch_size
        if any(type(value) is not int or value <= 0 for value in (frames,height,width)):
            raise ValueError('missing resolved target geometry')
        if type(count) is not int or count < 0 or patch_t != 1 or height % patch_h or width % patch_w:
            raise ValueError('invalid VDN target geometry')
        target = fields['video_indices'][count:]
        grid = (height//patch_h, width//patch_w)
        layout = upstream.layout_from_indices(target, frames, grid[0]*grid[1],
                    fields['position_ids'].shape[0], frame_size=grid, text_indices=fields['text_indices'])
        if layout.video_end != layout.seq_len:
            raise ValueError('generated video must be the final packed segment')
        upstream.set_layout(transformer, layout)
        transformer._vdn_layout_calls += 1
        print(json.dumps({"stage": "dit_layout", "step": transformer._vdn_layout_calls,
                          "sequence_tokens": layout.seq_len, "target_frames": frames,
                          "tokens_per_frame": layout.tokens_per_frame}), flush=True)

    transformer._prepare_vdn_layout = prepare
    return len(hybrids)
