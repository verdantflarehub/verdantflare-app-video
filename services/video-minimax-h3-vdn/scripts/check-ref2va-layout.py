"""CPU geometry regression using the pinned Diffusers and OpenVDN packers."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3Ref2VAPrepareLayoutStep
from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3Ref2VALoopDenoiser

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vdn_layout import attach_layout_bridge

spec=importlib.util.spec_from_file_location('upstream_layout','/opt/upstream/vdn/src/models/sequence_layout.py')
upstream=importlib.util.module_from_spec(spec);sys.modules[spec.name]=upstream;spec.loader.exec_module(upstream)
layout_from_indices=upstream.layout_from_indices


def iter_hybrids(model):return iter(model.hybrids)


def set_layout(model,layout):
    for hybrid in model.hybrids:hybrid.layout=layout


class GeometryModel:
    def __init__(self):
        self.config=SimpleNamespace(patch_size=(1,2,2))
        self.hybrids=[SimpleNamespace(teacher_mode=False,linear_attention_enabled=True,linear_attention=torch.nn.Linear(2,2))]


def main():
    refs=[SimpleNamespace(kind='image'),SimpleNamespace(kind='audio'),SimpleNamespace(kind='video',has_audio=True)]
    packed=MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
        text_token_tags=torch.ones(7,dtype=torch.long),references=refs,
        condition_latents=[torch.zeros(1,16,1,8,12),torch.zeros(1,16,3,12,8)],
        audio_condition_latents=[torch.zeros(8,64),torch.zeros(6,64)],num_latent_frames=22,
        latent_height=8,latent_width=12,num_audio_latents=5,patch_size=(1,2,2),
        audio_channels=2,audio_tag=2,video_tag=0)
    positions,tags,video,audio,text,condition_video,condition_audio=packed
    state=SimpleNamespace(num_latent_frames=22,latent_height=8,latent_width=12,num_condition_video_rows=condition_video)
    fields=dict(position_ids=positions,video_indices=video,text_indices=text)
    model=GeometryModel();assert attach_layout_bridge(model)==1
    model._prepare_vdn_layout(state,fields)
    layout=model.hybrids[0].layout
    assert layout.video_end==positions.shape[0]
    assert layout.video_start==positions.shape[0]-22*4*6
    assert layout.text_range==(0,7) and layout.frame_size==(4,6)
    assert layout.video_start>7+condition_video+condition_audio
    assert model._vdn_layout_calls==1
    model.hybrids[0].linear_attention(torch.ones(1,2))
    assert model._vdn_linear_calls==1
    state.num_condition_video_rows=0
    try:model._prepare_vdn_layout(state,fields)
    except ValueError:pass
    else:raise AssertionError('reference video rows were treated as generated video')
    inputs={item.name for item in MiniMaxH3Ref2VALoopDenoiser().inputs}
    assert {'num_latent_frames','latent_height','latent_width','num_condition_video_rows'}<=inputs
    print('Ref2VA layout verified: references remain global, VDN targets only generated frames')


if __name__=='__main__':main()
