"""Rebuild Comfy H3 conditioning from retained normalized Diffusers outputs."""
import json
import torch
from safetensors.torch import load_file
from .resources import ResourceError


def load_conditioning(resources):
    tensors = load_file(str(resources['files']['conditions']), device='cpu')
    geometry = json.loads(resources['files']['geometry'].read_text())
    embeddings, tags = tensors['prompt_embeds'], tensors['text_token_tags']
    if embeddings.ndim == 2:
        embeddings = embeddings.unsqueeze(0)
    if embeddings.ndim != 3 or embeddings.shape[0] != 1 or tags.numel() != embeddings.shape[1]:
        raise ResourceError('invalid_prompt_embedding_geometry')
    if not torch.isfinite(embeddings).all():
        raise ResourceError('nonfinite_condition')
    video_names, audio_names = iter(geometry['condition_latents']), iter(geometry['audio_condition_latents'])
    refs = []
    try:
        for reference in geometry['normalized_references']:
            kind = reference['kind']
            if kind not in {'image', 'video', 'audio'}:
                raise ResourceError('unsupported_reference')
            entry = {'kind': kind}
            if kind in {'image', 'video'}:
                video = tensors[next(video_names)]
                if video.ndim != 5 or video.shape[:2] != (1, 24) or not torch.isfinite(video).all():
                    raise ResourceError('invalid_reference_latent')
                entry.update(latent=video, latent_t=video.shape[2], latent_h=video.shape[3], latent_w=video.shape[4])
            if reference['has_audio']:
                audio = tensors[next(audio_names)]
                if audio.ndim != 2 or audio.shape[1] != 32 or audio.shape[0] % 2 or not torch.isfinite(audio).all():
                    raise ResourceError('invalid_reference_audio')
                length = audio.shape[0] // 2
                entry.update(audio_latent=audio.reshape(2, length, 32).permute(2, 0, 1).unsqueeze(0).contiguous(),
                             ref_audio_t=length)
                if kind == 'video':
                    entry['kind'] = 'video_audio'
            refs.append(entry)
    except (KeyError, StopIteration, TypeError) as exc:
        raise ResourceError('incomplete_reference_conditions') from exc
    if next(video_names, None) is not None or next(audio_names, None) is not None:
        raise ResourceError('unmatched_reference_conditions')
    return [[embeddings, {'minimax_token_tags': tags.reshape(-1), 'minimax_refs': refs}]]
