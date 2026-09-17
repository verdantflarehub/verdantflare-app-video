"""Export clean generation state for same-node post-processing, never MP4 encoding."""
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    out = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            out.update(block)
    return out.hexdigest()


class ExportingPipeline:
    """Request the modular pipeline's retained outputs without another generation."""
    FIELDS = ('normalized_references', 'latents', 'audio_latents', 'prompt_embeds', 'text_token_tags',
              'condition_latents', 'audio_condition_latents', 'position_ids',
              'token_tags', 'video_indices', 'audio_indices', 'text_indices',
              'num_condition_video_rows', 'num_condition_audio_rows',
              'height', 'width', 'num_frames', 'num_audio_latents')

    def __init__(self, pipeline, directory):
        self.pipeline, self.directory = pipeline, Path(directory)
        self.export_error = None

    def __call__(self, **kwargs):
        requested = kwargs.get('output', [])
        kwargs['output'] = list(dict.fromkeys([*requested, *self.FIELDS]))
        result = self.pipeline(**kwargs)
        try:
            self._export(result)
        except Exception as exc:
            # Optional post-processing resources must not invalidate a generated
            # source video. Never repeat inference after an export failure.
            self.export_error = type(exc).__name__
        return {key: result[key] for key in requested}

    def _export(self, result):
        import torch
        from safetensors.torch import save_file
        directory = self.directory / 'latent-bundle'
        directory.mkdir(exist_ok=False)
        metadata, conditions = {}, {}
        for name in self.FIELDS:
            value = result[name]
            if name == 'normalized_references':
                metadata[name] = [{'kind': ref.kind, 'has_audio': bool(ref.has_audio)} for ref in value]
            elif isinstance(value, torch.Tensor):
                tensor = value.detach().cpu().contiguous()
                if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                    raise ValueError('nonfinite_export_tensor')
                if name == 'latents':
                    if tensor.ndim != 5 or tensor.shape[:2] != (1, 24):
                        raise ValueError('invalid_video_latent_shape')
                    save_file({'samples': tensor}, directory / 'video.safetensors')
                elif name == 'audio_latents':
                    if tensor.ndim != 3 or tensor.shape[:2] != (2, 32):
                        raise ValueError('invalid_audio_latent_shape')
                    save_file({'samples': tensor.permute(1, 0, 2).unsqueeze(0).contiguous()}, directory / 'audio.safetensors')
                else:
                    conditions[name] = tensor
            elif isinstance(value, list):
                names = []
                for index, tensor in enumerate(value):
                    if tensor is None:
                        names.append(None)
                        continue
                    if not isinstance(tensor, torch.Tensor):
                        raise ValueError('unsupported_condition_export')
                    key = f'{name}.{index}'
                    conditions[key] = tensor.detach().cpu().contiguous()
                    if conditions[key].is_floating_point() and not torch.isfinite(conditions[key]).all():
                        raise ValueError('nonfinite_condition_tensor')
                    names.append(key)
                metadata[name] = names
            elif type(value) in (int, float, str, bool) or value is None:
                metadata[name] = value
            else:
                raise ValueError(f'unsupported_export_field:{name}')
        save_file(conditions, directory / 'conditions.safetensors')
        (directory / 'geometry.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')


def finalize(directory, task, provenance, media):
    """Publish a manifest only after the original media has passed validation."""
    directory = Path(directory)
    bundle = directory / 'latent-bundle'
    root = Path(os.environ.get('VDN_PROJECTS_ROOT', '/projects')).resolve()
    request = task['request']
    project = request.get('project_id')
    public_id = task['idempotency_key']
    node = os.environ.get('VDN_NODE_NAME')
    if not project or not node:
        raise ValueError('latent_export_requires_project_and_node')
    manifest = {'schema': 'h3-latent-bundle/v1', 'representation': 'h3-normalized-av/v1',
                'project_id': project, 'source_video_task_id': public_id, 'runtime_task_id': task['id'],
                'node': node, 'latent_state': 'clean', 'source_route': 'h3-vdn',
                'media': {'fps': 24, 'width': 768, 'height': 1344, 'frames': int(next(stream for stream in media['streams'] if stream['codec_type'] == 'video')['nb_read_frames'])},
                'request': request, 'provenance': provenance, 'files': {}}
    files = {'video_latent': bundle / 'video.safetensors', 'audio_latent': bundle / 'audio.safetensors',
             'conditions': bundle / 'conditions.safetensors', 'geometry': bundle / 'geometry.json',
             'source_video': directory / 'video.mp4'}
    for index, ref in enumerate(request['conditions']):
        suffix = {'image': '.image', 'video': '.video', 'audio': '.audio'}[ref['type']]
        files[f'reference_{index}'] = directory / f'reference-{index}{suffix}'
    for key, path in files.items():
        manifest['files'][key] = {'path': str(path.resolve().relative_to(root)),
                                  'size': path.stat().st_size, 'sha256': digest(path)}
    path = bundle / 'manifest.json'
    with path.open('x') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return {'manifest_path': str(path.resolve().relative_to(root)), 'manifest_sha256': digest(path),
            'node': node, 'schema': manifest['schema'], 'source_video_task_id': public_id}
