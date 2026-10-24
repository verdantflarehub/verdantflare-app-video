"""Read-only proofs from fixed native tasks; no collection or generation."""
import hashlib


def proof(record, artifacts):
    p = record.request
    if record.status not in {'succeeded', 'completed'} or p.get('postprocess') or record.delivery:
        raise ValueError('execution_proof_not_ready')
    prompt = p.get('prompt')
    if not isinstance(prompt, str) or not prompt:
        raise ValueError('execution_proof_not_ready')
    route = record.runtime_route
    model = p.get('model', 'minimax-h3-ref2va')
    def media(artifact_id):
        a = artifacts.get(artifact_id, record.project_id)
        if a.project_id != record.project_id:
            raise ValueError('execution_proof_project_mismatch')
        return {'sha256': a.sha256, 'size': a.size, 'mime': a.media_type}
    if route == 'h3-singularity' and record.service == 'h3-singularity' and model == 'minimax-h3-ref2va':
        if not record.artifact_id or not p.get('resolved_output'):
            raise ValueError('execution_proof_not_ready')
        inputs = [media(r['artifact_id']) for group in ('images', 'videos', 'audios')
                  for r in p.get('references', {}).get(group, [])]
        outputs = [media(record.artifact_id)]
        params = {k: p[k] for k in ('duration_seconds', 'aspect_ratio', 'quality_profile') if k in p}
        params['fps'] = 24
    elif record.service == 'higgsfield' and route == 'higgsfield' and model == 'seedance_2_5':
        if p.get('mode') != 'text_to_video' or any(p.get('references', {}).values()):
            raise ValueError('execution_proof_not_ready')
        m = record.media or {}
        if not (record.provider_state or {}).get('content_ref') or not m.get('sha256') or not m.get('size'):
            raise ValueError('execution_proof_not_ready')
        inputs = []
        outputs = [{'sha256': m['sha256'], 'size': m['size'], 'mime': 'video/mp4'}]
        params = {k: p[k] for k in ('duration_seconds', 'aspect_ratio', 'resolution', 'mode')}
    else:
        raise ValueError('execution_proof_not_ready')
    return {'schema_version': 1, 'service_id': 'video', 'run_id': record.video_task_id,
            'project_id': record.project_id, 'state': 'succeeded', 'model_id': model,
            'route': route, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'parameters': params, 'inputs': inputs, 'outputs': outputs}
