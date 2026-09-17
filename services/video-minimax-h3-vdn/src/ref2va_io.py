"""Ref2VA transport validation and verified local reference downloads."""
import hashlib
from pathlib import Path
import re
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

FRAMES = {5:124, 10:243, 15:345}


def validate(payload, source):
    fields = {'model','task','prompt','seconds','conditions','target','num_outputs_per_prompt',
              'num_inference_steps','flow_shift','audio_flow_shift','seed','idempotency_key'}
    if not isinstance(payload,dict) or set(payload) - {'project_id'} != fields:
        raise ValueError('invalid Ref2VA fields')
    if 'project_id' in payload and (not isinstance(payload['project_id'], str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}', payload['project_id'])):
        raise ValueError('invalid project id')
    if payload['model']!='MiniMaxAI/MiniMax-H3' or payload['task']!='ref2va':
        raise ValueError('only Ref2VA is served')
    if type(payload['seconds']) is not int or payload['seconds'] not in FRAMES:
        raise ValueError('duration must be 5, 10, or 15 seconds')
    if not isinstance(payload['prompt'],str) or not payload['prompt'].strip() or len(payload['prompt'])>24000:
        raise ValueError('invalid prompt')
    if payload['target']!={'short_edge':768,'aspect_ratio':'9:16','duration_seconds':float(payload['seconds'])}:
        raise ValueError('unsupported target')
    for key, expected in [('seed',7),('num_outputs_per_prompt',1),('num_inference_steps',8)]:
        if type(payload[key]) is not int or payload[key]!=expected:
            raise ValueError('invalid inference profile')
    if payload['flow_shift']!=12.0 or payload['audio_flow_shift']!=3.0:
        raise ValueError('unsupported schedules')
    key=payload['idempotency_key']
    if not isinstance(key,str) or not key.strip() or len(key)>128:
        raise ValueError('invalid idempotency key')
    refs=payload['conditions']
    if not isinstance(refs,list) or not 1<=len(refs)<=12:
        raise ValueError('Ref2VA requires 1 to 12 references')
    counts={'image':0,'video':0,'audio':0}
    for ref in refs:
        if not isinstance(ref,dict) or set(ref)!={'type','uri','role','sha256','size'}:
            raise ValueError('invalid reference')
        if not isinstance(ref['type'],str) or ref['type'] not in counts or ref['role']!='reference':
            raise ValueError('invalid reference modality')
        counts[ref['type']]+=1
        if not isinstance(ref['uri'],str) or not re.fullmatch(re.escape(source.rstrip('/'))+r'/runtime-artifacts/art_[0-9a-f]{32}/content',ref['uri']):
            raise ValueError('reference source forbidden')
        if not isinstance(ref['sha256'],str) or not re.fullmatch(r'[0-9a-f]{64}',ref['sha256']):
            raise ValueError('invalid reference digest')
        if type(ref['size']) is not int or not 0<ref['size']<=1024**3:
            raise ValueError('invalid reference size')
    if not counts['image']+counts['video'] or counts['image']>9 or counts['video']>3 or counts['audio']>3:
        raise ValueError('unsupported reference counts')
    if sum(ref['size'] for ref in refs)>2*1024**3:
        raise ValueError('combined references too large')
    return payload


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*_args,**_kwargs):
        raise ValueError('reference redirects forbidden')


def download(request,directory):
    opener=build_opener(ProxyHandler({}),NoRedirect())
    paths=[]
    for index,ref in enumerate(request['conditions']):
        suffix={'image':'.image','video':'.video','audio':'.audio'}[ref['type']]
        path=Path(directory)/f'reference-{index}{suffix}'
        count,digest=0,hashlib.sha256()
        with opener.open(Request(ref['uri']),timeout=60) as response,path.open('xb') as stream:
            if response.status!=200:raise ValueError('reference unavailable')
            while chunk:=response.read(1024**2):
                count+=len(chunk)
                if count>ref['size']:raise ValueError('reference exceeds declared size')
                digest.update(chunk);stream.write(chunk)
        if count!=ref['size'] or digest.hexdigest()!=ref['sha256']:
            raise ValueError('reference integrity failed')
        paths.append((ref['type'],str(path)))
    return paths
