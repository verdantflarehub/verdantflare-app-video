#!/usr/bin/env python3
"""One torchrun per Deployment; two engines remain loaded across queued requests."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import timedelta
from urllib.request import build_opener, HTTPRedirectHandler, Request

from resident_api import State, Store, serve
from sol_common import ROOT, read_json, sha256, verify_source


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):raise ValueError('reference redirects forbidden')


def verify_package(models,output):
    verify_source(Path('/opt/sol-h3'))
    manifest=read_json('/opt/sol-h3-experimental/patches.manifest.json')
    expected=dict(read_json(ROOT/'upstream.lock.json')['source_sha256'])
    series=read_json(ROOT/'patches/series.json')
    for entry in series['patches']:
        if sha256(ROOT/'patches'/entry['file'])!=entry['sha256']:raise ValueError('patch mismatch')
        expected[entry['source']]=entry['after_sha256']
    if manifest['files']!=expected:raise ValueError('patched source manifest mismatch')
    for file,digest in expected.items():
        if sha256(Path('/opt/sol-h3-experimental')/file)!=digest:raise ValueError('patched source mismatch')
    check=subprocess.run([sys.executable,str(ROOT/'model-integrity.py'),'--models',str(models),
                          '--lock',str(ROOT/'models.lock.json'),'--output',str(output)],check=False)
    if check.returncode not in {0,2}:raise ValueError('model checker failed')
    report=read_json(output)
    allowed={'file':'docs/QA-about-License.md','error':'missing_file_or_download_receipt'}
    if (len(report['errors'])>1 or any(e!=allowed for e in report['errors'])
        or len(report['files'])!=64-len(report['errors']) or not all(r['matches_receipt'] for r in report['files'].values())
        or report['adapter_sha256']!=read_json(ROOT/'models.lock.json')['adapter']['sha256']):
        raise ValueError('model integrity failed')
    return not bool(report['errors'])


def download(task,root):
    directory=root/task['id'];directory.mkdir(exist_ok=True)
    paths=[];opener=build_opener(NoRedirect())
    # No proxy environment or redirects may change the trusted in-cluster source.
    from urllib.request import ProxyHandler
    opener=build_opener(ProxyHandler({}),NoRedirect())
    for index,ref in enumerate(task['request']['conditions']):
        suffix={'image':'.png','video':'.mp4','audio':'.wav'}[ref['type']]
        path=directory/f'reference-{index}{suffix}';digest=hashlib.sha256();size=0
        with opener.open(Request(ref['uri']),timeout=60) as response,path.open('wb') as stream:
            if response.status!=200:raise ValueError('reference unavailable')
            while chunk:=response.read(1024*1024):
                size+=len(chunk)
                if size>ref['size']:raise ValueError('reference too large')
                digest.update(chunk);stream.write(chunk)
        if size!=ref['size'] or digest.hexdigest()!=ref['sha256']:raise ValueError('reference integrity failed')
        if ref['type']=='image':
            from PIL import Image
            with Image.open(path) as image:
                image.verify()
        else:
            probe=subprocess.run(['ffprobe','-v','error','-show_entries','stream=codec_type','-of','json',str(path)],capture_output=True,text=True,check=True,timeout=30)
            if not any(s.get('codec_type')==ref['type'] for s in json.loads(probe.stdout).get('streams',[])):raise ValueError('wrong reference type')
        paths.append((ref['type'],str(path)))
    return paths


def main():
    import torch
    import torch.distributed as dist
    from gpu_guard import verify_gpu_allocation
    rank=int(os.environ['LOCAL_RANK']);root=Path(os.environ.get('SOL_TASK_ROOT','/data/projects/h3-sol/tasks'))
    instance=os.environ['SOL_INSTANCE_ID'];state=State(instance);store=None;server=None;current=None
    if rank==0:
        store=Store(root,instance)
        server=serve(state,store,os.environ.get('SOL_RUNTIME_TOKEN',''),os.environ['SOL_ARTIFACT_SOURCE'])
    def stop(*_):
        state.set_stage('stopping',False)
        if rank==0 and current:store.update(current,'failed','interrupted','instance_stopped')
        os._exit(0)
    signal.signal(signal.SIGTERM,stop)
    try:
        if torch.cuda.device_count()!=2:raise ValueError('two physical CUDA GPUs required')
        uuids=['GPU-'+str(torch.cuda.get_device_properties(i).uuid).removeprefix('GPU-') for i in range(2)]
        print(json.dumps({"stage":"gpu_uuid_observed","rank":rank,"gpu_uuids":uuids}),flush=True)
        for i in range(2):
            if torch.cuda.get_device_capability(i)!=(8,9) or torch.cuda.get_device_properties(i).total_memory<23*1024**3:
                raise ValueError('two full RTX4090 SM89 GPUs required')
        torch.cuda.set_device(rank)
        test=torch.ones((32,32),device=rank,dtype=torch.bfloat16)
        if not torch.all(test@test==32):raise ValueError('BF16 probe failed')
        del test;torch.cuda.empty_cache()
        print(json.dumps({'stage':'gpu_check','rank':rank,'pid':os.getpid(),'gpu_uuids':uuids,'passed':True}),flush=True)
        dist.init_process_group(backend='nccl',device_id=torch.device('cuda',rank),timeout=timedelta(minutes=45))
        control=dist.new_group(backend='gloo',timeout=timedelta(minutes=45))
        pids=[None,None];dist.all_gather_object(pids,os.getpid(),group=control)
        models=Path(os.environ.get('SOL_MODELS','/models/MiniMax-H3'))
        state.gpu_uuids=uuids;state.rank_pids=pids;state.set_stage('model_integrity')
        report=[None]
        if rank==0:
            report[0]=verify_package(models,root/f'integrity-{instance}.json')
        dist.broadcast_object_list(report,src=0,group=control)
        state.package_complete=report[0]
        state.set_stage('loading');started=time.monotonic()
        from retain_cpu_weights import install,verify
        install();verify(torch.device('cuda',rank))
        from h3_runtime import MiniMaxH3Inference
        with torch.inference_mode():
            engine=MiniMaxH3Inference(model_path=str(models/'MiniMax-H3-Diffusers'),
                adapter_path=models/'MiniMax-H3-Turbo/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors',
                task='ref2va',attention_backend='dense',cpu_offload=True,output_width=768,output_height=1344)
        torch.cuda.synchronize(rank);dist.barrier(group=control)
        state.load_seconds=round(time.monotonic()-started,2);state.load_count=1;state.ready_at=time.time();state.set_stage('ready',True)
        print(json.dumps({'stage':'ready','rank':rank,'pid':os.getpid(),'load_count':1,'load_seconds':state.load_seconds}),flush=True)
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference
        warm_keys=set()
        while True:
            command=[None]
            if rank==0:
                task=store.take()
                if task:
                    current=task['id'];state.set_stage('downloading')
                    try:
                        refs=download(task,root)
                        command[0]={'task':task,'references':refs}
                    except Exception:
                        store.update(current,'failed','download_failed','reference_validation_failed');current=None;state.set_stage('ready')
                if command[0] is None:command[0]={'idle':True}
            dist.broadcast_object_list(command,src=0,group=control)
            state.last_heartbeat=time.monotonic()
            if command[0].get('idle'):
                time.sleep(1);continue
            task=command[0]['task'];current=task['id'];request=task['request'];times={}
            # Conservatively reuse warmup only for identical prompt/reference contents and duration.
            signature=hashlib.sha256(json.dumps([request['seconds'],request['prompt'],[(r['type'],r['sha256']) for r in request['conditions']]],sort_keys=True).encode()).hexdigest()
            references=[MiniMaxH3Reference(**{kind:path}) for kind,path in command[0]['references']]
            with torch.inference_mode():
                hit=signature in warm_keys
                if not hit:
                    state.set_stage('warming')
                    if rank==0:store.update(current,'in_progress','warming')
                    begin=time.monotonic();engine.warmup(duration=request['seconds'],prompt=request['prompt'],references=references)
                    times['warmup_seconds']=round(time.monotonic()-begin,3)
                    if len(warm_keys)>=8:warm_keys.clear()
                    warm_keys.add(signature)
                else:times['warmup_seconds']=0
                times['warmup_hit']=hit;times['load_count']=1
                # New reference objects and upstream generator/row-count reset isolate requests.
                references=[MiniMaxH3Reference(**{kind:path}) for kind,path in command[0]['references']]
                state.set_stage('generating')
                if rank==0:store.update(current,'in_progress','generating',timing=times)
                result=engine.generate(request['prompt'],duration=request['seconds'],seed=request['seed'],references=references)
                if rank==0:
                    state.set_stage('saving');output=root/current/'output.mp4';result.save(output)
                    probe=subprocess.run(['ffprobe','-v','error','-show_entries','stream=codec_type,codec_name,width,height,r_frame_rate:format=duration','-of','json',str(output)],check=True,capture_output=True,text=True,timeout=60)
                    media=json.loads(probe.stdout);v=next(s for s in media['streams'] if s['codec_type']=='video')
                    if (v['width'],v['height'],v['codec_name'],v['r_frame_rate'])!=(768,1344,'h264','24/1') or abs(float(media['format']['duration'])-request['seconds'])>1:
                        raise ValueError('generated media contract failed')
                    times['inference_seconds']=result.elapsed_s;times['sha256']=sha256(output)
                    store.update(current,'completed','completed',timing=times)
                del result,references
                dist.barrier(group=control)
            current=None;state.last_heartbeat=time.monotonic();state.set_stage('ready')
    except BaseException as error:
        state.set_stage('failed',False)
        if rank==0 and current:store.update(current,'failed','engine_failed',type(error).__name__)
        print(json.dumps({'stage':'failed','rank':rank,'error_type':type(error).__name__}),flush=True)
        # A poisoned rank invalidates the entire process group. torchrun supervises its peer.
        raise


if __name__=='__main__':main()
