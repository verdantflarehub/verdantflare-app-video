"""CPU-only HTTP and durable queue for the single resident distributed engine."""
from __future__ import annotations
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

VERSION = 'video-minimax-h3-sol-v0.2.10'
ID = re.compile(r'sol_[0-9a-f]{32}$')
SHA = re.compile(r'[0-9a-f]{64}$')


def validate(payload, source):
    expected = {'model','task','prompt','seconds','conditions','target','num_outputs_per_prompt',
                'num_inference_steps','flow_shift','audio_flow_shift','seed','idempotency_key'}
    if not isinstance(payload,dict) or set(payload)!=expected:
        raise ValueError('invalid fields')
    if payload['model']!='MiniMaxAI/MiniMax-H3' or payload['task']!='ref2va':
        raise ValueError('invalid model')
    if type(payload['seconds']) is not int or payload['seconds'] not in {5,10,15}:
        raise ValueError('supported durations are 5,10,15')
    if type(payload['seed']) is not int or payload['seed']!=7:
        raise ValueError('seed must be 7')
    if not isinstance(payload['prompt'],str) or not payload['prompt'].strip() or len(payload['prompt'])>24000:
        raise ValueError('invalid prompt')
    if not isinstance(payload['idempotency_key'],str) or not ID.fullmatch(payload['idempotency_key'].replace('video_task_','sol_',1)):
        raise ValueError('invalid attempt identity')
    if payload['target']!={'short_edge':768,'aspect_ratio':'9:16','duration_seconds':float(payload['seconds'])}:
        raise ValueError('portrait profile required')
    if type(payload['num_outputs_per_prompt']) is not int or type(payload['num_inference_steps']) is not int:
        raise ValueError('integer profile values required')
    if payload['num_outputs_per_prompt']!=1 or payload['num_inference_steps']!=4 or payload['flow_shift']!=12.0 or payload['audio_flow_shift']!=3.0:
        raise ValueError('invalid inference profile')
    if not isinstance(payload['conditions'],list) or not 1<=len(payload['conditions'])<=15:
        raise ValueError('invalid references')
    counts={'image':0,'video':0,'audio':0}
    for ref in payload['conditions']:
        if not isinstance(ref,dict) or set(ref)!={'type','uri','role','sha256','size'}:
            raise ValueError('invalid reference fields')
        kind=ref['type']
        if kind not in counts or ref['role']!='reference':raise ValueError('invalid reference kind')
        counts[kind]+=1
        if not isinstance(ref['uri'],str) or not re.fullmatch(re.escape(source.rstrip('/'))+r'/runtime-artifacts/art_[0-9a-f]{32}/content',ref['uri']):
            raise ValueError('reference source is not permitted')
        if not isinstance(ref['sha256'],str) or not SHA.fullmatch(ref['sha256']):raise ValueError('invalid reference digest')
        if type(ref['size']) is not int or not 0<ref['size']<=1024**3:raise ValueError('invalid reference size')
    if not counts['image']+counts['video'] or counts['image']>9 or counts['video']>3 or counts['audio']>3:
        raise ValueError('invalid reference counts')
    if sum(r['size'] for r in payload['conditions'])>2*1024**3:raise ValueError('combined references too large')
    return payload


class QueueFull(Exception):pass
class Conflict(Exception):pass


class Store:
    def __init__(self,root,instance_id):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.instance_id=instance_id;self.lock=threading.RLock()
        self.db=sqlite3.connect(self.root/'tasks.sqlite3',check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, key TEXT UNIQUE, digest TEXT, request TEXT, status TEXT, stage TEXT, instance TEXT, created REAL, updated REAL, error TEXT, timing TEXT)')
        with self.db:
            self.db.execute("UPDATE tasks SET status='failed',stage='interrupted',error='runtime_restarted',updated=? WHERE status IN ('queued','in_progress')",(time.time(),))

    def submit(self,request):
        encoded=json.dumps(request,sort_keys=True,separators=(',',':'));digest=hashlib.sha256(encoded.encode()).hexdigest()
        with self.lock,self.db:
            existing=self.db.execute('SELECT id,digest FROM tasks WHERE key=?',(request['idempotency_key'],)).fetchone()
            if existing:
                if existing[1]!=digest:raise Conflict()
                return self.get(existing[0])
            if self.db.execute("SELECT count(*) FROM tasks WHERE status IN ('queued','in_progress')").fetchone()[0]>=4:raise QueueFull()
            task_id='sol_'+uuid.uuid4().hex;now=time.time()
            self.db.execute('INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?)',(task_id,request['idempotency_key'],digest,encoded,'queued','queued',None,now,now,None,'{}'))
            return self.get(task_id)

    def get(self,task_id,internal=False):
        if not ID.fullmatch(task_id):raise KeyError(task_id)
        with self.lock:
            row=self.db.execute('SELECT id,status,stage,instance,created,updated,error,timing,request FROM tasks WHERE id=?',(task_id,)).fetchone()
        if not row:raise KeyError(task_id)
        value=dict(zip(['id','status','stage','execution_instance_id','created_at','updated_at','error','timing','request'],row))
        value['runtime_version']=VERSION;value['timing']=json.loads(value['timing'])
        if internal:value['request']=json.loads(value['request'])
        else:value.pop('request')
        return value

    def take(self):
        with self.lock,self.db:
            row=self.db.execute("SELECT id FROM tasks WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
            if not row:return None
            self.db.execute("UPDATE tasks SET status='in_progress',stage='downloading',instance=?,updated=? WHERE id=?",(self.instance_id,time.time(),row[0]))
            return self.get(row[0],True)

    def update(self,task_id,status,stage,error=None,timing=None):
        with self.lock,self.db:
            self.db.execute('UPDATE tasks SET status=?,stage=?,error=?,updated=?,timing=? WHERE id=?',
                            (status,stage,error,time.time(),json.dumps(timing or {}),task_id))

    def cancel(self,task_id):
        with self.lock,self.db:
            row=self.get(task_id)
            if row['status']=='cancelled':return row
            if row['status']!='queued':raise Conflict()
            self.db.execute("UPDATE tasks SET status='cancelled',stage='cancelled',updated=? WHERE id=?",(time.time(),task_id))
            return self.get(task_id)

    def output(self,task_id):
        record=self.get(task_id)
        if record['status']!='completed':raise KeyError(task_id)
        path=self.root/task_id/'output.mp4'
        if not path.is_file() or not path.resolve().is_relative_to(self.root.resolve()):raise KeyError(task_id)
        return path


class State:
    def __init__(self,instance_id):
        self.lock=threading.RLock();self.ready=False;self.stage='gpu_check';self.instance_id=instance_id
        self.gpu_uuids=[];self.rank_pids=[];self.load_count=0;self.load_seconds=None;self.ready_at=None
        self.started_at=time.time();self.package_complete=False;self.last_heartbeat=time.monotonic()
    def set_stage(self,stage,ready=None):
        with self.lock:
            self.stage=stage
            if ready is not None:self.ready=ready
    def snapshot(self):
        with self.lock:
            return {'service':'H3-Sol','version':VERSION,'ready':self.ready,'stage':self.stage,'execution_instance_id':self.instance_id,
                    'gpu_uuids':self.gpu_uuids,'rank_pids':self.rank_pids,'model_load_count':self.load_count,'load_seconds':self.load_seconds,
                    'started_at':self.started_at,'ready_at':self.ready_at,'model_package_complete':self.package_complete,
                    'quality_review':'pending','heartbeat_age_seconds':round(time.monotonic()-self.last_heartbeat,1)}


def serve(state,store,token,source,port=8000):
    if not token:raise ValueError('runtime token must be configured')
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup();self.connection.settimeout(15)
        def log_message(self,*args):pass
        def reply(self,code,value):
            body=json.dumps(value).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def authorized(self):
            if hmac.compare_digest(self.headers.get('Authorization',''),f'Bearer {token}'):return True
            self.reply(401,{'error':'unauthorized'});return False
        def do_GET(self):
            if self.path in {'/live','/health'}:
                data=state.snapshot();healthy=data['ready'] if self.path=='/health' else data['stage']!='failed'
                return self.reply(200 if healthy else 503,data)
            if not self.authorized():return
            parts=urlsplit(self.path).path.strip('/').split('/')
            try:
                if len(parts) not in {3,4} or parts[:2]!=['v1','videos']:raise KeyError()
                if len(parts)==3:return self.reply(200,store.get(parts[2]))
                if parts[3]!='content':raise KeyError()
                path=store.output(parts[2]);self.send_response(200);self.send_header('Content-Type','video/mp4');self.send_header('Content-Length',str(path.stat().st_size));self.end_headers()
                with path.open('rb') as stream:
                    for chunk in iter(lambda:stream.read(1024*1024),b''):self.wfile.write(chunk)
            except KeyError:self.reply(404,{'error':'not_found'})
        def do_DELETE(self):
            if not self.authorized():return
            parts=self.path.strip('/').split('/')
            try:
                if len(parts)!=3 or parts[:2]!=['v1','videos']:raise KeyError()
                return self.reply(200,store.cancel(parts[2]))
            except KeyError:self.reply(404,{'error':'not_found'})
            except Conflict:self.reply(409,{'error':'only_queued_tasks_can_be_cancelled'})
        def do_POST(self):
            if not self.authorized():return
            if self.path!='/v1/videos':return self.reply(404,{'error':'not_found'})
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=65536 or self.headers.get('Transfer-Encoding'):return self.reply(413,{'error':'invalid_body_size'})
                payload=validate(json.loads(self.rfile.read(size)),source)
                if not state.snapshot()['ready']:return self.reply(503,{'error':'model_not_ready'})
                return self.reply(200,store.submit(payload))
            except (ValueError,TypeError,KeyError):self.reply(400,{'error':'invalid_request'})
            except Conflict:self.reply(409,{'error':'idempotency_conflict'})
            except QueueFull:self.reply(429,{'error':'queue_full'})
    server=ThreadingHTTPServer(('0.0.0.0',port),Handler);server.daemon_threads=True
    threading.Thread(target=server.serve_forever,daemon=True).start()
    return server


