import copy
import json
from pathlib import Path
import tempfile
import unittest
from urllib.request import Request,urlopen
from urllib.error import HTTPError

from resident_api import Conflict, QueueFull, State, Store, serve, validate

SOURCE='http://mcp.example:8000'

def payload(n=0):
 return {'model':'MiniMaxAI/MiniMax-H3','task':'ref2va','prompt':'Approved camera movement','seconds':5,
         'conditions':[{'type':'image','role':'reference','uri':SOURCE+'/runtime-artifacts/art_'+'a'*32+'/content','sha256':'b'*64,'size':12}],
         'target':{'short_edge':768,'aspect_ratio':'9:16','duration_seconds':5.0},'num_outputs_per_prompt':1,'num_inference_steps':4,
         'flow_shift':12.0,'audio_flow_shift':3.0,'seed':7,'idempotency_key':'video_task_'+f'{n:032x}'}

class ResidentTest(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.store=Store(Path(self.temp.name),'11111111-1111-1111-1111-111111111111')
 def tearDown(self):self.store.db.close();self.temp.cleanup()
 def test_strict_profile_and_source(self):
  validate(payload(),SOURCE)
  for change in [{'seconds':True},{'seconds':4},{'seconds':16},{'seed':8},{'model':'h3-sol'},{'num_inference_steps':21}]:
   with self.assertRaises(ValueError):validate({**payload(),**change},SOURCE)
  six=payload();six['seconds']=6;six['target']={'short_edge':768,'aspect_ratio':'9:16','duration_seconds':6.0};validate(six,SOURCE)
  for uri in ['file:///etc/passwd',SOURCE+'/runtime-artifacts/../secret',SOURCE+'.evil/runtime-artifacts/art_'+'a'*32+'/content',SOURCE+'/runtime-artifacts/art_'+'a'*32+'/content?token=x']:
   p=payload();p['conditions'][0]['uri']=uri
   with self.assertRaises(ValueError):validate(p,SOURCE)
  p=payload();p['conditions'][0]['type']='audio'
  with self.assertRaises(ValueError):validate(p,SOURCE)
 def test_idempotency_queue_and_assignment(self):
  first=self.store.submit(payload());self.assertIsNone(first['execution_instance_id'])
  self.assertEqual(first['id'],self.store.submit(payload())['id'])
  p=payload();p['prompt']='different'
  with self.assertRaises(Conflict):self.store.submit(p)
  running=self.store.take();self.assertEqual(running['execution_instance_id'],self.store.instance_id)
  for i in range(1,4):self.store.submit(payload(i))
  with self.assertRaises(QueueFull):self.store.submit(payload(4))
 def test_restart_fails_unfinished_without_replay(self):
  record=self.store.submit(payload());self.store.take()
  other=Store(Path(self.temp.name),'new-instance')
  self.assertIsNone(other.take());self.assertEqual(other.get(record['id'])['status'],'failed')
  self.assertEqual(other.submit(payload())['id'],record['id']);other.db.close()
 def test_cancel_only_queued(self):
  row=self.store.submit(payload());self.assertEqual(self.store.cancel(row['id'])['status'],'cancelled')
  self.assertIsNone(self.store.take())
  row=self.store.submit(payload(1));self.store.take()
  with self.assertRaises(Conflict):self.store.cancel(row['id'])
 def test_content_only_for_completed_task_and_no_path_traversal(self):
  row=self.store.submit(payload())
  with self.assertRaises(KeyError):self.store.output(row['id'])
  with self.assertRaises(KeyError):self.store.get('../tasks.sqlite3')
  self.store.take();directory=Path(self.temp.name)/row['id'];directory.mkdir();(directory/'output.mp4').write_bytes(b'test')
  self.store.update(row['id'],'completed','completed')
  self.assertEqual(self.store.output(row['id']).read_bytes(),b'test')
 def test_http_auth_readiness_and_queue(self):
  state=State(self.store.instance_id);server=serve(state,self.store,'test-only',SOURCE,port=0)
  base='http://127.0.0.1:'+str(server.server_address[1])
  try:
   self.assertEqual(urlopen(base+'/live').status,200)
   with self.assertRaises(HTTPError) as e:urlopen(base+'/health')
   self.assertEqual(e.exception.code,503)
   with self.assertRaises(HTTPError) as e:urlopen(Request(base+'/v1/videos',data=json.dumps(payload()).encode()))
   self.assertEqual(e.exception.code,401)
   req=lambda:Request(base+'/v1/videos',data=json.dumps(payload()).encode(),headers={'Authorization':'Bearer test-only'})
   with self.assertRaises(HTTPError) as e:urlopen(req())
   self.assertEqual(e.exception.code,503)
   state.set_stage('ready',True)
   row=json.load(urlopen(req()));self.assertEqual(row['status'],'queued')
   self.assertNotIn('request',row)
  finally:server.shutdown();server.server_close()
