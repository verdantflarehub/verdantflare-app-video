import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import httpx
from app.artifacts import ArtifactStore
from app.tasks import TaskStore,TaskConflict
from app.executor import VideoExecutor

class SolRouteTest(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();root=Path(self.temp.name)
  self.assets=ArtifactStore(root);self.tasks=TaskStore(root);self.calls=[]
  def handle(request):
   self.calls.append(request)
   if request.url.path=='/health':return httpx.Response(200,json={'ready':True})
   if request.method=='POST':return httpx.Response(200,json={'id':'sol_'+'c'*32})
   return httpx.Response(200,json={'status':'completed','execution_instance_id':'11111111-1111-1111-1111-111111111111','stage':'completed'})
  self.http=httpx.Client(transport=httpx.MockTransport(handle))
  self.env=patch.dict(os.environ,{'H3_SOL_RUNTIME_URL':'http://sol.example:8000','H3_SOL_RUNTIME_TOKEN':'test-sol-token','H3_SOL_RUNTIME_ROUTE':'minimax-h3-sol-ref2va'});self.env.start()
  self.executor=VideoExecutor(self.assets,self.tasks,self.http)
  asset=self.assets.create_from_chunks(project_id='demo',operation='test',filename='image.png',media_type='image/png',chunks=[b'fixture'])
  self.kw={'project_id':'demo','idempotency_key':'attempt','model':'minimax-h3-ref2va','prompt':'Approved motion','duration_seconds':5,'aspect_ratio':'9:16','references':{'images':[{'artifact_id':asset.artifact_id,'purpose':'identity'}]}}
 def tearDown(self):self.http.close();self.env.stop();self.temp.cleanup()
 def test_sol_route_and_instance_persist_without_fallback(self):
  row=self.executor.generate(**self.kw,route='h3-sol')
  self.assertEqual(row.service,'h3-sol');self.assertEqual(row.runtime_route,'minimax-h3-sol-ref2va');self.assertEqual(self.calls[0].url.host,'sol.example')
  self.assertEqual(self.calls[0].headers['Authorization'],'Bearer test-sol-token')
  body=json.loads(self.calls[0].content);self.assertEqual(body['num_inference_steps'],4)
  self.assertNotIn('service', row.request)
  self.assertEqual(body['idempotency_key'],row.video_task_id);self.assertIn('sha256',body['conditions'][0])
  row=self.executor.status(row.video_task_id);self.assertEqual(row.execution_instance_id,'11111111-1111-1111-1111-111111111111')
  self.assertEqual(self.calls[-1].url.host,'sol.example')
  with self.assertRaises(TaskConflict):self.executor.generate(**self.kw,route='h3')
 def test_h3_default_remains_legacy_payload_and_digest(self):
  expected=self.executor._normalize(self.kw['project_id'],self.kw['model'],self.kw['prompt'],5,'9:16',self.kw['references'], 'h3')[1]
  row=self.executor.generate(**self.kw)
  self.assertEqual(row.input_digest,expected);self.assertEqual(row.service,'h3')
  self.assertNotIn('Authorization',self.calls[0].headers);self.assertNotEqual(self.calls[0].url.host,'sol.example')
  body=json.loads(self.calls[0].content);self.assertEqual(body['num_inference_steps'],21);self.assertNotIn('idempotency_key',body)
 def test_sol_duration_rejects_before_network(self):
  with self.assertRaises(ValueError):self.executor.generate(**{**self.kw,'duration_seconds':6},route='h3-sol')
  self.assertFalse(self.calls)
 def test_explicit_4090_route_keeps_business_model_type(self):
  with patch.dict(os.environ, {'H3_RUNTIME_ROUTES': json.dumps({
   'h3-sol-4090': {'url':'http://sol4090.example:8000','version':'video-minimax-h3-sol-v0.2.10','requires_token':True}})}):
   executor=VideoExecutor(self.assets,self.tasks,self.http)
   row=executor.generate(**self.kw,route='h3-sol-4090')
   self.assertEqual(row.request['model'],'minimax-h3-ref2va')
   self.assertEqual(row.request['route'],'h3-sol-4090')
   self.assertEqual(row.runtime_route,'h3-sol-4090')
   self.assertEqual(self.calls[-1].url.host,'sol4090.example')
