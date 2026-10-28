import hashlib
import json
from types import SimpleNamespace
import unittest
from app.execution_proof import proof

class ProofTests(unittest.TestCase):
    def setUp(self):
        self.task=SimpleNamespace(status='succeeded',artifact_id='out',service='h3-singularity',
            runtime_route='h3-singularity',video_task_id='task1',project_id='project1',delivery=None,
            request={'prompt':'向前推进','duration_seconds':4,'aspect_ratio':'9:16','quality_profile':None,
                'resolved_output':{'frames':90},'references':{'images':[{'artifact_id':'one'}]},'secret':'PRIVATE'})
        self.store=SimpleNamespace(get=lambda artifact_id,project_id:SimpleNamespace(
            project_id=project_id,sha256=('a' if artifact_id=='one' else 'b')*64,size=100,media_type='video/mp4'))
    def test_native_h3_is_fixed_read_only(self):
        p=proof(self.task,self.store)
        self.assertEqual(p['model_id'],'minimax-h3-ref2va');self.assertEqual(p['parameters']['fps'],24)
        self.assertEqual(p['prompt_sha256'],hashlib.sha256('向前推进'.encode()).hexdigest())
        self.assertEqual(p['inputs'][0]['sha256'],'a'*64);self.assertNotIn('PRIVATE',json.dumps(p))
    def test_rejects_failed_substituted_and_processed_tasks(self):
        for attr,value in [('status','failed'),('runtime_route','h3-vdn'),('delivery',{'status':'succeeded'})]:
            old=getattr(self.task,attr);setattr(self.task,attr,value)
            with self.assertRaises(ValueError):proof(self.task,self.store)
            setattr(self.task,attr,old)
    def test_seedance_exact_model_no_draft_substitution(self):
        self.task.service=self.task.runtime_route='higgsfield'
        self.task.request.update(model='seedance_2_5',mode='text_to_video',resolution='720p',references={})
        self.task.media={'sha256':'c'*64,'size':200};self.task.provider_state={'content_ref':{'version_id':'id'},'token':'PRIVATE'}
        p=proof(self.task,self.store);self.assertEqual(p['outputs'][0]['sha256'],'c'*64);self.assertNotIn('PRIVATE',json.dumps(p))
        self.task.request['model']='seedance_2_5_draft'
        with self.assertRaises(ValueError):proof(self.task,self.store)

class ProofBoundaryTests(unittest.TestCase):
    def test_proof_does_not_resume_or_collect(self):
        from unittest.mock import patch,MagicMock
        from app import server
        task=SimpleNamespace(project_id='project1',authority={'organization_id':'org1'})
        store=MagicMock();store.get.return_value=task
        with patch.object(server,'_active_tasks',return_value=store),patch.object(server,'_active_artifacts'), \
                patch.object(server,'_is_task_nsfw',return_value=False),patch.object(server,'authorize',return_value={'organization_id':'org1'}) as auth, \
                patch('app.execution_proof.proof',return_value={}),patch.object(server.executor,'result') as collect:
            server.video_result(task_id='task1',execution_proof_only=True)
            collect.assert_not_called();self.assertEqual(auth.call_args.kwargs,{'require_write':False})
        with patch.object(server,'_active_tasks',return_value=store),patch.object(server,'_active_artifacts'), \
                patch.object(server,'_is_task_nsfw',return_value=False),patch.object(server,'authorize',side_effect=PermissionError('project access denied')), \
                patch('app.execution_proof.proof') as build:
            result = server.video_result(task_id='task1',execution_proof_only=True)
            self.assertTrue(result.is_error)
            self.assertEqual(result.structured_content['error']['code'], 'project_access_denied')
            self.assertEqual(json.loads(result.content[0].text), result.structured_content)
            build.assert_not_called()
