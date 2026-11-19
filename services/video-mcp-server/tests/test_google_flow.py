"""Exercise Video's normal task entry points with a fake Flow gateway and Station."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import secrets
import tempfile
import unittest
from unittest.mock import patch

import httpx

from app.artifact_output import uuid7
from app.artifacts import ArtifactStore
from app.dashboard import task_channel, channel_label
from app.delivery import CALLER
from app.executor import VideoExecutor
from app.tasks import TaskStore, TaskConflict


class GoogleFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "sample.mp4"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "color=s=32x18:r=5",
                            "-t", "4", "-c:v", "mpeg4", "-movflags", "+faststart", str(target)], check=True)
            cls.media = target.read_bytes()

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.org, self.user, self.project, self.call = uuid7(), uuid7(), uuid7(), secrets.token_hex(16)
        env = {"GOOGLE_FLOW_GATEWAY_URL":"http://flow.invalid","GOOGLE_FLOW_INTERNAL_TOKEN":secrets.token_hex(),
               "GOOGLE_FLOW_ORGANIZATION_ID":self.org,"VIDEO_PROJECT_SERVICE_URL":"http://project.invalid",
               "VIDEO_PROJECT_SERVICE_TOKEN":secrets.token_hex(),"VIDEO_ARTIFACT_SERVICE_URL":"http://artifact.invalid",
               "VIDEO_ARTIFACT_SERVICE_TOKEN":secrets.token_hex()}
        self.enterContext(patch.dict(os.environ,env))
        ctx = CALLER.set({"user_id":self.user,"organization_id":self.org})
        self.addCleanup(CALLER.reset,ctx)
        self.client = httpx.Client(transport=httpx.MockTransport(self.respond))
        self.addCleanup(self.client.close)
        self.exe = VideoExecutor(ArtifactStore(root), TaskStore(root), self.client)
        self.adapter = self.exe.google_flow
        self.adapter.client.close()
        self.adapter.client = self.client
        self.gate = self.enterContext(patch.object(self.exe,"require_channel"))
        self.loss = None
        self.lost = self.denied = False
        self.posts = self.reads = self.checks = 0
        self.payloads, self.uploads, self.prepares = [], [], []
        self.digest = hashlib.sha256(self.media).hexdigest()
        self.upload, self.receipt = uuid7(), dict(store_id=uuid7(),artifact_id=uuid7(),version_id=uuid7())
        self.committed = False
        self.accepted, self.stage = True,"succeeded"
        self.transport = httpx.MockTransport(self.artifact)
        self.client_type = httpx.Client

    def artifact(self, req):
        path = req.url.path
        self.assertEqual(req.headers["X-User-Id"],self.user)
        if path.endswith("/uploads"):
            data = json.loads(req.content)
            self.prepares.append(data["write_id"])
            self.assertEqual(data["source"], {"kind":"task_output","service_id":"video","run_id":self.key,
                                             "task_id":self.call,"project_id":self.project})
            self.source = data["source"]
            phase, result = "prepare", {"upload_id":self.upload}
        elif path.endswith(self.upload):
            return httpx.Response(200,json={"state":"committed" if self.committed else "prepared"})
        elif req.method=="PUT":
            self.assertEqual(req.content,self.media)
            phase,result = "put", {}
        elif path.endswith("/commit"):
            self.committed = True
            phase,result = "commit", self.receipt | dict(source=self.source,organization_id=self.org,
                created_by=self.user,mime="video/mp4",sha256=self.digest,size=len(self.media))
        else:
            return httpx.Response(200,content=self.media)
        if self.loss == phase and not self.lost:
            self.lost = True
            raise httpx.ReadTimeout("fixture")
        return httpx.Response(200,json=result)

    def respond(self, req):
        path = req.url.path
        if path.endswith("/project/open"):
            self.assertEqual(json.loads(req.content),{"project_id":self.project,"require_write":True})
            return httpx.Response(403 if self.denied else 200,json={})
        self.checks += 1
        self.assertEqual((req.headers["X-Organization-Id"],req.headers["X-User-Id"],req.headers["X-Project-Id"],
                          req.headers["X-Project-Write"]), (self.org,self.user,self.project,"true"))
        if path=="/v1/capabilities":
            return httpx.Response(200,json={"models":{"video":["veo-lite"]}})
        if path=="/v1/account":
            return httpx.Response(200,json={"authorized":True,"credits":99})
        if path=="/v1/cost":
            return httpx.Response(200,json={"credits":3})
        if path=="/v1/inputs":
            digest=hashlib.sha256(req.content).hexdigest()
            self.uploads.append(digest)
            return httpx.Response(200,json={"sha256":digest,"size":len(req.content)})
        if path.endswith("/content"):
            self.reads += 1
            return httpx.Response(200,content=self.media,headers={"Content-Type":"video/mp4","X-Content-SHA256":self.digest})
        if req.method=="POST":
            self.assertEqual(path,"/v1/jobs")
            self.posts += 1
            data = json.loads(req.content)
            self.payloads.append(data)
            self.key = data["request_id"]
            if self.loss=="submit":
                raise httpx.ReadTimeout("fixture")
        else:
            self.assertEqual(path,"/v1/jobs/"+self.key)
        return httpx.Response(200,json=dict(request_id=self.key,call_id=self.call,accepted=self.accepted,
                                           state=self.stage,credits=3))

    def create(self, **changes):
        args = dict(route="google-flow",project_id=self.project,idempotency_key="scene",model="veo-lite",
                    prompt="paper boat",duration_seconds=4,aspect_ratio="16:9",references={},budget_limit="6")
        return self.exe.generate(**(args | changes))

    def poll(self, record):
        with patch("app.artifact_output.httpx.Client",side_effect=lambda **kw:self.client_type(transport=self.transport,**kw)):
            return self.exe.status(record.video_task_id)

    def test_project_preflight_and_reference_bytes(self):
        asset = self.exe.artifacts.create_from_chunks(project_id=self.project,operation="import",filename="ref.png",
                                                      media_type="image/png",chunks=[b"fixture"])
        rec=self.create(references={"images":[{"artifact_id":asset.artifact_id,"purpose":"subject"}]})
        self.assertFalse(self.uploads)
        self.assertEqual(self.poll(rec).status,"succeeded")
        self.assertEqual(self.payloads,[dict(request_id=rec.video_task_id,capability="video",model="veo-lite",
            parameters=dict(prompt="paper boat",duration=4,aspect="16:9",reference_images=self.uploads),budget_limit=6)])
        self.assertEqual(self.uploads,[asset.sha256])
        self.assertEqual(channel_label(task_channel(rec)),"Google Flow")

    def test_unknown_polls_original_job_and_rejects_changed_intent(self):
        self.accepted,self.stage=None,"dispatching"
        rec=self.create()
        self.assertEqual(self.create().video_task_id,rec.video_task_id)
        with self.assertRaises(TaskConflict):
            self.create(budget_limit="7")
        self.assertEqual(self.poll(rec).runtime_stage,"submission_unknown")
        self.assertEqual(self.poll(rec).runtime_stage,"submission_unknown")
        self.assertEqual(self.posts,1)

    def test_submit_response_lost_reopens_original_task(self):
        self.loss="submit"
        rec=self.create()
        self.assertEqual(self.poll(rec).runtime_stage,"submission_unknown")
        self.assertEqual(self.poll(rec).status,"succeeded")
        self.assertEqual(self.posts,1)

    def test_denied_project_and_foreign_org_never_reach_gateway(self):
        self.denied=True
        with self.assertRaises(PermissionError):
            self.create()
        token=CALLER.set({"organization_id":uuid7(),"user_id":self.user})
        try:
            with self.assertRaises(PermissionError):
                self.create()
        finally:
            CALLER.reset(token)
        self.assertEqual(self.checks,0)

    def check_artifact_loss(self,phase):
        self.loss=phase
        rec=self.create()
        partial=self.poll(rec)
        self.assertNotEqual(partial.status,"succeeded")
        self.assertTrue(self.lost)
        ready=self.poll(partial)
        self.assertEqual(ready.status,"succeeded")
        self.assertEqual(self.posts,1)
        self.assertEqual(self.reads,1)
        self.assertEqual(set(self.prepares),{partial.provider_state["artifact_write_id"]})
        self.assertEqual(ready.provider_state["artifact_upload_id"],self.upload)
        self.assertEqual(ready.provider_state["content_ref"],self.receipt)

    def test_prepare_response_lost(self):
        self.check_artifact_loss("prepare")

    def test_put_response_lost(self):
        self.check_artifact_loss("put")

    def test_commit_response_lost(self):
        self.check_artifact_loss("commit")
