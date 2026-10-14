from datetime import UTC, datetime, timedelta
import json
import unittest
from unittest.mock import patch

import httpx
from starlette.applications import Starlette
from starlette.testclient import TestClient

import test_higgsfield as fixture
from app.dashboard import Dashboard
from app.member_dashboard import MemberDashboard
from app.delivery import CALLER


class MemberDashboardTest(unittest.TestCase):
    def setUp(self):
        fixture.HiggsfieldTest.setUp(self)
        self.member = MemberDashboard(Dashboard(self.executor))
        self.http = TestClient(Starlette(routes=self.member.routes()))
        self.price = 2
        self.body = {'project_id':fixture.PROJECT,'idempotency_key':'member/one',
                     'title':'测试视频','model':'grok_video_v15_lite','route':'higgsfield',
                     'prompt':'paper boat','duration_seconds':2,'aspect_ratio':'16:9',
                     'resolution':'480p','references':{'images':[],'videos':[],'audios':[]}}

    def tearDown(self):
        self.http.close()
        fixture.HiggsfieldTest.tearDown(self)

    def respond(self, request):
        if request.url.path == '/v1/cost':
            now = datetime.now(UTC)
            return httpx.Response(200,json={'credits':self.price,'can_submit':True,'blocked_reason':None,
                'observed_at':now.isoformat(),'expires_at':(now+timedelta(seconds=60)).isoformat()})
        return fixture.HiggsfieldTest.respond(self, request)

    def post(self, path, body=None):
        return self.http.post('/api/higgsfield/'+path, json=self.body if body is None else body)

    def test_quote_creates_no_task_and_changed_input_requires_new_quote(self):
        self.assertEqual(self.post('preflight').status_code,200)
        self.assertEqual(self.member.dashboard.records(),[])
        self.body['prompt']='different boat'
        response=self.post('tasks')
        self.assertEqual(response.status_code,409)
        self.assertEqual(response.json()['error'],'quote_changed')
        self.assertEqual(self.member.dashboard.records(),[])

    def test_submit_is_durable_and_recovers_once_after_quote_expiry(self):
        self.assertEqual(self.post('preflight').status_code,200)
        first=self.post('tasks')
        self.assertEqual(first.status_code,200,first.text)
        self.price=9
        second=self.post('tasks')
        self.assertEqual(second.json()['video_task_id'],first.json()['video_task_id'])
        record=self.executor.tasks.get(first.json()['video_task_id'])
        self.assertEqual(record.request['price_ceiling'],2)
        self.assertEqual(self.executor.higgsfield.payload(record)['price_ceiling'],2)
        self.assertEqual(len(self.member.dashboard.records()),1)
        self.assertEqual(self.post('attempt', {k:self.body[k] for k in ('project_id','idempotency_key')}).json()['state'],'found')

    def test_quote_price_increase_does_not_submit(self):
        self.post('preflight')
        self.price=3
        self.assertEqual(self.post('tasks').json()['error'],'quote_changed')
        self.assertEqual(self.member.dashboard.records(),[])

    def test_cross_project_and_missing_identity_cannot_resolve_assets_or_quote(self):
        self.deny_project=True
        self.assertEqual(self.post('preflight').status_code,403)
        self.assertEqual(self.http.get('/api/higgsfield/assets?project_id='+fixture.PROJECT).status_code,403)
        marker=CALLER.set(None)
        try:
            self.assertEqual(self.http.get('/api/higgsfield/capabilities').status_code,403)
        finally:
            CALLER.reset(marker)

    def test_attempt_query_never_creates_and_rejects_other_channel(self):
        self.assertEqual(self.post('attempt', {k:self.body[k] for k in ('project_id','idempotency_key')}).json(),{'state':'not_found'})
        self.assertEqual(self.member.dashboard.records(),[])
        record=self.executor.tasks.create(project_id=fixture.PROJECT,idempotency_key='other',request={},input_digest='x',runtime_task_id='r',status='succeeded')
        self.assertEqual(self.http.get('/api/higgsfield/tasks/'+record.video_task_id).status_code,404)
        self.assertEqual(self.http.get('/api/higgsfield/tasks').json(),{'tasks':[]})

    def test_expired_quote_and_wrong_method_are_rejected(self):
        self.post('preflight')
        with self.member.quotes() as db:
            identity=self.member.identity(fixture.PROJECT,'member/one')
            value=json.loads(db.execute('SELECT value FROM quotes WHERE identity=?',(identity,)).fetchone()[0])
            value['quote']['expires_at']='2000-01-01T00:00:00+00:00'
            db.execute('UPDATE quotes SET value=? WHERE identity=?',(json.dumps(value),identity))
        self.assertEqual(self.post('tasks').json()['error'],'quote_expired')
        self.assertEqual(self.http.get('/api/higgsfield/preflight').status_code,405)
        self.assertEqual(self.http.get('/api/higgsfield/account').status_code,404)
