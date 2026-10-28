from datetime import UTC, datetime, timedelta
import json
import unittest
from unittest.mock import patch

import httpx
from starlette.applications import Starlette
from starlette.testclient import TestClient

import test_higgsfield as fixture
from app.dashboard import Dashboard
from app.channel_errors import ChannelError
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

    def test_disabled_model_rejects_quote_and_new_submit_before_supplier_call(self):
        asset = self.executor.artifacts.create_from_chunks(project_id=fixture.PROJECT, operation='import',
            filename='identity.png', media_type='image/png', chunks=[b'synthetic-media'])
        self.body.update(model='seedance_2_5_draft', duration_seconds=4, mode='reference_to_video',
                         references={'images':[{'artifact_id':asset.artifact_id, 'purpose':'reference'}]})
        error = ChannelError('provider_not_found', source='higgsfield', accepted=False)
        self.registry_gate.side_effect = error
        with patch.object(self.executor.higgsfield, 'quote') as quote:
            for endpoint in ('preflight', 'tasks'):
                response = self.post(endpoint)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(response.json(), {'error':error.public()})
            quote.assert_not_called()
        self.registry_gate.assert_called_with('higgsfield', 'seedance_2_5_draft')
        self.assertEqual(self.member.dashboard.records(), [])

    def test_existing_task_can_be_recovered_after_model_is_disabled(self):
        self.post('preflight')
        first = self.post('tasks').json()
        self.registry_gate.side_effect = ChannelError('provider_not_found', source='higgsfield', accepted=False)
        with patch.object(self.executor.higgsfield, 'quote') as quote:
            recovered = self.post('tasks')
            self.assertEqual(recovered.status_code, 200, recovered.text)
            self.assertEqual(recovered.json()['video_task_id'], first['video_task_id'])
            quote.assert_not_called()

    def test_channel_errors_retain_safe_envelope_in_member_requests(self):
        from app.channel_errors import ChannelError
        error = ChannelError('limits_unavailable', source='higgsfield', http_status=503, accepted=False)
        with patch.object(self.member, 'dispatch', side_effect=error):
            response = self.post('preflight')
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {'error': error.public()})
        self.assertEqual(response.headers['cache-control'], 'no-store')

    def test_member_models_use_live_effective_routes_and_do_not_fallback_on_missing_directory(self):
        registered = {'route':'higgsfield', 'model':'seedance_2_5', 'display_name':'Registered name',
                      'can_submit':False, 'modes':['text_to_video'], 'capability_id':'registered-capability'}
        with patch.object(self.executor, 'capabilities', return_value={'routes':[registered,
                {'route':'google','model':'unrelated'}]}):
            value = self.http.get('/api/higgsfield/capabilities').json()
        self.assertEqual(value, {'routes':[registered]})
        with patch.object(self.executor, 'capabilities', return_value={'routes':[]}):
            self.assertEqual(self.http.get('/api/higgsfield/capabilities').json(), {'routes':[]})

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

    def test_pending_result_returns_original_task_without_creating_again(self):
        self.post('preflight')
        first = self.post('tasks').json()
        with patch.object(self.executor.higgsfield, 'status', side_effect=lambda record: record):
            reply = self.post('tasks/' + first['video_task_id'] + '/result', {})
        self.assertEqual(reply.status_code, 200)
        self.assertEqual(reply.json()['video_task_id'], first['video_task_id'])
        self.assertEqual(len(self.member.dashboard.records()), 1)

    def test_project_listing_omits_empty_cursor_and_always_requires_write(self):
        with patch.object(self.executor.higgsfield,'project_request',return_value={'items':[], 'next_cursor':''}) as listing:
            response = self.http.get('/api/higgsfield/projects')
            self.assertEqual(response.status_code,200)
            listing.assert_called_with('list',{'require_write':True,'limit':50})
            self.http.get('/api/higgsfield/projects?cursor=next')
            listing.assert_called_with('list',{'require_write':True,'limit':50,'cursor':'next'})

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
