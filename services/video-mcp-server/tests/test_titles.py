import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from app import server, titles
from app.tasks import TaskStore
import test_dashboard


class TitleTest(unittest.TestCase):
    setUp = test_dashboard.DashboardTest.setUp
    tearDown = test_dashboard.DashboardTest.tearDown

    def create(self, title=None, key=None):
        return self.client.post('/api/tasks', headers=self.headers, json={**self.payload, 'title': title,
            'idempotency_key': key or self.payload['idempotency_key']})

    def test_title_is_not_generation_input_or_idempotency(self):
        first = self.create('湖面漂流的纸船').json()
        raw = (self.tasks.root / (first['video_task_id'] + '.json')).read_bytes()
        second = self.create('其他名称').json()
        self.assertEqual(first['video_task_id'], second['video_task_id'])
        self.assertEqual(second['title'], first['title'])
        self.assertEqual(len(self.calls), 1)
        self.assertNotIn('title', json.loads(self.calls[0].content))
        self.assertNotIn('title', self.tasks.get(first['video_task_id']).request)
        duplicate = self.create(first['title'], 'another-key').json()
        self.assertNotEqual(first['video_task_id'], duplicate['video_task_id'])
        self.assertEqual(first['title'], duplicate['title'])
        store = TaskStore(self.tasks.root.parent)
        record = store.get(first['video_task_id'])
        titles.update(store, record, '手改的纸船', first['title_revision'])
        self.assertEqual(raw, (store.root / (record.video_task_id + '.json')).read_bytes())
        self.assertEqual(titles.project(TaskStore(store.root.parent), record)['title'], '手改的纸船')
        for query in ('手改的纸船', first['video_task_id'], self.payload['idempotency_key']):
            result = self.client.get('/api/dashboard', params={'q': query}, headers=self.headers).json()
            self.assertIn(record.video_task_id, [row['video_task_id'] for row in result['tasks']])

    def test_legacy_fallback_and_invalid_optional_title(self):
        for i, title in enumerate(('', '<b>bad</b>', 'one\ntwo', 12, 'a'*65)):
            response = self.create(title, str(i))
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()['title_source'], 'fallback')
            self.assertIn('title_warning', response.json())
        record = self.tasks.create(project_id='demo', idempotency_key='old', request={
            'prompt': '<Picture 1> video_task_'+'a'*32+' https://example.invalid/a 湖面漂流的纸船。镜头'},
            input_digest='original', runtime_task_id='original', status='queued')
        info = titles.project(self.tasks, record)
        self.assertEqual(info['title'], '湖面漂流的纸船')
        self.assertEqual(info['title_revision'], 0)

    def test_atomic_revision_and_processing_inheritance(self):
        task = self.create('湖面漂流的纸船').json()
        record = self.tasks.get(task['video_task_id'])
        def edit(name):
            try:
                return titles.update(TaskStore(self.tasks.root.parent), record, name, 1)
            except titles.TitleConflict:
                return 'conflict'
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(edit, ('甲', '乙')))
        self.assertEqual(results.count('conflict'), 1)
        record = self.tasks.update(record, artifact_id=self.image.artifact_id)
        child = self.tasks.create(project_id='demo', idempotency_key='processing', request={
            'service':'interpolate', 'source_artifact_id':self.image.artifact_id}, input_digest='process',
            runtime_task_id='process', status='queued')
        self.assertEqual(titles.project(self.tasks, child)['title'], titles.project(self.tasks, record)['title']+' · 补帧')
        foreign = self.tasks.create(project_id='foreign', idempotency_key='processing', request={
            'service':'interpolate', 'source_artifact_id':self.image.artifact_id}, input_digest='foreign',
            runtime_task_id='process', status='queued')
        self.assertEqual(titles.project(self.tasks, foreign)['title'], '视频补帧')

    def test_rename_requires_write_permission_and_exact_body(self):
        task = self.create('纸船').json()
        path = '/api/tasks/'+task['video_task_id']+'/title'
        body = {'title':'湖面漂流的纸船', 'expected_title_revision':1}
        self.assertEqual(self.client.post(path, json=body, headers=self.headers).status_code, 403)
        headers = {**self.headers, 'X-User-Id':'actor', 'X-Organization-Id':'organization'}
        with mock.patch('app.delivery.authorize', side_effect=PermissionError):
            self.assertEqual(self.client.post(path, json=body, headers=headers).status_code, 403)
        with mock.patch('app.delivery.authorize') as authorize:
            response = self.client.post(path, json=body, headers=headers)
            self.assertEqual(response.status_code, 200, response.text)
            authorize.assert_called_once_with(self.executor.client, 'demo')
            self.assertEqual(response.json()['title_revision'], 2)
            self.assertEqual(self.client.post(path, json=body, headers=headers).status_code, 409)
            for invalid in ({'title':'\n'}, {'title':'<b>x</b>'}, {'expected_title_revision':True}, {'extra':'bad'}):
                response = self.client.post(path, json={**body, **invalid}, headers=headers)
                self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(len(self.calls), 1)

    def test_mcp_create_status_and_schema_share_title(self):
        with mock.patch.object(server, 'executor', self.executor), mock.patch.object(server, 'tasks', self.tasks):
            args = {**self.payload, 'title':'纸船'}
            created = server.video_create(**args).structured_content
            status = server.video_status(created['video_task_id']).structured_content
            self.assertEqual(status['title'], created['title'])
        schema = next(s for s in server.VIDEO_TOOLS_SCHEMA if s['name']=='video.create')
        self.assertIn('title', schema['inputSchema']['properties'])
        self.assertIn('video.title_update', [s['name'] for s in server.VIDEO_TOOLS_SCHEMA])
