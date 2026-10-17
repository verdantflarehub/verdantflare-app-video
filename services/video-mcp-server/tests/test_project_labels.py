import json
import os
import unittest
from unittest.mock import patch

import httpx

from app.delivery import CALLER
from app.project_labels import project_options


class ProjectLabelsTest(unittest.TestCase):
    def setUp(self):
        self.actor = CALLER.set({'user_id': 'user-a', 'organization_id': 'org-a'})
        self.env = patch.dict(os.environ, {'VIDEO_PROJECT_SERVICE_URL': 'https://project.test',
                                         'VIDEO_PROJECT_SERVICE_TOKEN': 'test-token'})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        CALLER.reset(self.actor)

    def test_read_only_pagination_authorized_labels_and_duplicates(self):
        calls = []
        def handler(request):
            body = json.loads(request.content)
            self.assertNotIn('require_write', body)
            self.assertTrue(body.get('cursor', 'initial'))
            self.assertEqual(request.headers['x-user-id'], 'user-a')
            self.assertEqual(request.headers['x-organization-id'], 'org-a')
            calls.append(body.get('cursor', ''))
            return httpx.Response(200, json={'items': [
                {'project_id': 'a' if not body.get('cursor', '') else 'b', 'name': '最后一块冰', 'category': 'creative_video'},
                {'project_id': 'unrelated', 'name': '不应返回', 'category': 'music'}],
                'next_cursor': 'page2' if not body.get('cursor', '') else ''})
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            items = project_options(client, ['b', 'a', 'missing'])
        self.assertEqual(calls, ['', 'page2'])
        self.assertEqual([x['id'] for x in items], ['a', 'b', 'missing'])
        self.assertEqual(items[0]['label'], '创意视频制作 · 最后一块冰（同名项目 1）')
        self.assertEqual(items[1]['label'], '创意视频制作 · 最后一块冰（同名项目 2）')
        self.assertEqual(items[2]['label'], '未分类 · 历史项目 3（名称不可用）')

    def test_no_identity_makes_no_request(self):
        CALLER.set(None)
        def handler(request):
            self.fail('must not read Project without verified identity')
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            self.assertIn('名称不可用', project_options(client, ['id'])[0]['label'])

    def test_permission_revocation_has_no_cached_title(self):
        def handler(request):
            if request.headers['x-user-id'] == 'user-a':
                return httpx.Response(200, json={'items': [{'project_id': 'a', 'name': '私有标题', 'category': 'music'}]})
            return httpx.Response(403)
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            self.assertEqual(project_options(client, ['a'])[0]['title'], '私有标题')
            CALLER.set({'user_id': 'user-b', 'organization_id': 'org-b'})
            self.assertNotIn('私有标题', project_options(client, ['a'])[0]['label'])

    def test_malformed_service_response_and_repeated_cursor_are_bounded(self):
        for payload in (None, {'items': None}, {'items': [], 'next_cursor': 'same'}):
            calls = []
            def handler(request):
                calls.append(request)
                return httpx.Response(200, json=payload) if payload is not None else httpx.Response(200, content='invalid')
            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                self.assertIn('名称不可用', project_options(client, ['a'])[0]['label'])
            self.assertLessEqual(len(calls), 2)
