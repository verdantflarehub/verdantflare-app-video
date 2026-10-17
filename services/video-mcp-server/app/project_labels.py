"""Read project labels with the current Studio identity; never cache across callers."""
import os
import time
from collections import Counter

import httpx

from .delivery import CALLER, request_id

CATEGORIES = {
    'character': '角色设计', 'music': '音乐制作', 'video': '创意视频',
    'ai_short_drama': '短剧制作', 'creative_video': '创意视频制作',
}


def project_options(client, project_ids):
    ids = sorted(set(project_ids))
    metadata = {}
    actor = CALLER.get()
    url = os.environ.get('VIDEO_PROJECT_SERVICE_URL', '').rstrip('/')
    token = os.environ.get('VIDEO_PROJECT_SERVICE_TOKEN', '')
    if ids and isinstance(actor, dict) and all(actor.get(k) for k in ('user_id', 'organization_id')) and url and token:
        cursor, seen = '', set()
        deadline = time.monotonic() + 8
        try:
            for _ in range(100):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                response = client.post(url + '/internal/v1/project/list', headers={
                    'Authorization': 'Bearer ' + token, 'X-User-Id': actor['user_id'],
                    'X-Organization-Id': actor['organization_id'], 'X-Request-Id': request_id(),
                }, json={'require_write': False, 'limit': 100, 'cursor': cursor}, timeout=min(5, remaining))
                response.raise_for_status()
                data = response.json()
                for item in data.get('items', []):
                    if isinstance(item, dict) and item.get('project_id') in ids:
                        name, category = item.get('name'), item.get('category')
                        if isinstance(name, str) and name.strip() and isinstance(category, str):
                            metadata[item['project_id']] = (name.strip(), category)
                cursor = data.get('next_cursor')
                if not cursor or not isinstance(cursor, str) or cursor in seen or len(metadata) == len(ids):
                    break
                seen.add(cursor)
        except (httpx.HTTPError, ValueError, TypeError, AttributeError):
            metadata = {}
    options = []
    for index, project_id in enumerate(ids, 1):
        title, category = metadata.get(project_id, (f'历史项目 {index}（名称不可用）', ''))
        category_label = CATEGORIES.get(category, '未分类')
        options.append({'id': project_id, 'category': category, 'category_label': category_label,
                        'title': title, 'label': f'{category_label} · {title}'})
    counts = Counter(item['label'] for item in options)
    positions = Counter()
    for item in options:
        label = item['label']
        if counts[label] > 1:
            positions[label] += 1
            item['label'] += f'（同名项目 {positions[label]}）'
    return options
