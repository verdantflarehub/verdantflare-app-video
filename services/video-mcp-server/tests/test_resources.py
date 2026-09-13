from __future__ import annotations
import copy
import unittest
from unittest import mock

from verdantflare_video_mcp.resources import Resources, parse_metrics

GPU1 = 'GPU-11111111-1111-1111-1111-111111111111'
GPU2 = 'GPU-22222222-2222-2222-2222-222222222222'


def fixture():
    deployment = {'metadata': {'name': 'video-minimax-h3-api', 'uid': 'deployment-1'}, 'spec': {'replicas': 2}}
    rs = {'metadata': {'uid': 'rs-1', 'ownerReferences': [{'kind': 'Deployment', 'uid': 'deployment-1', 'controller': True}]}}
    pod = {'metadata': {'name': 'h3-instance', 'uid': 'pod-1', 'ownerReferences': [{'kind': 'ReplicaSet', 'uid': 'rs-1', 'controller': True}],
                        'annotations': {'hami.io/vgpu-devices-allocated': f'{GPU1},NVIDIA,24564,0:{GPU2},NVIDIA,24564,0:;'}},
           'spec': {'nodeName': 'node-a', 'containers': [{'image': 'registry.example/video:h3-v1.0.0'}]},
           'status': {'phase': 'Running', 'conditions': [{'type': 'Ready', 'status': 'True'}], 'containerStatuses': [{'restartCount': 0}]}}
    return [[deployment], [rs], [pod]]


def metrics():
    return '\n'.join(f'{name}{{UUID="{uid}",modelName="RTX 4090"}} {value}' for uid in (GPU1, GPU2) for name, value in (
        ('DCGM_FI_DEV_GPU_UTIL', 0), ('DCGM_FI_DEV_FB_USED', 18432), ('DCGM_FI_DEV_FB_FREE', 6144),
        ('DCGM_FI_DEV_GPU_TEMP', 61), ('DCGM_FI_DEV_POWER_USAGE', 230)))


class Source:
    def __init__(self):
        self.data, self.text = fixture(), metrics()
    def inventory(self):
        return copy.deepcopy(self.data)
    def metrics(self):
        return self.text


class ResourcesTest(unittest.TestCase):
    def setUp(self):
        self.now = 100000
        self.source = Source()
        self.resources = Resources(self.source, clock=lambda: self.now)
        self.resources.sync_inventory()
        self.resources.sync_metrics()

    def test_counts_owner_chain_and_gpu_assignment(self):
        h3, sol, b4090 = self.resources.snapshot()['models']
        self.assertEqual((h3['ready'], h3['current'], h3['desired'], h3['deployment_status']), (1, 1, 2, 'partial'))
        self.assertEqual(sol['deployment_status'], 'not_deployed')
        self.assertEqual((b4090['route'], b4090['model_type'], b4090['deployment_status']), ('h3-sol-4090', 'minimax-h3-ref2va', 'not_deployed'))
        rogue = copy.deepcopy(self.source.data[2][0]); rogue['metadata']['uid'] = 'rogue'
        rogue['metadata']['ownerReferences'][0]['uid'] = 'unrelated-rs'
        self.source.data[2].append(rogue)
        self.resources.sync_inventory()
        self.assertEqual(self.resources.snapshot()['models'][0]['current'], 1)
        detail = self.resources.instance('h3', 'pod-1')
        self.assertEqual(len(detail['gpus']), 2)
        self.assertEqual(detail['gpus'][0]['metrics']['memory_total_gib'], 24)
        self.assertEqual(detail['gpus'][0]['metrics']['utilization_percent'], 0)

    def test_unknown_does_not_become_zero_or_false_assignment(self):
        with mock.patch.object(self.source, 'inventory', side_effect=OSError):
            self.resources.sync_inventory()
        h3 = self.resources.snapshot()['models'][0]
        self.assertEqual(h3['deployment_status'], 'unknown')
        self.assertIsNone(h3['current'])
        self.assertEqual(self.resources.get_instances('h3')['instances'], [])
        self.assertIsNone(self.resources.gpu('h3', 'pod-1', GPU1, 15)['metrics'])

    def test_zero_replicas_and_terminated_pods(self):
        self.source.data[0][0]['spec']['replicas'] = 0
        self.source.data[2][0]['metadata']['deletionTimestamp'] = '2026-01-01T00:00:00Z'
        self.resources.sync_inventory()
        h3 = self.resources.snapshot()['models'][0]
        self.assertEqual((h3['current'], h3['deployment_status']), (0, 'scaled_zero'))
        self.assertFalse(self.resources.history)

    def test_rollout_surge_and_pending_instance(self):
        pending = copy.deepcopy(self.source.data[2][0]); pending['metadata']['uid'] = 'pod-2'
        pending['status'] = {'phase': 'Pending'}
        self.source.data[2].append(pending)
        self.source.data[0][0]['spec']['replicas'] = 1
        self.resources.sync_inventory()
        h3 = self.resources.snapshot()['models'][0]
        self.assertEqual((h3['ready'],h3['current'],h3['desired'],h3['deployment_status']), (1,2,1,'partial'))
        self.source.data[2][1]['status'] = {'phase':'Running','conditions':[{'type':'Ready','status':'True'}]}
        self.resources.sync_inventory()
        self.assertEqual(self.resources.snapshot()['models'][0]['deployment_status'], 'online')

    def test_invalid_values_and_foreign_uuid_do_not_leak(self):
        text = metrics() + f'\nDCGM_FI_DEV_GPU_UTIL{{UUID="{GPU1}"}} 99'
        parsed = parse_metrics(text, {GPU1}, self.now)
        self.assertNotIn(GPU2, parsed)
        self.assertNotIn('utilization_percent', parsed[GPU1])
        for val in ['NaN', 'Inf', '-1', '9223372036854775794']:
            row = parse_metrics(f'DCGM_FI_DEV_GPU_UTIL{{UUID="{GPU1}"}} {val}', {GPU1}, self.now)
            self.assertFalse(row)
        self.assertFalse(parse_metrics(f'DCGM_FI_DEV_GPU_UTIL{{UUID="{GPU1}"}} 20 1000', {GPU1}, self.now))

    def test_missing_metrics_gap_stale_and_bounded_history(self):
        self.source.text = ''
        self.now += 40
        self.resources.sync_inventory(); self.resources.sync_metrics()
        row = self.resources.gpu('h3', 'pod-1', GPU1, 15)
        self.assertEqual(row['state'], 'stale')
        self.assertIsNone(row['metrics'])
        self.assertIsNone(row['history'][-1]['utilization_percent'])
        self.source.text = metrics()
        for _ in range(365):
            self.now += 10
            self.resources.sync_inventory(); self.resources.sync_metrics()
        self.assertEqual(len(self.resources.gpu('h3','pod-1',GPU1,60)['history']),360)
        self.assertLessEqual(len(self.resources.gpu('h3','pod-1',GPU1,15)['history']),91)

    def test_foreign_model_instance_and_gpu_paths_rejected(self):
        for args in [('other','pod-1',GPU1),('h3-sol','pod-1',GPU1),('h3','foreign',GPU1),('h3','pod-1','GPU-foreign')]:
            with self.assertRaises(KeyError):self.resources.gpu(*args,15)

    def test_new_api_auth_and_window_validation(self):
        import os
        from starlette.testclient import TestClient
        from starlette.applications import Starlette
        from verdantflare_video_mcp.server import BearerAuthMiddleware
        app=Starlette(routes=self.resources.routes()); app.add_middleware(BearerAuthMiddleware)
        with mock.patch.dict(os.environ,{'VIDEO_MCP_BEARER_TOKEN':'test-only'}),TestClient(app) as client:
            paths=['/api/mcp/status','/api/models','/api/models/h3/instances', '/api/models/h3/instances/pod-1',f'/api/models/h3/instances/pod-1/gpus/{GPU1}']
            for path in paths:
                self.assertEqual(client.get(path).status_code,401)
                response=client.get(path,headers={'Authorization':'Bearer test-only'})
                self.assertEqual(response.status_code,200,response.text)
                self.assertEqual(response.headers['cache-control'],'no-store')
            self.assertEqual(client.get(paths[-1]+'?window=arbitrary',headers={'Authorization':'Bearer test-only'}).status_code,400)
