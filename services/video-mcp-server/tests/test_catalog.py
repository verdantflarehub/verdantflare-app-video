import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from app.catalog import ModelCatalog


def channel(channel_id):
    return {'schema_version':1,'channel_version':'0.1.0','updated_on':'2026-10-14',
            'channel':{'id':channel_id,'display_name':channel_id,'kind':'local','sort_order':0},
            'sources':[{'id':'source','kind':'adapter_or_design_declaration',
                        'evidence_ref':'evidence/test.json','sha256':'a'*64}],
            'models':[{'id':'h3','model_id':'minimax-h3','display_name':'MiniMax H3',
                'entry_kind':'model','business_type':'video_generation','description':'Reference video',
                'sort_order':0,'identity_review':'pending','upstream_model_id':'minimax-h3-ref2va',
                'catalog_state':'listed','supplier_profiles':[],
                'team_profiles':[{'id':'reference','mode':'reference_to_video','resolutions':['768p'],
                    'max_resolution':'768p','duration_seconds':{'minimum':4,'maximum':15},
                    'duration_choices_seconds':[],'aspect_ratios':['16:9'],'input_limits':None,
                    'audio_input':'unknown','audio_output':'unknown','review_status':'pending',
                    'verification_status':'unverified','source_refs':['source']}],
                'billing_modes':['local_resources'],'source_refs':['source'],'review_notes':['private note']} ]}


class CatalogTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, data, decision='approved', suffix=''):
        name = data['channel']['id'] + suffix
        raw = yaml.safe_dump(data,sort_keys=False).encode()
        digest = hashlib.sha256(raw).hexdigest()
        (self.root/(name+'.yaml')).write_bytes(raw)
        review = {'schema_version':1,'channel_id':data['channel']['id'],'channel_version':'0.1.0',
            'candidate_path':name+'.yaml','candidate_key':None,'candidate_sha256':digest,
            'decision':decision,'scope':'complete_channel','approved_by':'test-user','approved_at':'2026-10-14',
            'decision_evidence_ref':'evidence/decision.md','review_notes_ref':'private-notes.md'}
        (self.root/(name+'.review.yaml')).write_text(yaml.safe_dump(review),'utf-8')
        return {'channel_id':data['channel']['id'],'channel_version':'0.1.0','sha256':digest,
                'catalog_file':name+'.yaml','review_file':name+'.review.yaml'}

    def load(self, entries):
        (self.root/'channels.yaml').write_text(yaml.safe_dump({'schema_version':1,'channels':entries}),'utf-8')
        return ModelCatalog(self.root).snapshot()

    def test_projection_and_snapshot_are_immutable(self):
        entry = self.write(channel('local'))
        self.load([entry])
        catalog = ModelCatalog(self.root)
        value = catalog.snapshot()
        self.assertEqual(value['state'],'ready')
        text = json.dumps(value)
        for private in ('evidence/', 'private note', 'test-user','"source_refs"','"sources"'):
            self.assertNotIn(private,text)
        value['channels'].clear()
        self.assertEqual(len(catalog.snapshot()['channels']),1)

    def test_channel_removal_recomputes_groups(self):
        local, hf = channel('local'), channel('higgsfield')
        local['models'][0]['identity_review'] = hf['models'][0]['identity_review'] = 'approved'
        a,b = self.write(local),self.write(hf)
        both = self.load([a,b])
        self.assertEqual(len(both['groups']),1)
        self.assertEqual(len(both['groups'][0]['entries']),2)
        remaining = self.load([a])
        self.assertEqual([c['id'] for c in remaining['channels']],['local'])
        self.assertEqual(len(remaining['groups'][0]['entries']),1)
        self.assertNotEqual(both['snapshot_id'],remaining['snapshot_id'])

    def test_pending_identity_never_merges(self):
        data = self.load([self.write(channel('local')),self.write(channel('higgsfield'))])
        self.assertEqual(len(data['groups']),2)

    def test_conflicts_are_not_overwritten(self):
        a,b = channel('local'),channel('higgsfield')
        a['models'][0]['identity_review'] = b['models'][0]['identity_review'] = 'approved'
        b['models'][0]['display_name'] = 'Different product'
        data = self.load([self.write(a),self.write(b)])
        self.assertEqual(data['conflicts'],['minimax-h3'])
        self.assertEqual(len(data['groups']),2)

    def test_missing_required_channel_is_error_not_removal(self):
        a,b = self.write(channel('local')),self.write(channel('higgsfield'))
        (self.root/b['catalog_file']).unlink()
        data = self.load([a,b])
        self.assertEqual(data['state'],'error')
        self.assertEqual(data['channels'],[])
        self.assertNotIn(str(self.root),json.dumps(data))

    def test_empty_selection_and_retired_entries(self):
        self.assertEqual(self.load([])['state'],'empty')
        data = channel('local'); data['models'][0]['catalog_state']='retired'
        self.assertEqual(self.load([self.write(data)])['groups'],[])

    def test_tampering_and_unapproved_review_rejected(self):
        a = self.write(channel('local'))
        with (self.root/a['catalog_file']).open('ab') as f: f.write(b'\n')
        self.assertEqual(self.load([a])['state'],'error')
        for decision in ('pending','rejected'):
            self.assertEqual(self.load([self.write(channel('local'),decision)])['state'],'error')

    def test_duplicate_keys_and_aliases_rejected_even_with_valid_hash(self):
        for extra in ('schema_version: 1\n','extra: &x [1]\nalias: *x\n'):
            a = self.write(channel('local'))
            path = self.root/a['catalog_file']
            path.write_bytes(path.read_bytes()+extra.encode())
            a['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(self.load([a])['state'],'error')

    def test_duplicate_channel_and_path_escape_rejected(self):
        a = self.write(channel('local'))
        self.assertEqual(self.load([a,a])['state'],'error')
        a['catalog_file']='../outside.yaml'
        self.assertEqual(self.load([a])['state'],'error')

    def test_semantics_and_unknown_fields_rejected(self):
        for mutation in ('unknown-source','bad-duration','bad-resolution','extra-field','duplicate-entry'):
            data = channel('local'); m=data['models'][0]; p=m['team_profiles'][0]
            if mutation=='unknown-source': p['source_refs']=['unknown']
            elif mutation=='bad-duration': p['duration_seconds']['maximum']=2
            elif mutation=='bad-resolution': p['max_resolution']='1080p'
            elif mutation=='extra-field': m['credential']='must never be accepted'
            elif mutation=='duplicate-entry': data['models'].append(copy.deepcopy(m))
            self.assertEqual(self.load([self.write(data)])['state'],'error',mutation)


if __name__ == '__main__':
    unittest.main()
