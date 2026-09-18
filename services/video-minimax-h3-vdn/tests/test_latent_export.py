import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file
from latent_export import ExportingPipeline, finalize


class LatentExportTest(unittest.TestCase):
    def test_export_failure_preserves_source_outputs_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            def pipeline(**kwargs):
                calls.append(kwargs)
                return {'videos': 'original', 'audio': 'source-audio'}
            exporter = ExportingPipeline(pipeline, tmp)
            result = exporter(output=['videos', 'audio'])
            self.assertEqual(result, {'videos': 'original', 'audio': 'source-audio'})
            self.assertEqual(len(calls), 1)
            self.assertEqual(exporter.export_error, 'KeyError')
            self.assertFalse((Path(tmp) / 'latent-bundle/manifest.json').exists())

    def test_single_generation_preserves_original_outputs_and_exports_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = {'normalized_references': [], 'videos': 'original-videos', 'audio': 'original-audio', 'sampling_rate': 32000,
                      'latents': torch.zeros(1, 24, 37, 2, 4),
                      'audio_latents': torch.arange(2 * 32 * 207, dtype=torch.float32).reshape(2, 32, 207),
                      'prompt_embeds': torch.ones(3, 4), 'text_token_tags': torch.zeros(3, dtype=torch.long),
                      'condition_latents': [torch.ones(1, 24, 1, 2, 2)],
                      'audio_condition_latents': [], 'position_ids': torch.zeros(3, 3),
                      'token_tags': torch.zeros(3, dtype=torch.long),
                      'video_indices': torch.zeros(1, dtype=torch.long),
                      'audio_indices': torch.zeros(1, dtype=torch.long),
                      'text_indices': torch.zeros(1, dtype=torch.long),
                      'num_condition_video_rows': 4, 'num_condition_audio_rows': 0,
                      'height': 1344, 'width': 768, 'num_frames': 124, 'num_audio_latents': 207}
            calls = []
            def pipeline(**kwargs):
                calls.append(kwargs)
                return {name: output[name] for name in kwargs['output']}
            result = ExportingPipeline(pipeline, root)(output=['videos', 'audio', 'sampling_rate'])
            self.assertEqual(len(calls), 1)
            self.assertEqual(result, {name: output[name] for name in ('videos', 'audio', 'sampling_rate')})
            audio = load_file(root / 'latent-bundle/audio.safetensors')['samples']
            self.assertTrue(torch.equal(audio[0, :, 0], output['audio_latents'][0]))
            self.assertTrue(torch.equal(audio[0, :, 1], output['audio_latents'][1]))
            conditions = load_file(root / 'latent-bundle/conditions.safetensors')
            self.assertTrue(torch.equal(conditions['condition_latents.0'], output['condition_latents'][0]))
            self.assertTrue(torch.equal(conditions['prompt_embeds'], output['prompt_embeds']))
            (root / 'video.mp4').write_bytes(b'media-validation-is-performed-before-finalize')
            task = {'id': 'vdn_' + 'b' * 32, 'idempotency_key': 'video_task_' + 'a' * 32,
                    'request': {'project_id': 'project-a', 'conditions': []}}
            media = {'streams': [{'codec_type': 'video', 'nb_read_frames': '124',
                                  'width': 768, 'height': 1344, 'avg_frame_rate': '24/1'}]}
            with patch.dict(os.environ, {'VDN_PROJECTS_ROOT': tmp, 'VDN_NODE_NAME': 'node-a'}):
                descriptor = finalize(root, task, {'model_revision': 'fixed'}, media)
            manifest = json.loads((root / descriptor['manifest_path']).read_text())
            self.assertEqual(manifest['source_video_task_id'], task['idempotency_key'])
            self.assertEqual(manifest['media']['frames'], 124)
            self.assertEqual(manifest['media']['width'], 768)
            self.assertEqual(manifest['media']['height'], 1344)
            self.assertEqual(manifest['media']['fps'], '24/1')
            self.assertEqual(manifest['project_id'], 'project-a')
            with patch.dict(os.environ, {'VDN_PROJECTS_ROOT': tmp, 'VDN_NODE_NAME': 'node-a'}):
                with self.assertRaises(FileExistsError):
                    finalize(root, task, {}, media)


if __name__ == '__main__':
    unittest.main()
