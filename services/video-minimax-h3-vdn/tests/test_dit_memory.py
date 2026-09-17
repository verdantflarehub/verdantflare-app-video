import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
try:
    import torch
except ImportError:
    torch=None
from vdn_dit_memory import project_bank, gate_heads, install_dit_memory, validate_geometry

@unittest.skipIf(torch is None,'Pinned PyTorch required')
class BankTests(unittest.TestCase):
    def test_project_complete_rows_and_add_only_target_segment(self):
        torch.manual_seed(43)
        linear=torch.nn.Linear(12,8)
        bank=torch.randn(11,12)
        calls=[]
        def project(rows):
            calls.append(tuple(rows.shape))
            return linear(rows)
        with torch.inference_mode():
            expected=linear(bank)
            actual=project_bank(project,bank,'cpu',bank.dtype,4)
            torch.testing.assert_close(actual,expected)
            self.assertEqual(calls,[(4,12),(4,12),(3,12)])
            target=torch.randn(19,8);before=target.clone()
            result=project_bank(project,bank,'cpu',bank.dtype,4,output=target,start_row=3)
            self.assertIs(result,target)
            torch.testing.assert_close(target[3:14],before[3:14]+expected)
            self.assertTrue(torch.equal(target[:3],before[:3]))
            self.assertTrue(torch.equal(target[14:],before[14:]))

    def test_gate_chunks_keep_full_input_width_and_tail_heads(self):
        class Gate(torch.nn.Module):
            head_dim=2
            def forward(self,x):return torch.sigmoid(x).view(-1,3,2)
        x=torch.randn(13,6)
        with torch.inference_mode():
            actual=gate_heads(Gate(),x,slice(1,3),5)
        torch.testing.assert_close(actual,Gate()(x)[:,1:3])

    def test_reject_nonpositive_and_boolean_groups(self):
        for h,t in [(0,512),(4,0),(True,512),(4,-1)]:
            with self.assertRaises(ValueError):install_dit_memory(None,None,h,t)

    def test_geometry_accepts_distinct_trunk_qkv_widths_and_rejects_bad_projections(self):
        from types import SimpleNamespace
        attention=SimpleNamespace(num_heads=3,head_dim=6,
            orig=SimpleNamespace(to_q=torch.nn.Linear(12,18),to_k=torch.nn.Linear(12,18),
                 to_v=torch.nn.Linear(12,18),to_out=[torch.nn.Linear(18,12)]),
            linear_attention=SimpleNamespace(num_heads=3,head_dim=6),
            to_out_linear=torch.nn.Linear(18,12))
        validate_geometry(attention)
        attention.orig.to_k=torch.nn.Linear(11,18)
        with self.assertRaisesRegex(RuntimeError,'QKV projection'):validate_geometry(attention)
        attention.orig.to_k=torch.nn.Linear(12,18)
        attention.to_out_linear=torch.nn.Linear(18,13)
        with self.assertRaisesRegex(RuntimeError,'branch projection'):validate_geometry(attention)
