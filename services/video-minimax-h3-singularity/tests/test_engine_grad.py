import concurrent.futures
from pathlib import Path
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("PyTorch is required for Engine gradient tests") from error

from h3_singularity.engine import Engine, RuntimeErrorCode


class EngineGradTest(unittest.TestCase):
    def test_dual_sigma_split_reuses_one_boundary(self):
        sigmas = torch.tensor([0.95, 0.70, 0.45, 0.20, 0.0], dtype=torch.float64)
        low, high = Engine._split_dual_sigmas(sigmas)
        self.assertEqual(low.tolist(), [0.95, 0.70, 0.45])
        self.assertEqual(high.tolist(), [0.45, 0.20, 0.0])

    def test_dual_sigma_split_rejects_non_four_nfe_path(self):
        with self.assertRaisesRegex(RuntimeErrorCode, "dual_sample_requires_5_sigma_points"):
            Engine._split_dual_sigmas(torch.tensor([1.0, 0.5, 0.0]))

    def test_worker_reference_outputs_do_not_retain_graphs(self):
        engine = object.__new__(Engine)
        encoder = torch.nn.Conv2d(3, 4, 3)

        def encode(*args):
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(torch.is_inference_mode_enabled())
            return [encoder(torch.ones(1, 3, 8, 8)).cpu() for _ in range(3)]

        engine._generate = encode

        def worker():
            with torch.enable_grad():
                outputs = engine.generate({}, {}, Path("unused.mp4"), lambda _: None)
                self.assertTrue(torch.is_grad_enabled())
                return outputs

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            outputs = pool.submit(worker).result()
        for latent in outputs:
            self.assertFalse(latent.requires_grad)
            self.assertIsNone(latent.grad_fn)
            self.assertFalse(torch.is_inference(latent))

    def test_oom_preserves_diagnostics_and_restores_grad_context(self):
        engine = object.__new__(Engine)
        engine._gpu_snapshot = lambda: {"1": {"allocated_bytes": 123}}

        def fail(*args):
            self.assertFalse(torch.is_grad_enabled())
            engine._active_stage = "vae_reference_encode"
            raise torch.cuda.OutOfMemoryError("test allocation failure")

        engine._generate = fail
        with torch.enable_grad():
            with self.assertRaises(RuntimeErrorCode) as caught:
                engine.generate({}, {}, Path("unused.mp4"), lambda _: None)
            self.assertTrue(torch.is_grad_enabled())
        self.assertEqual(caught.exception.code, "vae_reference_encode_out_of_memory")
        self.assertIsInstance(caught.exception.__cause__, torch.cuda.OutOfMemoryError)
        self.assertEqual(caught.exception.runtime_metrics["failure_gpu"]["1"]["allocated_bytes"], 123)


if __name__ == "__main__":
    unittest.main()
