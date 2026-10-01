import sys
import types
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("PyTorch is required for low VRAM tests") from error

from h3_singularity.lowvram import (
    CoffH3ModelLoader,
    Q_CHUNKS,
    _is_oom,
    configure_comfy_low_vram,
)


class _FakePatcher:
    def __init__(self):
        self.load_device = None
        self.offload_device = None


class _FakeModel:
    def __init__(self):
        self.patcher = _FakePatcher()


class _FakeLoader:
    def load_unet(self, name, dtype):
        self.name = name
        self.dtype = dtype
        return (_FakeModel(),)


class _FakeNodes:
    UNETLoader = _FakeLoader


class LowVramTest(unittest.TestCase):
    def test_loader_sets_cpu_offload_and_records_contract(self):
        model, evidence = CoffH3ModelLoader(
            _FakeNodes(), "model.safetensors", torch.device("cuda:0")
        ).load("auto")
        self.assertEqual(str(model.patcher.load_device), "cuda:0")
        self.assertEqual(str(model.patcher.offload_device), "cpu")
        self.assertEqual(evidence["loader"], "CoffH3ModelLoader")
        self.assertEqual(evidence["backend"], "auto")

    def test_backend_rejects_silent_fallback(self):
        with self.assertRaises(ValueError):
            CoffH3ModelLoader(_FakeNodes(), "model.safetensors", torch.device("cpu")).load("split")

    def test_chunk_contract_and_oom_detection(self):
        self.assertEqual(Q_CHUNKS, (8192, 4096, 2048, 1024))
        self.assertTrue(_is_oom(RuntimeError("CUDA out of memory")))
        self.assertFalse(_is_oom(RuntimeError("kernel launch failed")))

    def test_configure_supports_aggressive_no_vram_state(self):
        class _State:
            LOW_VRAM = types.SimpleNamespace(name="LOW_VRAM")
            NO_VRAM = types.SimpleNamespace(name="NO_VRAM")

        management = types.SimpleNamespace(vram_state=_State.LOW_VRAM, VRAMState=_State)
        previous_management = sys.modules.get("comfy.model_management")
        previous_comfy = sys.modules.get("comfy")
        sys.modules["comfy"] = types.ModuleType("comfy")
        sys.modules["comfy.model_management"] = management
        try:
            evidence = configure_comfy_low_vram("no")
            self.assertIs(management.vram_state, _State.NO_VRAM)
            self.assertEqual(evidence["vram_state"], "NO_VRAM")
            self.assertEqual(evidence["requested_state"], "no")
        finally:
            if previous_management is None:
                sys.modules.pop("comfy.model_management", None)
            else:
                sys.modules["comfy.model_management"] = previous_management
            if previous_comfy is None:
                sys.modules.pop("comfy", None)
            else:
                sys.modules["comfy"] = previous_comfy

    def test_configure_rejects_unknown_state(self):
        class _State:
            LOW_VRAM = object()
            NO_VRAM = object()

        management = types.SimpleNamespace(vram_state=_State.LOW_VRAM, VRAMState=_State)
        previous_management = sys.modules.get("comfy.model_management")
        previous_comfy = sys.modules.get("comfy")
        sys.modules["comfy"] = types.ModuleType("comfy")
        sys.modules["comfy.model_management"] = management
        try:
            with self.assertRaisesRegex(RuntimeError, "invalid_low_vram_state"):
                configure_comfy_low_vram("turbo")
        finally:
            if previous_management is None:
                sys.modules.pop("comfy.model_management", None)
            else:
                sys.modules["comfy.model_management"] = previous_management
            if previous_comfy is None:
                sys.modules.pop("comfy", None)
            else:
                sys.modules["comfy"] = previous_comfy


if __name__ == "__main__":
    unittest.main()
