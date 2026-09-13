"""CPU unit fixtures test rejection paths, not real model or creative acceptance."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sol_common as common
import sglang_backend

spec = importlib.util.spec_from_file_location("runner", common.ROOT / "run-inference.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "image.bin").write_bytes(b"unit-test fixture; not generation input")
        self.request = {"schema_version": 1, "task": "ref2va", "status": "frozen",
                        "approval_id": "unit-test-only", "generation_unit_id": "fixture",
                        "prompt": "test prompt", "duration": 5, "seed": 7,
                        "references": [{"type": "image", "path": "image.bin",
                                        "sha256": common.sha256(self.root / "image.bin")}]}

    def validate(self):
        common.write_json(self.root / "request.json", self.request)
        return common.validate_request(self.root / "request.json", self.root)

    def test_reference_order_and_frozen_prompt_preserved(self):
        self.request["prompt"] = "exact  frozen prompt"
        request, refs = self.validate()
        self.assertEqual(request["prompt"], "exact  frozen prompt")
        self.assertEqual(refs, [("image", self.root / "image.bin")])

    def test_prompt_normalization_is_not_silent(self):
        self.request["prompt"] = " prompt "
        with self.assertRaisesRegex(ValueError, "whitespace"):
            self.validate()

    def test_modified_input_rejected(self):
        (self.root / "image.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.validate()

    def test_audio_only_rejected(self):
        self.request["references"][0]["type"] = "audio"
        with self.assertRaisesRegex(ValueError, "audio-only"):
            self.validate()

    def test_unknown_fields_rejected(self):
        self.request["width"] = 768
        with self.assertRaisesRegex(ValueError, "schema"):
            self.validate()

    def test_unsupported_duration_not_rounded(self):
        self.request["duration"] = 4
        with self.assertRaisesRegex(ValueError, "5/10/15"):
            self.validate()

    def test_unfrozen_input_rejected(self):
        self.request["status"] = "draft"
        with self.assertRaisesRegex(ValueError, "frozen"):
            self.validate()

    def test_symlink_escape_rejected(self):
        (self.root / "outside").symlink_to(Path(__file__).resolve())
        self.request["references"][0]["path"] = "outside"
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.validate()

    def test_absolute_path_rejected(self):
        self.request["references"][0]["path"] = str(self.root / "image.bin")
        with self.assertRaisesRegex(ValueError, "relative"):
            self.validate()

    def test_4090_profile_cannot_be_unblocked_by_external_json(self):
        path = common.ROOT / "profiles/ref2va-rtx4090-2gpu-bf16-4nfe-dense.json"
        profile = common.load_profile(path)
        self.assertIn("BLOCKED_4090_LOADING", profile["blocked_reason"])
        profile["blocked_reason"] = None
        common.write_json(self.root / "profile.json", profile)
        with self.assertRaisesRegex(ValueError, "registered"):
            common.load_profile(self.root / "profile.json")

    def test_source_modified_or_extra_module_rejected(self):
        source = self.root / "source"
        source.mkdir()
        (source / "engine.py").write_text("# fixture")
        lock = {"source_sha256": {"engine.py": common.sha256(source / "engine.py")}}
        common.write_json(self.root / "upstream.lock.json", lock)
        with patch.object(common, "ROOT", self.root):
            common.verify_source(source)
            (source / "extra.py").write_text("# unexpected")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                common.verify_source(source)
            (source / "extra.py").unlink()
            (source / "engine.py").write_text("# changed")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                common.verify_source(source)

    def create_models(self):
        models = self.root / "models"
        (models / "base/transformer_ref").mkdir(parents=True)
        (models / "adapter").mkdir()
        shard = "transformer_ref/shard.safetensors"
        index = "transformer_ref/weights.safetensors.index.json"
        (models / "base" / shard).write_bytes(b"fake unit fixture")
        common.write_json(models / "base" / index, {"weight_map": {"weight": "shard.safetensors"}})
        (models / "adapter/lora.safetensors").write_bytes(b"adapter unit fixture")
        lock = {"base": {"files": [shard, index]}, "adapter": {
            "file": "lora.safetensors", "sha256": common.sha256(models / "adapter/lora.safetensors")}}
        common.write_json(self.root / "models.lock.json", lock)
        receipt = {"lock_sha256": common.sha256(self.root / "models.lock.json"), "files": {}}
        for name in ["base/" + shard, "base/" + index, "adapter/lora.safetensors"]:
            path = models / name
            receipt["files"][name] = {"bytes": path.stat().st_size, "sha256": common.sha256(path)}
        common.write_json(models / "models.manifest.json", receipt)
        return models, receipt

    def test_missing_shard_rejected_even_with_manifest(self):
        models, _ = self.create_models()
        with patch.object(common, "ROOT", self.root):
            common.verify_models(models)
            (models / "base/transformer_ref/shard.safetensors").unlink()
            with self.assertRaises(ValueError):
                common.verify_models(models)

    def test_wrong_partition_rejected(self):
        models, receipt = self.create_models()
        entry = receipt["files"].pop("base/transformer_ref/shard.safetensors")
        receipt["files"]["base/transformer/shard.safetensors"] = entry
        common.write_json(models / "models.manifest.json", receipt)
        with patch.object(common, "ROOT", self.root):
            with self.assertRaisesRegex(ValueError, "partition"):
                common.verify_models(models)

    def test_wrong_adapter_rejected_even_with_rehashed_manifest(self):
        models, receipt = self.create_models()
        name = "adapter/lora.safetensors"
        (models / name).write_bytes(b"different adapter")
        receipt["files"][name] = {"bytes": (models / name).stat().st_size,
                                  "sha256": common.sha256(models / name)}
        common.write_json(models / "models.manifest.json", receipt)
        with patch.object(common, "ROOT", self.root):
            with self.assertRaisesRegex(ValueError, "adapter"):
                common.verify_models(models)

    def test_child_failure_is_not_success(self):
        with self.assertRaisesRegex(RuntimeError, "code 9"):
            runner.execute_process([sys.executable, "-c", "raise SystemExit(9)"], self.root, {}, 5)

    def test_timeout_terminates_child(self):
        import subprocess
        with self.assertRaises(subprocess.TimeoutExpired):
            runner.execute_process([sys.executable, "-c", "import time; time.sleep(30)"],
                                   self.root, {}, 0.05)

    def run_with_failed_gpu_probe(self, output):
        self.validate()
        common.write_json(self.root / "models.manifest.json", {})
        argv = ["run-inference.py", "--execute", "--profile", "unused.json",
                "--request", str(self.root / "request.json"), "--inputs", str(self.root),
                "--models", str(self.root), "--output", str(output)]
        from contextlib import ExitStack, redirect_stdout
        import io
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(patch.object(runner, "load_profile", return_value={
                "blocked_reason": None, "id": "test-only"}))
            stack.enter_context(patch.object(runner, "verify_source", return_value={"revision": "test-only"}))
            stack.enter_context(patch.object(runner, "verify_models", return_value=(self.root, self.root)))
            stack.enter_context(patch.object(runner, "gpu_probe", side_effect=ValueError("GPU gate failed")))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return runner.main()

    def test_gpu_failure_persists_failed_attempt(self):
        output = self.root / "attempt"
        self.assertEqual(self.run_with_failed_gpu_probe(output), 1)
        record = common.read_json(output / "manifest.json")
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["quality_review"], "pending")
        self.assertEqual(record["error"], "GPU gate failed")

    def test_existing_attempt_is_not_overwritten(self):
        output = self.root / "attempt"
        output.mkdir()
        marker = output / "manifest.json"
        marker.write_text("preserve original evidence")
        self.assertEqual(self.run_with_failed_gpu_probe(output), 1)
        self.assertEqual(marker.read_text(), "preserve original evidence")

    def test_silent_video_rejected(self):
        with patch.object(runner.subprocess, "check_output", return_value=json.dumps({
                "streams": [{"codec_type": "video"}]}).encode()):
            with self.assertRaisesRegex(ValueError, "native audio"):
                runner.validate_media(self.root / "not-real.mp4", 5)

    def test_sglang_adapter_preserves_ref2va_conditions(self):
        params = sglang_backend.SGLangH3Inference.sampling_params(
            "frozen prompt", references=[("image", "./reference.png"), ("audio", "voice.wav")],
            duration=10, seed=7, output=self.root / "output.mp4")
        self.assertEqual(params["task"], "ref2va")
        self.assertEqual([item["type"] for item in params["conditions"]], ["image", "audio"])
        self.assertEqual(params["target"]["aspect_ratio"], "9:16")

    def test_sglang_adapter_rejects_audio_only_ref2va(self):
        with self.assertRaisesRegex(ValueError, "image or video"):
            sglang_backend.SGLangH3Inference.sampling_params(
                "prompt", references=[("audio", "voice.wav")], duration=5,
                seed=7, output=self.root / "output.mp4")


if __name__ == "__main__":
    unittest.main()
