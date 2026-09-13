"""SGLang-Diffusion adapter for the H3 resident worker.

The adapter is intentionally lazy: importing the API never imports CUDA or
SGLang.  This keeps the CPU HTTP contract usable for probes and tests.
"""
from __future__ import annotations

import os
from pathlib import Path


class SGLangH3Inference:
    """One SGLang DiffGenerator instance with the service's video contract."""

    def __init__(self, model_path: str):
        from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
        from registration import register_runtime
        register_runtime()
        self._generator = DiffGenerator.from_pretrained(
            local_mode=True, model_path=model_path,
            model_subfolder=os.environ.get("H3_MODEL_SUBFOLDER", "FL2VA"),
            num_gpus=1, tp_size=1, ulysses_degree=1,
            performance_mode="memory",
            layerwise_offload_components=["dit", "text_encoder", "vae"],
            server_warmup=False, master_port=int(os.environ.get("H3_MASTER_PORT", "30005")),
        )

    def generate(self, prompt: str, *, duration: int, seed: int, output: Path, steps: int = 4):
        result = self._generator.generate(sampling_params_kwargs={
            "prompt": prompt, "task": "t2va", "conditions": [],
            "target": {"short_edge": 768, "aspect_ratio": "9:16", "duration_seconds": float(duration)},
            "num_outputs_per_prompt": 1, "num_inference_steps": steps,
            "flow_shift": 12.0, "audio_flow_shift": 3.0, "seed": seed,
            "output_path": str(output.parent), "output_file_name": output.name,
            "save_output": True, "return_file_paths_only": True,
        })
        if result is None or isinstance(result, list):
            raise RuntimeError("SGLang returned no single video result")
        return result

    def shutdown(self):
        self._generator.shutdown()
