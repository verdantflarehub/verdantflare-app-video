"""SGLang-Diffusion adapter for the H3 resident worker.

The adapter is intentionally lazy: importing the API never imports CUDA or
SGLang.  This keeps the CPU HTTP contract usable for probes and tests.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from collections.abc import Sequence


class SGLangH3Inference:
    """One SGLang DiffGenerator instance with the service's video contract."""

    def __init__(self, model_path: str):
        from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
        runtime_dir = Path(os.environ.get("H3_SGLANG_PROFILE_DIR", "/opt/sol-h3/models/minimax_h3/RTX4090"))
        sys.path.insert(0, str(runtime_dir))
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

    @staticmethod
    def sampling_params(
        prompt: str,
        *,
        references: Sequence[tuple[str, str]],
        duration: int,
        seed: int,
        output: Path,
        steps: int = 4,
    ) -> dict:
        """Build the SGLang request without importing CUDA or the runtime.

        Keeping this mapping pure gives the CPU contract tests a way to catch
        the most dangerous regression here: accidentally sending a Ref2VA
        request as text-to-video and silently dropping its references.
        """
        if duration not in {5, 10, 15}:
            raise ValueError("SGLang Ref2VA supports only 5, 10, or 15 seconds")
        if not references or not any(kind in {"image", "video"} for kind, _ in references):
            raise ValueError("Ref2VA requires an image or video reference")
        conditions = []
        for kind, path in references:
            if kind not in {"image", "video", "audio"} or not path:
                raise ValueError("reference kind must be image, video, or audio")
            conditions.append({"type": kind, "path": str(Path(path).resolve())})
        return {
            "prompt": prompt,
            "task": "ref2va",
            "conditions": conditions,
            "target": {"short_edge": 768, "aspect_ratio": "9:16", "duration_seconds": float(duration)},
            "num_outputs_per_prompt": 1,
            "num_inference_steps": steps,
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
            "seed": seed,
            "output_path": str(output.parent),
            "output_file_name": output.name,
            "save_output": True,
            "return_file_paths_only": True,
        }

    def generate(self, prompt: str, *, references: Sequence[tuple[str, str]], duration: int, seed: int, output: Path, steps: int = 4):
        params = self.sampling_params(prompt, references=references, duration=duration,
                                      seed=seed, output=output, steps=steps)
        result = self._generator.generate(sampling_params_kwargs={
            **params,
        })
        if result is None or isinstance(result, list):
            raise RuntimeError("SGLang returned no single video result")
        return result

    def shutdown(self):
        self._generator.shutdown()
