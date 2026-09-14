"""Render through the PUBLISHED diffusers components, not this repository's stack.

    python src/inference/infer_diffusers.py                       # prompts/example_0.pt
    python src/inference/infer_diffusers.py "a prompt" --out results/diffusers.mp4
    python src/inference/infer_diffusers.py "a prompt" --steps 50 \
        --transformer stage-b-step-2000/diffusers
    python src/inference/infer_diffusers.py "a prompt" \
        --first prompts/image/first.png --last prompts/image/last.png
    python src/inference/infer_diffusers.py --offload_dit          # a 24 or 32 GB card

This runs the README's `Load it with Diffusers` snippet. Everything it renders comes from
the Hub, so the patched diffusers that scripts/setup_diffusers.sh installs is the whole
setup; the only thing it reads from here is the default prompt, and a prompt of your own
replaces that. It deliberately shares NO code with infer.py and infer_ulysses.py -- those
are the fast stack (fp8, the decomposed window kernel, Ulysses); this is the portable
one, single-GPU bf16.

`workflow=` keeps the unused 61.7 GB transformer partition from being fetched. Every
model but the transformer is offloaded, always: the 62 GB Qwen3-VL text encoder comes
onto the GPU one layer at a time, and each decoder whole, while it runs. The transformer
stays on the GPU unless `--offload_dit` streams it in one block at a time too.
"""
import argparse
import os

import torch
from accelerate import cpu_offload_with_hook
from diffusers import ModularPipeline
from diffusers.hooks import apply_group_offloading
from diffusers.utils.export_utils import encode_video

REPO = "OpenVDN/vdn-minimax-h3"
FPS = 24
# The repository's own showcase prompt. Every prompt cache carries the text it was
# encoded from, so the default is that text rather than a second copy of it here.
DEFAULT_PROMPT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "prompts", "example_0.pt")


def offload(pipe, device, dit=False):
    """The text encoder streams in by layer. The decoders come in whole through
    accelerate's hook: the pipeline calls their `encode` and `decode`, and that is the
    only hook those fire. With `dit` the transformer streams in by block, and otherwise
    it stays on the GPU. It offloads whole or by block, never by leaf: its fused kernels
    read a child's weights without calling the child, so a leaf's hook never fires."""
    apply_group_offloading(pipe.text_encoder, onload_device=device, offload_device="cpu",
                           offload_type="leaf_level", use_stream=True, low_cpu_mem_usage=True)
    _, vae = cpu_offload_with_hook(pipe.vae, execution_device=device)
    _, audio_vae = cpu_offload_with_hook(pipe.audio_vae, execution_device=device, prev_module_hook=vae)

    def decoder_back(module, args):
        vae.offload()                 # reference media are encoded before denoising
        audio_vae.offload()

    transformer = getattr(pipe, "transformer_ref", None) or pipe.transformer
    transformer.register_forward_pre_hook(decoder_back)
    if not dit:
        transformer.to(device)
        return
    # fp8 keeps its weights in buffers; diffusers_patches/0002 is what sends a streamed
    # group's buffers back to the CPU along with its parameters.
    apply_group_offloading(transformer, onload_device=device, offload_device="cpu",
                           offload_type="block_level", num_blocks_per_group=1,
                           use_stream=False)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("prompt", nargs="?",
                   help=f"defaults to the text {os.path.basename(DEFAULT_PROMPT)} was "
                        "encoded from")
    p.add_argument("--models", required=True, help="verified local model tree")
    p.add_argument("--out", default="results/diffusers.mp4")
    p.add_argument("--steps", type=int, default=8,
                   help="model evaluations (NFE). The scheduler counts sigma grid "
                        "points, one more than that, and this passes the +1 for you")
    p.add_argument("--frames", type=int, default=345,
                   help="snapped up to the next 17n+5; 5 to 15 seconds at 24 fps. The "
                        "default is the 14.4 seconds the reported numbers use, which "
                        "one 140 GB GPU holds in bf16 with room to spare")
    p.add_argument("--transformer", default=None,
                   help="a checkpoint's diffusers/ subfolder. Default: whatever the "
                        "repository's index names, the 8-step model")
    p.add_argument("--first", help="keyframe the video starts from")
    p.add_argument("--last", help="keyframe the video ends on")
    p.add_argument("--fp8", action="store_true",
                   help="every wide Linear in fp8 e4m3: the weights drop from 62 GB "
                        "to 43 and the GEMMs roughly double")
    p.add_argument("--offload_dit", action="store_true",
                   help="stream the transformer onto the GPU one block at a time, which "
                        "a 24 or 32 GB card needs: 345 frames then peak at 20 GB. The "
                        "transformer offloads whole or by block, never by leaf")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    prompt = args.prompt or torch.load(DEFAULT_PROMPT, map_location="cpu",
                                       weights_only=True)["prompt"]

    pipe = load_pipeline(args, "fl2va" if args.first or args.last else "t2va")
    render(pipe, args, prompt)


def load_pipeline(args, workflow):
    """Load upstream components once for either the CLI or resident worker."""
    pipe = ModularPipeline.from_pretrained(args.models, workflow=workflow, local_files_only=True)

    load_kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16}
    if args.transformer:
        load_kwargs["subfolder"] = {"transformer": args.transformer}
    if args.fp8:
        # A dict keys a kwarg to one component; the text encoder would not know it.
        load_kwargs["fp8"] = {"transformer": True}
    # Local-path adaptation: modular indexes may still contain Hub repository IDs.
    # Use the same ComponentSpec loader, with every component bound to the
    # caller-verified local tree. Missing components must fail immediately.
    from pathlib import Path
    root = Path(args.models).resolve()
    for name in pipe.pretrained_component_names:
        spec = pipe.get_component_spec(name)
        folder = args.transformer if name in {"transformer", "transformer_ref"} else (spec.subfolder or name)
        if not folder:
            raise ValueError("explicit transformer subfolder required")
        path = (root / folder).resolve()
        if not path.is_relative_to(root) or not path.is_dir():
            raise ValueError(f"missing local component: {name}")
        spec.pretrained_model_name_or_path = str(root)
        spec.subfolder, spec.revision = folder, None
        kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16,
                  "local_files_only": True}
        if args.fp8 and name == "transformer":
            kwargs["fp8"] = True
        pipe.update_components(**{name: spec.load(**kwargs)})
    offload(pipe, torch.device(args.device), dit=args.offload_dit)

    return pipe


@torch.inference_mode()
def render(pipe, args, prompt):
    keyframes = {}
    if args.first or args.last:
        from diffusers.utils import load_image

        if args.first:
            keyframes["image"] = load_image(args.first)
        if args.last:
            keyframes["last_image"] = load_image(args.last)

    videos, audio, rate = pipe(
        prompt=prompt,
        num_frames=args.frames,
        num_inference_steps=args.steps + 1,
        generator=torch.Generator(args.device).manual_seed(args.seed),
        output=["videos", "audio", "sampling_rate"],
        **keyframes,
    ).values()

    import numpy as np

    frames = torch.from_numpy(np.stack([np.asarray(frame) for frame in videos[0]]))
    encode_video(frames, fps=FPS, output_path=args.out,
                 audio=audio[0].float().cpu(), audio_sample_rate=rate)
    print(f"wrote {args.out}: {len(videos[0])} frames, "
          f"{audio.shape[-1] / rate:.2f}s of audio")


@torch.inference_mode()
def render_ref2va(pipe, args, prompt, paths):
    # Pinned Diffusers MiniMax-H3 "Omni-references" example, adapted to local
    # verified artifacts and the VDN checkpoint's NFE (+1 sigma grid point).
    from diffusers.modular_pipelines.minimax_h3 import (
        MiniMaxH3AudioReference, MiniMaxH3ImageReference, MiniMaxH3VideoReference,
    )
    reference_types = {"image": MiniMaxH3ImageReference,
                       "video": MiniMaxH3VideoReference, "audio": MiniMaxH3AudioReference}
    references = [reference_types[kind].from_file(path) for kind, path in paths]
    results = pipe(
        prompt=prompt,
        references=references,
        num_frames=args.frames,
        height=1344, width=768,
        num_inference_steps=args.steps + 1,
        generator=torch.Generator(args.device).manual_seed(args.seed),
        output=["videos", "audio", "sampling_rate"],
    )
    encode_video(results["videos"][0], fps=FPS, output_path=args.out,
                 audio=results["audio"][0].float().cpu(), audio_sample_rate=results["sampling_rate"])


if __name__ == "__main__":
    main()
