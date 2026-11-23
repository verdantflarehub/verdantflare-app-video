"""Isolated LatentSync 1.6 inference entrypoint using local offline model weights."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from accelerate.utils import set_seed
from DeepCache import DeepCacheSDHelper
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature
from omegaconf import OmegaConf
import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latentsync_root", type=str, required=True)
    parser.add_argument("--model_root", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--temp_dir", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--enable_deepcache", action="store_true")
    args = parser.parse_args()

    latentsync_root = Path(args.latentsync_root)
    model_root = Path(args.model_root)
    ckpt_link = latentsync_root / "checkpoints"
    if not ckpt_link.exists():
        ckpt_link.symlink_to(model_root / "checkpoints")
    config = OmegaConf.load(str(latentsync_root / "configs/unet/stage2_512.yaml"))

    is_fp16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16 else torch.float32

    scheduler = DDIMScheduler.from_pretrained(str(latentsync_root / "configs"), local_files_only=True)
    whisper_name = "small.pt" if config.model.cross_attention_dim == 768 else "tiny.pt"
    whisper_path = str(model_root / "checkpoints/whisper" / whisper_name)

    audio_encoder = Audio2Feature(
        model_path=whisper_path,
        device="cuda",
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )

    vae = AutoencoderKL.from_pretrained(
        str(model_root / "sd-vae-ft-mse"),
        torch_dtype=dtype,
        local_files_only=True,
    )
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        str(model_root / "checkpoints/latentsync_unet.pt"),
        device="cpu",
    )
    unet = unet.to(dtype=dtype)

    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    )
    if os.environ.get("LIPSYNC_ENABLE_CPU_OFFLOAD", "1") == "1":
        try:
            pipeline.enable_model_cpu_offload()
        except Exception:
            pipeline = pipeline.to("cuda")
    else:
        pipeline = pipeline.to("cuda")

    if args.enable_deepcache:
        helper = DeepCacheSDHelper(pipe=pipeline)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        helper.enable()

    if args.seed != -1:
        set_seed(args.seed)
    else:
        torch.seed()

    Path(args.temp_dir).mkdir(parents=True, exist_ok=True)
    pipeline(
        video_path=args.video_path,
        audio_path=args.audio_path,
        video_out_path=args.video_out_path,
        num_frames=config.data.num_frames,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        weight_dtype=dtype,
        width=config.data.resolution,
        height=config.data.resolution,
        mask_image_path=config.data.mask_image_path,
        temp_dir=args.temp_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
