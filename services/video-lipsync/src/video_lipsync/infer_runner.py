"""Isolated LatentSync 1.6 inference entrypoint using local offline model weights."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from accelerate.utils import set_seed
import cv2
from DeepCache import DeepCacheSDHelper
from diffusers import AutoencoderKL, DDIMScheduler
from einops import rearrange
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature
import numpy as np
from omegaconf import OmegaConf
import torch
import torchvision
from torchvision import transforms

from video_lipsync.paste_back import (
    PasteBackStats,
    extract_align_landmarks3,
    paste_back_frame,
    smooth_landmarks_sequence,
)


def _align_warp_face_border(restorer, img: np.ndarray, landmarks3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Warp face crop with BORDER_REPLICATE instead of upstream 127-gray fill to prevent edge artifacts."""
    affine_matrix, _ = restorer.transformation_from_points(
        landmarks3, restorer.face_template, smooth=False, p_bias=None
    )
    mat = np.asarray(affine_matrix, dtype=np.float32)
    cropped_face = cv2.warpAffine(
        img,
        mat,
        (int(restorer.face_size[0]), int(restorer.face_size[1])),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return cropped_face, mat


class SeamlessLipsyncPipeline(LipsyncPipeline):
    """LipsyncPipeline subclass with sub-nasal soft-mask paste-back, Lab color matching, and yaw safety."""

    def __init__(self, *args, metrics_out_path: Path | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._frame_landmarks: list[np.ndarray | None] = []
        self._metrics_out_path = metrics_out_path
        self.paste_back_stats = PasteBackStats()

    def affine_transform_video(self, video_frames: np.ndarray):
        raw_landmarks: list[np.ndarray | None] = []
        for frame in video_frames:
            bbox, lmk_106 = self.image_processor.face_detector(frame)
            if bbox is None or lmk_106 is None:
                raw_landmarks.append(None)
            else:
                raw_landmarks.append(np.asarray(lmk_106, dtype=np.float32))

        first_valid = next((lmk for lmk in raw_landmarks if lmk is not None), None)
        if first_valid is None:
            raise RuntimeError("Face not detected in any video frame")

        smoothed_landmarks = smooth_landmarks_sequence(raw_landmarks)
        self._frame_landmarks = smoothed_landmarks

        # Build fallback landmarks solely for cropping missing-face frames so UNet batching stays intact;
        # paste_back_frame still sees None in self._frame_landmarks and passes those frames through untouched.
        crop_landmarks: list[np.ndarray] = []
        last_valid = first_valid
        for lmk in smoothed_landmarks:
            if lmk is not None:
                last_valid = lmk
            crop_landmarks.append(last_valid)

        faces = []
        boxes = []
        affine_matrices = []
        resolution = int(self.image_processor.resolution)
        for frame, lmk in zip(video_frames, crop_landmarks):
            landmarks3 = np.round(extract_align_landmarks3(lmk))
            face_crop, affine_mat = _align_warp_face_border(self.image_processor.restorer, frame, landmarks3)
            box = [0, 0, face_crop.shape[1], face_crop.shape[0]]
            face_resized = cv2.resize(face_crop, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4)
            face_tensor = rearrange(torch.from_numpy(face_resized), "h w c -> c h w")
            faces.append(face_tensor)
            boxes.append(box)
            affine_matrices.append(affine_mat)

        return torch.stack(faces), boxes, affine_matrices

    def loop_video(self, whisper_chunks: list, video_frames: np.ndarray):
        if len(whisper_chunks) > len(video_frames):
            faces, boxes, affine_matrices = self.affine_transform_video(video_frames)
            base_landmarks = list(self._frame_landmarks)
            num_loops = math.ceil(len(whisper_chunks) / len(video_frames))
            loop_video_frames = []
            loop_faces = []
            loop_boxes = []
            loop_affine_matrices = []
            loop_landmarks: list[np.ndarray | None] = []
            for i in range(num_loops):
                if i % 2 == 0:
                    loop_video_frames.append(video_frames)
                    loop_faces.append(faces)
                    loop_boxes += boxes
                    loop_affine_matrices += affine_matrices
                    loop_landmarks += base_landmarks
                else:
                    loop_video_frames.append(video_frames[::-1])
                    loop_faces.append(faces.flip(0))
                    loop_boxes += boxes[::-1]
                    loop_affine_matrices += affine_matrices[::-1]
                    loop_landmarks += base_landmarks[::-1]

            video_frames = np.concatenate(loop_video_frames, axis=0)[: len(whisper_chunks)]
            faces = torch.cat(loop_faces, dim=0)[: len(whisper_chunks)]
            boxes = loop_boxes[: len(whisper_chunks)]
            affine_matrices = loop_affine_matrices[: len(whisper_chunks)]
            self._frame_landmarks = loop_landmarks[: len(whisper_chunks)]
        else:
            video_frames = video_frames[: len(whisper_chunks)]
            faces, boxes, affine_matrices = self.affine_transform_video(video_frames)

        return video_frames, faces, boxes, affine_matrices

    def restore_video(self, faces: torch.Tensor, video_frames: np.ndarray, boxes: list, affine_matrices: list):
        video_frames = video_frames[: len(faces)]
        self.paste_back_stats = PasteBackStats()
        out_frames = []
        for index, face in enumerate(faces):
            x1, y1, x2, y2 = boxes[index]
            height = int(y2 - y1)
            width = int(x2 - x1)
            face_resized = torchvision.transforms.functional.resize(
                face, size=(height, width), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
            )
            face_rgb = rearrange(face_resized, "c h w -> h w c")
            face_rgb = ((face_rgb / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()

            mat = affine_matrices[index]
            if isinstance(mat, torch.Tensor):
                mat = mat.detach().cpu().numpy()
            lmk = self._frame_landmarks[index] if index < len(self._frame_landmarks) else None

            out_frame = paste_back_frame(
                video_frames[index],
                face_rgb,
                mat,
                lmk,
                stats=self.paste_back_stats,
            )
            out_frames.append(out_frame)

        stats_dict = self.paste_back_stats.to_dict()
        if stats_dict["high_yaw_degraded_frames"] > 0 or stats_dict["high_yaw_fallback_frames"] > 0 or stats_dict["faces_missing_frames"] > 0:
            print(
                f"[video-lipsync] paste-back warning: "
                f"missing_faces={stats_dict['faces_missing_frames']}, "
                f"high_yaw_degraded={stats_dict['high_yaw_degraded_frames']}, "
                f"high_yaw_fallback={stats_dict['high_yaw_fallback_frames']}",
                flush=True,
            )
        if self._metrics_out_path is not None:
            try:
                self._metrics_out_path.parent.mkdir(parents=True, exist_ok=True)
                self._metrics_out_path.write_text(json.dumps(stats_dict), encoding="utf-8")
            except OSError:
                pass
        return np.stack(out_frames, axis=0)


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

    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = temp_dir / "paste_back_metrics.json"

    pipeline = SeamlessLipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
        metrics_out_path=metrics_path,
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
        temp_dir=str(temp_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
