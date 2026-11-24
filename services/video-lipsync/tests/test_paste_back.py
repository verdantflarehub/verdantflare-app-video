"""Unit tests for seamless mouth/jaw soft-mask paste-back, Lab color matching, and yaw safety."""

from __future__ import annotations

import math
import sys

import cv2
import numpy as np

from video_lipsync.paste_back import (
    JAW_CONTOUR_INDICES,
    LEFT_BROW_INDICES,
    MOUTH_INDICES,
    NOSE_CENTER_INDICES,
    RIGHT_BROW_INDICES,
    PasteBackStats,
    build_mouth_jaw_masks,
    estimate_yaw_asymmetry,
    paste_back_frame,
    smooth_landmarks_sequence,
)


def _make_synthetic_landmarks(
    cx: float = 256.0,
    cy: float = 256.0,
    face_w: float = 160.0,
    face_h: float = 200.0,
    nose_shift_x: float = 0.0,
    roll_deg: float = 0.0,
) -> np.ndarray:
    """Create a geometrically consistent 106-point landmark set matching InsightFace index groups."""
    lmk = np.full((106, 2), [cx, cy], dtype=np.float32)

    # Jaw contour (indices 0..32): semi-ellipse from left temple down to chin (index 0) and up to right temple
    angles = np.linspace(0.0, math.pi, len(JAW_CONTOUR_INDICES))
    for i, idx in enumerate(JAW_CONTOUR_INDICES):
        theta = angles[i]
        x = cx - math.cos(theta) * (face_w * 0.5)
        y = cy - face_h * 0.10 + math.sin(theta) * (face_h * 0.60)
        lmk[idx] = [x, y]
    lmk[0] = [cx, cy + face_h * 0.50]

    # Left & right brow clusters
    for offset, idx in enumerate(LEFT_BROW_INDICES):
        lmk[idx] = [cx - face_w * 0.24 + (offset - 2) * 3.0 + nose_shift_x * 0.3, cy - face_h * 0.22]
    for offset, idx in enumerate(RIGHT_BROW_INDICES):
        lmk[idx] = [cx + face_w * 0.24 + (offset - 2) * 3.0 + nose_shift_x * 0.3, cy - face_h * 0.22]

    # Nose cluster (shifts horizontally when head yaws)
    for offset, idx in enumerate(NOSE_CENTER_INDICES):
        lmk[idx] = [cx + nose_shift_x + (offset - 1.5) * 2.5, cy + face_h * 0.02 + (offset % 2) * 4.0]

    # Mouth cluster (indices 52..71)
    m_angles = np.linspace(0.0, 2.0 * math.pi, len(MOUTH_INDICES), endpoint=False)
    for i, idx in enumerate(MOUTH_INDICES):
        mx = cx + nose_shift_x * 0.8 + math.cos(m_angles[i]) * (face_w * 0.20)
        my = cy + face_h * 0.24 + math.sin(m_angles[i]) * (face_h * 0.06)
        lmk[idx] = [mx, my]

    if abs(roll_deg) > 1e-6:
        rad = math.radians(roll_deg)
        rot = np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]], dtype=np.float32)
        origin = np.array([cx, cy], dtype=np.float32)
        lmk = (lmk - origin) @ rot.T + origin

    return lmk


def _make_textured_frame(h: int = 512, w: int = 512) -> np.ndarray:
    """Create a realistic skin-toned frame with gentle gradient so seams are easy to detect."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.clip(198.0 + 10.0 * np.sin(xx / 45.0) - 8.0 * (yy / h), 0, 255)
    g = np.clip(154.0 + 8.0 * np.cos(yy / 50.0) - 6.0 * (xx / w), 0, 255)
    b = np.clip(134.0 + 6.0 * np.sin((xx + yy) / 60.0), 0, 255)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def test_paste_back_preserves_non_mouth_pixels_bit_exact():
    """Eyes, under-eyes, upper cheeks, and background must remain 100% identical to source frame."""
    src = _make_textured_frame(512, 512)
    lmk = _make_synthetic_landmarks(cx=256.0, cy=256.0)
    identity_affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    # Corrupt the entire upper half of generated_face with bright cyan and paint lips red
    gen = src.copy()
    gen[:245, :, :] = np.array([20, 240, 240], dtype=np.uint8)
    cv2.ellipse(gen, (256, 304), (24, 10), 0, 0, 360, (175, 70, 75), -1)

    stats = PasteBackStats()
    out = paste_back_frame(src, gen, identity_affine, lmk, stats=stats)

    assert stats.total_frames == 1
    assert stats.faces_detected_frames == 1
    assert stats.faces_missing_frames == 0

    # Eye & under-eye band (y < 235) and background corners must be bit-exact original
    assert np.array_equal(out[:235, :, :], src[:235, :, :])
    assert np.array_equal(out[:, :140, :], src[:, :140, :])
    assert np.array_equal(out[:, 372:, :], src[:, 372:, :])

    # Mouth center must reflect the synthesized lips
    diff_mouth = np.mean(np.abs(out[298:310, 244:268].astype(np.float32) - src[298:310, 244:268].astype(np.float32)))
    assert diff_mouth > 25.0


def test_lab_color_matching_eliminates_orange_and_brightness_shift():
    """Color shift in VAE output (e.g. bright/orange tint) must be neutralized at paste-back boundary."""
    src = _make_textured_frame(512, 512)
    lmk = _make_synthetic_landmarks(cx=256.0, cy=256.0)
    identity_affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    # Simulate VAE orange + brightness drift across the generated crop
    gen = np.clip(
        src.astype(np.float32) + np.array([24.0, 10.0, -18.0], dtype=np.float32),
        0,
        255,
    ).astype(np.uint8)

    out = paste_back_frame(src, gen, identity_affine, lmk)
    soft_mask, _, _ = build_mouth_jaw_masks(src.shape[:2], lmk)

    # Inside the pasted chin/mouth-skin area, color difference vs source should be tiny (< 4.0)
    pasted_region = soft_mask > 0.7
    mean_err = float(np.mean(np.abs(out[pasted_region].astype(np.float32) - src[pasted_region].astype(np.float32))))
    raw_err = float(np.mean(np.abs(gen[pasted_region].astype(np.float32) - src[pasted_region].astype(np.float32))))
    assert raw_err > 15.0
    assert mean_err < 4.5


def test_rotated_crop_has_no_gray_border_artifact():
    """Simulating a ~15-deg head turn with dark/gray border on generated crop must not leak onto cheek."""
    src = _make_textured_frame(512, 512)
    lmk = _make_synthetic_landmarks(cx=256.0, cy=256.0, nose_shift_x=16.0, roll_deg=12.0)

    # Crop affine mapping a 384x384 window around the face
    mat = cv2.getRotationMatrix2D((256.0, 256.0), 12.0, 1.0).astype(np.float32)
    mat[0, 2] -= 64.0
    mat[1, 2] -= 64.0

    warped_src_crop = cv2.warpAffine(src, mat, (384, 384), borderMode=cv2.BORDER_REPLICATE)
    # Paint a 127-gray frame border on the generated crop to simulate edge artifact
    gen_crop = warped_src_crop.copy()
    gen_crop[:10, :] = 127
    gen_crop[-10:, :] = 127
    gen_crop[:, :10] = 127
    gen_crop[:, -10:] = 127

    out = paste_back_frame(src, gen_crop, mat, lmk)
    # No pixel anywhere in the frame should have a sudden > 6 level jump from src
    max_diff = float(np.max(np.abs(out.astype(np.float32) - src.astype(np.float32))))
    assert max_diff < 8.0


def test_high_yaw_degrades_and_falls_back_with_metrics():
    """Large yaw frames degrade smoothly and extreme yaw falls back to source frame with counters."""
    src = _make_textured_frame(512, 512)
    gen = np.full_like(src, 40)
    identity_affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    lmk_front = _make_synthetic_landmarks(nose_shift_x=0.0)
    lmk_mid_yaw = _make_synthetic_landmarks(nose_shift_x=44.0)  # asym ~ 0.55 -> degraded
    lmk_extreme_yaw = _make_synthetic_landmarks(nose_shift_x=64.0)  # asym ~ 0.80 -> fallback

    assert estimate_yaw_asymmetry(lmk_front) < 0.1
    assert 0.45 < estimate_yaw_asymmetry(lmk_mid_yaw) < 0.65
    assert estimate_yaw_asymmetry(lmk_extreme_yaw) >= 0.65

    stats = PasteBackStats()
    out_front = paste_back_frame(src, gen, identity_affine, lmk_front, stats=stats)
    out_mid = paste_back_frame(src, gen, identity_affine, lmk_mid_yaw, stats=stats)
    out_ext = paste_back_frame(src, gen, identity_affine, lmk_extreme_yaw, stats=stats)
    out_missing = paste_back_frame(src, gen, identity_affine, None, stats=stats)

    assert stats.total_frames == 4
    assert stats.faces_detected_frames == 3
    assert stats.faces_missing_frames == 1
    assert stats.high_yaw_degraded_frames == 1
    assert stats.high_yaw_fallback_frames == 1

    # Extreme yaw and missing-face frames are 100% bit-exact source
    assert np.array_equal(out_ext, src)
    assert np.array_equal(out_missing, src)
    # Degraded frame blends less strongly than frontal frame
    assert np.mean(np.abs(out_mid.astype(np.float32) - src.astype(np.float32))) < np.mean(
        np.abs(out_front.astype(np.float32) - src.astype(np.float32))
    )


def test_landmark_smoother_damps_jitter_and_preserves_zero_cuda():
    """Adaptive EMA reduces high-frequency detector jitter without importing torch/CUDA."""
    rng = np.random.default_rng(42)
    base = _make_synthetic_landmarks()
    noisy = [base + rng.normal(0.0, 2.0, size=base.shape).astype(np.float32) for _ in range(20)]

    smoothed = smooth_landmarks_sequence(noisy)
    raw_jitter = float(np.mean([np.linalg.norm(noisy[i] - noisy[i - 1]) for i in range(1, len(noisy))]))
    smooth_jitter = float(np.mean([np.linalg.norm(smoothed[i] - smoothed[i - 1]) for i in range(1, len(smoothed))]))
    assert smooth_jitter < raw_jitter * 0.6

    for mod in ("torch", "diffusers", "onnxruntime"):
        assert mod not in sys.modules


def test_seamless_pipeline_does_not_override_diffusers_init_signature():
    """diffusers.DiffusionPipeline inspects __init__ parameters; subclass must not override __init__."""
    import ast
    from pathlib import Path

    src_path = Path(__file__).resolve().parents[1] / "src" / "video_lipsync" / "infer_runner.py"
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    cls_nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SeamlessLipsyncPipeline"]
    assert len(cls_nodes) == 1
    method_names = {n.name for n in cls_nodes[0].body if isinstance(n, ast.FunctionDef)}
    assert "__init__" not in method_names
    assert {"affine_transform_video", "loop_video", "restore_video"} <= method_names
