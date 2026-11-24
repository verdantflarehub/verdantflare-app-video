"""Seamless mouth/jaw soft-mask paste-back, Lab color matching, and yaw safety for LatentSync."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import cv2
import numpy as np


# InsightFace 2d106det landmark index groups (matching LatentSync ImageProcessor / FaceDetector)
LEFT_BROW_INDICES = (43, 48, 49, 51, 50)
RIGHT_BROW_INDICES = (101, 102, 103, 104, 105)
NOSE_CENTER_INDICES = (74, 77, 83, 86)
JAW_CONTOUR_INDICES = tuple(range(0, 33))
MOUTH_INDICES = tuple(range(52, 72))

# Yaw asymmetry thresholds (|d_right - d_left| / (d_right + d_left))
# Front face: ~0.00-0.12; ~15 deg turn: ~0.15-0.28; >35 deg: ~0.45+; >50 deg: ~0.65+
YAW_DEGRADE_START = 0.45
YAW_FALLBACK_THRESHOLD = 0.65


@dataclass
class PasteBackStats:
    total_frames: int = 0
    faces_detected_frames: int = 0
    faces_missing_frames: int = 0
    high_yaw_degraded_frames: int = 0
    high_yaw_fallback_frames: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "total_frames": self.total_frames,
            "faces_detected_frames": self.faces_detected_frames,
            "faces_missing_frames": self.faces_missing_frames,
            "high_yaw_degraded_frames": self.high_yaw_degraded_frames,
            "high_yaw_fallback_frames": self.high_yaw_fallback_frames,
        }


def extract_align_landmarks3(lmk_106: np.ndarray) -> np.ndarray:
    """Compute the 3 alignment anchor points (left brow, right brow, nose) from 106 landmarks."""
    pts = np.asarray(lmk_106, dtype=np.float32)
    pt_left = np.mean(pts[list(LEFT_BROW_INDICES)], axis=0)
    pt_right = np.mean(pts[list(RIGHT_BROW_INDICES)], axis=0)
    pt_nose = np.mean(pts[list(NOSE_CENTER_INDICES)], axis=0)
    return np.stack([pt_left, pt_right, pt_nose], axis=0)


def smooth_landmarks_sequence(
    landmarks_seq: Sequence[np.ndarray | None],
    *,
    min_alpha: float = 0.35,
    max_alpha: float = 0.85,
    motion_scale: float = 0.04,
) -> list[np.ndarray | None]:
    """Velocity-adaptive EMA temporal smoothing for 106-point landmarks to eliminate seam jitter.

    Small frame-to-frame detector noise (1-2px) is strongly damped (alpha -> min_alpha),
    while genuine head movement increases alpha toward max_alpha to prevent tracking lag.
    """
    smoothed: list[np.ndarray | None] = []
    prev: np.ndarray | None = None
    for item in landmarks_seq:
        if item is None:
            smoothed.append(None)
            prev = None
            continue
        cur = np.asarray(item, dtype=np.float32)
        if prev is None or cur.shape != prev.shape:
            prev = cur.copy()
            smoothed.append(prev.copy())
            continue

        jaw = cur[list(JAW_CONTOUR_INDICES)]
        face_span = float(max(np.ptp(jaw[:, 0]), np.ptp(jaw[:, 1]), 1.0))
        displacement = float(np.mean(np.linalg.norm(cur - prev, axis=1)))
        norm_motion = displacement / (face_span * motion_scale)
        alpha = min_alpha + (max_alpha - min_alpha) * min(1.0, max(0.0, norm_motion))
        prev = alpha * cur + (1.0 - alpha) * prev
        smoothed.append(prev.copy())
    return smoothed


def _face_axes(lmk_106: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return local face unit axes (u_x right, u_y down), origin (nose), and face width."""
    pts = np.asarray(lmk_106, dtype=np.float32)
    left_brow = np.mean(pts[list(LEFT_BROW_INDICES)], axis=0)
    right_brow = np.mean(pts[list(RIGHT_BROW_INDICES)], axis=0)
    nose = np.mean(pts[list(NOSE_CENTER_INDICES)], axis=0)
    mouth_center = np.mean(pts[list(MOUTH_INDICES)], axis=0)

    dx = right_brow - left_brow
    norm_x = float(np.linalg.norm(dx))
    if norm_x < 1e-4:
        u_x = np.array([1.0, 0.0], dtype=np.float32)
    else:
        u_x = (dx / norm_x).astype(np.float32)

    u_y = np.array([-u_x[1], u_x[0]], dtype=np.float32)
    if float(np.dot(mouth_center - nose, u_y)) < 0.0:
        u_y = -u_y

    jaw_proj_x = (pts[list(JAW_CONTOUR_INDICES)] - nose) @ u_x
    face_width = float(max(np.max(jaw_proj_x) - np.min(jaw_proj_x), norm_x * 1.6, 16.0))
    return u_x, u_y, nose, face_width


def estimate_yaw_asymmetry(lmk_106: np.ndarray) -> float:
    """Estimate horizontal head turn asymmetry in [0, 1] from nose position inside jaw contour."""
    pts = np.asarray(lmk_106, dtype=np.float32)
    u_x, _, nose, _ = _face_axes(pts)
    jaw_proj_x = (pts[list(JAW_CONTOUR_INDICES)] - nose) @ u_x
    min_x = float(np.min(jaw_proj_x))
    max_x = float(np.max(jaw_proj_x))
    d_left = max(0.0, -min_x)
    d_right = max(0.0, max_x)
    total = d_left + d_right
    if total < 1e-4:
        return 1.0
    return float(abs(d_right - d_left) / total)


def yaw_blend_weight(
    yaw_asymmetry: float,
    *,
    degrade_start: float = YAW_DEGRADE_START,
    fallback_threshold: float = YAW_FALLBACK_THRESHOLD,
) -> float:
    """Compute smooth blend scale in [0, 1] for large yaw angles."""
    if yaw_asymmetry <= degrade_start:
        return 1.0
    if yaw_asymmetry >= fallback_threshold:
        return 0.0
    return float((fallback_threshold - yaw_asymmetry) / max(fallback_threshold - degrade_start, 1e-6))


def _odd_ksize(val: float, minimum: int = 3) -> int:
    k = max(minimum, int(round(val)))
    return k if k % 2 == 1 else k + 1


def build_mouth_jaw_masks(
    frame_shape: tuple[int, int],
    lmk_106: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (soft_blend_mask, core_binary_mask, color_ring_mask) in frame coordinates.

    All returned arrays have shape (H, W) with float32 values in [0, 1].
    The soft blend mask strictly covers only the sub-nasal mouth & chin region inside the
    face contour, leaving eyes, upper cheeks, and background 100% untouched.
    """
    h, w = int(frame_shape[0]), int(frame_shape[1])
    pts = np.asarray(lmk_106, dtype=np.float32)
    u_x, u_y, nose, face_width = _face_axes(pts)

    nose_pts = pts[list(NOSE_CENTER_INDICES)]
    mouth_pts = pts[list(MOUTH_INDICES)]
    jaw_pts = pts[list(JAW_CONTOUR_INDICES)]

    nose_y = (nose_pts - nose) @ u_y
    mouth_y = (mouth_pts - nose) @ u_y
    mouth_x = (mouth_pts - nose) @ u_x
    jaw_y = (jaw_pts - nose) @ u_y
    jaw_x = (jaw_pts - nose) @ u_x

    y_nose_bottom = float(np.max(nose_y))
    y_mouth_top = float(np.min(mouth_y))
    y_mouth_bottom = float(np.max(mouth_y))
    y_chin = float(np.max(jaw_y))

    # Keep the upper boundary strictly below the nose base (never reaching eyes/under-eyes).
    gap = max(0.0, y_mouth_top - y_nose_bottom)
    y_upper = y_nose_bottom + 0.15 * gap
    face_height = max(y_chin - y_nose_bottom, y_mouth_bottom - y_nose_bottom, face_width * 0.45)

    # Full face contour mask (shrunk slightly inward to stay off the silhouette/background edge).
    face_hull = cv2.convexHull(np.round(jaw_pts).astype(np.int32))
    face_contour_u8 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(face_contour_u8, face_hull, 255)
    contour_inset = _odd_ksize(face_width * 0.04, minimum=3)
    face_inner_u8 = cv2.erode(
        face_contour_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (contour_inset, contour_inset)),
    )

    # Lower-face polygon points: mouth + lower jaw points below y_upper + subtle chin drop margin
    # so wide-open vowels ("a") do not clip the moving chin.
    mouth_w = float(max(np.max(mouth_x) - np.min(mouth_x), face_width * 0.25))
    mouth_center_x = float(0.5 * (np.min(mouth_x) + np.max(mouth_x)))
    half_upper_w = min(mouth_w * 0.68, face_width * 0.36)

    # Clamp upper shoulder points inside the jaw width at that height.
    jaw_min_x = float(np.min(jaw_x)) + face_width * 0.06
    jaw_max_x = float(np.max(jaw_x)) - face_width * 0.06
    left_shoulder_x = max(jaw_min_x, mouth_center_x - half_upper_w)
    right_shoulder_x = min(jaw_max_x, mouth_center_x + half_upper_w)

    upper_left = nose + left_shoulder_x * u_x + y_upper * u_y
    upper_right = nose + right_shoulder_x * u_x + y_upper * u_y

    # Select jaw points below y_upper and pull them slightly inward laterally, while giving
    # the chin bottom a tiny downward margin for wide mouth openings.
    selected_jaw = []
    chin_margin = face_height * 0.04
    for pt, jx, jy in zip(jaw_pts, jaw_x, jaw_y):
        if jy < y_upper:
            continue
        # Pull lateral jaw slightly inward toward mouth center
        pulled_x = mouth_center_x + (jx - mouth_center_x) * 0.90
        pulled_y = jy + (chin_margin if jy >= y_mouth_bottom else 0.0)
        selected_jaw.append(nose + pulled_x * u_x + pulled_y * u_y)

    # Also include lower mouth corners expanded slightly outward for natural nasolabial transition.
    mouth_y_mid = 0.5 * (y_mouth_top + y_mouth_bottom)
    mid_left = nose + max(jaw_min_x, float(np.min(mouth_x)) - mouth_w * 0.20) * u_x + mouth_y_mid * u_y
    mid_right = nose + min(jaw_max_x, float(np.max(mouth_x)) + mouth_w * 0.20) * u_x + mouth_y_mid * u_y
    chin_center = nose + mouth_center_x * u_x + (y_chin + chin_margin) * u_y

    poly_points = [mouth_pts, np.stack([upper_left, upper_right, mid_left, mid_right, chin_center], axis=0)]
    if selected_jaw:
        poly_points.append(np.stack(selected_jaw, axis=0))
    all_pts = np.concatenate(poly_points, axis=0)
    hull = cv2.convexHull(np.round(all_pts).astype(np.int32))

    core_u8 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(core_u8, hull, 255)

    # Intersect with valid warped region (eroded to stay away from affine warp borders).
    valid_soft = np.ones((h, w), dtype=np.float32)
    if valid_mask is not None:
        valid_u8 = (np.asarray(valid_mask, dtype=np.float32) > 0.5).astype(np.uint8) * 255
        border_erode = _odd_ksize(face_width * 0.05, minimum=3)
        valid_inner_u8 = cv2.erode(
            valid_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_erode, border_erode)),
        )
        core_u8 = cv2.bitwise_and(core_u8, valid_inner_u8)
        valid_blur_k = _odd_ksize(face_width * 0.06, minimum=3)
        valid_soft = cv2.GaussianBlur(valid_inner_u8.astype(np.float32) / 255.0, (valid_blur_k, valid_blur_k), 0.0)

    # Also enforce strict upper-lip / sub-nasal clipping so Gaussian blur never bleeds near eyes.
    # Build feather mask: erode slightly inside core, then Gaussian blur.
    erode_k = _odd_ksize(face_width * 0.03, minimum=3)
    eroded_u8 = cv2.erode(core_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k)))
    if int(np.count_nonzero(eroded_u8)) == 0:
        eroded_u8 = core_u8

    sigma = max(2.5, face_width * 0.045)
    blur_k = _odd_ksize(sigma * 6.0 + 1.0, minimum=5)
    soft_mask = cv2.GaussianBlur(eroded_u8.astype(np.float32) / 255.0, (blur_k, blur_k), sigma)

    # Ensure core mouth lips stay near 1.0 so lip articulation is never washed out.
    lip_hull = cv2.convexHull(np.round(mouth_pts).astype(np.int32))
    lip_u8 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(lip_u8, lip_hull, 255)
    lip_dilate_k = _odd_ksize(face_width * 0.04, minimum=3)
    lip_u8 = cv2.dilate(lip_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lip_dilate_k, lip_dilate_k)))
    lip_sigma = max(1.5, face_width * 0.02)
    lip_blur_k = _odd_ksize(lip_sigma * 6.0 + 1.0, minimum=3)
    lip_soft = cv2.GaussianBlur(lip_u8.astype(np.float32) / 255.0, (lip_blur_k, lip_blur_k), lip_sigma)
    soft_mask = np.maximum(soft_mask, lip_soft)

    # Strictly zero out pixels outside the valid warped region and outside the upper-face cutoff.
    soft_mask = np.clip(soft_mask * valid_soft, 0.0, 1.0)
    # Zero out negligible feather tail (< 1/255) so pixels outside the mouth region are bit-exact original.
    soft_mask[soft_mask < (1.0 / 255.0)] = 0.0

    # Color-matching reference ring: skin band immediately surrounding the mouth core mask,
    # staying inside both the valid warped region and the inner face contour (no background/lips).
    ring_dilate_k = _odd_ksize(face_width * 0.14, minimum=7)
    outer_u8 = cv2.dilate(core_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_dilate_k, ring_dilate_k)))
    ring_u8 = cv2.subtract(outer_u8, core_u8)
    ring_u8 = cv2.bitwise_and(ring_u8, face_inner_u8)
    if valid_mask is not None:
        ring_u8 = cv2.bitwise_and(ring_u8, valid_inner_u8)

    # Fallback if face is tiny and ring is sparse: use non-lip region inside core_u8.
    if int(np.count_nonzero(ring_u8)) < 32:
        ring_u8 = cv2.subtract(core_u8, lip_u8)

    return (
        soft_mask.astype(np.float32),
        (core_u8.astype(np.float32) / 255.0),
        (ring_u8.astype(np.float32) / 255.0),
    )


def match_color_lab(
    generated_rgb: np.ndarray,
    source_rgb: np.ndarray,
    ring_mask: np.ndarray,
    *,
    max_mean_shift: float = 25.0,
    min_scale: float = 0.75,
    max_scale: float = 1.33,
) -> np.ndarray:
    """Match luminance & chroma of generated_rgb to source_rgb using surrounding skin ring in CIE Lab."""
    ring = np.asarray(ring_mask, dtype=np.float32) > 0.5
    if int(np.count_nonzero(ring)) < 16:
        return generated_rgb

    gen_u8 = np.clip(generated_rgb, 0, 255).astype(np.uint8)
    src_u8 = np.clip(source_rgb, 0, 255).astype(np.uint8)

    gen_lab = cv2.cvtColor(gen_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    src_lab = cv2.cvtColor(src_u8, cv2.COLOR_RGB2LAB).astype(np.float32)

    out_lab = gen_lab.copy()
    for ch in range(3):
        g_vals = gen_lab[:, :, ch][ring]
        s_vals = src_lab[:, :, ch][ring]
        g_mean = float(np.mean(g_vals))
        s_mean = float(np.mean(s_vals))
        g_std = float(np.std(g_vals))
        s_std = float(np.std(s_vals))

        if g_std > 1.0 and s_std > 1.0:
            scale = float(np.clip(s_std / g_std, min_scale, max_scale))
        else:
            scale = 1.0
        mean_shift = float(np.clip(s_mean - g_mean, -max_mean_shift, max_mean_shift))
        out_lab[:, :, ch] = (gen_lab[:, :, ch] - g_mean) * scale + g_mean + mean_shift

    out_lab = np.clip(out_lab, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)


def paste_back_frame(
    source_rgb: np.ndarray,
    generated_face_rgb: np.ndarray,
    affine_matrix: np.ndarray,
    lmk_106: np.ndarray | None,
    *,
    stats: PasteBackStats | None = None,
) -> np.ndarray:
    """Inverse-warp generated mouth/face crop and seamlessly blend into source_rgb.

    - Uses BORDER_REPLICATE during inverse warp to prevent 127-gray padding artifacts.
    - Applies sub-nasal mouth & jaw soft mask so non-mouth regions remain bit-exact source pixels.
    - Applies CIE Lab color/luminance matching on the surrounding skin ring before blending.
    - Degrades blend weight or falls back to source_rgb on extreme head yaw.
    """
    if stats is not None:
        stats.total_frames += 1

    if lmk_106 is None:
        if stats is not None:
            stats.faces_missing_frames += 1
        return source_rgb.copy()

    if stats is not None:
        stats.faces_detected_frames += 1

    yaw_asym = estimate_yaw_asymmetry(lmk_106)
    weight = yaw_blend_weight(yaw_asym)
    if weight <= 0.0:
        if stats is not None:
            stats.high_yaw_fallback_frames += 1
        return source_rgb.copy()
    if weight < 1.0 and stats is not None:
        stats.high_yaw_degraded_frames += 1

    h, w = source_rgb.shape[:2]
    mat = np.asarray(affine_matrix, dtype=np.float32)
    if mat.ndim == 3:
        mat = mat[0]
    if mat.shape != (2, 3):
        raise ValueError(f"Expected affine_matrix shape (2, 3), got {mat.shape}")

    gen_h, gen_w = generated_face_rgb.shape[:2]
    warped_face = cv2.warpAffine(
        np.clip(generated_face_rgb, 0, 255).astype(np.uint8),
        mat,
        (w, h),
        flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )
    valid_src = np.ones((gen_h, gen_w), dtype=np.float32)
    valid_mask = cv2.warpAffine(
        valid_src,
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )

    soft_mask, _, ring_mask = build_mouth_jaw_masks((h, w), lmk_106, valid_mask=valid_mask)
    if weight < 1.0:
        soft_mask = soft_mask * weight

    if float(np.max(soft_mask)) <= 0.0:
        return source_rgb.copy()

    matched_face = match_color_lab(warped_face, source_rgb, ring_mask)

    alpha = soft_mask[:, :, None]
    blended = alpha * matched_face.astype(np.float32) + (1.0 - alpha) * source_rgb.astype(np.float32)
    out = np.clip(np.rint(blended), 0, 255).astype(np.uint8)

    # Guarantee bit-exact preservation where soft_mask == 0.
    zero_region = soft_mask == 0.0
    out[zero_region] = source_rgb[zero_region]
    return out
