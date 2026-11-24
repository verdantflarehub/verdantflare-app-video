"""Explicit single-image time anchors; no pixel or camera-path guarantee."""
from .errors import RuntimeErrorCode

CAPABILITY = {"schema": "single-frame-v1", "maximum": 8, "index_base": 0}


def validate_guides(guides, image_count, frame_count, image_guidance=None):
    if guides is None:
        return
    if image_guidance not in (None, "reference"):
        raise RuntimeErrorCode("conflicting_image_guidance")
    if not isinstance(guides, list) or not 1 <= len(guides) <= CAPABILITY["maximum"]:
        raise RuntimeErrorCode("image_guides_must_be_1_to_8")
    previous = -1
    for guide in guides:
        if not isinstance(guide, dict) or set(guide) != {"image_index", "frame_index"}:
            raise RuntimeErrorCode("invalid_image_guide_fields")
        image, frame = guide["image_index"], guide["frame_index"]
        if type(image) is not int or not 0 <= image < image_count:
            raise RuntimeErrorCode("image_guide_image_out_of_range")
        if type(frame) is not int or not 0 <= frame < frame_count:
            raise RuntimeErrorCode("image_guide_frame_out_of_range")
        if frame <= previous:
            raise RuntimeErrorCode("image_guides_must_be_strictly_ordered")
        previous = frame


def validate_images(images, guides, width, height, frame_count):
    validate_guides(guides, len(images), frame_count)
    for index in {g["image_index"] for g in guides}:
        if tuple(images[index].shape) != (1, height, width, 3):
            raise RuntimeErrorCode("image_guide_canvas_mismatch")


def anchor_images(conditioning, images, guides, width, height, frame_count, encode):
    """Encode exact pixels separately from resized Ref2VA references."""
    validate_images(images, guides, width, height, frame_count)
    if not isinstance(conditioning, (list, tuple)) or not conditioning:
        raise RuntimeErrorCode("invalid_image_guide_conditioning")
    for item in conditioning:
        if not isinstance(item, (list, tuple)) or len(item) != 2 or not isinstance(item[1], dict):
            raise RuntimeErrorCode("invalid_image_guide_conditioning")
        existing = item[1].get("minimax_keyframes", [])
        if not isinstance(existing, list) or any(not isinstance(k, dict) for k in existing):
            raise RuntimeErrorCode("invalid_image_guide_conditioning")
        if any(k.get("image") is not None or k.get("latent") is not None for k in existing):
            raise RuntimeErrorCode("conflicting_image_guidance")
    encoded = {}
    for guide in guides:
        index = guide["image_index"]
        if index not in encoded:
            latent = encode(images[index])
            if tuple(latent.shape) != (1, 24, 1, height // 16, width // 16):
                raise RuntimeErrorCode("image_guide_latent_shape_mismatch")
            encoded[index] = latent
    return [[embedding, {**metadata, "minimax_keyframes": list(metadata.get("minimax_keyframes", [])) + [
        {"resolved_frame_index": g["frame_index"], "latent": encoded[g["image_index"]]} for g in guides
    ]}] for embedding, metadata in conditioning]
