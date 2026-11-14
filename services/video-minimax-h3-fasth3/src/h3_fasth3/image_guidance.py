"""Opt-in first-frame conditioning; not a pixel preservation guarantee."""
from .errors import RuntimeErrorCode


def validate_first_image(images, width, height):
    if not images:
        raise RuntimeErrorCode("first_frame_requires_image")
    shape = tuple(images[0].shape)
    if len(shape) != 4 or shape[0] != 1 or shape[1:3] != (height, width) or shape[3] != 3:
        raise RuntimeErrorCode("first_frame_canvas_mismatch")


def anchor_first_image(conditioning, width, height):
    if not conditioning:
        raise RuntimeErrorCode("invalid_first_frame_conditioning")
    result = []
    for embedding, metadata in conditioning:
        refs = [r for r in metadata.get("minimax_refs", []) if r.get("kind") == "image"]
        if not refs:
            raise RuntimeErrorCode("first_frame_requires_encoded_image")
        encoded = refs[0].get("latent")
        if encoded is None or tuple(encoded.shape) != (1, 24, 1, height // 16, width // 16):
            raise RuntimeErrorCode("first_frame_latent_shape_mismatch")
        keyframes = list(metadata.get("minimax_keyframes", []))
        if any(k.get("latent") is not None or k.get("image") is not None for k in keyframes):
            raise RuntimeErrorCode("conflicting_image_guidance")
        keyframes.append({"resolved_frame_index": 0, "latent": encoded})
        result.append([embedding, {**metadata, "minimax_keyframes": keyframes}])
    return result
