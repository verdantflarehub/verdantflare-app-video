"""Strict FastH3 single-frame guide contract and live capability checks."""
DESCRIPTION = (
    "Experimental h3-fasth3 single-image time anchors. Zero-based indices address references.images "
    "and base resolved_output frames, never postprocessed delivery_output. Exact base canvas required; "
    "not pixel preservation, a camera lock or a continuous hold guarantee."
)
SCHEMA = {"type": ["array", "null"], "minItems": 1, "maxItems": 8, "items": {
    "type": "object", "additionalProperties": False,
    "properties": {"image_index": {"type": "integer", "minimum": 0},
                   "frame_index": {"type": "integer", "minimum": 0}},
    "required": ["image_index", "frame_index"],
}}


def check_route(guides, service, image_guidance=None):
    if guides is None:
        return
    if service != "h3-fasth3":
        raise ValueError("image_guides_require_h3_fasth3")
    if image_guidance not in (None, "reference"):
        raise ValueError("conflicting_image_guidance")


def validate(guides, images, output):
    if not isinstance(guides, list) or not 1 <= len(guides) <= 8:
        raise ValueError("image_guides_must_be_1_to_8")
    previous = -1
    for guide in guides:
        if not isinstance(guide, dict) or set(guide) != {"image_index", "frame_index"}:
            raise ValueError("invalid_image_guide_fields")
        image, frame = guide["image_index"], guide["frame_index"]
        if type(image) is not int or not 0 <= image < len(images):
            raise ValueError("image_guide_image_out_of_range")
        if type(frame) is not int or not 0 <= frame < output["frames"]:
            raise ValueError("image_guide_frame_out_of_range")
        if frame <= previous:
            raise ValueError("image_guides_must_be_strictly_ordered")
        if any(images[image][key] != output[key] for key in ("width", "height")):
            raise ValueError("image_guide_canvas_mismatch")
        previous = frame
    return [dict(guide) for guide in guides]


def runtime_supports(health):
    caps = health.get("capabilities") if isinstance(health, dict) else None
    value = caps.get("image_guides") if isinstance(caps, dict) else None
    return (isinstance(value, dict) and value.get("schema") == "single-frame-v1"
            and type(value.get("maximum")) is int and value["maximum"] == 8
            and type(value.get("index_base")) is int and value["index_base"] == 0)
