from __future__ import annotations

import base64
import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MODELS = {
    "grok_video_v15_lite": {"minimum_duration": 2, "maximum_duration": 15,
                           "ratios": ["16:9", "9:16", "4:3", "3:4", "1:1"], "first_frame": True},
    "seedance_2_5": {"minimum_duration": 4, "maximum_duration": 15,
                     "ratios": ["16:9", "9:16", "4:3", "3:4", "1:1", "21:9"], "first_frame": False},
}


class SourceImage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    content_base64: str = Field(min_length=1, max_length=14 * 1024 * 1024)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    format: Literal["png", "jpeg", "webp"]

    def content(self):
        data = base64.b64decode(self.content_base64, validate=True)
        if not 0 < len(data) <= 10 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != self.sha256:
            raise ValueError("source_integrity_failed")
        return data


class Generation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: Literal["grok_video_v15_lite", "seedance_2_5"]
    prompt: str = Field(min_length=1, max_length=10000)
    duration_seconds: int
    aspect_ratio: str = "16:9"
    resolution: Literal["480p", "720p"] = "720p"
    mode: Literal["text_to_video", "image_to_video"] = "text_to_video"
    source: SourceImage | None = None

    @field_validator("prompt")
    @classmethod
    def prompt_valid(cls, value):
        if not value.strip() or "\x00" in value:
            raise ValueError("invalid_prompt")
        return value

    @model_validator(mode="after")
    def supported(self):
        model = MODELS[self.model]
        if not model["minimum_duration"] <= self.duration_seconds <= model["maximum_duration"]:
            raise ValueError("unsupported_duration")
        if self.aspect_ratio not in model["ratios"]:
            raise ValueError("unsupported_aspect_ratio")
        if self.model == "seedance_2_5" and self.resolution != "720p":
            raise ValueError("unsupported_resolution")
        if (self.source is not None) != (self.mode == "image_to_video"):
            raise ValueError("mode_reference_mismatch")
        if self.source and not model["first_frame"]:
            raise ValueError("unsupported_references")
        return self

    def arguments(self, operation: str, source_path=None):
        result = ["generate", operation, self.model, "--prompt", self.prompt,
                  "--duration", str(self.duration_seconds), "--resolution", self.resolution,
                  "--aspect_ratio", self.aspect_ratio]
        if self.model == "seedance_2_5":
            result.extend(["--mode", "t2v"])
        if source_path is not None:
            result.extend(["--start-image", str(source_path)])
        return result


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    request_id: str = Field(pattern=r"^video_task_[0-9a-f]{32}$")
    binding_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    organization_id: str = Field(pattern=r"^[0-9a-f-]{36}$")
    user_id: str = Field(pattern=r"^[0-9a-f-]{36}$")
    project_id: str = Field(pattern=r"^[0-9a-f-]{36}$")
    generation: Generation
