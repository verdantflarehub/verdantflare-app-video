"""Internal request contracts. Public adapters supply the persisted task ID."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

class ProcessingRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    project_id: str = Field(pattern='^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$')
    idempotency_key: str = Field(pattern='^video_task_[0-9a-f]{32}$')
    source_artifact_id: str = Field(pattern='^art_[0-9a-f]{32}$')
    source_sha256: str = Field(pattern='^[0-9a-f]{64}$')

class InterpolationRequest(ProcessingRequest):
    backend: Literal['rife'] = 'rife'
    target_fps_num: int = Field(default=48, ge=1, le=120000)
    target_fps_den: int = Field(default=1, ge=1, le=1001)
