"""Internal request contracts. Public adapters supply the persisted task ID."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

class ProcessingRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    project_id: str = Field(pattern='^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$')
    idempotency_key: str = Field(pattern='^video_task_[0-9a-f]{32}$')
    source_artifact_id: str = Field(pattern='^art_[0-9a-f]{32}$')
    source_sha256: str = Field(pattern='^[0-9a-f]{64}$')

class SRRequest(ProcessingRequest):
    backend: Literal['seedvr2'] = 'seedvr2'
    quality_mode: Literal['standard'] = 'standard'
    target_width: int = Field(ge=16, le=2048)
    target_height: int = Field(ge=16, le=2048)
    seed: int = Field(default=666, ge=0, le=4294967295)

    @field_validator('target_width', 'target_height')
    @classmethod
    def divisible_by_16(cls, value):
        if value % 16:
            raise ValueError('SeedVR2 target dimensions must be divisible by 16')
        return value
