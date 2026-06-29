from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Mode(str, Enum):
    pc = "pc"
    img = "img"


class JobState(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class Stage(str, Enum):
    sam3d = "sam3d"
    cadrille = "cadrille"


class ReconstructOptions(BaseModel):
    mode: Mode = Mode.pc
    n_candidates: int = Field(default=20, ge=1, le=64)
    seed: int = 42
    cleanup: bool = True


class JobView(BaseModel):
    job_id: str
    status: JobState
    stage: Optional[Stage] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str
