import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    sam3d_entry: str
    sam3d_ckpt: str
    device: str
    stage_timeout_s: float


def load_settings() -> Settings:
    return Settings(
        sam3d_entry=os.environ.get("SAM3D_ENTRY", "/opt/aiws/run_sam3d.py"),
        sam3d_ckpt=os.environ.get("SAM3D_CKPT", "/ckpt/sam3d"),
        device=os.environ.get("SAM3D_DEVICE", "cuda:0"),
        stage_timeout_s=float(os.environ.get("STAGE_TIMEOUT_S", "1800")),
    )
