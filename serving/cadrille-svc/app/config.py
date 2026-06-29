import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    cadrille_entry: str
    cadrille_ckpt: str
    device: str
    stage_timeout_s: float


def load_settings() -> Settings:
    return Settings(
        cadrille_entry=os.environ.get("CADRILLE_ENTRY", "/opt/aiws/run_cadrille.py"),
        cadrille_ckpt=os.environ.get("CADRILLE_CKPT", "/ckpt/cadrille"),
        device=os.environ.get("CADRILLE_DEVICE", "cuda:0"),
        stage_timeout_s=float(os.environ.get("STAGE_TIMEOUT_S", "1800")),
    )
