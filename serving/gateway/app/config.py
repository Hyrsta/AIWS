import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    sam3d_url: str
    cadrille_url: str
    artifacts_dir: str
    db_path: str
    workers: int
    stage_timeout_s: float
    max_images: int


def load_settings() -> Settings:
    return Settings(
        sam3d_url=os.environ.get("SAM3D_URL", "http://sam3d-svc:8000"),
        cadrille_url=os.environ.get("CADRILLE_URL", "http://cadrille-svc:8000"),
        artifacts_dir=os.environ.get("ARTIFACTS_DIR", "/artifacts"),
        db_path=os.environ.get("DB_PATH", "/artifacts/jobs.db"),
        workers=int(os.environ.get("WORKERS", "1")),
        stage_timeout_s=float(os.environ.get("STAGE_TIMEOUT_S", "1800")),
        max_images=int(os.environ.get("MAX_IMAGES", "16")),
    )
