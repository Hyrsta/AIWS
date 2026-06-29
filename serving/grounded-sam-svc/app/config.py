import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    default_prompt: str
    grounding_dino_ckpt: str
    sam_ckpt: str
    sam_model_type: str
    device: str
    box_threshold: float
    text_threshold: float


def load_settings() -> Settings:
    return Settings(
        default_prompt=os.environ.get("WORKPIECE_PROMPT", "workpiece. metal part."),
        grounding_dino_ckpt=os.environ.get("GROUNDING_DINO_CKPT", "/ckpt/grounding_dino"),
        sam_ckpt=os.environ.get("SAM_CKPT", "/ckpt/sam"),
        sam_model_type=os.environ.get("SAM_MODEL_TYPE", "vit_h"),
        device=os.environ.get("SEG_DEVICE", "cuda:0"),
        box_threshold=float(os.environ.get("BOX_THRESHOLD", "0.3")),
        text_threshold=float(os.environ.get("TEXT_THRESHOLD", "0.25")),
    )
