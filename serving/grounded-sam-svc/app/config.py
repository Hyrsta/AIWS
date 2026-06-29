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
    # GroundingDINO is loaded via HF transformers (zero-shot detection); this is
    # a model id downloaded to hf_home (use a mirror via hf_endpoint where direct
    # HF access is blocked). grounding_dino_ckpt is kept for a future local-weights
    # path but is unused by the transformers loader. New fields have defaults so
    # existing test Settings(...) construction keeps working.
    grounding_dino_model_id: str = "IDEA-Research/grounding-dino-tiny"
    hf_endpoint: str = "https://hf-mirror.com"
    hf_home: str = "/ckpt/hf-cache"


def load_settings() -> Settings:
    return Settings(
        default_prompt=os.environ.get("WORKPIECE_PROMPT", "workpiece. metal part."),
        grounding_dino_ckpt=os.environ.get("GROUNDING_DINO_CKPT", "/ckpt/grounding_dino"),
        sam_ckpt=os.environ.get("SAM_CKPT", "/ckpt/sam/sam_vit_h_4b8939.pth"),
        sam_model_type=os.environ.get("SAM_MODEL_TYPE", "vit_h"),
        device=os.environ.get("SEG_DEVICE", "cuda:0"),
        box_threshold=float(os.environ.get("BOX_THRESHOLD", "0.3")),
        text_threshold=float(os.environ.get("TEXT_THRESHOLD", "0.25")),
        grounding_dino_model_id=os.environ.get(
            "GROUNDING_DINO_MODEL_ID", "IDEA-Research/grounding-dino-tiny"
        ),
        hf_endpoint=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
        hf_home=os.environ.get("HF_HOME", "/ckpt/hf-cache"),
    )
