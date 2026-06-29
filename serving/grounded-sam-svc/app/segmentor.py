import os


class NoDetection(Exception):
    pass


class Segmentor:
    """Interface for GroundingDINO + SAM. The real model loading and inference
    are implemented in Phase 2 on the RXL host. Contract tests inject a fake.
    """

    def __init__(self):
        self.ready = False

    def warmup(self, settings) -> None:
        if not os.path.exists(settings.grounding_dino_ckpt):
            raise RuntimeError(f"grounding-dino checkpoint not found: {settings.grounding_dino_ckpt}")
        if not os.path.exists(settings.sam_ckpt):
            raise RuntimeError(f"sam checkpoint not found: {settings.sam_ckpt}")
        # Phase 2: load GroundingDINO + SAM here.
        self.ready = True

    def segment(self, image_bytes, prompt, box_threshold, text_threshold) -> dict:
        # Phase 2: run GroundingDINO detection then SAM masking; raise NoDetection
        # when nothing clears box_threshold. Return mask_png_base64, score, box,
        # label, num_detections.
        raise NotImplementedError("real segmentor implemented in Phase 2")
