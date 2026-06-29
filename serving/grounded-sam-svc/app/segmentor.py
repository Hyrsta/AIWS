import base64
import io
import os

import numpy as np
from PIL import Image


class NoDetection(Exception):
    pass


class Segmentor:
    """GroundingDINO (HF transformers, zero-shot detection) + SAM (mask).

    Validated approach on RXL: detection via transformers
    `IDEA-Research/grounding-dino-tiny` (downloaded through HF_ENDPOINT mirror,
    cached under a writable dir), masking via segment-anything `vit_h` with a
    local checkpoint. Returns one mask for the highest-confidence detection.
    """

    def __init__(self):
        self.ready = False
        self._processor = None
        self._detector = None
        self._predictor = None
        self._device = "cpu"

    def warmup(self, settings) -> None:
        # Set HF env before importing transformers so the mirror + cache dir are
        # honored when the detector weights are downloaded.
        if settings.hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)
        if settings.hf_home:
            os.environ.setdefault("HF_HOME", settings.hf_home)

        import torch  # noqa: F401
        from segment_anything import SamPredictor, sam_model_registry
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        if not os.path.exists(settings.sam_ckpt):
            raise RuntimeError(f"sam checkpoint not found: {settings.sam_ckpt}")

        self._device = settings.device

        model_id = settings.grounding_dino_model_id
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(self._device)
        self._detector.eval()

        sam = sam_model_registry[settings.sam_model_type](checkpoint=settings.sam_ckpt)
        sam.to(self._device)
        self._predictor = SamPredictor(sam)
        self.ready = True

    @staticmethod
    def _normalize_prompt(prompt: str) -> str:
        # GroundingDINO expects lowercase phrases, each ending with a period.
        text = prompt.strip().lower()
        if not text.endswith("."):
            text = text + "."
        return text

    def segment(self, image_bytes, prompt, box_threshold, text_threshold) -> dict:
        import torch

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)

        text = self._normalize_prompt(prompt)
        inputs = self._processor(images=image, text=text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._detector(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],  # (height, width)
        )[0]

        boxes = results["boxes"]
        scores = results["scores"]
        labels = results.get("labels") or results.get("text_labels") or []
        num_detections = int(len(boxes))
        if num_detections == 0:
            raise NoDetection(prompt)

        best = int(torch.argmax(scores).item())
        best_box = boxes[best].detach().cpu().numpy()  # [x0, y0, x1, y1]
        best_score = float(scores[best].item())
        best_label = str(labels[best]) if best < len(labels) else prompt

        self._predictor.set_image(image_np)
        masks, mask_scores, _ = self._predictor.predict(
            box=best_box[None, :],
            multimask_output=False,
        )
        mask = masks[0].astype(np.uint8) * 255

        buf = io.BytesIO()
        Image.fromarray(mask, mode="L").save(buf, format="PNG")
        mask_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "mask_png_base64": mask_b64,
            "score": round(best_score, 4),
            "box": [round(float(v), 2) for v in best_box.tolist()],
            "label": best_label,
            "num_detections": num_detections,
        }
