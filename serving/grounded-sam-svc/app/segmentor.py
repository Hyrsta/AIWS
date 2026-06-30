import base64
import io
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image


class NoDetection(Exception):
    pass


@dataclass
class _ImageState:
    image_np: object        # HxWx3 uint8 ndarray
    width: int
    height: int
    points: list = field(default_factory=list)       # list[(x, y, label)]
    box: object = None                                # [x0,y0,x1,y1] or None
    low_res_mask: object = None                       # SAM low-res mask logits or None
    sam_features: object = None                       # cached predictor.features
    sam_input_size: object = None
    sam_original_size: object = None


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

    @staticmethod
    def _mask_to_b64(mask_uint8) -> str:
        buf = io.BytesIO()
        Image.fromarray(mask_uint8, mode="L").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _detect(self, image_np, image_size, prompt, box_threshold, text_threshold) -> dict:
        import torch
        image = Image.fromarray(image_np)
        text = self._normalize_prompt(prompt)
        inputs = self._processor(images=image, text=text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._detector(**inputs)
        results = self._processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=box_threshold,
            text_threshold=text_threshold, target_sizes=[image_size[::-1]])[0]
        boxes = results["boxes"]
        scores = results["scores"]
        labels = results.get("labels") or results.get("text_labels") or []
        if int(len(boxes)) == 0:
            raise NoDetection(prompt)
        best = int(torch.argmax(scores).item())
        return {"box": boxes[best].detach().cpu().numpy(),
                "score": float(scores[best].item()),
                "label": str(labels[best]) if best < len(labels) else prompt,
                "num_detections": int(len(boxes))}

    def segment(self, image_bytes, prompt, box_threshold, text_threshold) -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)
        det = self._detect(image_np, image.size, prompt, box_threshold, text_threshold)
        self._predictor.set_image(image_np)
        masks, _, _ = self._predictor.predict(box=det["box"][None, :], multimask_output=False)
        mask = masks[0].astype(np.uint8) * 255
        return {"mask_png_base64": self._mask_to_b64(mask),
                "score": round(det["score"], 4),
                "box": [round(float(v), 2) for v in det["box"].tolist()],
                "label": det["label"], "num_detections": det["num_detections"]}

    def _restore(self, state):
        # Re-seat the cached image embedding into the shared predictor.
        p = self._predictor
        p.features = state.sam_features
        p.input_size = state.sam_input_size
        p.original_size = state.sam_original_size
        p.is_image_set = True

    def encode_image(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)
        self._predictor.set_image(image_np)
        st = _ImageState(image_np=image_np, width=image.size[0], height=image.size[1])
        st.sam_features = self._predictor.features
        st.sam_input_size = self._predictor.input_size
        st.sam_original_size = self._predictor.original_size
        return st

    def auto_mask(self, state, prompt, box_threshold, text_threshold):
        self._restore(state)
        try:
            det = self._detect(state.image_np, (state.width, state.height),
                               prompt, box_threshold, text_threshold)
        except NoDetection:
            empty = np.zeros((state.height, state.width), dtype=np.uint8)
            state.box = None
            state.low_res_mask = None
            state.points = []
            return {"mask_png_base64": self._mask_to_b64(empty), "score": 0.0,
                    "box": None, "width": state.width, "height": state.height,
                    "detected": False}
        masks, _, low = self._predictor.predict(box=det["box"][None, :], multimask_output=False)
        state.box = det["box"].tolist()
        state.points = []
        state.low_res_mask = low
        mask = masks[0].astype(np.uint8) * 255
        return {"mask_png_base64": self._mask_to_b64(mask),
                "score": round(det["score"], 4),
                "box": [round(float(v), 2) for v in det["box"].tolist()],
                "width": state.width, "height": state.height, "detected": True}

    def refine_mask(self, state, prompt, points, box, reset, box_threshold, text_threshold):
        self._restore(state)
        if reset:
            state.points = []
            state.box = None
            state.low_res_mask = None
            empty = np.zeros((state.height, state.width), dtype=np.uint8)
            return {"mask_png_base64": self._mask_to_b64(empty), "score": 0.0,
                    "width": state.width, "height": state.height}
        if prompt:
            # Fresh text detection replaces accumulated points/box.
            det = self._detect(state.image_np, (state.width, state.height),
                               prompt, box_threshold, text_threshold)
            state.points = []
            state.box = det["box"].tolist()
            masks, _, low = self._predictor.predict(box=det["box"][None, :], multimask_output=False)
        else:
            if points:
                for (x, y, lab) in points:
                    state.points.append((float(x), float(y), int(lab)))
            if box is not None:
                state.box = [float(v) for v in box]
            pc = np.array([[p[0], p[1]] for p in state.points], dtype=np.float32) if state.points else None
            pl = np.array([p[2] for p in state.points], dtype=np.int32) if state.points else None
            bx = np.array(state.box, dtype=np.float32)[None, :] if state.box is not None else None
            mi = state.low_res_mask if state.low_res_mask is not None else None
            masks, _, low = self._predictor.predict(
                point_coords=pc, point_labels=pl, box=bx,
                mask_input=mi, multimask_output=False)
        state.low_res_mask = low
        mask = masks[0].astype(np.uint8) * 255
        return {"mask_png_base64": self._mask_to_b64(mask), "score": 0.0,
                "width": state.width, "height": state.height}
