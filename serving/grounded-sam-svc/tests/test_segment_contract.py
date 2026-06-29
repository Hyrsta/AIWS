import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import io
from fastapi.testclient import TestClient
from app.main import build_app
from app.config import Settings
from app.segmentor import NoDetection


def settings():
    return Settings(default_prompt="workpiece. metal part.", grounding_dino_ckpt="x",
                    sam_ckpt="x", sam_model_type="vit_h", device="cuda:0",
                    box_threshold=0.3, text_threshold=0.25)


class OkSeg:
    ready = True
    def segment(self, image_bytes, prompt, box_threshold, text_threshold):
        assert prompt == "workpiece. metal part."
        return {"mask_png_base64": "AAAA", "score": 0.6, "box": [0, 0, 2, 2],
                "label": "workpiece", "num_detections": 1}


class NoneSeg:
    ready = True
    def segment(self, image_bytes, prompt, box_threshold, text_threshold):
        raise NoDetection("workpiece. metal part.")


def png():
    return io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"0" * 16)


def test_segment_returns_mask_with_default_prompt():
    client = TestClient(build_app(settings(), OkSeg()))
    r = client.post("/segment", files={"image": ("a.png", png(), "image/png")})
    assert r.status_code == 200
    assert r.json()["mask_png_base64"] == "AAAA"


def test_segment_override_prompt():
    captured = {}
    class CapSeg:
        ready = True
        def segment(self, image_bytes, prompt, box_threshold, text_threshold):
            captured["prompt"] = prompt
            return {"mask_png_base64": "BBBB", "score": 0.4, "box": [0, 0, 1, 1],
                    "label": "p", "num_detections": 1}
    client = TestClient(build_app(settings(), CapSeg()))
    r = client.post("/segment", files={"image": ("a.png", png(), "image/png")},
                    data={"prompt": "bracket"})
    assert r.status_code == 200
    assert captured["prompt"] == "bracket"


def test_segment_no_detection_returns_422():
    client = TestClient(build_app(settings(), NoneSeg()))
    r = client.post("/segment", files={"image": ("a.png", png(), "image/png")})
    assert r.status_code == 422
