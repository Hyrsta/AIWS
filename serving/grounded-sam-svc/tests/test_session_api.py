import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import io
from fastapi.testclient import TestClient
from app.main import build_app
from app.config import Settings


def settings():
    return Settings(default_prompt="workpiece. metal part.", grounding_dino_ckpt="x",
                    sam_ckpt="x", sam_model_type="vit_h", device="cuda:0",
                    box_threshold=0.3, text_threshold=0.25, max_sessions=4)


class FakeSeg:
    ready = True
    def segment(self, *a, **k):
        return {"mask_png_base64": "S", "score": 0.5, "box": [0, 0, 1, 1],
                "label": "x", "num_detections": 1}
    def encode_image(self, image_bytes):
        return {"img": image_bytes}
    def auto_mask(self, state, prompt, box_threshold, text_threshold):
        return {"mask_png_base64": "AUTO", "score": 0.5, "box": [0, 0, 1, 1],
                "width": 4, "height": 3, "detected": True}
    def refine_mask(self, state, prompt, points, box, reset, box_threshold, text_threshold):
        return {"mask_png_base64": "REF", "score": 0.6, "width": 4, "height": 3}


def png():
    return io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"0" * 16)


def client():
    return TestClient(build_app(settings(), FakeSeg()))


def test_session_create_returns_id_and_mask():
    r = client().post("/segment/session", files={"image": ("a.png", png(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["mask_png_base64"] == "AUTO"
    assert body["session_id"]


def test_refine_with_points_returns_mask():
    c = client()
    sid = c.post("/segment/session", files={"image": ("a.png", png(), "image/png")}).json()["session_id"]
    r = c.post("/segment/refine", json={"session_id": sid,
              "points": [{"x": 2, "y": 3, "label": 1}]})
    assert r.status_code == 200
    assert r.json()["mask_png_base64"] == "REF"


def test_refine_unknown_session_returns_409():
    r = client().post("/segment/refine", json={"session_id": "nope", "points": []})
    assert r.status_code == 409


def test_delete_session_returns_204():
    c = client()
    sid = c.post("/segment/session", files={"image": ("a.png", png(), "image/png")}).json()["session_id"]
    r = c.delete(f"/segment/session/{sid}")
    assert r.status_code == 204
    # refine after delete -> 409
    r2 = c.post("/segment/refine", json={"session_id": sid, "points": []})
    assert r2.status_code == 409


class NoDetectSeg(FakeSeg):
    def refine_mask(self, state, prompt, points, box, reset, box_threshold, text_threshold):
        from app.segmentor import NoDetection
        raise NoDetection(prompt)


def test_refine_no_detection_prompt_returns_422():
    c = TestClient(build_app(settings(), NoDetectSeg()))
    sid = c.post("/segment/session", files={"image": ("a.png", png(), "image/png")}).json()["session_id"]
    r = c.post("/segment/refine", json={"session_id": sid, "prompt": "ghost"})
    assert r.status_code == 422
