import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fastapi.testclient import TestClient
from app.main import build_app
from app.config import Settings


class FakeSeg:
    ready = True
    def segment(self, image_bytes, prompt, box_threshold, text_threshold):
        return {}


def test_healthz_ok():
    app = build_app(Settings(default_prompt="workpiece. metal part.",
                             grounding_dino_ckpt="x", sam_ckpt="x",
                             sam_model_type="vit_h", device="cuda:0",
                             box_threshold=0.3, text_threshold=0.25), FakeSeg())
    assert TestClient(app).get("/healthz").status_code == 200
