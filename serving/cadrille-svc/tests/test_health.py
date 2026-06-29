import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fastapi.testclient import TestClient
from app.main import build_app
from app.config import Settings


class FakeRunner:
    ready = True
    def run(self, settings, req): return {}


def test_healthz_ok():
    app = build_app(Settings(cadrille_entry="x", cadrille_ckpt="x",
                             device="cuda:0", stage_timeout_s=10), FakeRunner())
    assert TestClient(app).get("/healthz").status_code == 200
