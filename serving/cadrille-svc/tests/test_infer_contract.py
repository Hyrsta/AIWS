import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fastapi.testclient import TestClient
from app.main import build_app
from app.config import Settings


class OkRunner:
    ready = True
    def run(self, settings, req):
        assert req["mode"] == "pc"
        return {"cad_code_path": "/a/cad/model.py", "step_path": "/a/cad/model.step",
                "preview_path": "/a/preview/model.stl", "metrics": {"iou": 0.2}}


class BoomRunner:
    ready = True
    def run(self, settings, req):
        raise RuntimeError("materialize timed out")


def settings():
    return Settings(cadrille_entry="x", cadrille_ckpt="x", device="cuda:0", stage_timeout_s=10)


def test_infer_returns_paths_and_metrics():
    client = TestClient(build_app(settings(), OkRunner()))
    r = client.post("/infer", json={"job_id": "j", "mesh_path": "/a/mesh/sam3d_mesh.ply",
                                    "mode": "pc", "n_candidates": 20, "seed": 42, "cleanup": True})
    assert r.status_code == 200
    assert r.json()["metrics"]["iou"] == 0.2


def test_infer_failure_returns_500():
    client = TestClient(build_app(settings(), BoomRunner()))
    r = client.post("/infer", json={"job_id": "j", "mesh_path": "/a/mesh/sam3d_mesh.ply",
                                    "mode": "pc", "n_candidates": 20, "seed": 42, "cleanup": True})
    assert r.status_code == 500
    assert "timed out" in r.text
