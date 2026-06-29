import json
import os
import zipfile
from app import bundle
from app.models import ReconstructOptions


def seed_job_dir(job_dir):
    for rel in ["cad/model.py", "cad/model.step", "preview/model.stl",
                "preview/render.png", "mesh/sam3d_mesh.ply", "metrics.json"]:
        full = os.path.join(job_dir, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write("x")


def test_write_manifest_contains_options_and_metrics(tmp_path):
    jd = str(tmp_path)
    path = bundle.write_manifest(jd, "job1", ReconstructOptions(), {"iou": 0.22})
    data = json.load(open(path))
    assert data["job_id"] == "job1"
    assert data["options"]["mode"] == "pc"
    assert data["metrics"]["iou"] == 0.22


def test_build_zip_includes_all_artifacts(tmp_path):
    jd = str(tmp_path)
    seed_job_dir(jd)
    bundle.write_manifest(jd, "job1", ReconstructOptions(), {})
    zp = bundle.build_zip(jd, "job1")
    with zipfile.ZipFile(zp) as z:
        names = set(z.namelist())
    assert "cad/model.py" in names
    assert "cad/model.step" in names
    assert "preview/model.stl" in names
    assert "preview/render.png" in names
    assert "mesh/sam3d_mesh.ply" in names
    assert "metrics.json" in names
    assert "manifest.json" in names


def test_build_zip_skips_missing_artifacts(tmp_path):
    jd = str(tmp_path)
    os.makedirs(os.path.join(jd, "cad"))
    with open(os.path.join(jd, "cad", "model.py"), "w") as f:
        f.write("x")
    zp = bundle.build_zip(jd, "job1")
    with zipfile.ZipFile(zp) as z:
        names = set(z.namelist())
    assert names == {"cad/model.py"}


def test_build_zip_includes_auto_mask_when_present(tmp_path):
    jd = str(tmp_path)
    os.makedirs(os.path.join(jd, "mesh"))
    with open(os.path.join(jd, "mesh", "auto_mask.png"), "w") as f:
        f.write("x")
    zp = bundle.build_zip(jd, "job1")
    with zipfile.ZipFile(zp) as z:
        assert "mesh/auto_mask.png" in set(z.namelist())
