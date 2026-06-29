import json
import os
import zipfile

from .models import ReconstructOptions

ARTIFACT_NAMES = [
    "cad/model.py",
    "cad/model.step",
    "preview/model.stl",
    "preview/render.png",
    "mesh/sam3d_mesh.ply",
    "mesh/auto_mask.png",
    "metrics.json",
]


def write_manifest(job_dir: str, job_id: str, options: ReconstructOptions,
                   metrics: dict) -> str:
    manifest = {
        "job_id": job_id,
        "options": options.model_dump(mode="json"),
        "metrics": metrics,
        "artifacts": [n for n in ARTIFACT_NAMES if os.path.exists(os.path.join(job_dir, n))],
    }
    path = os.path.join(job_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def build_zip(job_dir: str, job_id: str) -> str:
    zip_path = os.path.join(job_dir, f"{job_id}.zip")
    members = ARTIFACT_NAMES + ["manifest.json"]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in members:
            full = os.path.join(job_dir, rel)
            if os.path.exists(full):
                z.write(full, rel)
    return zip_path
