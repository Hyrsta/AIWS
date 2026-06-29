import io
import os
import time
import zipfile

import httpx
import pytest

GATEWAY = os.environ.get("GATEWAY_URL", "http://localhost:8080")
SAMPLE = os.environ.get("SAMPLE_IMAGE", "")


@pytest.mark.integration
def test_full_pipeline_produces_complete_bundle():
    assert SAMPLE and os.path.exists(SAMPLE), "set SAMPLE_IMAGE to a real test image"
    with open(SAMPLE, "rb") as f:
        files = {"images": (os.path.basename(SAMPLE), f, "image/png")}
        r = httpx.post(f"{GATEWAY}/v1/reconstruct", files=files,
                       data={"mode": "pc"}, timeout=30)
    assert r.status_code == 202
    jid = r.json()["job_id"]

    deadline = time.time() + 2400
    status = None
    while time.time() < deadline:
        s = httpx.get(f"{GATEWAY}/v1/jobs/{jid}", timeout=30).json()
        status = s["status"]
        if status in ("succeeded", "failed"):
            break
        time.sleep(5)
    assert status == "succeeded", f"job ended {status}: {s.get('error')}"

    rr = httpx.get(f"{GATEWAY}/v1/jobs/{jid}/result", timeout=120)
    assert rr.status_code == 200
    z = zipfile.ZipFile(io.BytesIO(rr.content))
    names = set(z.namelist())
    for expected in ["cad/model.py", "cad/model.step", "preview/model.stl",
                     "mesh/sam3d_mesh.ply", "metrics.json", "manifest.json"]:
        assert expected in names, f"missing {expected}"
    step = z.read("cad/model.step").decode("latin-1")
    assert "ISO-10303" in step, "STEP file header missing"
