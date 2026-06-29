import base64
import json
import os

from app.jobstore import JobStore
from app.orchestrator import Orchestrator
from app.clients import ServiceError
from app.models import ReconstructOptions, JobState, Stage

PNG_1X1 = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000a49444154789c6360000000020001a5f645400000000049454e44ae426082"
    )
).decode()


class FakeGroundedSam:
    def __init__(self, fail=False):
        self.fail = fail

    def segment(self, image_path, prompt):
        if self.fail:
            raise ServiceError("segment", "no object")
        return {"mask_png_base64": PNG_1X1, "score": 0.7, "box": [0, 0, 1, 1],
                "label": "workpiece", "num_detections": 1}


class FakeSam3d:
    def __init__(self, fail=False):
        self.fail = fail

    def infer(self, job_id, input_dir, mask_path):
        assert mask_path.endswith("mask.png")
        if self.fail:
            raise ServiceError("sam3d", "sam3d boom")
        job_dir = os.path.dirname(input_dir)
        mesh = os.path.join(job_dir, "mesh", "sam3d_mesh.ply")
        os.makedirs(os.path.dirname(mesh), exist_ok=True)
        open(mesh, "w").write("ply")
        return mesh


class FakeCadrille:
    def __init__(self, fail=False):
        self.fail = fail

    def infer(self, job_id, mesh_path, options):
        if self.fail:
            raise ServiceError("cadrille", "cadrille boom")
        job_dir = os.path.dirname(os.path.dirname(mesh_path))
        cad = os.path.join(job_dir, "cad")
        os.makedirs(cad, exist_ok=True)
        open(os.path.join(cad, "model.py"), "w").write("import cadquery")
        return {"metrics": {"iou": 0.2}}


def build(tmp_path, sam_fail=False, cad_fail=False, seg_fail=False):
    store = JobStore(os.path.join(tmp_path, "jobs.db"))
    orch = Orchestrator(
        store, FakeGroundedSam(seg_fail), FakeSam3d(sam_fail), FakeCadrille(cad_fail),
        str(tmp_path)
    )
    return store, orch


def seed_input_image(orch, jid):
    input_dir = os.path.join(orch.job_dir(jid), "input")
    os.makedirs(input_dir, exist_ok=True)
    with open(os.path.join(input_dir, "000.png"), "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
    return input_dir


def test_happy_path_succeeds_with_zip(tmp_path):
    store, orch = build(tmp_path)
    jid = store.create(ReconstructOptions())
    seed_input_image(orch, jid)
    orch.run(jid)
    job = store.get(jid)
    assert job.status == JobState.succeeded
    assert job.stage is None
    assert os.path.exists(os.path.join(orch.job_dir(jid), f"{jid}.zip"))


def test_sam3d_failure_marks_stage(tmp_path):
    store, orch = build(tmp_path, sam_fail=True)
    jid = store.create(ReconstructOptions())
    seed_input_image(orch, jid)
    orch.run(jid)
    job = store.get(jid)
    assert job.status == JobState.failed
    assert job.stage == Stage.sam3d
    assert "sam3d boom" in job.error


def test_cadrille_failure_marks_stage(tmp_path):
    store, orch = build(tmp_path, cad_fail=True)
    jid = store.create(ReconstructOptions())
    seed_input_image(orch, jid)
    orch.run(jid)
    job = store.get(jid)
    assert job.status == JobState.failed
    assert job.stage == Stage.cadrille


def test_auto_segment_writes_mask_and_succeeds(tmp_path):
    store, orch = build(tmp_path)
    jid = store.create(ReconstructOptions())  # segment defaults to auto
    seed_input_image(orch, jid)
    orch.run(jid)
    job = store.get(jid)
    assert job.status == JobState.succeeded
    assert os.path.exists(os.path.join(orch.job_dir(jid), "input", "mask.png"))
    assert os.path.exists(os.path.join(orch.job_dir(jid), "mesh", "auto_mask.png"))
    manifest = json.load(open(os.path.join(orch.job_dir(jid), "manifest.json")))
    seg = manifest["metrics"]["segment"]
    assert seg["score"] == 0.7
    assert seg["box"] == [0, 0, 1, 1]
    assert "prompt" in seg


def test_segment_failure_marks_segment_stage(tmp_path):
    store, orch = build(tmp_path, seg_fail=True)
    jid = store.create(ReconstructOptions())
    seed_input_image(orch, jid)
    orch.run(jid)
    job = store.get(jid)
    assert job.status == JobState.failed
    assert job.stage == Stage.segment


def test_provided_mode_skips_segment(tmp_path):
    store, orch = build(tmp_path, seg_fail=True)  # seg would fail if called
    jid = store.create(ReconstructOptions(segment="provided"))
    input_dir = seed_input_image(orch, jid)
    with open(os.path.join(input_dir, "mask.png"), "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
    orch.run(jid)
    assert store.get(jid).status == JobState.succeeded
