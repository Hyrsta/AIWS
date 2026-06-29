import os
from app.jobstore import JobStore
from app.orchestrator import Orchestrator
from app.clients import ServiceError
from app.models import ReconstructOptions, JobState, Stage


class FakeSam3d:
    def __init__(self, fail=False):
        self.fail = fail

    def infer(self, job_id, input_dir):
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
        return {"cad_code_path": os.path.join(cad, "model.py"),
                "step_path": "", "preview_path": "", "metrics": {"iou": 0.2}}


def build(tmp_path, sam_fail=False, cad_fail=False):
    store = JobStore(os.path.join(tmp_path, "jobs.db"))
    orch = Orchestrator(store, FakeSam3d(sam_fail), FakeCadrille(cad_fail), str(tmp_path))
    return store, orch


def test_happy_path_succeeds_with_zip(tmp_path):
    store, orch = build(tmp_path)
    jid = store.create(ReconstructOptions())
    os.makedirs(os.path.join(orch.job_dir(jid), "input"), exist_ok=True)
    orch.run(jid)
    job = store.get(jid)
    assert job.status == JobState.succeeded
    assert job.stage is None
    assert os.path.exists(os.path.join(orch.job_dir(jid), f"{jid}.zip"))


def test_sam3d_failure_marks_stage(tmp_path):
    store, orch = build(tmp_path, sam_fail=True)
    jid = store.create(ReconstructOptions())
    os.makedirs(os.path.join(orch.job_dir(jid), "input"), exist_ok=True)
    orch.run(jid)
    job = store.get(jid)
    assert job.status == JobState.failed
    assert job.stage == Stage.sam3d
    assert "sam3d boom" in job.error


def test_cadrille_failure_marks_stage(tmp_path):
    store, orch = build(tmp_path, cad_fail=True)
    jid = store.create(ReconstructOptions())
    os.makedirs(os.path.join(orch.job_dir(jid), "input"), exist_ok=True)
    orch.run(jid)
    job = store.get(jid)
    assert job.status == JobState.failed
    assert job.stage == Stage.cadrille
