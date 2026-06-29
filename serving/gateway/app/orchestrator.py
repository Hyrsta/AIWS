import os

from . import bundle
from .clients import ServiceError
from .models import JobState, Stage


class Orchestrator:
    def __init__(self, store, sam3d, cadrille, artifacts_dir: str):
        self.store = store
        self.sam3d = sam3d
        self.cadrille = cadrille
        self.artifacts_dir = artifacts_dir

    def job_dir(self, job_id: str) -> str:
        return os.path.join(self.artifacts_dir, "jobs", job_id)

    def run(self, job_id: str):
        options = self.store.options(job_id)
        jd = self.job_dir(job_id)
        input_dir = os.path.join(jd, "input")
        try:
            self.store.set_status(job_id, JobState.running, stage=Stage.sam3d)
            mesh_path = self.sam3d.infer(job_id, input_dir)

            self.store.set_status(job_id, JobState.running, stage=Stage.cadrille)
            result = self.cadrille.infer(job_id, mesh_path, options)

            bundle.write_manifest(jd, job_id, options, result.get("metrics", {}))
            bundle.build_zip(jd, job_id)
            self.store.set_status(job_id, JobState.succeeded, stage=None)
        except ServiceError as e:
            self.store.set_status(job_id, JobState.failed, stage=Stage(e.stage), error=str(e))
        except Exception as e:  # noqa: BLE001 - any unexpected failure becomes a failed job
            self.store.set_status(job_id, JobState.failed, error=str(e))
