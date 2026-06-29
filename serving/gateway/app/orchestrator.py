import base64
import glob
import os
import shutil

from . import bundle
from .clients import ServiceError
from .models import JobState, Stage, SegmentMode


class Orchestrator:
    def __init__(self, store, grounded_sam, sam3d, cadrille, artifacts_dir: str):
        self.store = store
        self.grounded_sam = grounded_sam
        self.sam3d = sam3d
        self.cadrille = cadrille
        self.artifacts_dir = artifacts_dir

    def job_dir(self, job_id: str) -> str:
        return os.path.join(self.artifacts_dir, "jobs", job_id)

    def _first_image(self, input_dir: str) -> str:
        candidates = sorted(
            p for p in glob.glob(os.path.join(input_dir, "*"))
            if os.path.basename(p) != "mask.png"
        )
        if not candidates:
            raise RuntimeError("no input image found")
        return candidates[0]

    def run(self, job_id: str):
        try:
            options = self.store.options(job_id)
            jd = self.job_dir(job_id)
            input_dir = os.path.join(jd, "input")
            mask_path = os.path.join(input_dir, "mask.png")
            seg_meta = None

            if options.segment == SegmentMode.auto:
                self.store.set_status(job_id, JobState.running, stage=Stage.segment)
                result = self.grounded_sam.segment(
                    self._first_image(input_dir), options.detect_prompt
                )
                mask_bytes = base64.b64decode(result["mask_png_base64"])
                with open(mask_path, "wb") as f:
                    f.write(mask_bytes)
                mesh_dir = os.path.join(jd, "mesh")
                os.makedirs(mesh_dir, exist_ok=True)
                shutil.copyfile(mask_path, os.path.join(mesh_dir, "auto_mask.png"))
                seg_meta = {
                    "score": result.get("score"),
                    "prompt": options.detect_prompt,
                    "box": result.get("box"),
                }
            else:
                if not os.path.exists(mask_path):
                    raise RuntimeError(f"provided mask not found: {mask_path}")

            self.store.set_status(job_id, JobState.running, stage=Stage.sam3d)
            mesh_path = self.sam3d.infer(job_id, input_dir, mask_path)

            self.store.set_status(job_id, JobState.running, stage=Stage.cadrille)
            cad = self.cadrille.infer(job_id, mesh_path, options)

            metrics = cad.get("metrics", {})
            if seg_meta is not None:
                metrics = {**metrics, "segment": seg_meta}
            bundle.write_manifest(jd, job_id, options, metrics)
            bundle.build_zip(jd, job_id)
            self.store.set_status(job_id, JobState.succeeded, stage=None)
        except ServiceError as e:
            self.store.set_status(job_id, JobState.failed, stage=Stage(e.stage), error=str(e))
        except Exception as e:  # noqa: BLE001
            self.store.set_status(job_id, JobState.failed, error=str(e))
