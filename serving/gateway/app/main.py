import os
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from .clients import CadrilleClient, Sam3dClient
from .config import load_settings
from .jobstore import JobStore
from .models import JobState, ReconstructOptions
from .orchestrator import Orchestrator
from .worker import Worker


def build_app(settings, store, sam3d, cadrille, worker) -> FastAPI:

    @asynccontextmanager
    async def _lifespan(app):
        worker.start()
        yield
        worker.stop()

    app = FastAPI(title="AIWS CAD Reconstruction API", lifespan=_lifespan)

    @app.get("/livez")
    def livez():
        return {"status": "ok"}

    @app.get("/healthz")
    def healthz():
        if not (worker.is_alive() and sam3d.healthz() and cadrille.healthz()):
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return {"status": "ok"}

    @app.post("/v1/reconstruct", status_code=202)
    def reconstruct(
        images: list[UploadFile] = File(...),
        mode: str = Form("pc"),
        n_candidates: int = Form(20),
        seed: int = Form(42),
        cleanup: bool = Form(True),
    ):
        if not images:
            raise HTTPException(status_code=400, detail="at least one image is required")
        if len(images) > settings.max_images:
            raise HTTPException(status_code=400,
                                detail=f"too many images, max is {settings.max_images}")
        for up in images:
            if up.content_type is None or not up.content_type.startswith("image/"):
                raise HTTPException(status_code=400,
                                    detail=f"not an image: {up.filename}")
        try:
            options = ReconstructOptions(mode=mode, n_candidates=n_candidates,
                                         seed=seed, cleanup=cleanup)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=e.errors())

        job_id = store.create(options)
        input_dir = os.path.join(settings.artifacts_dir, "jobs", job_id, "input")
        os.makedirs(input_dir, exist_ok=True)
        for i, up in enumerate(images):
            ext = os.path.splitext(up.filename or f"img{i}.png")[1] or ".png"
            with open(os.path.join(input_dir, f"{i:03d}{ext}"), "wb") as f:
                shutil.copyfileobj(up.file, f)
        return {"job_id": job_id, "status": JobState.queued.value}

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    @app.get("/v1/jobs/{job_id}/result")
    def get_result(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if job.status != JobState.succeeded:
            raise HTTPException(status_code=409,
                                detail=f"job status is {job.status.value}")
        zip_path = os.path.join(settings.artifacts_dir, "jobs", job_id, f"{job_id}.zip")
        if not os.path.exists(zip_path):
            raise HTTPException(status_code=500, detail="result bundle missing")
        return FileResponse(zip_path, media_type="application/zip",
                            filename=f"{job_id}.zip")

    return app


def create_app() -> FastAPI:
    settings = load_settings()
    store = JobStore(settings.db_path)
    sam3d = Sam3dClient(settings.sam3d_url, settings.stage_timeout_s)
    cadrille = CadrilleClient(settings.cadrille_url, settings.stage_timeout_s)
    orchestrator = Orchestrator(store, sam3d, cadrille, settings.artifacts_dir)
    worker = Worker(store, orchestrator)
    return build_app(settings, store, sam3d, cadrille, worker)


app = create_app()
