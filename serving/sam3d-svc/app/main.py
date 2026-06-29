from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import load_settings
from .inference import Runner


class InferRequest(BaseModel):
    job_id: str
    input_dir: str
    mask_path: str


def build_app(settings, runner) -> FastAPI:
    app = FastAPI(title="sam3d-svc")

    @app.get("/healthz")
    def healthz():
        if not getattr(runner, "ready", False):
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return {"status": "ok"}

    @app.post("/infer")
    def infer(req: InferRequest):
        try:
            return runner.run(settings, req.model_dump())
        except Exception as e:  # noqa: BLE001 - surface stage failure as 500
            raise HTTPException(status_code=500, detail=str(e))

    return app


def create_app() -> FastAPI:
    settings = load_settings()
    runner = Runner()
    runner.warmup(settings)
    return build_app(settings, runner)
