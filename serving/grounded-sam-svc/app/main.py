from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .config import load_settings
from .segmentor import NoDetection, Segmentor


def build_app(settings, segmentor) -> FastAPI:
    app = FastAPI(title="grounded-sam-svc")

    @app.get("/healthz")
    def healthz():
        if not getattr(segmentor, "ready", False):
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return {"status": "ok"}

    @app.post("/segment")
    def segment(
        image: UploadFile = File(...),
        prompt: str = Form(None),
        box_threshold: float = Form(None),
        text_threshold: float = Form(None),
    ):
        data = image.file.read()
        used_prompt = prompt or settings.default_prompt
        bt = box_threshold if box_threshold is not None else settings.box_threshold
        tt = text_threshold if text_threshold is not None else settings.text_threshold
        try:
            return segmentor.segment(data, used_prompt, bt, tt)
        except NoDetection:
            raise HTTPException(status_code=422,
                                detail=f"no object matched prompt '{used_prompt}'")
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(e))

    return app


def create_app() -> FastAPI:
    settings = load_settings()
    segmentor = Segmentor()
    segmentor.warmup(settings)
    return build_app(settings, segmentor)
