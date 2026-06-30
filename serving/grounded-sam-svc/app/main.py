import threading

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import load_settings
from .segmentor import NoDetection, Segmentor
from .sessions import SessionExpired, SessionStore

_gpu_lock = threading.Lock()


class RefinePoint(BaseModel):
    x: float
    y: float
    label: int = 1


class RefineBody(BaseModel):
    session_id: str
    prompt: str | None = None
    points: list[RefinePoint] | None = None
    box: list[float] | None = None
    reset: bool = False


def build_app(settings, segmentor) -> FastAPI:
    app = FastAPI(title="grounded-sam-svc")

    store = SessionStore(segmentor,
                         max_sessions=getattr(settings, "max_sessions", 8),
                         ttl_seconds=getattr(settings, "session_ttl_seconds", 900))

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

    @app.post("/segment/session")
    def segment_session(image: UploadFile = File(...), prompt: str = Form(None)):
        data = image.file.read()
        used_prompt = prompt or settings.default_prompt
        try:
            with _gpu_lock:
                sid, result = store.create(data, used_prompt,
                                           settings.box_threshold, settings.text_threshold)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(e))
        return {"session_id": sid, **result}

    @app.post("/segment/refine")
    def segment_refine(body: RefineBody = Body(...)):
        pts = [(p.x, p.y, p.label) for p in body.points] if body.points else None
        try:
            with _gpu_lock:
                return store.refine(body.session_id, prompt=body.prompt, points=pts,
                                    box=body.box, reset=body.reset,
                                    box_threshold=settings.box_threshold,
                                    text_threshold=settings.text_threshold)
        except SessionExpired:
            raise HTTPException(status_code=409, detail="session_expired")
        except NoDetection:
            raise HTTPException(status_code=422, detail="no object matched the prompt")
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/segment/session/{session_id}", status_code=204)
    def segment_release(session_id: str):
        with _gpu_lock:
            store.release(session_id)
        return None

    return app


def create_app() -> FastAPI:
    settings = load_settings()
    segmentor = Segmentor()
    segmentor.warmup(settings)
    return build_app(settings, segmentor)
