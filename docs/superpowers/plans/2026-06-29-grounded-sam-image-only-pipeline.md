# Grounded-SAM Auto-Segmentation and Image-Only Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user submit an RGB image with no mask and still get a CAD reconstruction, by auto-generating the mask with Grounded-SAM, exposed in both the REST service and the GUI, and land the SAM3D/Cadrille adapters that make the REST pipeline run end to end.

**Architecture:** Add a fourth container `grounded-sam-svc` (GroundingDINO + SAM) that returns a mask for an image. The gateway gains a `segment` stage before `sam3d`; image-only is the default. The SAM3D and Cadrille adapters factor the proven single-sample pipeline out of `simple_reconstruct_job.py` into a shared module reused by the adapters and the GUI job. The GUI gains an RGB-only input mode.

**Tech Stack:** Python 3.10+ (CPU dev on 3.9.6 venv), FastAPI, uvicorn, httpx, pydantic v2, GroundingDINO, segment-anything, Docker, docker-compose, React/TypeScript (GUI).

## Global Constraints

- No em-dashes or en-dashes in any file, doc, or prose. Use comma, period, colon, or hyphen.
- Default REST path is image-only: `ReconstructOptions.segment` defaults to `auto`. `provided` means the client supplies a mask.
- Default detection prompt is `workpiece. metal part.` via env `WORKPIECE_PROMPT`, overridable per call.
- GroundingDINO picks the single highest-confidence detection above `box_threshold` (default 0.3, `text_threshold` default 0.25). No multi-object merging.
- `grounded-sam-svc /segment` returns the mask in the response body as `mask_png_base64` (not via the shared volume), so the GUI backend (no shared volume) can use it.
- Only the gateway and `grounded-sam-svc` are published on host ports; `sam3d-svc` and `cadrille-svc` stay internal.
- Adapters reuse the factored pipeline module, they do not reimplement SAM3D/Cadrille logic.
- Phases: Phase 1 (`serving/`) is CPU-testable and committed on this branch. Phase 2 (`scripts/`, `gui/`, model builds, integration) runs on the RXL GPU host against `origin/main`, where `gui/` and `scripts/` are tracked (they are gitignored in the local Mac worktree).
- CPU dev: a venv is at `serving/.venv` (Python 3.9.6). Phase 1 service code must run on 3.9.6 (use `Optional[...]`, PEP 585 builtins generics, no 3.10-only syntax). Tests that import a service `main` set `ARTIFACTS_DIR` and `DB_PATH` to temp paths because `app = create_app()` runs at import for the gateway.

## File Structure

Phase 1 (serving/, tracked in this repo):

```
serving/grounded-sam-svc/
  app/{__init__,config,segmentor,main}.py
  requirements.txt
  Dockerfile
  tests/{test_health,test_segment_contract}.py
serving/gateway/app/        (modify: models, clients, bundle, orchestrator, main, config)
serving/gateway/tests/      (extend)
serving/sam3d-svc/app/      (modify: main InferRequest, inference cmd)
serving/sam3d-svc/tests/    (extend)
serving/docker-compose.yml  (add grounded-sam-svc; gateway env)
serving/.env.example        (add seg config)
```

Phase 2 (origin/main, RXL):

```
scripts/aiws_pipeline_core.py            (factor SAM3D inference, normalize, cadrille, postscale)
gui/backend/simple_reconstruct_job.py    (refactor to use the core module)
serving/sam3d-svc/run_sam3d.py           (adapter body using the core module)
serving/cadrille-svc/run_cadrille.py     (adapter body using the core module)
serving/grounded-sam-svc/app/segmentor.py (real GroundingDINO + SAM implementation)
gui/backend/app.py                       (input_mode="image")
gui/frontend/src/components/InputImagesPanel.tsx  (RGB-only option)
```

Test run convention (Phase 1): from a service dir, `PYTHONPATH=. ../.venv/bin/pytest` (gateway also needs `ARTIFACTS_DIR=$(mktemp -d) DB_PATH=$(mktemp -u)`).

---

# Phase 1: REST service (CPU-testable, this branch)

### Task 1: Gateway segment options and stage

**Files:**
- Modify: `serving/gateway/app/models.py`
- Test: `serving/gateway/tests/test_models.py`

**Interfaces:**
- Produces: `SegmentMode(auto, provided)` enum. `Stage` gains member `segment = "segment"`. `ReconstructOptions` gains `segment: SegmentMode = SegmentMode.auto` and `detect_prompt: Optional[str] = None`.

- [ ] **Step 1: Write the failing test**

Append to `serving/gateway/tests/test_models.py`:
```python
from app.models import SegmentMode, Stage


def test_segment_defaults_to_auto():
    o = ReconstructOptions()
    assert o.segment == SegmentMode.auto
    assert o.detect_prompt is None


def test_segment_provided_accepted():
    o = ReconstructOptions(segment="provided")
    assert o.segment == SegmentMode.provided


def test_invalid_segment_rejected():
    with pytest.raises(ValidationError):
        ReconstructOptions(segment="magic")


def test_stage_has_segment():
    assert Stage.segment.value == "segment"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd serving/gateway && PYTHONPATH=. ../.venv/bin/pytest tests/test_models.py -v`
Expected: FAIL with `ImportError: cannot import name 'SegmentMode'`.

- [ ] **Step 3: Write minimal implementation**

In `serving/gateway/app/models.py`, add the enum and fields:
```python
class SegmentMode(str, Enum):
    auto = "auto"
    provided = "provided"
```
Add `segment = "segment"` to the `Stage` enum (keep `sam3d` and `cadrille`):
```python
class Stage(str, Enum):
    segment = "segment"
    sam3d = "sam3d"
    cadrille = "cadrille"
```
Add the two fields to `ReconstructOptions`:
```python
class ReconstructOptions(BaseModel):
    mode: Mode = Mode.pc
    n_candidates: int = Field(default=20, ge=1, le=64)
    seed: int = 42
    cleanup: bool = True
    segment: SegmentMode = SegmentMode.auto
    detect_prompt: Optional[str] = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd serving/gateway && PYTHONPATH=. ../.venv/bin/pytest tests/test_models.py -v`
Expected: PASS (prior model tests plus 4 new).

- [ ] **Step 5: Commit**

```bash
git add serving/gateway/app/models.py serving/gateway/tests/test_models.py
git commit -m "feat(gateway): segment mode/stage and detect_prompt option"
```

---

### Task 2: Grounded-SAM client and Sam3d mask argument

**Files:**
- Modify: `serving/gateway/app/clients.py`
- Test: `serving/gateway/tests/test_clients.py`

**Interfaces:**
- Consumes: `ServiceError` (existing).
- Produces: `GroundedSamClient(base_url, timeout_s, client=None)` with `healthz() -> bool` and `segment(image_path, prompt) -> dict` returning the parsed JSON `{mask_png_base64, score, box, label, num_detections}`; non-200 and transport errors raise `ServiceError("segment", ...)`. `Sam3dClient.infer` gains a third parameter `mask_path` and sends it in the JSON body.

- [ ] **Step 1: Write the failing test**

Append to `serving/gateway/tests/test_clients.py`:
```python
from app.clients import GroundedSamClient


def test_grounded_sam_segment_parses_mask(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
    def handler(request):
        assert request.url.path == "/segment"
        return httpx.Response(200, json={
            "mask_png_base64": "AAAA", "score": 0.5,
            "box": [1, 2, 3, 4], "label": "workpiece", "num_detections": 1})
    c = GroundedSamClient("http://svc", 5, client=client_with(handler))
    out = c.segment(str(img), "workpiece")
    assert out["mask_png_base64"] == "AAAA"
    assert out["score"] == 0.5


def test_grounded_sam_segment_422_raises_segment_stage(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    def handler(request):
        return httpx.Response(422, text="no object matched")
    c = GroundedSamClient("http://svc", 5, client=client_with(handler))
    with pytest.raises(ServiceError) as ei:
        c.segment(str(img), "workpiece")
    assert ei.value.stage == "segment"


def test_sam3d_infer_sends_mask_path():
    seen = {}
    def handler(request):
        seen["body"] = request.read().decode()
        return httpx.Response(200, json={"mesh_path": "/a/mesh/sam3d_mesh.ply"})
    c = Sam3dClient("http://svc", 5, client=client_with(handler))
    c.infer("x", "/a/input", "/a/input/mask.png")
    assert "mask.png" in seen["body"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd serving/gateway && PYTHONPATH=. ../.venv/bin/pytest tests/test_clients.py -v`
Expected: FAIL with `ImportError: cannot import name 'GroundedSamClient'`.

- [ ] **Step 3: Write minimal implementation**

In `serving/gateway/app/clients.py`, update `Sam3dClient.infer` to accept and send `mask_path`:
```python
    def infer(self, job_id: str, input_dir: str, mask_path: str) -> str:
        try:
            r = self._client.post(
                f"{self.base_url}/infer",
                json={"job_id": job_id, "input_dir": input_dir, "mask_path": mask_path},
                timeout=self.timeout_s,
            )
        except httpx.HTTPError as e:
            raise ServiceError("sam3d", f"sam3d-svc unreachable: {e}")
        if r.status_code != 200:
            raise ServiceError("sam3d", f"sam3d-svc returned {r.status_code}: {r.text}")
        return r.json()["mesh_path"]
```
Add the new client:
```python
class GroundedSamClient(_Base):
    def healthz(self) -> bool:
        return self._healthz()

    def segment(self, image_path: str, prompt) -> dict:
        data = {}
        if prompt:
            data["prompt"] = prompt
        try:
            with open(image_path, "rb") as f:
                files = {"image": (os.path.basename(image_path), f, "image/png")}
                r = self._client.post(
                    f"{self.base_url}/segment", data=data, files=files,
                    timeout=self.timeout_s,
                )
        except httpx.HTTPError as e:
            raise ServiceError("segment", f"grounded-sam-svc unreachable: {e}")
        if r.status_code != 200:
            raise ServiceError("segment", f"grounded-sam-svc returned {r.status_code}: {r.text}")
        return r.json()
```
Add `import os` at the top of `clients.py` if not present.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd serving/gateway && PYTHONPATH=. ../.venv/bin/pytest tests/test_clients.py -v`
Expected: PASS. (The existing `test_sam3d_infer_returns_mesh_path` and the transport-error tests call `c.infer("x", "/in")` with two args; update those existing calls to pass a third arg `"/in/mask.png"`.)

- [ ] **Step 5: Commit**

```bash
git add serving/gateway/app/clients.py serving/gateway/tests/test_clients.py
git commit -m "feat(gateway): grounded-sam client; sam3d infer takes mask_path"
```

---

### Task 3: Bundle includes the auto mask

**Files:**
- Modify: `serving/gateway/app/bundle.py`
- Test: `serving/gateway/tests/test_bundle.py`

**Interfaces:**
- Produces: `ARTIFACT_NAMES` gains `"mesh/auto_mask.png"` (optional, skipped when absent by the existing missing-file logic).

- [ ] **Step 1: Write the failing test**

Append to `serving/gateway/tests/test_bundle.py`:
```python
def test_build_zip_includes_auto_mask_when_present(tmp_path):
    jd = str(tmp_path)
    import os
    os.makedirs(os.path.join(jd, "mesh"))
    with open(os.path.join(jd, "mesh", "auto_mask.png"), "w") as f:
        f.write("x")
    zp = bundle.build_zip(jd, "job1")
    import zipfile
    with zipfile.ZipFile(zp) as z:
        assert "mesh/auto_mask.png" in set(z.namelist())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd serving/gateway && PYTHONPATH=. ../.venv/bin/pytest tests/test_bundle.py::test_build_zip_includes_auto_mask_when_present -v`
Expected: FAIL (auto_mask.png not in zip).

- [ ] **Step 3: Write minimal implementation**

In `serving/gateway/app/bundle.py`, add the entry to `ARTIFACT_NAMES`:
```python
ARTIFACT_NAMES = [
    "cad/model.py",
    "cad/model.step",
    "preview/model.stl",
    "preview/render.png",
    "mesh/sam3d_mesh.ply",
    "mesh/auto_mask.png",
    "metrics.json",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd serving/gateway && PYTHONPATH=. ../.venv/bin/pytest tests/test_bundle.py -v`
Expected: PASS (existing bundle tests still green; the includes-all test already only asserts membership, not exact set).

- [ ] **Step 5: Commit**

```bash
git add serving/gateway/app/bundle.py serving/gateway/tests/test_bundle.py
git commit -m "feat(gateway): include auto_mask.png in result bundle"
```

---

### Task 4: Orchestrator segment stage

**Files:**
- Modify: `serving/gateway/app/orchestrator.py`
- Test: `serving/gateway/tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `GroundedSamClient`, updated `Sam3dClient.infer(job_id, input_dir, mask_path)`, `SegmentMode`, `Stage.segment`.
- Produces: `Orchestrator(store, grounded_sam, sam3d, cadrille, artifacts_dir)` (note the new `grounded_sam` first model arg). On `segment=auto`: stage `segment`, call `grounded_sam.segment(image_path, prompt)`, decode `mask_png_base64` to `<job_dir>/input/mask.png` and copy to `<job_dir>/mesh/auto_mask.png`, fold `{score, prompt, box}` into metrics under key `segment`. On `segment=provided`: skip seg, use `<job_dir>/input/mask.png` (written by main.py). Then `sam3d.infer(job_id, input_dir, mask_path)` and the existing cadrille stage.

- [ ] **Step 1: Write the failing test**

Replace the fakes and add cases in `serving/gateway/tests/test_orchestrator.py`. Add a base64 1x1 PNG and fakes:
```python
import base64, glob

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
    def infer(self, job_id, input_dir, mask_path):
        assert mask_path.endswith("mask.png")
        job_dir = os.path.dirname(input_dir)
        mesh = os.path.join(job_dir, "mesh", "sam3d_mesh.ply")
        os.makedirs(os.path.dirname(mesh), exist_ok=True)
        open(mesh, "w").write("ply")
        return mesh


class FakeCadrille:
    def infer(self, job_id, mesh_path, options):
        job_dir = os.path.dirname(os.path.dirname(mesh_path))
        cad = os.path.join(job_dir, "cad")
        os.makedirs(cad, exist_ok=True)
        open(os.path.join(cad, "model.py"), "w").write("import cadquery")
        return {"metrics": {"iou": 0.2}}


def build(tmp_path, seg_fail=False):
    store = JobStore(os.path.join(tmp_path, "jobs.db"))
    orch = Orchestrator(store, FakeGroundedSam(seg_fail), FakeSam3d(), FakeCadrille(), str(tmp_path))
    return store, orch


def seed_input_image(orch, jid):
    input_dir = os.path.join(orch.job_dir(jid), "input")
    os.makedirs(input_dir, exist_ok=True)
    with open(os.path.join(input_dir, "000.png"), "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
    return input_dir


def test_auto_segment_writes_mask_and_succeeds(tmp_path):
    store, orch = build(tmp_path)
    jid = store.create(ReconstructOptions())  # segment defaults to auto
    seed_input_image(orch, jid)
    orch.run(jid)
    job = store.get(jid)
    assert job.status == JobState.succeeded
    assert os.path.exists(os.path.join(orch.job_dir(jid), "input", "mask.png"))
    assert os.path.exists(os.path.join(orch.job_dir(jid), "mesh", "auto_mask.png"))


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
```
Update the existing happy/failure tests in this file to construct `Orchestrator(store, FakeGroundedSam(), FakeSam3d(), FakeCadrille(), str(tmp_path))` with the new arg order, and update `FakeSam3d.infer` signatures to `(self, job_id, input_dir, mask_path)`. Import `Stage` and `ServiceError` at the top.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd serving/gateway && PYTHONPATH=. ../.venv/bin/pytest tests/test_orchestrator.py -v`
Expected: FAIL (`Orchestrator.__init__` takes the old arg count, or `segment` handling missing).

- [ ] **Step 3: Write minimal implementation**

Rewrite `serving/gateway/app/orchestrator.py`:
```python
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
                result = self.grounded_sam.segment(self._first_image(input_dir),
                                                   options.detect_prompt)
                mask_bytes = base64.b64decode(result["mask_png_base64"])
                with open(mask_path, "wb") as f:
                    f.write(mask_bytes)
                mesh_dir = os.path.join(jd, "mesh")
                os.makedirs(mesh_dir, exist_ok=True)
                shutil.copyfile(mask_path, os.path.join(mesh_dir, "auto_mask.png"))
                seg_meta = {"score": result.get("score"),
                            "prompt": options.detect_prompt,
                            "box": result.get("box")}

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd serving/gateway && PYTHONPATH=. ../.venv/bin/pytest tests/test_orchestrator.py -v`
Expected: PASS (all cases including the three new ones).

- [ ] **Step 5: Commit**

```bash
git add serving/gateway/app/orchestrator.py serving/gateway/tests/test_orchestrator.py
git commit -m "feat(gateway): orchestrator segment stage (auto and provided)"
```

---

### Task 5: Gateway wiring, config, and routes

**Files:**
- Modify: `serving/gateway/app/config.py`, `serving/gateway/app/main.py`
- Test: `serving/gateway/tests/test_api.py`

**Interfaces:**
- Consumes: `GroundedSamClient`, updated `Orchestrator`.
- Produces: `Settings` gains `grounded_sam_url`. `build_app(settings, store, grounded_sam, sam3d, cadrille, worker)` (new `grounded_sam` arg). `POST /v1/reconstruct` accepts optional `mask: UploadFile` and form fields `segment` and `detect_prompt`, saves an uploaded mask to `<job_dir>/input/mask.png`. `/healthz` also checks `grounded_sam.healthz()`.

- [ ] **Step 1: Write the failing test**

In `serving/gateway/tests/test_api.py`, update `make_client` to construct settings with `grounded_sam_url="http://gsam"` and call `build_app(settings, store, StubHealth(...), StubHealth(...), StubHealth(...), StubWorker())` where the first health stub is the grounded-sam one. Add tests:
```python
def test_reconstruct_defaults_segment_auto(tmp_path):
    client, store = make_client(tmp_path)
    r = client.post("/v1/reconstruct",
                    files={"images": ("a.png", png_bytes(), "image/png")},
                    data={"mode": "pc"})
    assert r.status_code == 202
    jid = r.json()["job_id"]
    assert store.options(jid).segment.value == "auto"


def test_reconstruct_provided_saves_mask(tmp_path):
    client, store = make_client(tmp_path)
    r = client.post("/v1/reconstruct",
                    files={"images": ("a.png", png_bytes(), "image/png"),
                           "mask": ("m.png", png_bytes(), "image/png")},
                    data={"mode": "pc", "segment": "provided"})
    assert r.status_code == 202
    jid = r.json()["job_id"]
    import os
    assert os.path.exists(os.path.join(str(tmp_path), "jobs", jid, "input", "mask.png"))


def test_healthz_503_when_grounded_sam_down(tmp_path):
    client, _ = make_client(tmp_path, gsam_ok=False)
    assert client.get("/healthz").status_code == 503
```
Update `make_client` signature to accept `gsam_ok=True` and pass the right stubs.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd serving/gateway && ARTIFACTS_DIR=$(mktemp -d) DB_PATH=$(mktemp -u) PYTHONPATH=. ../.venv/bin/pytest tests/test_api.py -v`
Expected: FAIL (build_app arity / missing form fields).

- [ ] **Step 3: Write minimal implementation**

In `serving/gateway/app/config.py`, add to `Settings` and `load_settings`:
```python
    grounded_sam_url: str
```
```python
        grounded_sam_url=os.environ.get("GROUNDED_SAM_URL", "http://grounded-sam-svc:8000"),
```
In `serving/gateway/app/main.py`, change `build_app` to take `grounded_sam` and wire it into `healthz` and the orchestrator-less route logic. Replace the signature and healthz:
```python
def build_app(settings, store, grounded_sam, sam3d, cadrille, worker) -> FastAPI:
    ...
    @app.get("/healthz")
    def healthz():
        if not (worker.is_alive() and grounded_sam.healthz()
                and sam3d.healthz() and cadrille.healthz()):
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return {"status": "ok"}
```
Update `reconstruct` to accept the mask and new fields and save the mask:
```python
    @app.post("/v1/reconstruct", status_code=202)
    def reconstruct(
        images: list[UploadFile] = File(...),
        mask: UploadFile = File(None),
        mode: str = Form("pc"),
        n_candidates: int = Form(20),
        seed: int = Form(42),
        cleanup: bool = Form(True),
        segment: str = Form("auto"),
        detect_prompt: str = Form(None),
    ):
        if not images:
            raise HTTPException(status_code=400, detail="at least one image is required")
        if len(images) > settings.max_images:
            raise HTTPException(status_code=400,
                                detail=f"too many images, max is {settings.max_images}")
        for up in images:
            if up.content_type is None or not up.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail=f"not an image: {up.filename}")
        try:
            options = ReconstructOptions(mode=mode, n_candidates=n_candidates, seed=seed,
                                         cleanup=cleanup, segment=segment,
                                         detect_prompt=detect_prompt)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=e.errors())
        if options.segment.value == "provided" and mask is None:
            raise HTTPException(status_code=400,
                                detail="segment=provided requires a mask upload")

        job_id = store.create(options)
        input_dir = os.path.join(settings.artifacts_dir, "jobs", job_id, "input")
        os.makedirs(input_dir, exist_ok=True)
        for i, up in enumerate(images):
            ext = os.path.splitext(up.filename or f"img{i}.png")[1] or ".png"
            with open(os.path.join(input_dir, f"{i:03d}{ext}"), "wb") as f:
                shutil.copyfileobj(up.file, f)
        if mask is not None:
            with open(os.path.join(input_dir, "mask.png"), "wb") as f:
                shutil.copyfileobj(mask.file, f)
        return {"job_id": job_id, "status": JobState.queued.value}
```
Update `create_app` to build the grounded-sam client and pass it:
```python
def create_app() -> FastAPI:
    settings = load_settings()
    store = JobStore(settings.db_path)
    grounded_sam = GroundedSamClient(settings.grounded_sam_url, settings.stage_timeout_s)
    sam3d = Sam3dClient(settings.sam3d_url, settings.stage_timeout_s)
    cadrille = CadrilleClient(settings.cadrille_url, settings.stage_timeout_s)
    orchestrator = Orchestrator(store, grounded_sam, sam3d, cadrille, settings.artifacts_dir)
    worker = Worker(store, orchestrator)
    return build_app(settings, store, grounded_sam, sam3d, cadrille, worker)
```
Add `from .clients import GroundedSamClient` to the imports.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd serving/gateway && ARTIFACTS_DIR=$(mktemp -d) DB_PATH=$(mktemp -u) PYTHONPATH=. ../.venv/bin/pytest -v`
Expected: PASS (full gateway suite; update any existing healthz test stubs to include the grounded-sam stub).

- [ ] **Step 5: Commit**

```bash
git add serving/gateway/app/config.py serving/gateway/app/main.py serving/gateway/tests/test_api.py
git commit -m "feat(gateway): wire grounded-sam, accept mask upload and segment options"
```

---

### Task 6: sam3d-svc accepts a mask

**Files:**
- Modify: `serving/sam3d-svc/app/main.py`, `serving/sam3d-svc/app/inference.py`
- Test: `serving/sam3d-svc/tests/test_infer_contract.py`

**Interfaces:**
- Produces: `InferRequest` gains `mask_path: str`. `Runner.run` passes `--input-mask <mask_path>` to the adapter command.

- [ ] **Step 1: Write the failing test**

Update `serving/sam3d-svc/tests/test_infer_contract.py` `OkRunner.run` to assert the mask and the POST body to include it:
```python
class OkRunner:
    ready = True
    def run(self, settings, req):
        assert req["mask_path"].endswith("mask.png")
        return {"mesh_path": "/artifacts/jobs/j/mesh/sam3d_mesh.ply"}
```
And update the POST payloads in that file to include `"mask_path": "/artifacts/jobs/j/input/mask.png"`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd serving/sam3d-svc && PYTHONPATH=. ../.venv/bin/pytest -v`
Expected: FAIL (`mask_path` missing from `InferRequest`, 422 on the request).

- [ ] **Step 3: Write minimal implementation**

In `serving/sam3d-svc/app/main.py`, add `mask_path` to `InferRequest`:
```python
class InferRequest(BaseModel):
    job_id: str
    input_dir: str
    mask_path: str
```
In `serving/sam3d-svc/app/inference.py`, add the image discovery and the `--input-mask` flag. The `Runner.run` builds the command with the first input image and the mask:
```python
    def run(self, settings, req: dict) -> dict:
        input_dir = req["input_dir"]
        mask_path = req["mask_path"]
        job_dir = os.path.dirname(input_dir)
        mesh_dir = os.path.join(job_dir, "mesh")
        os.makedirs(mesh_dir, exist_ok=True)
        mesh_path = os.path.join(mesh_dir, "sam3d_mesh.ply")
        image_path = self._first_image(input_dir)
        cmd = [
            "python", settings.sam3d_entry,
            "--input-image", image_path,
            "--input-mask", mask_path,
            "--seed", str(settings.__dict__.get("seed", 42)),
            "--out-mesh", mesh_path,
        ]
        try:
            subprocess.run(cmd, check=True, timeout=settings.stage_timeout_s,
                           capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            raise RuntimeError("sam3d inference timed out")
        except subprocess.CalledProcessError as e:
            tail = (e.stderr or "")[-2000:]
            raise RuntimeError(f"sam3d inference failed: {tail}")
        if not os.path.exists(mesh_path):
            raise RuntimeError("sam3d produced no mesh")
        return {"mesh_path": mesh_path}

    def _first_image(self, input_dir):
        import glob
        candidates = sorted(
            p for p in glob.glob(os.path.join(input_dir, "*"))
            if os.path.basename(p) != "mask.png"
        )
        if not candidates:
            raise RuntimeError("no input image found")
        return candidates[0]
```
(The `--input-image`/`--input-mask`/`--out-mesh` flags match the `run_sam3d.py` adapter contract authored in Phase 2.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd serving/sam3d-svc && PYTHONPATH=. ../.venv/bin/pytest -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add serving/sam3d-svc/app/main.py serving/sam3d-svc/app/inference.py serving/sam3d-svc/tests/test_infer_contract.py
git commit -m "feat(sam3d-svc): accept mask_path and pass it to the adapter"
```

---

### Task 7: grounded-sam-svc service

**Files:**
- Create: `serving/grounded-sam-svc/app/{__init__,config,segmentor,main}.py`, `serving/grounded-sam-svc/requirements.txt`, `serving/grounded-sam-svc/Dockerfile`
- Test: `serving/grounded-sam-svc/tests/{test_health,test_segment_contract}.py`

**Interfaces:**
- Produces: FastAPI `build_app(settings, segmentor)` with `GET /healthz` and `POST /segment` (multipart `image` + optional form `prompt`, `box_threshold`, `text_threshold`) returning `{mask_png_base64, score, box, label, num_detections}`; raises `422` when the segmentor reports no detection. `segmentor.segment(image_bytes, prompt, box_threshold, text_threshold) -> dict` is the mockable interface; the real GroundingDINO+SAM implementation lands in Phase 2. A `NoDetection` exception type maps to 422.

- [ ] **Step 1: Write the failing tests**

`serving/grounded-sam-svc/tests/test_health.py`:
```python
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fastapi.testclient import TestClient
from app.main import build_app
from app.config import Settings


class FakeSeg:
    ready = True
    def segment(self, image_bytes, prompt, box_threshold, text_threshold):
        return {}


def test_healthz_ok():
    app = build_app(Settings(default_prompt="workpiece. metal part.",
                             grounding_dino_ckpt="x", sam_ckpt="x",
                             sam_model_type="vit_h", device="cuda:0",
                             box_threshold=0.3, text_threshold=0.25), FakeSeg())
    assert TestClient(app).get("/healthz").status_code == 200
```

`serving/grounded-sam-svc/tests/test_segment_contract.py`:
```python
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import io
from fastapi.testclient import TestClient
from app.main import build_app
from app.config import Settings
from app.segmentor import NoDetection


def settings():
    return Settings(default_prompt="workpiece. metal part.", grounding_dino_ckpt="x",
                    sam_ckpt="x", sam_model_type="vit_h", device="cuda:0",
                    box_threshold=0.3, text_threshold=0.25)


class OkSeg:
    ready = True
    def segment(self, image_bytes, prompt, box_threshold, text_threshold):
        assert prompt == "workpiece. metal part."
        return {"mask_png_base64": "AAAA", "score": 0.6, "box": [0, 0, 2, 2],
                "label": "workpiece", "num_detections": 1}


class NoneSeg:
    ready = True
    def segment(self, image_bytes, prompt, box_threshold, text_threshold):
        raise NoDetection("workpiece. metal part.")


def png():
    return io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"0" * 16)


def test_segment_returns_mask_with_default_prompt():
    client = TestClient(build_app(settings(), OkSeg()))
    r = client.post("/segment", files={"image": ("a.png", png(), "image/png")})
    assert r.status_code == 200
    assert r.json()["mask_png_base64"] == "AAAA"


def test_segment_override_prompt():
    captured = {}
    class CapSeg:
        ready = True
        def segment(self, image_bytes, prompt, box_threshold, text_threshold):
            captured["prompt"] = prompt
            return {"mask_png_base64": "BBBB", "score": 0.4, "box": [0, 0, 1, 1],
                    "label": "p", "num_detections": 1}
    client = TestClient(build_app(settings(), CapSeg()))
    r = client.post("/segment", files={"image": ("a.png", png(), "image/png")},
                    data={"prompt": "bracket"})
    assert r.status_code == 200
    assert captured["prompt"] == "bracket"


def test_segment_no_detection_returns_422():
    client = TestClient(build_app(settings(), NoneSeg()))
    r = client.post("/segment", files={"image": ("a.png", png(), "image/png")})
    assert r.status_code == 422
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd serving/grounded-sam-svc && PYTHONPATH=. ../.venv/bin/pytest -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.main'`.

- [ ] **Step 3: Write minimal implementation**

`serving/grounded-sam-svc/app/__init__.py`: empty.

`serving/grounded-sam-svc/app/config.py`:
```python
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    default_prompt: str
    grounding_dino_ckpt: str
    sam_ckpt: str
    sam_model_type: str
    device: str
    box_threshold: float
    text_threshold: float


def load_settings() -> Settings:
    return Settings(
        default_prompt=os.environ.get("WORKPIECE_PROMPT", "workpiece. metal part."),
        grounding_dino_ckpt=os.environ.get("GROUNDING_DINO_CKPT", "/ckpt/grounding_dino"),
        sam_ckpt=os.environ.get("SAM_CKPT", "/ckpt/sam"),
        sam_model_type=os.environ.get("SAM_MODEL_TYPE", "vit_h"),
        device=os.environ.get("SEG_DEVICE", "cuda:0"),
        box_threshold=float(os.environ.get("BOX_THRESHOLD", "0.3")),
        text_threshold=float(os.environ.get("TEXT_THRESHOLD", "0.25")),
    )
```

`serving/grounded-sam-svc/app/segmentor.py`:
```python
import os


class NoDetection(Exception):
    pass


class Segmentor:
    """Interface for GroundingDINO + SAM. The real model loading and inference
    are implemented in Phase 2 on the RXL host. Contract tests inject a fake.
    """

    def __init__(self):
        self.ready = False

    def warmup(self, settings) -> None:
        if not os.path.exists(settings.grounding_dino_ckpt):
            raise RuntimeError(f"grounding-dino checkpoint not found: {settings.grounding_dino_ckpt}")
        if not os.path.exists(settings.sam_ckpt):
            raise RuntimeError(f"sam checkpoint not found: {settings.sam_ckpt}")
        # Phase 2: load GroundingDINO + SAM here.
        self.ready = True

    def segment(self, image_bytes, prompt, box_threshold, text_threshold) -> dict:
        # Phase 2: run GroundingDINO detection then SAM masking; raise NoDetection
        # when nothing clears box_threshold. Return mask_png_base64, score, box,
        # label, num_detections.
        raise NotImplementedError("real segmentor implemented in Phase 2")
```

`serving/grounded-sam-svc/app/main.py`:
```python
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
```

`serving/grounded-sam-svc/requirements.txt`:
```
fastapi==0.111.0
uvicorn[standard]==0.30.1
pydantic==2.7.4
python-multipart==0.0.9
pytest==8.2.2
```

`serving/grounded-sam-svc/Dockerfile`:
```dockerfile
# Base provides torch + CUDA. Build on RXL. GroundingDINO and segment-anything
# are installed here; their weights are mounted at runtime.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Phase 2 adds: pip install GroundingDINO + segment-anything and their deps.

COPY app ./app

ENV WORKPIECE_PROMPT="workpiece. metal part."
ENV SEG_DEVICE=cuda:0
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=20 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd serving/grounded-sam-svc && PYTHONPATH=. ../.venv/bin/pytest -v`
Expected: PASS (4 tests). Docker build is deferred to RXL.

- [ ] **Step 5: Commit**

```bash
git add serving/grounded-sam-svc
git commit -m "feat(grounded-sam-svc): FastAPI segment service shell, contract tests, Dockerfile"
```

---

### Task 8: Compose and configuration

**Files:**
- Modify: `serving/docker-compose.yml`, `serving/.env.example`

**Interfaces:**
- Produces: a `grounded-sam-svc` service (built, published on `${SEG_PORT:-8081}:8000`, GPU reservation, seg checkpoint mounts); the gateway gains `GROUNDED_SAM_URL`.

- [ ] **Step 1: Add the service and env**

In `serving/.env.example`, add:
```
# Host port for the grounded-sam segmentation service (used by the GUI backend)
SEG_PORT=8081
SEG_DEVICE=cuda:0
WORKPIECE_PROMPT=workpiece. metal part.
GROUNDING_DINO_CKPT_HOST=/data/ckpt/grounding_dino
SAM_CKPT_HOST=/data/ckpt/sam
```
In `serving/docker-compose.yml`, add to the gateway `environment:`:
```yaml
      GROUNDED_SAM_URL: http://grounded-sam-svc:8000
```
And add the service:
```yaml
  grounded-sam-svc:
    build: ./grounded-sam-svc
    image: aiws-grounded-sam-svc:dev
    ports:
      - "${SEG_PORT:-8081}:8000"
    environment:
      SEG_DEVICE: ${SEG_DEVICE:-cuda:0}
      WORKPIECE_PROMPT: ${WORKPIECE_PROMPT:-workpiece. metal part.}
      GROUNDING_DINO_CKPT: /ckpt/grounding_dino
      SAM_CKPT: /ckpt/sam
    volumes:
      - artifacts:/artifacts
      - ${GROUNDING_DINO_CKPT_HOST}:/ckpt/grounding_dino:ro
      - ${SAM_CKPT_HOST}:/ckpt/sam:ro
    networks:
      - aiws
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: ["gpu"]
```
Add `grounded-sam-svc` to the gateway `depends_on:` list.

- [ ] **Step 2: Validate the compose file**

Run: `cd serving && SAM3D_CKPT_HOST=/tmp CADRILLE_CKPT_HOST=/tmp GROUNDING_DINO_CKPT_HOST=/tmp SAM_CKPT_HOST=/tmp docker compose config -q && echo OK`
Expected: prints `OK`.

- [ ] **Step 3: Commit**

```bash
git add serving/docker-compose.yml serving/.env.example
git commit -m "build(serving): add grounded-sam-svc to the compose stack"
```

---

# Phase 2: Model wiring, adapters, GUI, integration (RXL, origin/main)

These tasks run on the RXL GPU host against `origin/main`, where `gui/` and `scripts/` are tracked and the models, weights, and `cadrille:latest` image exist. They cannot be unit-tested on the CPU dev box. Each lists exact files, the contract, and the validation to run on RXL.

### Task 9: Factor the shared pipeline core

**Files:**
- Create: `scripts/aiws_pipeline_core.py`
- Modify: `gui/backend/simple_reconstruct_job.py`

**Work:** Extract these functions from `simple_reconstruct_job.py` into `aiws_pipeline_core.py`, unchanged in behavior, and import them back in the GUI job:
- `configure_sam3d_determinism(...)`
- the SAM3D `inference(image, mask, seed)` invocation that produces `mesh.glb` / `mesh.stl`
- `normalize` STL to unit cube
- the Cadrille invocation (`run_cadrille_on_split` / docker `cadrille:latest`) and candidate selection
- the postscale step (`cadrille_metric_postscale.py`)
- preview/decimation and result-path helpers (`decimate_sam3d_preview`, `build_simple_result_paths`)

**Validation (RXL):** run the existing GUI backend tests and a manual `image_mask` job; confirm byte-for-byte equivalent outputs to before the refactor (no behavior change). Commit on `origin/main`.

### Task 10: SAM3D and Cadrille adapter bodies

**Files:**
- Create: `serving/sam3d-svc/run_sam3d.py`, `serving/cadrille-svc/run_cadrille.py`
- Modify: `serving/sam3d-svc/Dockerfile`, `serving/cadrille-svc/Dockerfile` (COPY `scripts/aiws_pipeline_core.py` and its `scripts/` deps into the image; set `PYTHONPATH`)

**Contract (must match the Phase 1 service `inference.py` commands):**
- `run_sam3d.py --input-image P --input-mask P --seed N --out-mesh P`: load image + mask, call the core SAM3D inference, write the mesh to `--out-mesh`.
- `run_cadrille.py --mesh P --mode pc|img --n-candidates N --seed N --ckpt P --device D --out-dir P [--cleanup]`: normalize, run Cadrille via the core module, postscale, write `cad/model.py`, `cad/model.step`, `preview/model.stl`, `preview/render.png`, `metrics.json` under `--out-dir`.

**Validation (RXL):** build both images; run each adapter on one sample by hand; confirm the canonical bundle layout appears.

### Task 11: Real Grounded-SAM segmentor

**Files:**
- Modify: `serving/grounded-sam-svc/app/segmentor.py`, `serving/grounded-sam-svc/Dockerfile`, `serving/grounded-sam-svc/requirements.txt`

**Work:** Implement `Segmentor.warmup` (load GroundingDINO with `grounding_dino_ckpt`, SAM with `sam_ckpt` + `sam_model_type`, onto `device`) and `Segmentor.segment(image_bytes, prompt, box_threshold, text_threshold)`: run GroundingDINO detection, keep the highest-confidence box above `box_threshold`, run SAM to get its mask, encode the mask PNG as base64, return `{mask_png_base64, score, box, label, num_detections}`; raise `NoDetection` when nothing clears the threshold. Add GroundingDINO + segment-anything install to the Dockerfile/requirements. Provide weights on the host and mount them.

**Validation (RXL):** build the image; `POST /segment` a real workpiece photo; confirm a sensible mask and score; confirm a blank image returns 422.

### Task 12: GUI image-only mode

**Files:**
- Modify: `gui/backend/app.py`, `gui/backend/simple_reconstruct_job.py` (or its caller), `gui/frontend/src/components/InputImagesPanel.tsx`, related frontend types/api

**Work:**
- Backend: `create_simple_reconstruct` accepts `input_mode="image"`; on that mode call `grounded-sam-svc /segment` (URL from config, default `http://localhost:${SEG_PORT}`) with the uploaded image, save the returned mask, then run the existing `image_mask` job path. Add the seg URL to `/health`.
- Frontend: add an "RGB only (auto-segment)" choice to the input-mode selector; when active hide the mask uploader and show an optional detection-prompt field; surface a clear error if segmentation finds nothing.

**Validation (RXL):** run the GUI; upload an RGB-only image; confirm the auto mask is generated and the reconstruction completes; confirm the existing `image_mask` and `mesh` modes are unchanged. Run the frontend test suite.

### Task 13: End-to-end build and integration smoke

**Files:**
- Use: `serving/tests/integration/test_end_to_end.py` (extend for image-only)

**Work:** Build all four images; `docker compose up`; wait for `/healthz`. Extend the integration test to also submit an image-only request (`segment=auto`, no mask) and assert the bundle contains a valid STEP and `mesh/auto_mask.png`. Run the GUI RGB-only path once manually.

**Validation (RXL):** `GATEWAY_URL=... SAMPLE_IMAGE=... pytest serving/tests/integration -m integration` passes; the GUI RGB-only job completes.

---

## Self-Review

**Spec coverage:**
- grounded-sam-svc component: Task 7 (shell + contract) and Task 11 (real model).
- POST /segment contract, 422 no-detection, default+override prompt: Tasks 7, 11.
- REST segment stage, segment=auto default, provided path: Tasks 1, 4, 5.
- GroundedSamClient and ServiceError(stage=segment): Task 2.
- sam3d-svc mask_path: Tasks 2 (client), 6 (service).
- bundle auto_mask.png + detection metadata in metrics: Tasks 3, 4.
- compose published seg service + gateway env: Task 8.
- adapters reusing simple_reconstruct_job: Tasks 9, 10.
- GUI image-only mode (backend + frontend): Task 12.
- error handling (no detection, seg down): Tasks 4, 5, 7.
- testing (contract, orchestrator, GUI, integration): Tasks 4, 5, 7, 12, 13.
- deployment sequencing on RXL: Tasks 9 to 13.

**Placeholder scan:** Phase 1 steps contain full code. Phase 2 tasks are specifications, not micro-TDD, because they import RXL-only modules (the SAM3D repo, `cadrille:latest`, GroundingDINO/SAM weights, the GUI runtime) that cannot be exercised on the CPU dev box; each names exact files, the exact CLI contract that the Phase 1 `inference.py` already calls, and the on-RXL validation. This is an environment boundary, not a placeholder.

**Type consistency:** `SegmentMode(auto, provided)`, `Stage.segment`, and `ReconstructOptions.segment/detect_prompt` (Task 1) are used unchanged in Tasks 4 and 5. `GroundedSamClient.segment(image_path, prompt) -> dict` (Task 2) matches the orchestrator call (Task 4) and the service response shape (Task 7). `Sam3dClient.infer(job_id, input_dir, mask_path)` (Task 2) matches the orchestrator call (Task 4) and `sam3d-svc` `InferRequest.mask_path` (Task 6). The `run_sam3d.py` / `run_cadrille.py` flags (Task 10) match the commands built in `sam3d-svc`/`cadrille-svc` `inference.py` (Task 6 and the prior PR #36 cadrille-svc).
