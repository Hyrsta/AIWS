# Dockerized REST CAD Reconstruction Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the offline SAM3D then Cadrille pipeline as a Dockerized REST service: one async full-pipeline endpoint, image input in, CAD result bundle out.

**Architecture:** Three containers wired by docker-compose on an internal network with one shared artifact volume. A lightweight FastAPI gateway (no ML deps) owns the public API, an async job queue, orchestration, and bundle assembly. Two internal services run the models: sam3d-svc (image to mesh) and cadrille-svc (mesh to CAD). Services exchange file paths on the shared volume. Each model service shells out to the existing AIWS batch scripts under a hard subprocess timeout, so the heavy dependency stacks stay isolated in their own images.

**Tech Stack:** Python 3.10+, FastAPI, uvicorn, httpx, pydantic v2, sqlite3 (stdlib), pytest, Docker, docker-compose, NVIDIA Container Toolkit.

## Global Constraints

- No em-dashes or en-dashes in any file, doc, or prose. Use comma, period, colon, or hyphen.
- Cadrille default mode is pc with body-cleanup rerank on. img is an optional override.
- Default n_candidates is 20 (matches the canonical x20 decode configuration). Default seed is 42.
- Result bundle must contain: generated CAD code, materialized STEP, preview mesh and render, intermediate SAM3D mesh, metrics, manifest.
- Host hardware is unknown. Do not hardcode a GPU count. Each service reads its own device from env.
- Only the gateway publishes a host port. sam3d-svc and cadrille-svc are reachable only on the internal compose network.
- Mesh and artifacts move via the shared /artifacts volume, never base64 over HTTP.
- Single GPU-bound worker by default (WORKERS=1) to serialize GPU use and avoid OOM.
- Service code is not tracked in the local worktree. Implementation lands via a PR to origin/main and builds on RXL. Where a step shells out to an existing AIWS script, confirm the flag names against the live scripts in repos/ on RXL before the integration smoke.

## File Structure

```
serving/
  docker-compose.yml          three services, shared volume, internal network
  .env.example                documented configuration
  README.md                   client-facing API and deployment docs
  gateway/
    Dockerfile                slim python, no ML deps
    requirements.txt
    app/
      __init__.py
      config.py               env-driven Settings
      models.py               pydantic schemas and enums
      jobstore.py             SQLite-backed job metadata store
      clients.py              httpx clients for the two services
      bundle.py               manifest + zip assembly
      orchestrator.py         per-job stage orchestration
      worker.py               FIFO single-worker loop
      main.py                 FastAPI app and routes
    tests/
      conftest.py
      test_models.py
      test_jobstore.py
      test_clients.py
      test_bundle.py
      test_orchestrator.py
      test_api.py
  cadrille-svc/
    Dockerfile                FROM cadrille:latest
    requirements.txt
    app/
      __init__.py
      config.py
      inference.py            subprocess adapter to the cadrille flow
      main.py                 FastAPI: POST /infer, GET /healthz
    tests/
      test_health.py
      test_infer_contract.py
  sam3d-svc/
    Dockerfile                from the SAM3D conda base, MKL fix baked in
    requirements.txt
    app/
      __init__.py
      config.py
      inference.py            subprocess adapter to sam3d_batch
      main.py                 FastAPI: POST /infer, GET /healthz
    tests/
      test_health.py
      test_infer_contract.py
  tests/
    integration/
      test_end_to_end.py      one real image, full pipeline, GPU box only
```

Test run convention: from a service directory, run `PYTHONPATH=. pytest`.

---

### Task 1: Gateway config and schemas

**Files:**
- Create: `serving/gateway/app/__init__.py`
- Create: `serving/gateway/app/config.py`
- Create: `serving/gateway/app/models.py`
- Create: `serving/gateway/tests/conftest.py`
- Test: `serving/gateway/tests/test_models.py`

**Interfaces:**
- Produces: `Settings` dataclass with fields `sam3d_url, cadrille_url, artifacts_dir, db_path, workers, stage_timeout_s, max_images`; `load_settings() -> Settings`. Enums `Mode(pc,img)`, `JobState(queued,running,succeeded,failed)`, `Stage(sam3d,cadrille)`. `ReconstructOptions(mode: Mode=pc, n_candidates: int=20 [1..64], seed: int=42, cleanup: bool=True)`. `JobView(job_id, status: JobState, stage: Optional[Stage], error: Optional[str], created_at, updated_at)`.

- [ ] **Step 1: Write the failing test**

`serving/gateway/tests/conftest.py`:
```python
import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
```

`serving/gateway/tests/test_models.py`:
```python
import pytest
from pydantic import ValidationError
from app.models import ReconstructOptions, Mode, JobState

def test_defaults_match_canonical():
    o = ReconstructOptions()
    assert o.mode == Mode.pc
    assert o.n_candidates == 20
    assert o.seed == 42
    assert o.cleanup is True

def test_n_candidates_lower_bound_rejected():
    with pytest.raises(ValidationError):
        ReconstructOptions(n_candidates=0)

def test_n_candidates_upper_bound_rejected():
    with pytest.raises(ValidationError):
        ReconstructOptions(n_candidates=65)

def test_invalid_mode_rejected():
    with pytest.raises(ValidationError):
        ReconstructOptions(mode="solid")

def test_job_state_values():
    assert JobState.queued.value == "queued"
    assert JobState.succeeded.value == "succeeded"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd serving/gateway && PYTHONPATH=. pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app'`.

- [ ] **Step 3: Write minimal implementation**

`serving/gateway/app/__init__.py`: empty file.

`serving/gateway/app/config.py`:
```python
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    sam3d_url: str
    cadrille_url: str
    artifacts_dir: str
    db_path: str
    workers: int
    stage_timeout_s: float
    max_images: int


def load_settings() -> Settings:
    return Settings(
        sam3d_url=os.environ.get("SAM3D_URL", "http://sam3d-svc:8000"),
        cadrille_url=os.environ.get("CADRILLE_URL", "http://cadrille-svc:8000"),
        artifacts_dir=os.environ.get("ARTIFACTS_DIR", "/artifacts"),
        db_path=os.environ.get("DB_PATH", "/artifacts/jobs.db"),
        workers=int(os.environ.get("WORKERS", "1")),
        stage_timeout_s=float(os.environ.get("STAGE_TIMEOUT_S", "1800")),
        max_images=int(os.environ.get("MAX_IMAGES", "16")),
    )
```

`serving/gateway/app/models.py`:
```python
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Mode(str, Enum):
    pc = "pc"
    img = "img"


class JobState(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class Stage(str, Enum):
    sam3d = "sam3d"
    cadrille = "cadrille"


class ReconstructOptions(BaseModel):
    mode: Mode = Mode.pc
    n_candidates: int = Field(default=20, ge=1, le=64)
    seed: int = 42
    cleanup: bool = True


class JobView(BaseModel):
    job_id: str
    status: JobState
    stage: Optional[Stage] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd serving/gateway && PYTHONPATH=. pytest tests/test_models.py -v`
Expected: PASS, 5 passed.

- [ ] **Step 5: Commit**

```bash
git add serving/gateway/app/__init__.py serving/gateway/app/config.py serving/gateway/app/models.py serving/gateway/tests/conftest.py serving/gateway/tests/test_models.py
git commit -m "feat(gateway): config and request schemas"
```

---

### Task 2: Job store (SQLite)

**Files:**
- Create: `serving/gateway/app/jobstore.py`
- Test: `serving/gateway/tests/test_jobstore.py`

**Interfaces:**
- Consumes: `ReconstructOptions, JobState, Stage, JobView` from Task 1.
- Produces: `JobStore(db_path)` with `create(options) -> str (job_id)`, `get(job_id) -> Optional[JobView]`, `set_status(job_id, status, stage=None, error=None)`, `next_queued() -> Optional[str]`, `options(job_id) -> ReconstructOptions`.

- [ ] **Step 1: Write the failing test**

`serving/gateway/tests/test_jobstore.py`:
```python
import os
from app.jobstore import JobStore
from app.models import ReconstructOptions, JobState, Stage, Mode


def make_store(tmp_path):
    return JobStore(os.path.join(tmp_path, "jobs.db"))


def test_create_and_get_roundtrip(tmp_path):
    s = make_store(tmp_path)
    jid = s.create(ReconstructOptions(mode=Mode.img, n_candidates=8))
    job = s.get(jid)
    assert job.job_id == jid
    assert job.status == JobState.queued
    assert job.stage is None


def test_options_persisted(tmp_path):
    s = make_store(tmp_path)
    jid = s.create(ReconstructOptions(mode=Mode.img, n_candidates=8, seed=7))
    opts = s.options(jid)
    assert opts.mode == Mode.img
    assert opts.n_candidates == 8
    assert opts.seed == 7


def test_status_transition_records_stage_and_error(tmp_path):
    s = make_store(tmp_path)
    jid = s.create(ReconstructOptions())
    s.set_status(jid, JobState.failed, stage=Stage.cadrille, error="boom")
    job = s.get(jid)
    assert job.status == JobState.failed
    assert job.stage == Stage.cadrille
    assert job.error == "boom"


def test_next_queued_is_fifo(tmp_path):
    s = make_store(tmp_path)
    a = s.create(ReconstructOptions())
    b = s.create(ReconstructOptions())
    assert s.next_queued() == a
    s.set_status(a, JobState.running, stage=Stage.sam3d)
    assert s.next_queued() == b


def test_get_missing_returns_none(tmp_path):
    s = make_store(tmp_path)
    assert s.get("nope") is None


def test_persists_across_reopen(tmp_path):
    path = os.path.join(tmp_path, "jobs.db")
    jid = JobStore(path).create(ReconstructOptions())
    assert JobStore(path).get(jid).job_id == jid
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd serving/gateway && PYTHONPATH=. pytest tests/test_jobstore.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.jobstore'`.

- [ ] **Step 3: Write minimal implementation**

`serving/gateway/app/jobstore.py`:
```python
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

from .models import JobState, Stage, ReconstructOptions, JobView

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    stage TEXT,
    error TEXT,
    options TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create(self, options: ReconstructOptions) -> str:
        job_id = uuid.uuid4().hex
        now = _now()
        with self._conn() as c:
            c.execute(
                "INSERT INTO jobs (job_id, status, stage, error, options, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (job_id, JobState.queued.value, None, None,
                 options.model_dump_json(), now, now),
            )
        return job_id

    def get(self, job_id: str) -> Optional[JobView]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            return None
        return JobView(
            job_id=row["job_id"],
            status=JobState(row["status"]),
            stage=Stage(row["stage"]) if row["stage"] else None,
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def set_status(self, job_id, status: JobState, stage: Optional[Stage] = None,
                   error: Optional[str] = None):
        with self._conn() as c:
            c.execute(
                "UPDATE jobs SET status=?, stage=?, error=?, updated_at=? WHERE job_id=?",
                (status.value, stage.value if stage else None, error, _now(), job_id),
            )

    def next_queued(self) -> Optional[str]:
        with self._conn() as c:
            row = c.execute(
                "SELECT job_id FROM jobs WHERE status=? ORDER BY created_at ASC, rowid ASC LIMIT 1",
                (JobState.queued.value,),
            ).fetchone()
        return row["job_id"] if row else None

    def options(self, job_id) -> ReconstructOptions:
        with self._conn() as c:
            row = c.execute("SELECT options FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return ReconstructOptions.model_validate_json(row["options"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd serving/gateway && PYTHONPATH=. pytest tests/test_jobstore.py -v`
Expected: PASS, 6 passed.

- [ ] **Step 5: Commit**

```bash
git add serving/gateway/app/jobstore.py serving/gateway/tests/test_jobstore.py
git commit -m "feat(gateway): SQLite job store"
```

---

### Task 3: Service clients

**Files:**
- Create: `serving/gateway/app/clients.py`
- Test: `serving/gateway/tests/test_clients.py`

**Interfaces:**
- Consumes: `ReconstructOptions` from Task 1.
- Produces: `ServiceError(stage: str, message)` exception. `Sam3dClient(base_url, timeout_s, client=None)` with `healthz() -> bool` and `infer(job_id, input_dir) -> str (mesh_path)`. `CadrilleClient(base_url, timeout_s, client=None)` with `healthz() -> bool` and `infer(job_id, mesh_path, options) -> dict` containing keys `cad_code_path, step_path, preview_path, metrics`.

- [ ] **Step 1: Write the failing test**

`serving/gateway/tests/test_clients.py`:
```python
import httpx
import pytest
from app.clients import Sam3dClient, CadrilleClient, ServiceError
from app.models import ReconstructOptions


def client_with(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://svc")


def test_sam3d_infer_returns_mesh_path():
    def handler(request):
        assert request.url.path == "/infer"
        return httpx.Response(200, json={"mesh_path": "/artifacts/jobs/x/mesh/sam3d_mesh.ply"})
    c = Sam3dClient("http://svc", 5, client=client_with(handler))
    assert c.infer("x", "/artifacts/jobs/x/input") == "/artifacts/jobs/x/mesh/sam3d_mesh.ply"


def test_sam3d_infer_raises_service_error_on_500():
    def handler(request):
        return httpx.Response(500, text="model crashed")
    c = Sam3dClient("http://svc", 5, client=client_with(handler))
    with pytest.raises(ServiceError) as ei:
        c.infer("x", "/in")
    assert ei.value.stage == "sam3d"


def test_cadrille_infer_passes_options_and_parses_result():
    def handler(request):
        body = request.read().decode()
        assert '"mode": "pc"' in body or '"mode":"pc"' in body
        return httpx.Response(200, json={
            "cad_code_path": "/a/cad/model.py",
            "step_path": "/a/cad/model.step",
            "preview_path": "/a/preview/model.stl",
            "metrics": {"iou": 0.22},
        })
    c = CadrilleClient("http://svc", 5, client=client_with(handler))
    out = c.infer("x", "/a/mesh/sam3d_mesh.ply", ReconstructOptions())
    assert out["metrics"]["iou"] == 0.22


def test_healthz_false_on_transport_error():
    def handler(request):
        raise httpx.ConnectError("down")
    c = CadrilleClient("http://svc", 5, client=client_with(handler))
    assert c.healthz() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd serving/gateway && PYTHONPATH=. pytest tests/test_clients.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.clients'`.

- [ ] **Step 3: Write minimal implementation**

`serving/gateway/app/clients.py`:
```python
import httpx

from .models import ReconstructOptions


class ServiceError(Exception):
    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(message)


class _Base:
    def __init__(self, base_url: str, timeout_s: float, client=None):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._client = client or httpx.Client(timeout=timeout_s)

    def _healthz(self) -> bool:
        try:
            r = self._client.get(f"{self.base_url}/healthz", timeout=5)
            return r.status_code == 200
        except httpx.HTTPError:
            return False


class Sam3dClient(_Base):
    def healthz(self) -> bool:
        return self._healthz()

    def infer(self, job_id: str, input_dir: str) -> str:
        try:
            r = self._client.post(
                f"{self.base_url}/infer",
                json={"job_id": job_id, "input_dir": input_dir},
                timeout=self.timeout_s,
            )
        except httpx.HTTPError as e:
            raise ServiceError("sam3d", f"sam3d-svc unreachable: {e}")
        if r.status_code != 200:
            raise ServiceError("sam3d", f"sam3d-svc returned {r.status_code}: {r.text}")
        return r.json()["mesh_path"]


class CadrilleClient(_Base):
    def healthz(self) -> bool:
        return self._healthz()

    def infer(self, job_id: str, mesh_path: str, options: ReconstructOptions) -> dict:
        try:
            r = self._client.post(
                f"{self.base_url}/infer",
                json={
                    "job_id": job_id,
                    "mesh_path": mesh_path,
                    "mode": options.mode.value,
                    "n_candidates": options.n_candidates,
                    "seed": options.seed,
                    "cleanup": options.cleanup,
                },
                timeout=self.timeout_s,
            )
        except httpx.HTTPError as e:
            raise ServiceError("cadrille", f"cadrille-svc unreachable: {e}")
        if r.status_code != 200:
            raise ServiceError("cadrille", f"cadrille-svc returned {r.status_code}: {r.text}")
        return r.json()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd serving/gateway && PYTHONPATH=. pytest tests/test_clients.py -v`
Expected: PASS, 4 passed.

- [ ] **Step 5: Commit**

```bash
git add serving/gateway/app/clients.py serving/gateway/tests/test_clients.py
git commit -m "feat(gateway): httpx clients for sam3d and cadrille services"
```

---

### Task 4: Bundle assembly

**Files:**
- Create: `serving/gateway/app/bundle.py`
- Test: `serving/gateway/tests/test_bundle.py`

**Interfaces:**
- Consumes: `ReconstructOptions` from Task 1.
- Produces: `ARTIFACT_NAMES` list; `write_manifest(job_dir, job_id, options, metrics) -> str (path)`; `build_zip(job_dir, job_id) -> str (zip_path)`. The zip contains every artifact that exists plus manifest.json.

- [ ] **Step 1: Write the failing test**

`serving/gateway/tests/test_bundle.py`:
```python
import json
import os
import zipfile
from app import bundle
from app.models import ReconstructOptions


def seed_job_dir(job_dir):
    for rel in ["cad/model.py", "cad/model.step", "preview/model.stl",
                "preview/render.png", "mesh/sam3d_mesh.ply", "metrics.json"]:
        full = os.path.join(job_dir, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write("x")


def test_write_manifest_contains_options_and_metrics(tmp_path):
    jd = str(tmp_path)
    path = bundle.write_manifest(jd, "job1", ReconstructOptions(), {"iou": 0.22})
    data = json.load(open(path))
    assert data["job_id"] == "job1"
    assert data["options"]["mode"] == "pc"
    assert data["metrics"]["iou"] == 0.22


def test_build_zip_includes_all_artifacts(tmp_path):
    jd = str(tmp_path)
    seed_job_dir(jd)
    bundle.write_manifest(jd, "job1", ReconstructOptions(), {})
    zp = bundle.build_zip(jd, "job1")
    with zipfile.ZipFile(zp) as z:
        names = set(z.namelist())
    assert "cad/model.py" in names
    assert "cad/model.step" in names
    assert "preview/model.stl" in names
    assert "preview/render.png" in names
    assert "mesh/sam3d_mesh.ply" in names
    assert "metrics.json" in names
    assert "manifest.json" in names


def test_build_zip_skips_missing_artifacts(tmp_path):
    jd = str(tmp_path)
    os.makedirs(os.path.join(jd, "cad"))
    with open(os.path.join(jd, "cad", "model.py"), "w") as f:
        f.write("x")
    zp = bundle.build_zip(jd, "job1")
    with zipfile.ZipFile(zp) as z:
        names = set(z.namelist())
    assert names == {"cad/model.py"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd serving/gateway && PYTHONPATH=. pytest tests/test_bundle.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.bundle'`.

- [ ] **Step 3: Write minimal implementation**

`serving/gateway/app/bundle.py`:
```python
import json
import os
import zipfile

from .models import ReconstructOptions

ARTIFACT_NAMES = [
    "cad/model.py",
    "cad/model.step",
    "preview/model.stl",
    "preview/render.png",
    "mesh/sam3d_mesh.ply",
    "metrics.json",
]


def write_manifest(job_dir: str, job_id: str, options: ReconstructOptions,
                   metrics: dict) -> str:
    manifest = {
        "job_id": job_id,
        "options": options.model_dump(mode="json"),
        "metrics": metrics,
        "artifacts": [n for n in ARTIFACT_NAMES if os.path.exists(os.path.join(job_dir, n))],
    }
    path = os.path.join(job_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def build_zip(job_dir: str, job_id: str) -> str:
    zip_path = os.path.join(job_dir, f"{job_id}.zip")
    members = ARTIFACT_NAMES + ["manifest.json"]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in members:
            full = os.path.join(job_dir, rel)
            if os.path.exists(full):
                z.write(full, rel)
    return zip_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd serving/gateway && PYTHONPATH=. pytest tests/test_bundle.py -v`
Expected: PASS, 3 passed.

- [ ] **Step 5: Commit**

```bash
git add serving/gateway/app/bundle.py serving/gateway/tests/test_bundle.py
git commit -m "feat(gateway): result bundle manifest and zip assembly"
```

---

### Task 5: Orchestrator and worker

**Files:**
- Create: `serving/gateway/app/orchestrator.py`
- Create: `serving/gateway/app/worker.py`
- Test: `serving/gateway/tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `JobStore` (Task 2), `Sam3dClient/CadrilleClient/ServiceError` (Task 3), `bundle` (Task 4), `JobState/Stage` (Task 1).
- Produces: `Orchestrator(store, sam3d, cadrille, artifacts_dir)` with `job_dir(job_id) -> str` and `run(job_id)`. On success the job ends `succeeded` with stage cleared and a zip on disk. On a stage failure the job ends `failed` with `stage` set to the failing stage. `Worker(store, orchestrator, poll_interval_s=0.5)` with `start()` and `stop()`.

- [ ] **Step 1: Write the failing test**

`serving/gateway/tests/test_orchestrator.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd serving/gateway && PYTHONPATH=. pytest tests/test_orchestrator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.orchestrator'`.

- [ ] **Step 3: Write minimal implementation**

`serving/gateway/app/orchestrator.py`:
```python
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
```

`serving/gateway/app/worker.py`:
```python
import threading
import time


class Worker:
    def __init__(self, store, orchestrator, poll_interval_s: float = 0.5):
        self.store = store
        self.orchestrator = orchestrator
        self.poll_interval_s = poll_interval_s
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _loop(self):
        while not self._stop.is_set():
            job_id = self.store.next_queued()
            if job_id is None:
                time.sleep(self.poll_interval_s)
                continue
            self.orchestrator.run(job_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd serving/gateway && PYTHONPATH=. pytest tests/test_orchestrator.py -v`
Expected: PASS, 3 passed.

- [ ] **Step 5: Commit**

```bash
git add serving/gateway/app/orchestrator.py serving/gateway/app/worker.py serving/gateway/tests/test_orchestrator.py
git commit -m "feat(gateway): job orchestrator and single-worker loop"
```

---

### Task 6: Gateway API routes

**Files:**
- Create: `serving/gateway/app/main.py`
- Test: `serving/gateway/tests/test_api.py`

**Interfaces:**
- Consumes: all gateway modules from Tasks 1 to 5.
- Produces: FastAPI `app` with `POST /v1/reconstruct` (multipart images plus form fields mode, n_candidates, seed, cleanup; returns 202 {job_id, status}), `GET /v1/jobs/{id}` (JobView or 404), `GET /v1/jobs/{id}/result` (zip FileResponse, 409 if not succeeded, 404 if unknown), `GET /healthz`, `GET /livez`. A module-level `build_app(settings, store, sam3d, cadrille, worker)` factory enables tests to inject fakes.

- [ ] **Step 1: Write the failing test**

`serving/gateway/tests/test_api.py`:
```python
import io
import os
from fastapi.testclient import TestClient
from app.main import build_app
from app.jobstore import JobStore
from app.config import Settings
from app.models import JobState, Stage


class StubWorker:
    def start(self): pass
    def stop(self): pass


class StubHealth:
    def __init__(self, ok): self.ok = ok
    def healthz(self): return self.ok


def make_client(tmp_path, sam_ok=True, cad_ok=True):
    settings = Settings(
        sam3d_url="http://sam3d", cadrille_url="http://cadrille",
        artifacts_dir=str(tmp_path), db_path=os.path.join(tmp_path, "jobs.db"),
        workers=1, stage_timeout_s=10, max_images=4,
    )
    store = JobStore(settings.db_path)
    app = build_app(settings, store, StubHealth(sam_ok), StubHealth(cad_ok), StubWorker())
    return TestClient(app), store


def png_bytes():
    return io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"0" * 16)


def test_livez_ok(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.get("/livez").status_code == 200


def test_reconstruct_enqueues_and_returns_job_id(tmp_path):
    client, store = make_client(tmp_path)
    r = client.post("/v1/reconstruct",
                    files={"images": ("a.png", png_bytes(), "image/png")},
                    data={"mode": "pc"})
    assert r.status_code == 202
    jid = r.json()["job_id"]
    assert store.get(jid).status == JobState.queued
    assert os.path.exists(os.path.join(str(tmp_path), "jobs", jid, "input"))


def test_reconstruct_rejects_no_images(tmp_path):
    client, _ = make_client(tmp_path)
    r = client.post("/v1/reconstruct", data={"mode": "pc"})
    assert r.status_code == 422  # FastAPI required-field validation


def test_reconstruct_rejects_bad_mode(tmp_path):
    client, _ = make_client(tmp_path)
    r = client.post("/v1/reconstruct",
                    files={"images": ("a.png", png_bytes(), "image/png")},
                    data={"mode": "solid"})
    assert r.status_code == 400


def test_reconstruct_rejects_too_many_images(tmp_path):
    client, _ = make_client(tmp_path)
    files = [("images", (f"{i}.png", png_bytes(), "image/png")) for i in range(5)]
    r = client.post("/v1/reconstruct", files=files, data={"mode": "pc"})
    assert r.status_code == 400


def test_get_job_404_for_unknown(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.get("/v1/jobs/nope").status_code == 404


def test_result_409_when_not_done(tmp_path):
    client, store = make_client(tmp_path)
    r = client.post("/v1/reconstruct",
                    files={"images": ("a.png", png_bytes(), "image/png")},
                    data={"mode": "pc"})
    jid = r.json()["job_id"]
    assert client.get(f"/v1/jobs/{jid}/result").status_code == 409


def test_result_returns_zip_when_succeeded(tmp_path):
    client, store = make_client(tmp_path)
    r = client.post("/v1/reconstruct",
                    files={"images": ("a.png", png_bytes(), "image/png")},
                    data={"mode": "pc"})
    jid = r.json()["job_id"]
    zpath = os.path.join(str(tmp_path), "jobs", jid, f"{jid}.zip")
    open(zpath, "wb").write(b"PK\x03\x04zip")
    store.set_status(jid, JobState.succeeded, stage=None)
    rr = client.get(f"/v1/jobs/{jid}/result")
    assert rr.status_code == 200
    assert rr.headers["content-type"] == "application/zip"


def test_healthz_503_when_downstream_down(tmp_path):
    client, _ = make_client(tmp_path, cad_ok=False)
    assert client.get("/healthz").status_code == 503
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd serving/gateway && PYTHONPATH=. pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.main'`.

- [ ] **Step 3: Write minimal implementation**

`serving/gateway/app/main.py`:
```python
import os
import shutil

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
    app = FastAPI(title="AIWS CAD Reconstruction API")

    @app.on_event("startup")
    def _startup():
        worker.start()

    @app.on_event("shutdown")
    def _shutdown():
        worker.stop()

    @app.get("/livez")
    def livez():
        return {"status": "ok"}

    @app.get("/healthz")
    def healthz():
        if not (sam3d.healthz() and cadrille.healthz()):
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd serving/gateway && PYTHONPATH=. pytest tests/test_api.py -v`
Expected: PASS, 9 passed. Then run the whole gateway suite: `PYTHONPATH=. pytest -v` and expect all green.

Note: `app = create_app()` runs at import. In tests we import `build_app`, not `app`, and tests inject fakes, so no live service connection is attempted. The full-suite run will still import main and call `create_app()`, which constructs (but does not call) real clients and opens a SQLite file at the default `/artifacts/jobs.db`. To keep the suite hermetic, set `ARTIFACTS_DIR` and `DB_PATH` to a temp path for the suite run:
Run: `cd serving/gateway && ARTIFACTS_DIR=$(mktemp -d) DB_PATH=$(mktemp -u) PYTHONPATH=. pytest -v`

- [ ] **Step 5: Commit**

```bash
git add serving/gateway/app/main.py serving/gateway/tests/test_api.py
git commit -m "feat(gateway): REST routes for reconstruct, job status, and result"
```

---

### Task 7: Gateway Dockerfile and dependencies

**Files:**
- Create: `serving/gateway/requirements.txt`
- Create: `serving/gateway/Dockerfile`

**Interfaces:**
- Produces: a buildable gateway image exposing port 8000, running uvicorn on `app.main:app`.

- [ ] **Step 1: Write the dependency and image files**

`serving/gateway/requirements.txt`:
```
fastapi==0.111.0
uvicorn[standard]==0.30.1
httpx==0.27.0
pydantic==2.7.4
python-multipart==0.0.9
pytest==8.2.2
```

`serving/gateway/Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV ARTIFACTS_DIR=/artifacts
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=10 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/livez').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Build the image**

Run: `cd serving/gateway && docker build -t aiws-gateway:dev .`
Expected: build succeeds, final line `naming to docker.io/library/aiws-gateway:dev`.

- [ ] **Step 3: Smoke test the container liveness**

Run:
```bash
docker run -d --name gw-smoke -p 18080:8000 -e ARTIFACTS_DIR=/tmp/art aiws-gateway:dev
sleep 3
curl -fsS http://localhost:18080/livez
docker rm -f gw-smoke
```
Expected: `{"status":"ok"}`.

- [ ] **Step 4: Commit**

```bash
git add serving/gateway/requirements.txt serving/gateway/Dockerfile
git commit -m "build(gateway): Dockerfile and pinned requirements"
```

---

### Task 8: cadrille-svc service

**Files:**
- Create: `serving/cadrille-svc/app/__init__.py`
- Create: `serving/cadrille-svc/app/config.py`
- Create: `serving/cadrille-svc/app/inference.py`
- Create: `serving/cadrille-svc/app/main.py`
- Create: `serving/cadrille-svc/requirements.txt`
- Create: `serving/cadrille-svc/Dockerfile`
- Test: `serving/cadrille-svc/tests/test_health.py`
- Test: `serving/cadrille-svc/tests/test_infer_contract.py`

**Interfaces:**
- Produces: FastAPI `build_app(settings, runner)` with `GET /healthz` (200 when ready) and `POST /infer` accepting `{job_id, mesh_path, mode, n_candidates, seed, cleanup}` and returning `{cad_code_path, step_path, preview_path, metrics}`. `inference.run(settings, req) -> dict` shells out to the cadrille flow under a hard timeout and writes the canonical `cad/`, `preview/`, and `metrics.json` layout into the job directory.

- [ ] **Step 1: Write the failing tests**

`serving/cadrille-svc/tests/test_health.py`:
```python
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fastapi.testclient import TestClient
from app.main import build_app
from app.config import Settings


class FakeRunner:
    ready = True
    def run(self, settings, req): return {}


def test_healthz_ok():
    app = build_app(Settings(cadrille_entry="x", cadrille_ckpt="x",
                             device="cuda:0", stage_timeout_s=10), FakeRunner())
    assert TestClient(app).get("/healthz").status_code == 200
```

`serving/cadrille-svc/tests/test_infer_contract.py`:
```python
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fastapi.testclient import TestClient
from app.main import build_app
from app.config import Settings


class OkRunner:
    ready = True
    def run(self, settings, req):
        assert req["mode"] == "pc"
        return {"cad_code_path": "/a/cad/model.py", "step_path": "/a/cad/model.step",
                "preview_path": "/a/preview/model.stl", "metrics": {"iou": 0.2}}


class BoomRunner:
    ready = True
    def run(self, settings, req):
        raise RuntimeError("materialize timed out")


def settings():
    return Settings(cadrille_entry="x", cadrille_ckpt="x", device="cuda:0", stage_timeout_s=10)


def test_infer_returns_paths_and_metrics():
    client = TestClient(build_app(settings(), OkRunner()))
    r = client.post("/infer", json={"job_id": "j", "mesh_path": "/a/mesh/sam3d_mesh.ply",
                                    "mode": "pc", "n_candidates": 20, "seed": 42, "cleanup": True})
    assert r.status_code == 200
    assert r.json()["metrics"]["iou"] == 0.2


def test_infer_failure_returns_500():
    client = TestClient(build_app(settings(), BoomRunner()))
    r = client.post("/infer", json={"job_id": "j", "mesh_path": "/a/mesh/sam3d_mesh.ply",
                                    "mode": "pc", "n_candidates": 20, "seed": 42, "cleanup": True})
    assert r.status_code == 500
    assert "timed out" in r.text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd serving/cadrille-svc && PYTHONPATH=. pytest -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.main'`.

- [ ] **Step 3: Write minimal implementation**

`serving/cadrille-svc/app/__init__.py`: empty file.

`serving/cadrille-svc/app/config.py`:
```python
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    cadrille_entry: str
    cadrille_ckpt: str
    device: str
    stage_timeout_s: float


def load_settings() -> Settings:
    return Settings(
        cadrille_entry=os.environ.get("CADRILLE_ENTRY", "/opt/aiws/run_cadrille.py"),
        cadrille_ckpt=os.environ.get("CADRILLE_CKPT", "/ckpt/cadrille"),
        device=os.environ.get("CADRILLE_DEVICE", "cuda:0"),
        stage_timeout_s=float(os.environ.get("STAGE_TIMEOUT_S", "1800")),
    )
```

`serving/cadrille-svc/app/inference.py`:
```python
import json
import os
import subprocess


class Runner:
    """Shells out to the cadrille flow under a hard timeout.

    The hard timeout converts the known OCC materialization deadlock into a
    clean failure instead of a hung request.
    """

    def __init__(self):
        self.ready = False

    def warmup(self, settings) -> None:
        if not os.path.exists(settings.cadrille_ckpt):
            raise RuntimeError(f"cadrille checkpoint not found: {settings.cadrille_ckpt}")
        self.ready = True

    def run(self, settings, req: dict) -> dict:
        mesh_path = req["mesh_path"]
        job_dir = os.path.dirname(os.path.dirname(mesh_path))
        # CADRILLE_ENTRY is a thin CLI on the image that wraps the existing
        # cadrille_infer_wrapper + cadrille_evaluate_wrapper to emit the canonical
        # cad/, preview/, metrics.json layout. Confirm flag names against the live
        # wrappers in repos/cadrille on RXL before the integration smoke.
        cmd = [
            "python", settings.cadrille_entry,
            "--mesh", mesh_path,
            "--mode", req["mode"],
            "--n-candidates", str(req["n_candidates"]),
            "--seed", str(req["seed"]),
            "--ckpt", settings.cadrille_ckpt,
            "--device", settings.device,
            "--out-dir", job_dir,
        ]
        if req.get("cleanup", True):
            cmd.append("--cleanup")
        try:
            subprocess.run(cmd, check=True, timeout=settings.stage_timeout_s,
                           capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            raise RuntimeError("cadrille inference timed out")
        except subprocess.CalledProcessError as e:
            tail = (e.stderr or "")[-2000:]
            raise RuntimeError(f"cadrille inference failed: {tail}")

        metrics_path = os.path.join(job_dir, "metrics.json")
        metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
        return {
            "cad_code_path": os.path.join(job_dir, "cad", "model.py"),
            "step_path": os.path.join(job_dir, "cad", "model.step"),
            "preview_path": os.path.join(job_dir, "preview", "model.stl"),
            "metrics": metrics,
        }
```

`serving/cadrille-svc/app/main.py`:
```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import load_settings
from .inference import Runner


class InferRequest(BaseModel):
    job_id: str
    mesh_path: str
    mode: str = "pc"
    n_candidates: int = 20
    seed: int = 42
    cleanup: bool = True


def build_app(settings, runner) -> FastAPI:
    app = FastAPI(title="cadrille-svc")

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
```

Note: tests construct `build_app` with a fake runner whose `ready=True`, so the `create_app` warmup that checks the checkpoint path is not exercised in unit tests.

`serving/cadrille-svc/requirements.txt`:
```
fastapi==0.111.0
uvicorn[standard]==0.30.1
pydantic==2.7.4
pytest==8.2.2
```

`serving/cadrille-svc/Dockerfile`:
```dockerfile
FROM cadrille:latest

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
# run_cadrille.py is the thin adapter CLI that wraps the existing cadrille
# wrappers and emits the canonical bundle layout. It is provided by the AIWS
# integration repo and copied to CADRILLE_ENTRY at build time on RXL.
COPY run_cadrille.py /opt/aiws/run_cadrille.py

ENV CADRILLE_ENTRY=/opt/aiws/run_cadrille.py
ENV CADRILLE_DEVICE=cuda:0
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=20 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Note on `run_cadrille.py`: this adapter lives in the AIWS repo alongside the existing wrappers (it is not reproduced here because it imports `cadrille_infer_wrapper` and `cadrille_evaluate_wrapper`, which are only present on RXL). Its contract is fixed by this task: read `--mesh --mode --n-candidates --seed --ckpt --device --out-dir [--cleanup]`, run point-cloud sampling or 4-view render per mode, run Cadrille inference, materialize the best candidate to `out-dir/cad/model.py` and `model.step`, write `out-dir/preview/model.stl` and `render.png`, and write `out-dir/metrics.json`. Build it by reusing the logic already in `scripts/e2e_sam3d_to_cadrille.py` for the cadrille half.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd serving/cadrille-svc && PYTHONPATH=. pytest -v`
Expected: PASS, 3 passed.

- [ ] **Step 5: Commit**

```bash
git add serving/cadrille-svc
git commit -m "feat(cadrille-svc): FastAPI infer wrapper, timeout runner, Dockerfile"
```

---

### Task 9: sam3d-svc service

**Files:**
- Create: `serving/sam3d-svc/app/__init__.py`
- Create: `serving/sam3d-svc/app/config.py`
- Create: `serving/sam3d-svc/app/inference.py`
- Create: `serving/sam3d-svc/app/main.py`
- Create: `serving/sam3d-svc/requirements.txt`
- Create: `serving/sam3d-svc/Dockerfile`
- Test: `serving/sam3d-svc/tests/test_health.py`
- Test: `serving/sam3d-svc/tests/test_infer_contract.py`

**Interfaces:**
- Produces: FastAPI `build_app(settings, runner)` with `GET /healthz` and `POST /infer` accepting `{job_id, input_dir}` and returning `{mesh_path}`. `inference.run(settings, req) -> dict` shells out to sam3d_batch under a hard timeout and writes the mesh to `mesh/sam3d_mesh.ply` in the job directory.

- [ ] **Step 1: Write the failing tests**

`serving/sam3d-svc/tests/test_health.py`:
```python
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fastapi.testclient import TestClient
from app.main import build_app
from app.config import Settings


class FakeRunner:
    ready = True
    def run(self, settings, req): return {}


def test_healthz_ok():
    app = build_app(Settings(sam3d_entry="x", sam3d_ckpt="x",
                             device="cuda:0", stage_timeout_s=10), FakeRunner())
    assert TestClient(app).get("/healthz").status_code == 200
```

`serving/sam3d-svc/tests/test_infer_contract.py`:
```python
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fastapi.testclient import TestClient
from app.main import build_app
from app.config import Settings


class OkRunner:
    ready = True
    def run(self, settings, req):
        assert req["input_dir"].endswith("input")
        return {"mesh_path": "/artifacts/jobs/j/mesh/sam3d_mesh.ply"}


class BoomRunner:
    ready = True
    def run(self, settings, req):
        raise RuntimeError("sam3d timed out")


def settings():
    return Settings(sam3d_entry="x", sam3d_ckpt="x", device="cuda:0", stage_timeout_s=10)


def test_infer_returns_mesh_path():
    client = TestClient(build_app(settings(), OkRunner()))
    r = client.post("/infer", json={"job_id": "j", "input_dir": "/artifacts/jobs/j/input"})
    assert r.status_code == 200
    assert r.json()["mesh_path"].endswith("sam3d_mesh.ply")


def test_infer_failure_returns_500():
    client = TestClient(build_app(settings(), BoomRunner()))
    r = client.post("/infer", json={"job_id": "j", "input_dir": "/artifacts/jobs/j/input"})
    assert r.status_code == 500
    assert "timed out" in r.text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd serving/sam3d-svc && PYTHONPATH=. pytest -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.main'`.

- [ ] **Step 3: Write minimal implementation**

`serving/sam3d-svc/app/__init__.py`: empty file.

`serving/sam3d-svc/app/config.py`:
```python
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    sam3d_entry: str
    sam3d_ckpt: str
    device: str
    stage_timeout_s: float


def load_settings() -> Settings:
    return Settings(
        sam3d_entry=os.environ.get("SAM3D_ENTRY", "/opt/aiws/run_sam3d.py"),
        sam3d_ckpt=os.environ.get("SAM3D_CKPT", "/ckpt/sam3d"),
        device=os.environ.get("SAM3D_DEVICE", "cuda:0"),
        stage_timeout_s=float(os.environ.get("STAGE_TIMEOUT_S", "1800")),
    )
```

`serving/sam3d-svc/app/inference.py`:
```python
import os
import subprocess


class Runner:
    def __init__(self):
        self.ready = False

    def warmup(self, settings) -> None:
        if not os.path.exists(settings.sam3d_ckpt):
            raise RuntimeError(f"sam3d checkpoint not found: {settings.sam3d_ckpt}")
        self.ready = True

    def run(self, settings, req: dict) -> dict:
        input_dir = req["input_dir"]
        job_dir = os.path.dirname(input_dir)
        mesh_dir = os.path.join(job_dir, "mesh")
        os.makedirs(mesh_dir, exist_ok=True)
        mesh_path = os.path.join(mesh_dir, "sam3d_mesh.ply")
        # SAM3D_ENTRY is a thin CLI on the image that wraps sam3d_batch.py.
        # Confirm flag names against the live script in repos/sam-3d-objects on RXL.
        cmd = [
            "python", settings.sam3d_entry,
            "--input-dir", input_dir,
            "--ckpt", settings.sam3d_ckpt,
            "--device", settings.device,
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
```

`serving/sam3d-svc/app/main.py`:
```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import load_settings
from .inference import Runner


class InferRequest(BaseModel):
    job_id: str
    input_dir: str


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
```

`serving/sam3d-svc/requirements.txt`:
```
fastapi==0.111.0
uvicorn[standard]==0.30.1
pydantic==2.7.4
pytest==8.2.2
```

`serving/sam3d-svc/Dockerfile`:
```dockerfile
# Base image is the SAM3D conda environment. Build this base on RXL from the
# existing SAM3D environment, with the MKL fix baked in: the environment must
# NOT carry the base-conda sequential MKL via LD_PRELOAD. The verification step
# below fails the build if a leaked LD_PRELOAD is present.
FROM aiws-sam3d-base:latest

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY run_sam3d.py /opt/aiws/run_sam3d.py

# Verify no leaked MKL preload is baked into the image environment.
RUN test -z "${LD_PRELOAD}" || (echo "LD_PRELOAD must be empty in sam3d image" && exit 1)

ENV SAM3D_ENTRY=/opt/aiws/run_sam3d.py
ENV SAM3D_DEVICE=cuda:0
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=20 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Note on `run_sam3d.py`: thin adapter in the AIWS repo wrapping `scripts/sam3d_batch.py`. Contract fixed by this task: read `--input-dir --ckpt --device --out-mesh`, run SAM3D on the images in input-dir, write the reconstructed mesh to `--out-mesh`. Seed handling follows the existing seeded sam3d_batch path.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd serving/sam3d-svc && PYTHONPATH=. pytest -v`
Expected: PASS, 3 passed.

- [ ] **Step 5: Commit**

```bash
git add serving/sam3d-svc
git commit -m "feat(sam3d-svc): FastAPI infer wrapper, timeout runner, Dockerfile"
```

---

### Task 10: docker-compose and configuration

**Files:**
- Create: `serving/docker-compose.yml`
- Create: `serving/.env.example`

**Interfaces:**
- Produces: a compose stack with gateway (published), sam3d-svc, cadrille-svc, one shared `artifacts` volume, an internal network, healthchecks, and GPU reservations that do not hardcode a count.

- [ ] **Step 1: Write the compose and env files**

`serving/.env.example`:
```
# Host port the gateway listens on
GATEWAY_PORT=8080

# Per-service device. On a single-GPU host set both to cuda:0.
SAM3D_DEVICE=cuda:0
CADRILLE_DEVICE=cuda:0

# Checkpoint paths on the host, mounted read-only into the services
SAM3D_CKPT_HOST=/data/ckpt/sam3d
CADRILLE_CKPT_HOST=/data/ckpt/cadrille

# Hard per-stage timeout in seconds
STAGE_TIMEOUT_S=1800

# Gateway worker count. Keep at 1 to serialize GPU work.
WORKERS=1
```

`serving/docker-compose.yml`:
```yaml
services:
  gateway:
    build: ./gateway
    image: aiws-gateway:dev
    ports:
      - "${GATEWAY_PORT:-8080}:8000"
    environment:
      SAM3D_URL: http://sam3d-svc:8000
      CADRILLE_URL: http://cadrille-svc:8000
      ARTIFACTS_DIR: /artifacts
      DB_PATH: /artifacts/jobs.db
      WORKERS: ${WORKERS:-1}
      STAGE_TIMEOUT_S: ${STAGE_TIMEOUT_S:-1800}
    volumes:
      - artifacts:/artifacts
    networks:
      - aiws
    depends_on:
      - sam3d-svc
      - cadrille-svc

  sam3d-svc:
    build: ./sam3d-svc
    image: aiws-sam3d-svc:dev
    environment:
      SAM3D_DEVICE: ${SAM3D_DEVICE:-cuda:0}
      SAM3D_CKPT: /ckpt/sam3d
      STAGE_TIMEOUT_S: ${STAGE_TIMEOUT_S:-1800}
    volumes:
      - artifacts:/artifacts
      - ${SAM3D_CKPT_HOST}:/ckpt/sam3d:ro
    networks:
      - aiws
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: ["gpu"]

  cadrille-svc:
    build: ./cadrille-svc
    image: aiws-cadrille-svc:dev
    environment:
      CADRILLE_DEVICE: ${CADRILLE_DEVICE:-cuda:0}
      CADRILLE_CKPT: /ckpt/cadrille
      STAGE_TIMEOUT_S: ${STAGE_TIMEOUT_S:-1800}
    volumes:
      - artifacts:/artifacts
      - ${CADRILLE_CKPT_HOST}:/ckpt/cadrille:ro
    networks:
      - aiws
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: ["gpu"]

volumes:
  artifacts:

networks:
  aiws:
    driver: bridge
```

- [ ] **Step 2: Validate the compose file**

Run: `cd serving && cp .env.example .env && SAM3D_CKPT_HOST=/tmp CADRILLE_CKPT_HOST=/tmp docker compose config -q && echo OK`
Expected: prints `OK` with no schema errors.

- [ ] **Step 3: Commit**

```bash
git add serving/docker-compose.yml serving/.env.example
git commit -m "build(serving): docker-compose stack, shared volume, GPU reservation"
```

---

### Task 11: Client-facing README

**Files:**
- Create: `serving/README.md`

**Interfaces:**
- Produces: usage docs for the remote client and host operator. No code dependency.

- [ ] **Step 1: Write the README**

`serving/README.md`:
```markdown
# AIWS CAD Reconstruction REST Service

Submit RGB images, get a CAD reconstruction bundle back. The SAM3D then Cadrille
pipeline runs server-side as one async job. Only the gateway is exposed; the two
model services are internal.

## Host prerequisites

- Docker and docker-compose.
- NVIDIA Container Toolkit, so containers can reach the GPU.
- The cadrille and sam3d base images built on the host (see each service Dockerfile).
- Model checkpoints on the host, referenced by SAM3D_CKPT_HOST and CADRILLE_CKPT_HOST.

## Configure and run

```bash
cd serving
cp .env.example .env
# edit .env: set GATEWAY_PORT, device per GPU layout, checkpoint paths
docker compose up --build -d
curl -fsS http://localhost:8080/healthz
```

On a single-GPU host, set both SAM3D_DEVICE and CADRILLE_DEVICE to cuda:0. On a
multi-GPU host, set them to distinct devices.

## API

### Submit a job

```bash
curl -s -X POST http://localhost:8080/v1/reconstruct \
  -F "images=@view0.png" \
  -F "images=@view1.png" \
  -F "mode=pc" \
  -F "n_candidates=20" \
  -F "seed=42" \
  -F "cleanup=true"
```

Response:

```json
{ "job_id": "ab12cd...", "status": "queued" }
```

Options: mode is pc (default, best) or img. n_candidates default 20. seed default
42. cleanup default true.

### Poll status

```bash
curl -s http://localhost:8080/v1/jobs/ab12cd...
```

```json
{ "job_id": "ab12cd...", "status": "running", "stage": "cadrille", "error": null }
```

status is queued, running, succeeded, or failed. stage is sam3d or cadrille while
running, and identifies the failing stage when status is failed.

### Fetch the result

```bash
curl -s -o result.zip http://localhost:8080/v1/jobs/ab12cd.../result
```

Returns 409 until the job has succeeded. The zip contains:

- cad/model.py: generated CAD code
- cad/model.step: materialized STEP
- preview/model.stl and preview/render.png: preview of the reconstruction
- mesh/sam3d_mesh.ply: intermediate SAM3D mesh
- metrics.json: candidate metrics and chosen-candidate info
- manifest.json: job parameters and artifact index

## Notes

- Jobs run one at a time by default to serialize GPU use. Raise WORKERS only if
  the host has the VRAM to run both stages concurrently.
- Authentication and TLS are out of scope here. Put the gateway behind a reverse
  proxy for any non-trusted network.
```

- [ ] **Step 2: Scan the README for forbidden dashes**

Run: `grep -n "—\|–" serving/README.md || echo "clean"`
Expected: prints `clean`.

- [ ] **Step 3: Commit**

```bash
git add serving/README.md
git commit -m "docs(serving): client-facing API and deployment README"
```

---

### Task 12: Integration smoke test (GPU box)

**Files:**
- Create: `serving/tests/integration/test_end_to_end.py`

**Interfaces:**
- Consumes: a running compose stack on a GPU host (RXL or GXD), reachable at `GATEWAY_URL`.
- Produces: an end-to-end assertion that one real image yields a complete bundle with a valid STEP.

This task runs on a GPU host with the stack up. It is not part of the CPU unit suite.

- [ ] **Step 1: Write the integration test**

`serving/tests/integration/test_end_to_end.py`:
```python
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
```

- [ ] **Step 2: Bring up the stack on the GPU host**

Run:
```bash
cd serving && cp .env.example .env
# edit .env with real checkpoint paths and device layout
docker compose up --build -d
# wait for readiness
until curl -fsS http://localhost:8080/healthz; do sleep 5; done
```
Expected: `/healthz` returns `{"status":"ok"}` once both services finish loading.

- [ ] **Step 3: Run the integration test**

Run: `cd serving && GATEWAY_URL=http://localhost:8080 SAMPLE_IMAGE=/path/to/real/view.png pytest tests/integration/test_end_to_end.py -v -m integration`
Expected: PASS. The job reaches succeeded and the bundle contains every artifact with a valid STEP header.

- [ ] **Step 4: Commit**

```bash
git add serving/tests/integration/test_end_to_end.py
git commit -m "test(serving): end-to-end integration smoke for the full pipeline"
```

---

## Self-Review

**Spec coverage check:**

- One full-pipeline async endpoint: Tasks 1, 5, 6 (POST /v1/reconstruct, queue, orchestrator).
- Split into two internal services behind a gateway: Tasks 6, 8, 9, 10.
- Async job plus poll: Tasks 2, 6 (job store, GET /v1/jobs/{id}).
- PC default with cleanup, img override, n_candidates 20, seed 42: Tasks 1, 8.
- Result bundle with all six artifact groups plus manifest: Task 4, asserted in Task 12.
- Internal service contracts on the shared volume: Tasks 3, 8, 9, mesh handoff via /artifacts.
- Portable GPU config, no hardcoded count, per-service device: Tasks 8, 9, 10.
- Error handling, per-stage failure, hard subprocess timeout, readiness 503: Tasks 5, 8, 9, 6.
- Packaging: cadrille-svc FROM cadrille:latest, sam3d MKL fix baked and verified, slim gateway, compose with healthchecks: Tasks 7, 8, 9, 10.
- Testing: unit, contract, integration smoke: Tasks 1 to 6 unit, 8 and 9 contract, 12 integration.
- Out of scope respected: no auth or TLS, no GUI rewire, no online pipeline. Confirmed across tasks.

**Placeholder scan:** No TBD, TODO, or vague error-handling steps. The two adapter scripts (run_cadrille.py, run_sam3d.py) have fixed CLI contracts defined in Tasks 8 and 9; they are not reproduced inline because they import RXL-only wrappers, and that constraint is stated in the Global Constraints and in each task note. This is an integration seam, not a placeholder: the calling code, request and response shapes, and the adapter CLI contract are all fully specified.

**Type consistency:** `ReconstructOptions`, `JobState`, `Stage`, `JobView` are defined in Task 1 and used unchanged in Tasks 2, 4, 5, 6. `ServiceError(stage, message)` is defined in Task 3 and consumed in Task 5. The service `/infer` request and response shapes in Tasks 8 and 9 match what the clients in Task 3 send and parse (`mesh_path` for sam3d; `cad_code_path, step_path, preview_path, metrics` for cadrille). The artifact path layout (`cad/model.py`, `cad/model.step`, `preview/model.stl`, `preview/render.png`, `mesh/sam3d_mesh.ply`, `metrics.json`) is identical in Tasks 4, 8, 9, and 12.
