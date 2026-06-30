from __future__ import annotations

import io
import json
import os
import shlex
import shutil
import socket
import subprocess
import tempfile
import time
import base64
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import numpy as np
from pydantic import BaseModel, Field
import re
import trimesh
import yaml


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
JOBS_ROOT = WORKSPACE_ROOT / "outputs" / "gui-jobs"

DEFAULT_REMOTE_HOST = "RXL"
DEFAULT_REMOTE_WORKDIR = "/ssd1/rxl/zhankaiming/AIWS"
DEFAULT_REMOTE_PYTHON = "/ssd1/rxl/zhankaiming/envs/sam3d-objects/bin/python"
DEFAULT_REMOTE_SAM3D_OUTPUT_ROOT = "/ssd1/rxl/zhankaiming/AIWS/outputs/sam3d-aiws52-clean-mesh-stl-20260410-193527"
DEFAULT_REMOTE_DATASET_ROOT = "/ssd1/rxl/zhankaiming/AIWS/data/aiws5.2-usable"
DEFAULT_REMOTE_CADRILLE_ROOT = "/ssd1/rxl/zhankaiming/AIWS/repos/cadrille"
DEFAULT_REMOTE_CADRILLE_IMAGE = "cadrille:latest"
DEFAULT_CADRILLE_CHECKPOINT = "ckpt/cadrille_sft"
DEFAULT_CADRILLE_PROCESSOR_PATH = "ckpt/Qwen2-VL-2B-Instruct"
DEFAULT_CADRILLE_DOCKER_EXTRA_ARGS = "--ipc=host --shm-size=16g"

DEFAULT_CATALOG_PATH = f"{DEFAULT_REMOTE_WORKDIR}/docs/workpiece-dimensions.md"
DEFAULT_SIMPLE_REMOTE_ROOT = f"{DEFAULT_REMOTE_WORKDIR}/outputs/gui-simple"
DEFAULT_SIMPLE_CADRILLE_MODE = "pc"
DEFAULT_SIMPLE_CADRILLE_N_SAMPLES = 5
DEFAULT_SIMPLE_CADRILLE_BATCH_SIZE = 64
DEFAULT_SIMPLE_CADRILLE_RUNTIME: Literal["auto", "docker", "host"] = "docker"
DEFAULT_SIMPLE_CADRILLE_CHECKPOINT = "ckpt/cadrille_rl"
DEFAULT_SIMPLE_CADRILLE_CHECKPOINT_PRESET: Literal["SFT", "RL"] = "RL"
SIMPLE_CADRILLE_CHECKPOINT_PRESETS = {
    "SFT": "ckpt/cadrille_sft",
    "RL": "ckpt/cadrille_rl",
}
DEFAULT_SIMPLE_EXPORT_BREP = True
DEFAULT_SIMPLE_SELECTION_MODE: Literal["evaluate", "index"] = "evaluate"
DEFAULT_SIMPLE_SELECTED_CANDIDATE_INDEX = 0

GROUNDED_SAM_URL = os.environ.get("GROUNDED_SAM_URL", "http://127.0.0.1:18090")

ALLOWED_UPLOAD_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ALLOWED_MESH_UPLOAD_EXTS = {".stl", ".glb", ".obj", ".ply"}
LOCAL_HOST_ALIASES = {"local", "localhost", "127.0.0.1", "::1", socket.gethostname(), os.uname().nodename}


app = FastAPI(title="AIWS GUI Backend", version="0.2.0")


class FullRunRequest(BaseModel):
    ssh_host: str = DEFAULT_REMOTE_HOST
    remote_workdir: str = DEFAULT_REMOTE_WORKDIR
    remote_python: str = DEFAULT_REMOTE_PYTHON
    sam3d_output_root: str = DEFAULT_REMOTE_SAM3D_OUTPUT_ROOT
    output_root: str = Field(..., min_length=1)
    split_prefix: str = "sam3d_bridge_sft_gui"
    modalities: str = "pc,img"
    gpus: str = "0,1,2,3"
    pc_n_samples: int = 5
    img_n_samples: int = 1
    cadrille_batch_size: int = 64
    selection_mode: Literal["evaluate", "index"] = "evaluate"
    allow_selection_fallback: bool = False
    cadrille_runtime: Literal["auto", "docker", "host"] = "docker"
    cadrille_docker_image: str = DEFAULT_REMOTE_CADRILLE_IMAGE
    cadrille_docker_extra_args: str = DEFAULT_CADRILLE_DOCKER_EXTRA_ARGS
    cadrille_checkpoint: str = DEFAULT_CADRILLE_CHECKPOINT
    cadrille_processor_path: str = DEFAULT_CADRILLE_PROCESSOR_PATH
    export_brep: bool = True
    force: bool = False
    dry_run: bool = False


class E2ERunRequest(BaseModel):
    ssh_host: str = DEFAULT_REMOTE_HOST
    remote_workdir: str = DEFAULT_REMOTE_WORKDIR
    remote_python: str = DEFAULT_REMOTE_PYTHON
    sam3d_output_root: str = DEFAULT_REMOTE_SAM3D_OUTPUT_ROOT
    dataset_root: str = DEFAULT_REMOTE_DATASET_ROOT
    cadrille_root: str = DEFAULT_REMOTE_CADRILLE_ROOT
    cadrille_output_root: str = Field(..., min_length=1)
    bridge_split_name: str = "sam3d_bridge_sft_gui_single"
    skip_sam3d: bool = True
    cadrille_runtime: Literal["auto", "docker", "host"] = "docker"
    cadrille_docker_image: str = DEFAULT_REMOTE_CADRILLE_IMAGE
    cadrille_docker_extra_args: str = DEFAULT_CADRILLE_DOCKER_EXTRA_ARGS
    cadrille_docker_gpus: str = "device=2"
    cadrille_checkpoint: str = DEFAULT_CADRILLE_CHECKPOINT
    cadrille_processor_path: str = DEFAULT_CADRILLE_PROCESSOR_PATH
    cadrille_mode: Literal["pc", "img"] = "pc"
    cadrille_n_samples: int = 5
    cadrille_batch_size: int = 64
    limit: Optional[int] = None
    selection_mode: Literal["evaluate", "index"] = "evaluate"
    allow_selection_fallback: bool = False
    selected_candidate_index: int = 0
    export_brep: bool = True
    force: bool = False
    dry_run: bool = False


class JobSummary(BaseModel):
    job_id: str
    kind: str
    status: str
    ssh_host: str
    output_root: str
    created_at: float
    updated_at: float
    remote_pid: Optional[int] = None
    exit_code: Optional[int] = None
    command: list[str]
    log_path: str
    stage: Optional[str] = None
    stage_label: Optional[str] = None
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    stage_timings: Optional[dict[str, Any]] = None
    result_paths: Optional[dict[str, Any]] = None
    request: Optional[dict[str, Any]] = None
    error: Optional[str] = None


@app.on_event("startup")
def _startup() -> None:
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)


def _probe_gpus() -> list[dict[str, Any]]:
    """Best-effort GPU inventory via nvidia-smi. Empty list on any failure."""
    gpus: list[dict[str, Any]] = []
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4)
        if r.returncode == 0:
            for line in r.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 5:
                    continue
                try:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_total_mb": float(parts[2]),
                        "memory_free_mb": float(parts[3]),
                        "utilization": float(parts[4]),
                    })
                except ValueError:
                    continue
    except Exception:
        pass
    return gpus


def _pick_least_busy_gpu() -> int:
    """Index of the GPU with the most free memory (tie-broken by lower
    utilization). Falls back to 0 when nvidia-smi is unavailable."""
    gpus = _probe_gpus()
    if not gpus:
        return 0
    best = max(gpus, key=lambda g: (g["memory_free_mb"], -g["utilization"]))
    return int(best["index"])


def _probe_docker() -> dict[str, Any]:
    """Best-effort docker availability + cadrille image presence check."""
    out: dict[str, Any] = {"docker_ok": False, "docker_version": None,
                            "cadrille_image_present": False, "cadrille_image": DEFAULT_REMOTE_CADRILLE_IMAGE}
    try:
        v = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"],
                           capture_output=True, text=True, timeout=3)
        if v.returncode == 0:
            out["docker_ok"] = True
            out["docker_version"] = v.stdout.strip()
            i = subprocess.run(["docker", "image", "inspect", DEFAULT_REMOTE_CADRILLE_IMAGE],
                               capture_output=True, text=True, timeout=3)
            out["cadrille_image_present"] = (i.returncode == 0)
    except Exception:
        pass
    return out


def _probe_grounded_sam() -> bool:
    """Best-effort reachability check for grounded-sam-svc."""
    try:
        req = urllib.request.Request(
            f"{GROUNDED_SAM_URL}/health",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=2):
            return True
    except Exception:
        return False


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "workspace_root": str(WORKSPACE_ROOT),
        "jobs_root": str(JOBS_ROOT),
        "defaults": {
            "ssh_host": DEFAULT_REMOTE_HOST,
            "remote_workdir": DEFAULT_REMOTE_WORKDIR,
            "remote_python": DEFAULT_REMOTE_PYTHON,
            "sam3d_output_root": DEFAULT_REMOTE_SAM3D_OUTPUT_ROOT,
            "dataset_root": DEFAULT_REMOTE_DATASET_ROOT,
            "cadrille_root": DEFAULT_REMOTE_CADRILLE_ROOT,
            "cadrille_docker_image": DEFAULT_REMOTE_CADRILLE_IMAGE,
            "cadrille_docker_extra_args": DEFAULT_CADRILLE_DOCKER_EXTRA_ARGS,
            "cadrille_checkpoint": DEFAULT_CADRILLE_CHECKPOINT,
            "cadrille_processor_path": DEFAULT_CADRILLE_PROCESSOR_PATH,
        },
        "simple_defaults": {
            "ssh_host": "local",
            "remote_workdir": DEFAULT_REMOTE_WORKDIR,
            "remote_root": DEFAULT_SIMPLE_REMOTE_ROOT,
            "cadrille_mode": DEFAULT_SIMPLE_CADRILLE_MODE,
            "cadrille_checkpoint_preset": DEFAULT_SIMPLE_CADRILLE_CHECKPOINT_PRESET,
            "cadrille_checkpoint": DEFAULT_SIMPLE_CADRILLE_CHECKPOINT,
            "cadrille_n_samples": DEFAULT_SIMPLE_CADRILLE_N_SAMPLES,
            "cadrille_batch_size": DEFAULT_SIMPLE_CADRILLE_BATCH_SIZE,
        },
        "grounded_sam_url": GROUNDED_SAM_URL,
        "grounded_sam_reachable": _probe_grounded_sam(),
        "runtime": _probe_docker(),
        "gpus": _probe_gpus(),
        "catalog_path": DEFAULT_CATALOG_PATH,
    }


@app.get("/catalog")
def get_catalog() -> dict[str, Any]:
    """Return the workpiece-dimensions catalog parsed from docs/workpiece-dimensions.md.

    Frontend uses this so its workpiece/model dropdowns always reflect what the
    post-scaling pipeline actually accepts."""
    catalog_path = Path(DEFAULT_CATALOG_PATH)
    if not catalog_path.exists():
        raise HTTPException(status_code=500, detail=f"Catalog file not found: {catalog_path}")
    text = catalog_path.read_text(encoding="utf-8")
    m = re.search(r"```yaml\s*(.*?)```", text, re.DOTALL)
    if not m:
        raise HTTPException(status_code=500, detail="No ```yaml ... ``` block in catalog")
    try:
        data = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse catalog yaml: {exc}")

    classes: dict[str, dict[str, Any]] = {}
    for cls_name, models in data.items():
        if not isinstance(models, dict):
            continue
        entries: dict[str, dict[str, Any]] = {}
        for model_code, bbox_m in models.items():
            if isinstance(bbox_m, list) and len(bbox_m) == 3:
                bbox_m_floats = [float(v) for v in bbox_m]
                entries[str(model_code)] = {
                    "bbox_m": bbox_m_floats,
                    "bbox_mm": [round(v * 1000.0, 4) for v in bbox_m_floats],
                }
        classes[str(cls_name)] = {
            "models": list(entries.keys()),
            "entries": entries,
        }
    return {"source": str(catalog_path), "classes": classes}


@app.get("/jobs/{job_id}/file")
def download_job_file(job_id: str, path: str = Query(..., min_length=1)) -> FileResponse:
    """Serve a single file from inside a job's output root. Path must be under
    that root (defense against arbitrary path traversal)."""
    job = load_job(job_path(job_id))
    output_root = Path(job["output_root"]).resolve()
    target = Path(path).resolve()
    try:
        target.relative_to(output_root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Path is not under the job output root")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {target}")
    return FileResponse(target, filename=target.name)


@app.get("/jobs/{job_id}/inputs")
def get_job_inputs(job_id: str) -> dict[str, Any]:
    """Return absolute paths to the uploaded photo + mask for a job.

    Both files live under `<job_root>/input/` and are named `input.{ext}`
    and `mask.{ext}` where ext depends on what the user uploaded. We glob
    rather than relying on result_paths so this works for jobs created
    before result_paths started tracking inputs."""
    job = load_job(job_path(job_id))
    output_root = Path(job["output_root"]).resolve()
    input_dir = output_root / "input"
    image_path: str | None = None
    mask_path: str | None = None
    mesh_path: str | None = None
    if input_dir.is_dir():
        for p in sorted(input_dir.iterdir()):
            stem = p.stem.lower()
            if stem == "input" and p.is_file() and image_path is None:
                image_path = str(p)
            elif stem == "mask" and p.is_file() and mask_path is None:
                mask_path = str(p)
            elif stem == "mesh" and p.is_file() and mesh_path is None:
                mesh_path = str(p)
    return {"job_id": job_id, "input_image": image_path, "input_mask": mask_path, "input_mesh": mesh_path}


@app.get("/jobs", response_model=list[JobSummary])
def list_jobs() -> list[JobSummary]:
    # Resilient: one corrupt/empty job file (e.g. a status.json truncated by a
    # killed worker) must not 500 the entire history list — skip and log it.
    summaries: list[JobSummary] = []
    for path in sorted(JOBS_ROOT.glob("*.json"), reverse=True):
        try:
            summaries.append(JobSummary(**refresh_job(load_job(path))))
        except Exception as exc:  # noqa: BLE001
            print(f"[list_jobs] skipping unreadable job file {path.name}: {exc!r}", flush=True)
    return summaries


@app.get("/jobs/{job_id}", response_model=JobSummary)
def get_job(job_id: str) -> JobSummary:
    try:
        job = refresh_job(load_job(job_path(job_id)))
        return JobSummary(**job)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        # A corrupt/truncated tracking file must not 500 the endpoint; surface a
        # clean 422 so the UI shows "unreadable job" instead of a server error.
        print(f"[get_job] unreadable job {job_id}: {exc!r}", flush=True)
        raise HTTPException(status_code=422, detail=f"Job record unreadable: {job_id}")


@app.post("/jobs/full-run", response_model=JobSummary)
def create_full_run(request: FullRunRequest) -> JobSummary:
    command = build_full_run_command(request)
    job = create_ssh_job(
        kind="full_run",
        ssh_host=request.ssh_host,
        remote_workdir=request.remote_workdir,
        output_root=request.output_root,
        command=command,
        request_payload=request.model_dump(),
    )
    return JobSummary(**job)


@app.post("/jobs/e2e", response_model=JobSummary)
def create_e2e_run(request: E2ERunRequest) -> JobSummary:
    command = build_e2e_command(request)
    job = create_ssh_job(
        kind="single_e2e",
        ssh_host=request.ssh_host,
        remote_workdir=request.remote_workdir,
        output_root=request.cadrille_output_root,
        command=command,
        request_payload=request.model_dump(),
    )
    return JobSummary(**job)


@app.post("/jobs/simple-reconstruct", response_model=JobSummary)
async def create_simple_reconstruct(
    image: Optional[UploadFile] = File(None),
    mask: Optional[UploadFile] = File(None),
    mesh: Optional[UploadFile] = File(None),
    input_mode: Literal["image_mask", "image", "mesh"] = Form("image_mask"),
    detect_prompt: Optional[str] = Form(None),
    cadrille_checkpoint_preset: Literal["SFT", "RL"] = Form(DEFAULT_SIMPLE_CADRILLE_CHECKPOINT_PRESET),
    cadrille_mode: Literal["PC", "IMG"] = Form(DEFAULT_SIMPLE_CADRILLE_MODE.upper()),
    workpiece_class: Optional[str] = Form(None),
    model_code: Optional[str] = Form(None),
    gpu_index: Optional[int] = Form(None),
) -> JobSummary:
    image_name: str | None = None
    mask_name: str | None = None
    mesh_name: str | None = None
    image_ext: str | None = None
    mask_ext: str | None = None
    mesh_ext: str | None = None

    if input_mode == "mesh":
        if mesh is None:
            raise HTTPException(status_code=400, detail="Mesh upload is required for mesh input mode")
        mesh_name = sanitize_upload_name(mesh.filename or "mesh.stl")
        mesh_ext = Path(mesh_name).suffix.lower()
        if mesh_ext not in ALLOWED_MESH_UPLOAD_EXTS:
            raise HTTPException(status_code=400, detail=f"Unsupported mesh type: {mesh_ext}")
    elif input_mode == "image":
        if image is None:
            raise HTTPException(status_code=400, detail="Image upload is required for RGB-only auto-segment mode")
        image_name = sanitize_upload_name(image.filename or "input.png")
        image_ext = Path(image_name).suffix.lower()
        if image_ext not in ALLOWED_UPLOAD_EXTS:
            raise HTTPException(status_code=400, detail=f"Unsupported image type: {image_ext}")
    else:
        if image is None or mask is None:
            raise HTTPException(status_code=400, detail="Image and mask uploads are required for image/mask input mode")
        image_name = sanitize_upload_name(image.filename or "input.png")
        mask_name = sanitize_upload_name(mask.filename or "mask.png")
        image_ext = Path(image_name).suffix.lower()
        mask_ext = Path(mask_name).suffix.lower()
        if image_ext not in ALLOWED_UPLOAD_EXTS:
            raise HTTPException(status_code=400, detail=f"Unsupported image type: {image_ext}")
        if mask_ext not in ALLOWED_UPLOAD_EXTS:
            raise HTTPException(status_code=400, detail=f"Unsupported mask type: {mask_ext}")

    job_id = f"simple_reconstruct-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    local_job_root = JOBS_ROOT / job_id
    local_input_root = local_job_root / "input"
    local_input_root.mkdir(parents=True, exist_ok=True)

    ssh_host = "local"
    remote_workdir = DEFAULT_REMOTE_WORKDIR
    remote_job_root = f"{DEFAULT_SIMPLE_REMOTE_ROOT}/{job_id}"
    remote_input_root = f"{remote_job_root}/input"
    status_path = f"{remote_job_root}/status.json"
    log_path = f"{remote_job_root}/job.log"

    local_image_path: Path | None = None
    local_mask_path: Path | None = None
    local_mesh_path: Path | None = None
    remote_image_path: str | None = None
    remote_mask_path: str | None = None
    remote_mesh_path: str | None = None

    if input_mode == "mesh":
        local_mesh_path = local_input_root / f"mesh{mesh_ext}"
        local_mesh_path.write_bytes(await mesh.read())
        remote_mesh_path = f"{remote_input_root}/mesh{mesh_ext}"
    elif input_mode == "image":
        local_image_path = local_input_root / f"input{image_ext}"
        local_image_path.write_bytes(await image.read())
        remote_image_path = f"{remote_input_root}/input{image_ext}"
        # Auto-segment: call grounded-sam-svc to obtain a mask PNG.
        _img_bytes = local_image_path.read_bytes()
        _boundary = b"AIWS_GSAM_BOUNDARY"
        _parts: list[bytes] = []
        _parts.append(
            b"--" + _boundary + b"\r\n"
            b'Content-Disposition: form-data; name="image"; filename="input.png"\r\n'
            b"Content-Type: image/png\r\n\r\n"
            + _img_bytes + b"\r\n"
        )
        if detect_prompt:
            _parts.append(
                b"--" + _boundary + b"\r\n"
                b'Content-Disposition: form-data; name="prompt"\r\n\r\n'
                + detect_prompt.encode() + b"\r\n"
            )
        _parts.append(b"--" + _boundary + b"--\r\n")
        _seg_body = b"".join(_parts)
        _seg_req = urllib.request.Request(
            f"{GROUNDED_SAM_URL}/segment",
            data=_seg_body,
            headers={"Content-Type": f"multipart/form-data; boundary={_boundary.decode()}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(_seg_req, timeout=60) as _resp:
                _seg_data = json.loads(_resp.read())
        except urllib.error.HTTPError as _exc:
            if _exc.code == 422:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "No object detected by auto-segmentation. "
                        "Try a different detection prompt, or upload a mask manually."
                    ),
                ) from _exc
            raise HTTPException(
                status_code=502,
                detail=f"grounded-sam-svc error: HTTP {_exc.code}",
            ) from _exc
        except Exception as _exc:
            raise HTTPException(
                status_code=502,
                detail=f"grounded-sam-svc unreachable: {_exc}",
            ) from _exc
        _mask_b64 = _seg_data.get("mask_png_base64")
        if not _mask_b64:
            raise HTTPException(
                status_code=502,
                detail="grounded-sam-svc returned no mask_png_base64 field",
            )
        _mask_bytes = base64.b64decode(_mask_b64)
        local_mask_path = local_input_root / "mask.png"
        local_mask_path.write_bytes(_mask_bytes)
        mask_name = "mask.png"
        mask_ext = ".png"
        remote_mask_path = f"{remote_input_root}/mask.png"
        # Proceed as image_mask from here.
        input_mode = "image_mask"
    else:
        local_image_path = local_input_root / f"input{image_ext}"
        local_mask_path = local_input_root / f"mask{mask_ext}"
        local_image_path.write_bytes(await image.read())
        local_mask_path.write_bytes(await mask.read())
        remote_image_path = f"{remote_input_root}/input{image_ext}"
        remote_mask_path = f"{remote_input_root}/mask{mask_ext}"

    run_ssh_script(
        ssh_host,
        f"mkdir -p {shlex.quote(remote_job_root)} {shlex.quote(remote_input_root)}",
    )
    if input_mode == "mesh":
        upload_file_to_remote(ssh_host, local_mesh_path, remote_mesh_path)
    else:
        upload_file_to_remote(ssh_host, local_image_path, remote_image_path)
        upload_file_to_remote(ssh_host, local_mask_path, remote_mask_path)

    selected_checkpoint = SIMPLE_CADRILLE_CHECKPOINT_PRESETS[cadrille_checkpoint_preset]
    selected_mode = cadrille_mode.lower()
    # GPU selection: explicit pick, else least-busy auto. Drives both SAM3D
    # (CUDA_VISIBLE_DEVICES in the job runner) and the Cadrille docker --gpus.
    chosen_gpu = gpu_index if gpu_index is not None else _pick_least_busy_gpu()

    command = [
        DEFAULT_REMOTE_PYTHON,
        f"{DEFAULT_REMOTE_WORKDIR}/gui/backend/simple_reconstruct_job.py",
        "--repo-root",
        DEFAULT_REMOTE_WORKDIR,
        "--input-mode",
        input_mode,
    ]
    if input_mode == "mesh":
        command.extend(["--input-mesh", remote_mesh_path])
    else:
        command.extend(["--input-image", remote_image_path, "--input-mask", remote_mask_path])
    command.extend([
        "--job-root",
        remote_job_root,
        "--status-path",
        status_path,
        "--cadrille-runtime",
        DEFAULT_SIMPLE_CADRILLE_RUNTIME,
        "--cadrille-docker-image",
        DEFAULT_REMOTE_CADRILLE_IMAGE,
        "--cadrille-docker-extra-args",
        DEFAULT_CADRILLE_DOCKER_EXTRA_ARGS,
        "--cadrille-docker-gpus",
        f"device={chosen_gpu}",
        "--gpu-index",
        str(chosen_gpu),
        "--cadrille-checkpoint",
        selected_checkpoint,
        "--cadrille-processor-path",
        DEFAULT_CADRILLE_PROCESSOR_PATH,
        "--cadrille-mode",
        selected_mode,
        "--cadrille-n-samples",
        str(DEFAULT_SIMPLE_CADRILLE_N_SAMPLES),
        "--cadrille-batch-size",
        str(DEFAULT_SIMPLE_CADRILLE_BATCH_SIZE),
        "--selection-mode",
        DEFAULT_SIMPLE_SELECTION_MODE,
        "--selected-candidate-index",
        str(DEFAULT_SIMPLE_SELECTED_CANDIDATE_INDEX),
    ])
    command.append("--export-brep" if DEFAULT_SIMPLE_EXPORT_BREP else "--no-export-brep")
    # Optional post-scaling args. The job runner only runs the stage when a
    # workpiece_class is provided; model_code is required for non-h_beam.
    if workpiece_class:
        command.extend(["--workpiece-class", workpiece_class])
        if model_code:
            command.extend(["--model-code", model_code])

    remote_pid = launch_remote_job(
        ssh_host=ssh_host,
        remote_workdir=remote_workdir,
        output_root=remote_job_root,
        command=command,
        status_path=status_path,
        log_path=log_path,
    )

    job = {
        "job_id": job_id,
        "kind": "simple_reconstruct",
        "status": "running",
        "stage": "queued",
        "stage_label": "Queued",
        "ssh_host": ssh_host,
        "remote_workdir": remote_workdir,
        "output_root": remote_job_root,
        "command": command,
        "command_text": shell_join(command),
        "log_path": log_path,
        "status_path": status_path,
        "remote_pid": remote_pid,
        "created_at": now_ts(),
        "updated_at": now_ts(),
        "exit_code": None,
        "request": {
            "input_mode": input_mode,
            "image_filename": image_name,
            "mask_filename": mask_name,
            "detect_prompt": detect_prompt,
            "mesh_filename": mesh_name,
            "cadrille_checkpoint_preset": cadrille_checkpoint_preset,
            "cadrille_checkpoint": selected_checkpoint,
            "cadrille_mode": selected_mode,
            "cadrille_mode_label": selected_mode.upper(),
            "workpiece_class": workpiece_class,
            "model_code": model_code,
            "postscale_enabled": bool(workpiece_class),
            "gpu_index": chosen_gpu,
        },
    }
    save_job(job)
    return JobSummary(**job)


# ---------------------------------------------------------------------------
# Grounded-SAM session proxy endpoints
# Thin pass-through so the browser only talks to the GUI origin.
# ---------------------------------------------------------------------------

@app.post("/segment/session")
async def segment_session(image: UploadFile = File(...), prompt: Optional[str] = Form(None)):
    data = await image.read()
    boundary = b"AIWS_GSAM_REFINE_BOUNDARY"
    parts = [b"--" + boundary,
             b'Content-Disposition: form-data; name="image"; filename="upload.png"',
             b"Content-Type: application/octet-stream", b"", data]
    if prompt:
        parts += [b"--" + boundary,
                  b'Content-Disposition: form-data; name="prompt"', b"",
                  prompt.encode()]
    parts += [b"--" + boundary + b"--", b""]
    payload = b"\r\n".join(parts)
    req = urllib.request.Request(
        f"{GROUNDED_SAM_URL}/segment/session", data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return JSONResponse(status_code=resp.status, content=json.loads(resp.read()))
    except urllib.error.HTTPError as exc:
        raise HTTPException(status_code=exc.code, detail=exc.read().decode("utf-8", "replace"))
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=503, detail=f"grounded-sam-svc unreachable: {exc}")


@app.post("/segment/refine")
async def segment_refine(body: dict = Body(...)):
    req = urllib.request.Request(
        f"{GROUNDED_SAM_URL}/segment/refine", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return JSONResponse(status_code=resp.status, content=json.loads(resp.read()))
    except urllib.error.HTTPError as exc:
        # Propagate 409 so the frontend can re-create the session.
        raise HTTPException(status_code=exc.code, detail=exc.read().decode("utf-8", "replace"))
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=503, detail=f"grounded-sam-svc unreachable: {exc}")


@app.delete("/segment/session/{session_id}", status_code=204)
async def segment_release(session_id: str):
    req = urllib.request.Request(
        f"{GROUNDED_SAM_URL}/segment/session/{session_id}", method="DELETE")
    try:
        urllib.request.urlopen(req, timeout=30)
    except Exception:  # noqa: BLE001
        pass  # best-effort release; TTL will sweep otherwise
    return None


@app.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: str, tail_lines: int = Query(200, ge=0, le=200000)) -> dict[str, Any]:
    """Return the job log. `tail_lines=0` means full file (no `tail -n` cap).
    The cap was previously 2000 — too short for jobs that emit thousands of
    lines per stage, which made the GUI's log viewer drop earlier output."""
    job = refresh_job(load_job(job_path(job_id)))
    log_text = read_remote_tail(job["ssh_host"], job["log_path"], tail_lines=tail_lines)
    return {
        "job_id": job_id,
        "status": job["status"],
        "stage": job.get("stage"),
        "tail_lines": tail_lines,
        "log": log_text,
    }


@app.get("/jobs/{job_id}/metrics")
def get_job_metrics(job_id: str) -> dict[str, Any]:
    """Aggregate runtime metrics from the job's output files. Cheap, partial:
    any missing/unreadable section is reported as {"available": False}.
    Surfaces SAM3D runtime+GPU, Cadrille mean IoU / median CD / runtime / GPU,
    and a tiny postscale summary."""
    job = load_job(job_path(job_id))
    output_root = Path(job["output_root"]).resolve()

    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    # SAM3D meta.json (sam3d/GUI/user_upload/<stem>__obj01/meta.json).
    sam3d: dict[str, Any] = {"available": False}
    sam3d_dir = output_root / "sam3d"
    if sam3d_dir.is_dir():
        for meta_path in sorted(sam3d_dir.glob("**/meta.json")):
            data = _read_json(meta_path)
            if not data:
                continue
            sam3d = {
                "available": True,
                "duration_sec": data.get("duration_sec"),
                "model_init_sec": data.get("model_init_sec"),
                "peak_memory_reserved_mb": data.get("peak_memory_reserved_mb"),
                "peak_memory_allocated_mb": data.get("peak_memory_allocated_mb"),
                "cuda_visible_devices": data.get("cuda_visible_devices"),
            }
            break

    # Cadrille metrics + GPU memory.
    cadrille: dict[str, Any] = {"available": False}
    cadrille_dir = output_root / "cadrille"
    metrics_data = _read_json(cadrille_dir / "metrics.json") or {}
    gpu_data = _read_json(cadrille_dir / "gpu_memory.json") or {}
    if metrics_data or gpu_data:
        summary = metrics_data.get("summary") or {}
        devices = gpu_data.get("devices") or []
        device = devices[0] if devices else {}
        cadrille = {
            "available": True,
            "mean_iou": summary.get("mean_iou"),
            "median_cd": summary.get("median_cd"),
            "invalid_cd": summary.get("invalid_cd"),
            "invalid_iou": summary.get("invalid_iou"),
            "n_samples": gpu_data.get("n_samples"),
            "duration_sec": gpu_data.get("duration_sec"),
            "peak_memory_reserved_mb": device.get("peak_memory_reserved_mb"),
            "peak_memory_allocated_mb": device.get("peak_memory_allocated_mb"),
            "device_name": device.get("device_name"),
            "device_total_memory_mb": device.get("total_memory_mb"),
        }

    # Post-scaling summary (optional stage).
    postscale: dict[str, Any] = {"available": False}
    ps_dir = output_root / "postscale"
    ps_summary = _read_json(ps_dir / "_postscale_summary.json")
    if ps_summary:
        postscale = {"available": True}
        for key in ("count", "ok", "failed", "skipped", "workpiece_class", "model_code", "rewrite_mode"):
            if key in ps_summary:
                postscale[key] = ps_summary[key]

    # Per-sample postscale metadata holds the actual bbox numbers:
    # canonical_bbox (Cadrille's native units, "scale before scaling"),
    # catalog.bbox_mm (the catalog target dimensions selected for this
    # workpiece), and after_scale_bbox_mm (the actual dimensions after
    # the affine rewrite). Surface all three so the GUI can show them
    # as cards instead of burying them in a details expander.
    if ps_dir.is_dir():
        for meta_path in sorted(ps_dir.glob("*__metadata.json")):
            if meta_path.name.startswith("_"):
                continue
            data = _read_json(meta_path)
            if not data:
                continue
            cat = data.get("catalog") or {}
            canonical = data.get("canonical_bbox") or {}
            after = data.get("after_scale_bbox_mm") or {}
            scale = data.get("scale") or {}
            target_bbox = cat.get("bbox_mm") or [None, None, None]
            actual = [after.get("xlen"), after.get("ylen"), after.get("zlen")]
            canonical_extents = [canonical.get("xlen"), canonical.get("ylen"), canonical.get("zlen")]

            # Per-axis match. Tolerance matches the existing expander
            # (1e-3 relative error); any axis missing data → match=False.
            max_rel: float | None = None
            match_ok: bool | None = None
            if any(a is not None for a in actual) and any(t is not None for t in target_bbox):
                match_ok = True
                max_rel = 0.0
                for t, a in zip(target_bbox, actual):
                    if t is None or a is None or t == 0:
                        match_ok = False
                        continue
                    rel = abs(a - t) / t
                    if rel > max_rel:
                        max_rel = rel
                    if rel >= 1e-3:
                        match_ok = False

            postscale.update({
                "available": True,
                "workpiece_class": cat.get("workpiece_class") or postscale.get("workpiece_class"),
                "model_code": cat.get("model_code") or postscale.get("model_code"),
                "rewrite_mode": data.get("rewrite_mode") or scale.get("mode") or postscale.get("rewrite_mode"),
                "canonical_extents": canonical_extents,
                "catalog_target_mm": target_bbox,
                "after_scale_mm": actual,
                "max_rel_error": max_rel,
                "match_ok": match_ok,
            })
            break

    return {"job_id": job_id, "sam3d": sam3d, "cadrille": cadrille, "postscale": postscale}


@app.post("/jobs/{job_id}/terminate")
def terminate_job(job_id: str) -> dict[str, Any]:
    job = load_job(job_path(job_id))
    pid = job.get("remote_pid")
    if pid is None:
        raise HTTPException(status_code=400, detail="Job has no remote pid")

    # Cancel must tear the whole job down, not just the bash wrapper remote_pid
    # points at. (1) Kill the runner's process group (created via setsid at
    # launch) so the python child and its docker-CLI clients die too; fall back
    # to the bare pid for jobs launched before the setsid change. (2) Kill any
    # docker container bind-mounting this job's dir -- containers run under
    # dockerd, outside the process group, so a process kill alone leaves them
    # (and the GPU) running. (3) Stamp the live status.json terminated so
    # refresh_job stops reverting it to "running" on the next poll.
    teardown = (
        f"pid={int(pid)}\n"
        f"root={shlex.quote(job.get('output_root') or '')}\n"
        f"status={shlex.quote(job.get('status_path') or '')}\n"
        + r'''
kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
sleep 2
kill -KILL -"$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
if [ -n "$root" ]; then
  for cid in $(docker ps -q 2>/dev/null); do
    while IFS= read -r src; do
      case "$src" in
        "$root"|"$root"/*) docker kill "$cid" 2>/dev/null || true; break;;
      esac
    done < <(docker inspect -f '{{range .Mounts}}{{println .Source}}{{end}}' "$cid" 2>/dev/null)
  done
fi
if [ -n "$status" ]; then
  python3 - "$status" <<'PY' 2>/dev/null || true
import json, pathlib, sys, time
p = pathlib.Path(sys.argv[1])
d = {}
if p.exists():
    try:
        d = json.loads(p.read_text())
    except Exception:
        d = {}
d.update({"status": "terminated", "ended_at": time.time()})
p.write_text(json.dumps(d, indent=2))
PY
fi
'''
    )
    run_ssh_script(job["ssh_host"], teardown, check=False)
    job["status"] = "terminated"
    job["updated_at"] = now_ts()
    save_job(job)
    return {"ok": True, "job_id": job_id, "status": job["status"]}


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str) -> dict[str, Any]:
    """Delete a reconstruction from history: remove its output dir and the
    job-metadata file. Best-effort on the output dir (an old Docker-written
    dir may contain root-owned files we can't unlink); the metadata is always
    removed so the job leaves the history list regardless."""
    jpath = job_path(job_id)
    if not jpath.exists():
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    job = load_job(jpath)

    output_root = job.get("output_root")
    removed_dir = False
    dir_error: str | None = None
    if output_root:
        try:
            root = Path(output_root).resolve()
            # Safety: only delete dirs that live under the known job output roots.
            allowed = (str(JOBS_ROOT.resolve()),
                       f"{DEFAULT_REMOTE_WORKDIR}/outputs/gui-simple",
                       f"{DEFAULT_REMOTE_WORKDIR}/outputs/gui-jobs")
            if is_under_allowed_root(root, allowed):
                ssh_host = job.get("ssh_host", "local")
                if ssh_host in (None, "", "local"):
                    if root.is_dir():
                        shutil.rmtree(root, ignore_errors=True)
                        removed_dir = not root.exists()
                else:
                    rm = run_ssh_command(ssh_host, f"rm -rf {shlex.quote(str(root))}", check=False)
                    removed_dir = rm.returncode == 0
            else:
                dir_error = f"output_root outside allowed roots: {root}"
        except Exception as exc:  # noqa: BLE001
            dir_error = f"{type(exc).__name__}: {exc}"

    try:
        jpath.unlink()
    except FileNotFoundError:
        pass

    return {"ok": True, "job_id": job_id, "removed_dir": removed_dir, "dir_error": dir_error}


@app.get("/jobs/{job_id}/summary")
def get_job_summary(job_id: str) -> dict[str, Any]:
    job = refresh_job(load_job(job_path(job_id)))
    return summarize_output_root(job["ssh_host"], job["output_root"])


@app.get("/outputs/summary")
def get_output_summary(
    ssh_host: str = Query(DEFAULT_REMOTE_HOST),
    root: str = Query(..., min_length=1),
) -> dict[str, Any]:
    return summarize_output_root(ssh_host, root)


@app.get("/preview/files")
def get_preview_files(
    ssh_host: str = Query(DEFAULT_REMOTE_HOST),
    directory: str = Query(..., min_length=1),
    pattern: str = Query("*.stl", min_length=1),
) -> dict[str, Any]:
    return {
        "ssh_host": ssh_host,
        "directory": directory,
        "pattern": pattern,
        "files": list_remote_files(ssh_host, directory, pattern),
    }


@app.get("/preview/mesh")
def get_preview_mesh(
    ssh_host: str = Query(DEFAULT_REMOTE_HOST),
    path: str = Query(..., min_length=1),
    max_faces: int = Query(15000, ge=100, le=100000),
) -> dict[str, Any]:
    mesh_bytes = read_remote_file_bytes(ssh_host, path)
    mesh = load_mesh_from_bytes(mesh_bytes, path)
    return mesh_to_payload(mesh, path=path, max_faces=max_faces)


def now_ts() -> float:
    return time.time()


def shell_join(parts: list[str]) -> str:
    return shlex.join([str(part) for part in parts])


def sanitize_upload_name(name: str) -> str:
    clean = Path(name).name.strip().replace(" ", "_")
    return clean or "upload.bin"


JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_job_id(job_id: str) -> str:
    """Reject any job_id that is not a plain identifier.

    Every ``/jobs/{job_id}/*`` route maps job_id straight onto the filesystem
    via ``job_path``; this guard keeps that mapping from depending on router
    quirks (e.g. ``..`` segments or a leading ``-``) for its safety.
    """
    if not JOB_ID_RE.fullmatch(job_id or ""):
        raise HTTPException(status_code=400, detail=f"Invalid job id: {job_id!r}")
    return job_id


def job_path(job_id: str) -> Path:
    return JOBS_ROOT / f"{validate_job_id(job_id)}.json"


def is_under_allowed_root(root: Path, allowed_roots: tuple[str, ...]) -> bool:
    """True only when ``root`` is one of, or nested inside, an allowed root.

    Uses path-component containment (``os.path.commonpath``) rather than a bare
    string prefix, so a sibling like ``.../outputs/gui-simple-backup`` does NOT
    match the ``.../outputs/gui-simple`` root.
    """
    root_str = str(root)
    for allowed in allowed_roots:
        try:
            if os.path.commonpath([root_str, allowed]) == allowed:
                return True
        except ValueError:
            # Different drives / one relative and one absolute: not contained.
            continue
    return False


def load_job(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Job not found: {path.stem}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_job(job: dict[str, Any]) -> None:
    job["updated_at"] = now_ts()
    p = job_path(job["job_id"])
    # Atomic write: serialize to a unique temp file in the same dir, then
    # os.replace() (atomic on one filesystem). Prevents the 0-byte truncation and
    # "extra data" corruption a bare write_text() suffers under concurrent
    # refreshes or a full disk.
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f"{p.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(job, indent=2, ensure_ascii=False))
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def build_full_run_command(request: FullRunRequest) -> list[str]:
    cmd = [
        request.remote_python,
        f"{request.remote_workdir}/scripts/cadrille_batch.py",
        "--sam3d-output-root",
        request.sam3d_output_root,
        "--output-root",
        request.output_root,
        "--split-prefix",
        request.split_prefix,
        "--modalities",
        request.modalities,
        "--gpus",
        request.gpus,
        "--pc-n-samples",
        str(request.pc_n_samples),
        "--img-n-samples",
        str(request.img_n_samples),
        "--cadrille-batch-size",
        str(request.cadrille_batch_size),
        "--selection-mode",
        request.selection_mode,
        "--cadrille-runtime",
        request.cadrille_runtime,
        "--cadrille-docker-image",
        request.cadrille_docker_image,
        f"--cadrille-docker-extra-args={request.cadrille_docker_extra_args}",
        "--cadrille-checkpoint",
        request.cadrille_checkpoint,
        "--cadrille-processor-path",
        request.cadrille_processor_path,
    ]
    if request.allow_selection_fallback:
        cmd.append("--allow-selection-fallback")
    cmd.append("--export-brep" if request.export_brep else "--no-export-brep")
    if request.force:
        cmd.append("--force")
    if request.dry_run:
        cmd.append("--dry-run")
    return cmd


def build_e2e_command(request: E2ERunRequest) -> list[str]:
    cmd = [
        request.remote_python,
        f"{request.remote_workdir}/scripts/e2e_sam3d_to_cadrille.py",
        "--sam3d-output-root",
        request.sam3d_output_root,
        "--dataset-root",
        request.dataset_root,
        "--cadrille-root",
        request.cadrille_root,
        "--cadrille-output-root",
        request.cadrille_output_root,
        "--bridge-split-name",
        request.bridge_split_name,
        "--cadrille-runtime",
        request.cadrille_runtime,
        "--cadrille-docker-image",
        request.cadrille_docker_image,
        f"--cadrille-docker-extra-args={request.cadrille_docker_extra_args}",
        "--cadrille-docker-gpus",
        request.cadrille_docker_gpus,
        "--cadrille-checkpoint",
        request.cadrille_checkpoint,
        "--cadrille-processor-path",
        request.cadrille_processor_path,
        "--cadrille-mode",
        request.cadrille_mode,
        "--cadrille-n-samples",
        str(request.cadrille_n_samples),
        "--cadrille-batch-size",
        str(request.cadrille_batch_size),
        "--selection-mode",
        request.selection_mode,
        "--selected-candidate-index",
        str(request.selected_candidate_index),
    ]
    if request.skip_sam3d:
        cmd.append("--skip-sam3d")
    if request.limit is not None:
        cmd.extend(["--limit", str(request.limit)])
    if request.allow_selection_fallback:
        cmd.append("--allow-selection-fallback")
    cmd.append("--export-brep" if request.export_brep else "--no-export-brep")
    if request.force:
        cmd.append("--force")
    if request.dry_run:
        cmd.append("--dry-run")
    return cmd


def create_ssh_job(
    *,
    kind: str,
    ssh_host: str,
    remote_workdir: str,
    output_root: str,
    command: list[str],
    request_payload: dict[str, Any],
) -> dict[str, Any]:
    job_id = f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    remote_meta_root = f"{remote_workdir.rstrip('/')}/outputs/gui-jobs/{job_id}"
    status_path = f"{remote_meta_root}/status.json"
    log_path = f"{remote_meta_root}/job.log"

    remote_pid = launch_remote_job(
        ssh_host=ssh_host,
        remote_workdir=remote_workdir,
        output_root=output_root,
        command=command,
        status_path=status_path,
        log_path=log_path,
    )

    job = {
        "job_id": job_id,
        "kind": kind,
        "status": "running",
        "ssh_host": ssh_host,
        "remote_workdir": remote_workdir,
        "output_root": output_root,
        "command": command,
        "command_text": shell_join(command),
        "log_path": log_path,
        "status_path": status_path,
        "remote_pid": remote_pid,
        "created_at": now_ts(),
        "updated_at": now_ts(),
        "exit_code": None,
        "request": request_payload,
    }
    save_job(job)
    return job


def launch_remote_job(
    *,
    ssh_host: str,
    remote_workdir: str,
    output_root: str,
    command: list[str],
    status_path: str,
    log_path: str,
) -> int:
    command_text = shell_join(command)
    runner_path = f"{Path(status_path).parent.as_posix()}/runner.sh"
    script = f"""#!/usr/bin/env bash
set -euo pipefail
OUTPUT_ROOT={shlex.quote(output_root)}
STATUS_PATH={shlex.quote(status_path)}
LOG_PATH={shlex.quote(log_path)}
RUNNER_PATH={shlex.quote(runner_path)}
mkdir -p "$(dirname \"$STATUS_PATH\")" "$(dirname \"$LOG_PATH\")"
python3 - "$STATUS_PATH" <<'PY'
import json
import pathlib
import sys
import time
pathlib.Path(sys.argv[1]).write_text(json.dumps({{"status": "running", "stage": "queued", "stage_label": "Queued", "started_at": time.time()}}, indent=2))
PY
cat > "$RUNNER_PATH" <<'BASH'
#!/usr/bin/env bash
set -uo pipefail
cd {shlex.quote(remote_workdir)}
unset LD_PRELOAD  # strip base-conda MKL preload that breaks SAM3D MoGe FFT
{command_text}
rc=$?
python3 - {shlex.quote(status_path)} "$rc" <<'PY'
import json
import pathlib
import sys
import time
path = pathlib.Path(sys.argv[1])
rc = int(sys.argv[2])
payload = {{}}
if path.exists():
    try:
        payload = json.loads(path.read_text())
    except Exception:
        payload = {{}}
payload.update({{
    'status': 'completed' if rc == 0 else 'failed',
    'exit_code': rc,
    'ended_at': time.time(),
}})
path.write_text(json.dumps(payload, indent=2))
PY
exit "$rc"
BASH
chmod +x "$RUNNER_PATH"
# setsid (not nohup) so the runner leads its OWN session/process group: $! is
# both its pid and pgid, letting terminate_job signal the whole tree (python
# child + docker-CLI clients) with `kill -- -<pgid>`, not just the bash wrapper.
setsid bash "$RUNNER_PATH" > "$LOG_PATH" 2>&1 < /dev/null &
echo $!
"""
    result = run_ssh_script(ssh_host, script)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise HTTPException(status_code=500, detail="Failed to capture remote pid")
    try:
        return int(lines[-1])
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid remote pid output: {lines[-1]}") from exc


def is_local_host(host: str) -> bool:
    return (host or "").strip() in LOCAL_HOST_ALIASES


def ssh_argv(ssh_host: str, *args: str) -> list[str]:
    """Build an ``ssh`` argv with a ``--`` separator before the host.

    Without ``--`` OpenSSH parses any token starting with ``-`` as an option,
    so a host value like ``-oProxyCommand=...`` would be executed as an ssh
    option (argument injection). The ``--`` makes such a value be treated as a
    hostname (and rejected), while leaving every legitimate host unchanged.
    """
    return ["ssh", "--", ssh_host, *args]


def scp_argv(*args: str) -> list[str]:
    """Build an ``scp`` argv with a ``--`` separator before its operands."""
    return ["scp", "--", *args]


def run_local_script(script: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(["bash", "-s"], input=script, text=True, capture_output=True)
    if check and result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Local script failed",
                "returncode": result.returncode,
                "stderr": result.stderr,
                "stdout": result.stdout,
            },
        )
    return result


def run_ssh_script(ssh_host: str, script: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    if is_local_host(ssh_host):
        return run_local_script(script, check=check)
    result = subprocess.run(
        ssh_argv(ssh_host, "bash", "-s"),
        input=script,
        text=True,
        capture_output=True,
    )
    if check and result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "SSH script failed",
                "ssh_host": ssh_host,
                "returncode": result.returncode,
                "stderr": result.stderr,
                "stdout": result.stdout,
            },
        )
    return result


def run_ssh_command(ssh_host: str, command: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    if is_local_host(ssh_host):
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
    else:
        result = subprocess.run(
            ssh_argv(ssh_host, command),
            text=True,
            capture_output=True,
        )
    if check and result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "SSH command failed",
                "ssh_host": ssh_host,
                "returncode": result.returncode,
                "stderr": result.stderr,
                "stdout": result.stdout,
            },
        )
    return result


def upload_file_to_remote(ssh_host: str, local_path: Path, remote_path: str) -> None:
    if is_local_host(ssh_host):
        remote = Path(remote_path)
        remote.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, remote)
        return
    result = subprocess.run(
        scp_argv(str(local_path), f"{ssh_host}:{remote_path}"),
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to upload file",
                "ssh_host": ssh_host,
                "local_path": str(local_path),
                "remote_path": remote_path,
                "stderr": result.stderr,
                "stdout": result.stdout,
            },
        )


def read_remote_file_bytes(ssh_host: str, path: str) -> bytes:
    if is_local_host(ssh_host):
        try:
            return Path(path).read_bytes()
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail={"message": "Failed to read local file", "path": path, "error": str(exc)},
            ) from exc
    result = subprocess.run(
        ssh_argv(ssh_host, f"cat {shlex.quote(path)}"),
        capture_output=True,
    )
    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to read remote file",
                "ssh_host": ssh_host,
                "path": path,
                "returncode": result.returncode,
                "stderr": result.stderr.decode("utf-8", errors="ignore"),
            },
        )
    return result.stdout


def list_remote_files(ssh_host: str, directory: str, pattern: str) -> list[str]:
    source = f"""
import fnmatch
import json
import os

directory = {json.dumps(directory)}
pattern = {json.dumps(pattern)}
if not os.path.isdir(directory):
    print(json.dumps([]))
else:
    files = [name for name in sorted(os.listdir(directory)) if fnmatch.fnmatch(name, pattern)]
    print(json.dumps(files))
"""
    if is_local_host(ssh_host):
        result = subprocess.run(["python3", "-"], input=source, text=True, capture_output=True)
    else:
        result = subprocess.run(ssh_argv(ssh_host, "python3", "-"), input=source, text=True, capture_output=True)
    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to list remote files",
                "ssh_host": ssh_host,
                "directory": directory,
                "stderr": result.stderr,
            },
        )
    return json.loads(result.stdout or "[]")


def load_mesh_from_bytes(data: bytes, path: str) -> trimesh.Trimesh:
    file_type = Path(path).suffix.lower().lstrip(".") or "stl"
    loaded = trimesh.load(io.BytesIO(data), file_type=file_type, force="mesh")

    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded

    if isinstance(mesh, list):
        mesh = trimesh.util.concatenate([part for part in mesh if isinstance(part, trimesh.Trimesh)])

    if not isinstance(mesh, trimesh.Trimesh) or mesh.vertices is None or mesh.faces is None:
        raise HTTPException(status_code=400, detail=f"Unsupported mesh payload for preview: {path}")
    if len(mesh.faces) == 0:
        raise HTTPException(status_code=400, detail=f"Mesh has no faces: {path}")
    return mesh


def simplify_mesh_for_preview(mesh: trimesh.Trimesh, *, max_faces: int) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if len(faces) <= max_faces:
        return vertices, faces

    try:
        simplified = mesh.simplify_quadric_decimation(max_faces)
        simple_vertices = np.asarray(simplified.vertices)
        simple_faces = np.asarray(simplified.faces)
        if len(simple_faces) > 0:
            return simple_vertices, simple_faces
    except Exception:
        pass

    bounds = mesh.bounds
    if bounds is None:
        return vertices, faces[:max_faces]

    extents = np.maximum(bounds[1] - bounds[0], 1e-6)
    target_vertices = max(int(max_faces * 0.6), 1000)
    bins_per_axis = max(int(round(target_vertices ** (1.0 / 3.0))), 8)
    pitch = extents / float(bins_per_axis)
    quantized = np.floor((vertices - bounds[0]) / pitch).astype(np.int64)
    _, inverse = np.unique(quantized, axis=0, return_inverse=True)

    compact_vertices = np.zeros((inverse.max() + 1, 3), dtype=np.float64)
    counts = np.bincount(inverse)
    for axis in range(3):
        compact_vertices[:, axis] = np.bincount(inverse, weights=vertices[:, axis]) / counts

    compact_faces = inverse[faces]
    nondegenerate = (
        (compact_faces[:, 0] != compact_faces[:, 1])
        & (compact_faces[:, 0] != compact_faces[:, 2])
        & (compact_faces[:, 1] != compact_faces[:, 2])
    )
    compact_faces = compact_faces[nondegenerate]
    if len(compact_faces) == 0:
        return vertices, faces[:max_faces]

    compact_faces = np.unique(np.sort(compact_faces, axis=1), axis=0)
    if len(compact_faces) > max_faces:
        step = max(int(np.ceil(len(compact_faces) / max_faces)), 1)
        compact_faces = compact_faces[::step][:max_faces]

    used_vertices, remapped = np.unique(compact_faces.reshape(-1), return_inverse=True)
    compact_vertices = compact_vertices[used_vertices]
    compact_faces = remapped.reshape(-1, 3)
    return compact_vertices, compact_faces


def mesh_to_payload(mesh: trimesh.Trimesh, *, path: str, max_faces: int) -> dict[str, Any]:
    original_face_count = int(len(mesh.faces))
    vertices, faces = simplify_mesh_for_preview(mesh, max_faces=max_faces)

    return {
        "path": path,
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "original_face_count": original_face_count,
        "bounds": mesh.bounds.tolist() if mesh.bounds is not None else None,
        "extents": mesh.extents.tolist() if mesh.extents is not None else None,
        "vertices": vertices.astype(float).tolist(),
        "faces": faces.astype(int).tolist(),
    }


def read_remote_tail(ssh_host: str, log_path: str, tail_lines: int) -> str:
    """Return either the last N lines of the log (when `tail_lines > 0`) or
    the full file (when `tail_lines == 0`). The GUI uses `tail_lines=0` so
    the in-browser scrollable log container retains every line."""
    quoted = shlex.quote(log_path)
    if int(tail_lines) <= 0:
        script = f"if [ -f {quoted} ]; then cat {quoted}; fi"
    else:
        script = f"if [ -f {quoted} ]; then tail -n {int(tail_lines)} {quoted}; fi"
    return run_ssh_command(ssh_host, script, check=False).stdout


def refresh_job(job: dict[str, Any]) -> dict[str, Any]:
    status_path = job.get("status_path")
    remote_pid = int(job.get("remote_pid") or 0)
    script = f"""#!/usr/bin/env bash
set -euo pipefail
if [ -f {shlex.quote(status_path)} ]; then
  cat {shlex.quote(status_path)}
  exit 0
fi
if kill -0 {remote_pid} 2>/dev/null; then
  printf '{{"status":"running","stage":"queued","stage_label":"Queued"}}'
else
  printf '{{"status":"unknown"}}'
fi
"""
    result = run_ssh_script(job["ssh_host"], script, check=False)
    if result.returncode == 0 and result.stdout.strip():
        try:
            remote_state = json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            remote_state = {"status": "unknown"}
        job["status"] = remote_state.get("status", job["status"])
        for key in ("exit_code", "ended_at", "started_at", "updated_at", "stage", "stage_label", "stage_timings", "result_paths", "error"):
            if key in remote_state:
                job[key] = remote_state[key]
        save_job(job)
    return job


def summarize_output_root(ssh_host: str, root: str) -> dict[str, Any]:
    source = f"""
import glob
import json
import os

root = {json.dumps(root)}
result = {{"root": root, "modalities": {{}}}}
for mode in ("pc", "img"):
    mode_root = os.path.join(root, mode)
    if not os.path.isdir(mode_root):
        result["modalities"][mode] = {{"exists": False, "total_shards": 0, "done_shards": 0, "shards": []}}
        continue
    shards = []
    for shard in sorted(glob.glob(os.path.join(mode_root, "shard-*"))):
        summary_path = os.path.join(shard, "pipeline_summary.json")
        item = {{
            "name": os.path.basename(shard),
            "pipeline_summary_exists": os.path.exists(summary_path),
            "tmp_py_count": len(glob.glob(os.path.join(shard, "tmp_py", "*.py"))),
            "tmp_mesh_count": len(glob.glob(os.path.join(shard, "tmp_mesh", "*.stl"))),
            "tmp_brep_count": len(glob.glob(os.path.join(shard, "tmp_brep", "*.step"))),
            "selected_py_count": len(glob.glob(os.path.join(shard, "selected_py", "*.py"))),
            "selected_mesh_count": len(glob.glob(os.path.join(shard, "selected_mesh", "*.stl"))),
            "selected_brep_count": len(glob.glob(os.path.join(shard, "selected_brep", "*.step"))),
        }}
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            selection = data.get("selection", {{}})
            evaluate = selection.get("evaluate", {{}})
            item["metrics_summary"] = evaluate.get("summary")
            item["best_candidate_count"] = selection.get("best_candidate_count")
            item["selected_rows_count"] = len(data.get("selected_rows", []))
        shards.append(item)
    result["modalities"][mode] = {{
        "exists": True,
        "total_shards": len(shards),
        "done_shards": sum(1 for shard in shards if shard["pipeline_summary_exists"]),
        "shards": shards,
    }}
print(json.dumps(result))
"""
    if is_local_host(ssh_host):
        result = subprocess.run(["python3", "-"], input=source, text=True, capture_output=True)
    else:
        result = subprocess.run(ssh_argv(ssh_host, "python3", "-"), input=source, text=True, capture_output=True)
    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to summarize remote output root",
                "ssh_host": ssh_host,
                "root": root,
                "stderr": result.stderr,
                "stdout": result.stdout,
            },
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid summary JSON: {result.stdout[:500]}") from exc


# --- serve the built React SPA (same-origin; no CORS). Guarded so dev still runs without a build. ---
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
from starlette.responses import Response as _Response  # noqa: E402


class _SPAStaticFiles(StaticFiles):
    """StaticFiles that marks index.html as non-cacheable so a redeploy is
    picked up immediately, while letting content-hashed assets cache normally."""

    async def get_response(self, path: str, scope):  # type: ignore[override]
        response = await super().get_response(path, scope)
        if path in ("", ".", "index.html") or path.endswith("/index.html"):
            if isinstance(response, _Response):
                response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response


_FRONTEND_DIST = _Path(__file__).resolve().parents[1] / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/", _SPAStaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
