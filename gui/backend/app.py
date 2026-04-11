from __future__ import annotations

import json
import shlex
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
JOBS_ROOT = WORKSPACE_ROOT / "outputs" / "gui-jobs"

DEFAULT_REMOTE_HOST = "RXL"
DEFAULT_REMOTE_WORKDIR = "/ssd1/rxl/zhankaiming/AIWS"
DEFAULT_REMOTE_PYTHON = "/home/rxl/anaconda3/envs/sam3d-objects/bin/python"
DEFAULT_REMOTE_SAM3D_OUTPUT_ROOT = "/ssd1/rxl/zhankaiming/AIWS/outputs/sam3d-aiws52-clean-mesh-stl-20260410-193527"
DEFAULT_REMOTE_DATASET_ROOT = "/ssd1/rxl/zhankaiming/AIWS/data/aiws5.2-usable-materialized"
DEFAULT_REMOTE_CADRILLE_ROOT = "/ssd1/rxl/zhankaiming/AIWS/repos/cadrille"
DEFAULT_REMOTE_CADRILLE_IMAGE = "cadrille:latest"


app = FastAPI(title="AIWS E2E GUI Backend", version="0.1.0")


class FullRunRequest(BaseModel):
    ssh_host: str = DEFAULT_REMOTE_HOST
    remote_workdir: str = DEFAULT_REMOTE_WORKDIR
    remote_python: str = DEFAULT_REMOTE_PYTHON
    sam3d_output_root: str = DEFAULT_REMOTE_SAM3D_OUTPUT_ROOT
    output_root: str = Field(..., min_length=1)
    split_prefix: str = "sam3d_bridge_gui"
    modalities: str = "pc,img"
    gpus: str = "0,1,2,3"
    pc_n_samples: int = 5
    img_n_samples: int = 1
    cadrille_batch_size: int = 64
    selection_mode: Literal["evaluate", "index"] = "evaluate"
    allow_selection_fallback: bool = False
    cadrille_runtime: Literal["auto", "docker", "host"] = "docker"
    cadrille_docker_image: str = DEFAULT_REMOTE_CADRILLE_IMAGE
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
    cadrille_split_name: str = "sam3d_bridge_gui_single"
    skip_sam3d: bool = True
    cadrille_runtime: Literal["auto", "docker", "host"] = "docker"
    cadrille_docker_image: str = DEFAULT_REMOTE_CADRILLE_IMAGE
    cadrille_docker_gpus: str = "device=0"
    cadrille_mode: Literal["pc", "img"] = "pc"
    cadrille_input_source: Literal["mesh", "point_cloud", "multi_view"] = "mesh"
    cadrille_n_samples: int = 5
    cadrille_batch_size: int = 64
    sample_offset: int = 0
    max_samples: Optional[int] = None
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


@app.on_event("startup")
def _startup() -> None:
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)


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
        },
    }


@app.get("/jobs", response_model=list[JobSummary])
def list_jobs() -> list[JobSummary]:
    jobs = [refresh_job(load_job(path)) for path in sorted(JOBS_ROOT.glob("*.json"), reverse=True)]
    return [JobSummary(**job) for job in jobs]


@app.get("/jobs/{job_id}", response_model=JobSummary)
def get_job(job_id: str) -> JobSummary:
    job = refresh_job(load_job(job_path(job_id)))
    return JobSummary(**job)


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


@app.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: str, tail_lines: int = Query(200, ge=1, le=2000)) -> dict[str, Any]:
    job = refresh_job(load_job(job_path(job_id)))
    log_text = read_remote_tail(job["ssh_host"], job["log_path"], tail_lines=tail_lines)
    return {
        "job_id": job_id,
        "status": job["status"],
        "tail_lines": tail_lines,
        "log": log_text,
    }


@app.post("/jobs/{job_id}/terminate")
def terminate_job(job_id: str) -> dict[str, Any]:
    job = load_job(job_path(job_id))
    pid = job.get("remote_pid")
    if pid is None:
        raise HTTPException(status_code=400, detail="Job has no remote pid")

    script = f"kill -TERM {int(pid)} 2>/dev/null || true"
    run_ssh_command(job["ssh_host"], script, check=False)
    job["status"] = "terminated"
    job["updated_at"] = now_ts()
    save_job(job)
    return {"ok": True, "job_id": job_id, "status": job["status"]}


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


def now_ts() -> float:
    return time.time()


def shell_join(parts: list[str]) -> str:
    return shlex.join([str(part) for part in parts])


def job_path(job_id: str) -> Path:
    return JOBS_ROOT / f"{job_id}.json"


def load_job(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Job not found: {path.stem}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_job(job: dict[str, Any]) -> None:
    job["updated_at"] = now_ts()
    job_path(job["job_id"]).write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")


def build_full_run_command(request: FullRunRequest) -> list[str]:
    cmd = [
        request.remote_python,
        f"{request.remote_workdir}/scripts/run_cadrille_full_modalities_4gpu.py",
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
        f"{request.remote_workdir}/scripts/run_sam3d_to_cadrille_e2e.py",
        "--sam3d-output-root",
        request.sam3d_output_root,
        "--dataset-root",
        request.dataset_root,
        "--cadrille-root",
        request.cadrille_root,
        "--cadrille-output-root",
        request.cadrille_output_root,
        "--cadrille-split-name",
        request.cadrille_split_name,
        "--cadrille-runtime",
        request.cadrille_runtime,
        "--cadrille-docker-image",
        request.cadrille_docker_image,
        "--cadrille-docker-gpus",
        request.cadrille_docker_gpus,
        "--cadrille-mode",
        request.cadrille_mode,
        "--cadrille-input-source",
        request.cadrille_input_source,
        "--cadrille-n-samples",
        str(request.cadrille_n_samples),
        "--cadrille-batch-size",
        str(request.cadrille_batch_size),
        "--selection-mode",
        request.selection_mode,
        "--selected-candidate-index",
        str(request.selected_candidate_index),
        "--sample-offset",
        str(request.sample_offset),
    ]
    if request.skip_sam3d:
        cmd.append("--skip-sam3d")
    if request.max_samples is not None:
        cmd.extend(["--max-samples", str(request.max_samples)])
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
    status_path = f"{output_root.rstrip('/')}/.gui_job_status.json"
    log_path = f"{output_root.rstrip('/')}/gui_job.log"

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
    script = f"""#!/usr/bin/env bash
set -euo pipefail
OUTPUT_ROOT={shlex.quote(output_root)}
STATUS_PATH={shlex.quote(status_path)}
LOG_PATH={shlex.quote(log_path)}
WORKDIR={shlex.quote(remote_workdir)}
CMD={shlex.quote(command_text)}
mkdir -p "$OUTPUT_ROOT"
python3 - "$STATUS_PATH" <<'PY'
import json
import pathlib
import sys
import time
pathlib.Path(sys.argv[1]).write_text(json.dumps({{"status": "running", "started_at": time.time()}}, indent=2))
PY
nohup bash -lc "cd \"$WORKDIR\" && $CMD; rc=$?; python3 - \"$STATUS_PATH\" \"$rc\" <<'PY'
import json
import pathlib
import sys
import time
rc = int(sys.argv[2])
payload = {{
    'status': 'completed' if rc == 0 else 'failed',
    'exit_code': rc,
    'ended_at': time.time(),
}}
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload, indent=2))
PY
exit $rc" > "$LOG_PATH" 2>&1 < /dev/null &
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


def run_ssh_script(ssh_host: str, script: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["ssh", ssh_host, "bash", "-s"],
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
    result = subprocess.run(
        ["ssh", ssh_host, command],
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


def read_remote_tail(ssh_host: str, log_path: str, tail_lines: int) -> str:
    script = f"if [ -f {shlex.quote(log_path)} ]; then tail -n {int(tail_lines)} {shlex.quote(log_path)}; fi"
    return run_ssh_command(ssh_host, script, check=False).stdout


def refresh_job(job: dict[str, Any]) -> dict[str, Any]:
    if job.get("status") in {"completed", "failed", "terminated"}:
        return job

    status_path = job.get("status_path")
    remote_pid = int(job.get("remote_pid") or 0)
    script = f"""#!/usr/bin/env bash
set -euo pipefail
if [ -f {shlex.quote(status_path)} ]; then
  cat {shlex.quote(status_path)}
  exit 0
fi
if kill -0 {remote_pid} 2>/dev/null; then
  printf '{{"status":"running"}}'
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
        if "exit_code" in remote_state:
            job["exit_code"] = remote_state["exit_code"]
        if "ended_at" in remote_state:
            job["ended_at"] = remote_state["ended_at"]
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
    result = subprocess.run(["ssh", ssh_host, "python3", "-"], input=source, text=True, capture_output=True)
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
