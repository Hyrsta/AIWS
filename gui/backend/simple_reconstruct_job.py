#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one user-facing GUI reconstruction job: image + mask -> SAM3D -> Cadrille."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--input-mask", type=Path, required=True)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)

    parser.add_argument("--sam3d-python", default=sys.executable)
    parser.add_argument("--sam3d-repo-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize-stl", dest="normalize_stl", action="store_true", default=True)
    parser.add_argument("--no-normalize-stl", dest="normalize_stl", action="store_false")

    parser.add_argument("--cadrille-python", default=sys.executable)
    parser.add_argument("--cadrille-runtime", choices=("auto", "docker", "host"), default="docker")
    parser.add_argument("--cadrille-docker-image", default="cadrille:latest")
    parser.add_argument("--cadrille-docker-python", default="python")
    parser.add_argument("--cadrille-docker-gpus", default="device=0")
    parser.add_argument("--cadrille-docker-extra-args", default="--ipc=host --shm-size=16g")
    parser.add_argument("--cadrille-root", type=Path, default=None)
    parser.add_argument("--cadrille-checkpoint", default="ckpt/cadrille_rl")
    parser.add_argument("--cadrille-processor-path", default="ckpt/Qwen2-VL-2B-Instruct")
    parser.add_argument("--cadrille-mode", choices=("pc", "img"), default="pc")
    parser.add_argument("--cadrille-n-samples", type=int, default=5)
    parser.add_argument("--cadrille-batch-size", type=int, default=64)
    parser.add_argument("--selection-mode", choices=("evaluate", "index"), default="evaluate")
    parser.add_argument("--selected-candidate-index", type=int, default=0)
    parser.add_argument("--allow-selection-fallback", action="store_true")
    parser.add_argument("--export-brep", dest="export_brep", action="store_true", default=True)
    parser.add_argument("--no-export-brep", dest="export_brep", action="store_false")
    parser.add_argument("--brep-ext", default="step")
    parser.add_argument("--convert-timeout-sec", type=float, default=5.0)

    # Post-scaling stage (optional). When --workpiece-class is set, the
    # Cadrille selected_py is fed through scripts/cadrille_metric_postscale.py
    # against the catalog target dims in docs/workpiece-dimensions.md to
    # produce metric-scaled .step/.stl in <job_root>/postscale/.
    parser.add_argument("--workpiece-class", type=str, default=None,
                        choices=["cover_plate", "square_tube", "bellmouth", "h_beam"],
                        help="Catalog workpiece class. If omitted, post-scaling is skipped.")
    parser.add_argument("--model-code", type=str, default=None,
                        help="Catalog model code (e.g. G140). Ignored for h_beam (uses 'default').")
    parser.add_argument("--postscale-script", type=Path, default=None,
                        help="Override path to cadrille_metric_postscale.py (defaults to repo_root/scripts/).")
    parser.add_argument("--postscale-catalog", type=Path, default=None,
                        help="Override path to workpiece-dimensions.md (defaults to repo_root/docs/).")
    return parser.parse_args()


POSTSCALE_STAGE_LABEL = "Post-scaling: Aligning CAD to catalog (mm)"
BODY_CLEANUP_STAGE_LABEL = "Body cleanup: Removing hallucinated bodies"
PIPELINE_STAGE_LABELS = [
    "SAM3D: Loading checkpoints",
    "SAM3D: Generating mesh",
    "Cadrille: Preparing input",
    "Cadrille: Generating CAD result",
    BODY_CLEANUP_STAGE_LABEL,
    POSTSCALE_STAGE_LABEL,
]
TERMINAL_STATUSES = {"completed", "failed", "terminated"}


def write_status(
    path: Path,
    *,
    status: str,
    stage: str,
    stage_label: str,
    error: str | None = None,
    result_paths: dict[str, Any] | None = None,
) -> None:
    now = time.time()
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    started_at = existing.get("started_at") or now
    stage_timings = dict(existing.get("stage_timings") or {})
    previous_label = existing.get("stage_label")

    if previous_label in PIPELINE_STAGE_LABELS and previous_label != stage_label:
        previous_entry = dict(stage_timings.get(previous_label) or {})
        previous_entry.setdefault("started_at", existing.get("updated_at") or started_at)
        previous_entry.setdefault("ended_at", now)
        stage_timings[previous_label] = previous_entry

    if stage_label in PIPELINE_STAGE_LABELS:
        current_entry = dict(stage_timings.get(stage_label) or {})
        default_started_at = started_at if stage_label == PIPELINE_STAGE_LABELS[0] else now
        current_entry.setdefault("started_at", default_started_at)
        if status in TERMINAL_STATUSES:
            current_entry.setdefault("ended_at", now)
        else:
            current_entry["ended_at"] = None
        stage_timings[stage_label] = current_entry

    if status in TERMINAL_STATUSES:
        active_label = previous_label if previous_label in PIPELINE_STAGE_LABELS else stage_label
        if active_label in PIPELINE_STAGE_LABELS:
            active_entry = dict(stage_timings.get(active_label) or {})
            active_entry.setdefault("started_at", started_at)
            active_entry.setdefault("ended_at", now)
            stage_timings[active_label] = active_entry

    payload = {
        "status": status,
        "stage": stage,
        "stage_label": stage_label,
        "started_at": started_at,
        "updated_at": now,
        "stage_timings": stage_timings,
    }
    if status in TERMINAL_STATUSES:
        payload["ended_at"] = now
    if error:
        payload["error"] = error
    if result_paths:
        payload["result_paths"] = result_paths
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def patch_torch_hub_for_local_dinov2(torch: Any) -> None:
    local_repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    if not local_repo.exists():
        return

    original_load = torch.hub.load
    announced = {"done": False}

    def wrapped_load(repo_or_dir: str, model: str, *load_args: Any, **load_kwargs: Any):
        source = load_kwargs.get("source", "github")
        if repo_or_dir == "facebookresearch/dinov2" and source == "github":
            load_kwargs["source"] = "local"
            repo_or_dir = str(local_repo)
            if not announced["done"]:
                print(f"[torch.hub] redirecting facebookresearch/dinov2 to local cache: {local_repo}", flush=True)
                announced["done"] = True
        return original_load(repo_or_dir, model, *load_args, **load_kwargs)

    torch.hub.load = wrapped_load


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def first_match(path: Path, pattern: str) -> str | None:
    matches = sorted(path.glob(pattern))
    return str(matches[0]) if matches else None


def copy_result_file(src: str | None, dst: Path) -> str | None:
    if not src:
        return None
    src_path = Path(src)
    if not src_path.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst)
    return str(dst)


def build_simple_result_paths(
    *,
    job_root: Path,
    sam3d_mesh_glb: Path,
    sam3d_mesh_stl: Path,
    cadrille_output_root: Path,
    selected_mesh: str | None,
    selected_py: str | None,
    selected_brep: str | None,
) -> dict[str, Any]:
    results_root = job_root / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    result_paths = {
        "job_root": str(job_root),
        "results_root": str(results_root),
        "sam3d_mesh_glb": str(results_root / "sam3d_mesh.glb"),
        "sam3d_mesh_stl": str(results_root / "sam3d_mesh.stl"),
        "cadrille_output_root": str(cadrille_output_root),
        "selected_mesh": None,
        "selected_py": None,
        "selected_brep": None,
        # Body-cleanup slots — populated by run_body_cleanup_stage.
        "cleaned_brep_step": None,
        "cleaned_mesh_stl": None,
        "cleanup_metadata": None,
        "n_bodies_before": None,
        "n_bodies_after": None,
        # Post-scaling slots — populated by run_postscale_stage when applicable.
        "postscale_dir": None,
        "scaled_mesh_stl": None,
        "scaled_brep_step": None,
        "scaled_py": None,
        "scaled_metadata": None,
        "workpiece_class": None,
        "model_code": None,
    }
    shutil.copy2(sam3d_mesh_glb, results_root / "sam3d_mesh.glb")
    shutil.copy2(sam3d_mesh_stl, results_root / "sam3d_mesh.stl")
    result_paths["selected_mesh"] = copy_result_file(selected_mesh, results_root / "cadrille_selected_mesh.stl")
    result_paths["selected_py"] = copy_result_file(selected_py, results_root / "cadrille_selected.py")
    result_paths["selected_brep"] = copy_result_file(selected_brep, results_root / f"cadrille_selected.{Path(selected_brep).suffix.lstrip('.')}" if selected_brep else results_root / "cadrille_selected.step")
    return result_paths


def run_postscale_stage(
    *,
    repo_root: Path,
    job_root: Path,
    selected_py_host: Path,
    docker_image: str,
    workpiece_class: str,
    model_code: str | None,
    postscale_script_override: Path | None,
    postscale_catalog_override: Path | None,
) -> dict[str, str]:
    """Run cadrille_metric_postscale.py in docker on the selected_py and copy
    outputs into <job_root>/results/postscale_*. Returns a dict of populated
    result_paths keys (caller merges into the main result_paths)."""
    script_path = (postscale_script_override
                   or (repo_root / "scripts" / "cadrille_metric_postscale.py")).resolve()
    catalog_path = (postscale_catalog_override
                    or (repo_root / "docs" / "workpiece-dimensions.md")).resolve()
    if not script_path.exists():
        raise RuntimeError(f"post-scaling script not found: {script_path}")
    if not catalog_path.exists():
        raise RuntimeError(f"post-scaling catalog not found: {catalog_path}")

    postscale_work = (job_root / "postscale").resolve()
    postscale_work.mkdir(parents=True, exist_ok=True)
    selected_py_in_ctr = "/job/" + str(selected_py_host.resolve().relative_to(job_root))

    docker_cmd = [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{repo_root}:/repo:ro",
        "-v", f"{job_root}:/job",
        docker_image,
        "python",
        "/repo/scripts/cadrille_metric_postscale.py",
        "--py", selected_py_in_ctr,
        "--out-dir", "/job/postscale",
        "--dimensions", "/repo/docs/workpiece-dimensions.md",
        "--workpiece-class", workpiece_class,
        "--rewrite-mode", "axiswise",
        "--export-stl",
    ]
    if model_code and model_code not in ("(default)", "default", ""):
        docker_cmd.extend(["--model-code", model_code])
    run_cmd(docker_cmd)

    py_stem = selected_py_host.stem
    src_scaled_py = postscale_work / f"{py_stem}__scaled.py"
    src_scaled_step = postscale_work / f"{py_stem}__scaled.step"
    src_scaled_stl = postscale_work / f"{py_stem}__scaled.stl"
    src_metadata = postscale_work / f"{py_stem}__metadata.json"

    # Copy into results/ so the GUI's results_root has stable filenames
    results_root = job_root / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    populated: dict[str, str | None] = {
        "postscale_dir": str(postscale_work),
        "scaled_py": copy_result_file(str(src_scaled_py), results_root / "cadrille_scaled.py"),
        "scaled_brep_step": copy_result_file(str(src_scaled_step), results_root / "cadrille_scaled.step"),
        "scaled_mesh_stl": copy_result_file(str(src_scaled_stl), results_root / "cadrille_scaled.stl"),
        "scaled_metadata": copy_result_file(str(src_metadata), results_root / "cadrille_scaled_metadata.json"),
        "workpiece_class": workpiece_class,
        "model_code": (model_code if model_code and model_code not in ("(default)", "") else None),
    }
    return {k: v for k, v in populated.items() if v is not None}


def run_body_cleanup_stage(
    *,
    repo_root: Path,
    job_root: Path,
    selected_brep_host: Path,
    docker_image: str,
) -> dict[str, Any]:
    """Run scripts/cadrille_body_cleanup.py in docker on the selected .step and
    copy cleaned outputs into <job_root>/results/. Returns result_paths keys to
    merge. Raises on hard failure (caller treats cleanup as best-effort)."""
    cleanup_work = (job_root / "cleanup").resolve()
    cleanup_work.mkdir(parents=True, exist_ok=True)
    brep_in_ctr = "/job/" + str(selected_brep_host.resolve().relative_to(job_root))

    docker_cmd = [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{repo_root}:/repo:ro",
        "-v", f"{job_root}:/job",
        docker_image,
        "python",
        "/repo/scripts/cadrille_body_cleanup.py",
        "--in-step", brep_in_ctr,
        "--out-dir", "/job/cleanup",
        "--export-stl",
    ]
    run_cmd(docker_cmd)

    stem = selected_brep_host.stem  # e.g. "cadrille_selected"
    src_step = cleanup_work / f"{stem}__cleaned.step"
    src_stl = cleanup_work / f"{stem}__cleaned.stl"
    src_meta = cleanup_work / f"{stem}__cleanup_metadata.json"

    results_root = job_root / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    populated: dict[str, Any] = {
        "cleaned_brep_step": copy_result_file(str(src_step), results_root / "cadrille_cleaned.step"),
        "cleaned_mesh_stl": copy_result_file(str(src_stl), results_root / "cadrille_cleaned.stl"),
        "cleanup_metadata": copy_result_file(str(src_meta), results_root / "cadrille_cleanup_metadata.json"),
    }
    if src_meta.exists():
        try:
            meta = json.loads(src_meta.read_text(encoding="utf-8"))
            populated["n_bodies_before"] = meta.get("n_bodies_before")
            populated["n_bodies_after"] = meta.get("n_bodies_after")
        except Exception:  # noqa: BLE001
            pass
    return {k: v for k, v in populated.items() if v is not None}


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print("[RUN]", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def main() -> None:
    args = parse_args()

    repo_root = args.repo_root.resolve()
    sam3d_repo_root = (args.sam3d_repo_root or (repo_root / "repos" / "sam-3d-objects")).resolve()
    cadrille_root = (args.cadrille_root or (repo_root / "repos" / "cadrille")).resolve()
    job_root = args.job_root.resolve()
    status_path = args.status_path.resolve()
    input_image = args.input_image.resolve()
    input_mask = args.input_mask.resolve()

    sam3d_output_root = job_root / "sam3d"
    bridge_root = job_root / "bridge"
    cadrille_output_root = job_root / "cadrille"
    split_name = "gui_single_upload"
    split_dir = bridge_root / "data" / split_name
    manifest_jsonl = bridge_root / "input_manifest.jsonl"

    current_stage = "sam3d"
    try:
        write_status(status_path, status="running", stage="sam3d", stage_label="SAM3D: Loading checkpoints")
        job_root.mkdir(parents=True, exist_ok=True)

        os.environ.setdefault("CONDA_PREFIX", str(Path(sys.executable).resolve().parents[1]))
        os.environ.setdefault("ATTN_BACKEND", "flash_attn")
        os.environ.setdefault("SPARSE_ATTN_BACKEND", "flash_attn")

        sys.path.insert(0, str(repo_root / "scripts"))
        sys.path.insert(0, str(sam3d_repo_root))
        sys.path.insert(0, str(sam3d_repo_root / "notebook"))

        from inference import Inference  # type: ignore
        from sam3d_cadrille_bridge import ensure_clean_dir, prepare_cadrille_split, write_manifest_jsonl  # type: ignore
        import torch  # type: ignore

        patch_torch_hub_for_local_dinov2(torch)

        image = Image.open(input_image).convert("RGB")
        mask_image = Image.open(input_mask).convert("L")
        if image.size != mask_image.size:
            raise RuntimeError(f"Image/mask size mismatch: {image.size} vs {mask_image.size}")

        image_np = np.array(image)
        mask_np = (np.array(mask_image) > 0).astype(np.uint8)
        mask_pixels = int(mask_np.sum())
        if mask_pixels <= 0:
            raise RuntimeError("Uploaded mask is empty")

        config_path = sam3d_repo_root / "checkpoints" / "hf" / "pipeline.yaml"
        os.chdir(sam3d_repo_root)
        model_init_started = time.time()
        inference = Inference(str(config_path), compile=False)
        model_init_sec = time.time() - model_init_started

        sample_stem = input_image.stem or "upload"
        sample_out_dir = sam3d_output_root / "GUI" / "user_upload" / f"{sample_stem}__obj01"
        sample_out_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = sample_out_dir / "mesh.glb"
        stl_path = sample_out_dir / "mesh.stl"
        meta_path = sample_out_dir / "meta.json"
        results_path = sam3d_output_root / "results.jsonl"

        write_status(status_path, status="running", stage="sam3d", stage_label="SAM3D: Generating mesh")

        started_at = time.time()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        output = inference(image_np, mask_np, seed=args.seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        mesh = output.get("glb")
        if mesh is None:
            raise RuntimeError("SAM3D output did not include a GLB mesh")
        mesh.export(str(mesh_path))
        mesh.export(str(stl_path))
        duration = time.time() - started_at
        peak_allocated_mb = None
        peak_reserved_mb = None
        if torch.cuda.is_available():
            peak_allocated_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 2)
            peak_reserved_mb = round(torch.cuda.max_memory_reserved() / (1024**2), 2)

        record = {
            "global_index": 0,
            "task_index_in_shard": 1,
            "total_tasks_in_shard": 1,
            "total_tasks_global": 1,
            "num_shards": 1,
            "shard_index": 0,
            "task_id": f"GUI/user_upload/{sample_stem}__obj01",
            "split": "all",
            "subset": "GUI",
            "workpiece": "user_upload",
            "stem": sample_stem,
            "object_index": 1,
            "object_count_in_image": 1,
            "image_path": str(input_image),
            "annotation_path": None,
            "output_dir": str(sample_out_dir),
            "mesh_path": str(mesh_path),
            "stl_path": str(stl_path),
            "artifact_formats": ["glb", "stl"],
            "category": "user_upload",
            "group": 1,
            "bbox": [0.0, 0.0, float(image.width), float(image.height)],
            "area": float(mask_pixels),
            "width": int(image.width),
            "height": int(image.height),
            "image_pixels": int(image.width * image.height),
            "seed": args.seed,
            "dataset_layout": "gui_upload",
            "exclude_stems_file": None,
            "exclude_stems_count": 0,
            "started_at_epoch": started_at,
            "model_init_sec": round(model_init_sec, 3),
            "status": "ok",
            "hostname": os.uname().nodename,
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "duration_sec": round(duration, 3),
            "ended_at_epoch": round(time.time(), 3),
            "mask_pixels": mask_pixels,
            "mask_fraction": round(mask_pixels / float(image.width * image.height), 6),
            "sec_per_megapixel": round(duration / ((image.width * image.height) / 1_000_000), 6),
            "instances_per_hour": round(3600.0 / duration, 3) if duration > 0 else None,
            "peak_memory_allocated_mb": peak_allocated_mb,
            "peak_memory_reserved_mb": peak_reserved_mb,
            "mesh_size_bytes": mesh_path.stat().st_size if mesh_path.exists() else None,
            "stl_size_bytes": stl_path.stat().st_size if stl_path.exists() else None,
        }
        meta_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        append_jsonl(results_path, record)

        current_stage = "cadrille"
        write_status(status_path, status="running", stage="cadrille", stage_label="Cadrille: Preparing input")

        ensure_clean_dir(cadrille_output_root, force=True, dry_run=False, label="cadrille-output-root")
        bridge_root.mkdir(parents=True, exist_ok=True)
        ensure_clean_dir(split_dir, force=True, dry_run=False, label="bridge split directory")
        prepared_rows = prepare_cadrille_split(
            [record],
            split_dir=split_dir,
            normalize_stl=bool(args.normalize_stl),
            dry_run=False,
        )
        write_manifest_jsonl(manifest_jsonl, prepared_rows, dry_run=False)

        runner_script = repo_root / "scripts" / "run_cadrille_on_split.py"
        write_status(status_path, status="running", stage="cadrille", stage_label="Cadrille: Generating CAD result")

        cmd = [
            sys.executable,
            str(runner_script),
            "--prepared-split-name",
            split_name,
            "--prepared-split-dir",
            str(split_dir),
            "--bridge-manifest-jsonl",
            str(manifest_jsonl),
            "--cadrille-root",
            str(cadrille_root),
            "--cadrille-output-root",
            str(cadrille_output_root),
            "--cadrille-mode",
            args.cadrille_mode,
            "--cadrille-input-source",
            "mesh",
            "--cadrille-runtime",
            args.cadrille_runtime,
            "--cadrille-python",
            args.cadrille_python,
            "--cadrille-docker-image",
            args.cadrille_docker_image,
            "--cadrille-docker-python",
            args.cadrille_docker_python,
            "--cadrille-docker-gpus",
            args.cadrille_docker_gpus,
            f"--cadrille-docker-extra-args={args.cadrille_docker_extra_args}",
            "--cadrille-checkpoint",
            args.cadrille_checkpoint,
            "--cadrille-processor-path",
            args.cadrille_processor_path,
            "--cadrille-n-samples",
            str(args.cadrille_n_samples),
            "--cadrille-batch-size",
            str(args.cadrille_batch_size),
            "--selection-mode",
            args.selection_mode,
            "--selected-candidate-index",
            str(args.selected_candidate_index),
            "--brep-ext",
            args.brep_ext,
            "--convert-timeout-sec",
            str(args.convert_timeout_sec),
            "--sam3d-output-root",
            str(sam3d_output_root),
            "--records-found-ok",
            "1",
            "--records-selected-for-bridge",
            "1",
        ]
        if args.allow_selection_fallback:
            cmd.append("--allow-selection-fallback")
        cmd.append("--export-brep" if args.export_brep else "--no-export-brep")
        run_cmd(cmd)

        selected_mesh = first_match(cadrille_output_root, "selected_mesh/*.stl")
        selected_py = first_match(cadrille_output_root, "selected_py/*.py")
        selected_brep = first_match(cadrille_output_root, f"selected_brep/*.{args.brep_ext}")
        result_paths = build_simple_result_paths(
            job_root=job_root,
            sam3d_mesh_glb=mesh_path,
            sam3d_mesh_stl=stl_path,
            cadrille_output_root=cadrille_output_root,
            selected_mesh=selected_mesh,
            selected_py=selected_py,
            selected_brep=selected_brep,
        )

        # ─── Body cleanup stage (always-on when a BRep exists) ───
        if result_paths.get("selected_brep"):
            current_stage = "body_cleanup"
            write_status(
                status_path,
                status="running",
                stage="body_cleanup",
                stage_label=BODY_CLEANUP_STAGE_LABEL,
                result_paths=result_paths,
            )
            try:
                cleanup_results = run_body_cleanup_stage(
                    repo_root=repo_root,
                    job_root=job_root,
                    selected_brep_host=Path(result_paths["selected_brep"]),
                    docker_image=args.cadrille_docker_image,
                )
                result_paths.update(cleanup_results)
            except Exception as exc:  # noqa: BLE001
                print(f"[body-cleanup] stage skipped: {exc!r}", flush=True)

        # ─── Post-scaling stage (optional) ───
        if args.workpiece_class and selected_py:
            current_stage = "postscale"
            write_status(
                status_path,
                status="running",
                stage="postscale",
                stage_label=POSTSCALE_STAGE_LABEL,
                result_paths=result_paths,
            )
            postscale_results = run_postscale_stage(
                repo_root=repo_root,
                job_root=job_root,
                selected_py_host=Path(selected_py),
                docker_image=args.cadrille_docker_image,
                workpiece_class=args.workpiece_class,
                model_code=args.model_code,
                postscale_script_override=args.postscale_script,
                postscale_catalog_override=args.postscale_catalog,
            )
            result_paths.update(postscale_results)

        write_status(status_path, status="completed", stage="completed", stage_label="Done", result_paths=result_paths)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        write_status(
            status_path,
            status="failed",
            stage=current_stage,
            stage_label="Failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    main()
