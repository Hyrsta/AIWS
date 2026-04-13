#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
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
    return parser.parse_args()


def write_status(
    path: Path,
    *,
    status: str,
    stage: str,
    stage_label: str,
    error: str | None = None,
    result_paths: dict[str, Any] | None = None,
) -> None:
    payload = {
        "status": status,
        "stage": stage,
        "stage_label": stage_label,
        "updated_at": time.time(),
    }
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
        write_status(status_path, status="running", stage="sam3d", stage_label="Processing SAM3D")
        job_root.mkdir(parents=True, exist_ok=True)

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
        write_status(status_path, status="running", stage="cadrille", stage_label="Processing Cadrille")

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

        result_paths = {
            "job_root": str(job_root),
            "sam3d_mesh_glb": str(mesh_path),
            "sam3d_mesh_stl": str(stl_path),
            "cadrille_output_root": str(cadrille_output_root),
            "selected_mesh": first_match(cadrille_output_root, "selected_mesh/*.stl"),
            "selected_py": first_match(cadrille_output_root, "selected_py/*.py"),
            "selected_brep": first_match(cadrille_output_root, f"selected_brep/*.{args.brep_ext}"),
        }
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
