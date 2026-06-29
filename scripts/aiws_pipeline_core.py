#!/usr/bin/env python3
"""Shared single-sample reconstruction pipeline core.

This module factors the two halves of the proven GUI job
(`gui/backend/simple_reconstruct_job.py`) into callable functions so the REST
service adapters (`serving/sam3d-svc/run_sam3d.py`,
`serving/cadrille-svc/run_cadrille.py`) and the GUI share one implementation:

- `run_sam3d_inference(...)`  image + mask -> SAM3D mesh (glb + stl) + record
- `run_cadrille_pipeline(...)` SAM3D mesh + record -> Cadrille CAD result_paths

The proven stage helpers (determinism, bridge, reselect, body-cleanup,
postscale, stage-metrics, result-path assembly) are imported unchanged from
`simple_reconstruct_job` rather than copied, so there is a single source of
truth for them. `simple_reconstruct_job.py` itself is left unmodified to avoid
any regression in the live GUI job; once these functions are validated on a GPU
host its `main()` can be slimmed to call them.

NOTE: this code runs only on the RXL host, where the SAM3D repo, the cadrille
docker image, and the AIWS `scripts/` are present. It cannot be exercised on a
CPU-only box. Validate end to end on a GPU before relying on it.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _import_gui_helpers(repo_root: Path):
    """Make the proven helpers in gui/backend/simple_reconstruct_job importable.

    They are pure functions (determinism, docker stage runners, result-path
    assembly) and are reused verbatim so there is one implementation.
    """
    gui_backend = repo_root / "gui" / "backend"
    if str(gui_backend) not in sys.path:
        sys.path.insert(0, str(gui_backend))
    import simple_reconstruct_job as srj  # type: ignore

    return srj


def run_sam3d_inference(
    *,
    repo_root: Path,
    sam3d_repo_root: Path,
    input_image: Path,
    input_mask: Path,
    seed: int,
    out_glb: Path,
    out_stl: Path,
    gpu_index: int | None = None,
) -> dict[str, Any]:
    """Run SAM3D on one image + mask, export the mesh to out_glb and out_stl,
    and return the per-instance `record` dict the Cadrille bridge consumes.

    Mirrors the image_mask branch of simple_reconstruct_job.main().
    """
    repo_root = repo_root.resolve()
    sam3d_repo_root = sam3d_repo_root.resolve()
    input_image = input_image.resolve()
    input_mask = input_mask.resolve()
    out_glb = out_glb.resolve()
    out_stl = out_stl.resolve()
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    out_stl.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("CONDA_PREFIX", str(Path(sys.executable).resolve().parents[1]))
    # xformers is the reliable default on this host: the flash_attn wheel
    # requires GLIBC_2.32 which is not available on the RXL system libc.
    os.environ["ATTN_BACKEND"] = "xformers"
    os.environ["SPARSE_ATTN_BACKEND"] = "xformers"
    if gpu_index is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    else:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

    sys.path.insert(0, str(repo_root / "scripts"))
    sys.path.insert(0, str(sam3d_repo_root))
    sys.path.insert(0, str(sam3d_repo_root / "notebook"))

    srj = _import_gui_helpers(repo_root)

    from inference import Inference  # type: ignore
    import torch  # type: ignore

    srj.patch_torch_hub_for_local_dinov2(torch)
    srj.configure_sam3d_determinism(seed, torch_module=torch)

    image = Image.open(input_image).convert("RGB")
    mask_image = Image.open(input_mask).convert("L")
    if image.size != mask_image.size:
        raise RuntimeError(f"Image/mask size mismatch: {image.size} vs {mask_image.size}")

    image_np = np.array(image)
    mask_np = (np.array(mask_image) > 0).astype(np.uint8)
    mask_pixels = int(mask_np.sum())
    if mask_pixels <= 0:
        raise RuntimeError("mask is empty")

    config_path = sam3d_repo_root / "checkpoints" / "hf" / "pipeline.yaml"
    os.chdir(sam3d_repo_root)
    model_init_started = time.time()
    inference = Inference(str(config_path), compile=False)
    model_init_sec = time.time() - model_init_started

    srj.configure_sam3d_determinism(seed, inference=inference, torch_module=torch)
    started_at = time.time()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    output = inference(image_np, mask_np, seed=seed)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    mesh = output.get("glb")
    if mesh is None:
        raise RuntimeError("SAM3D output did not include a GLB mesh")
    mesh.export(str(out_glb))
    mesh.export(str(out_stl))
    duration = time.time() - started_at

    sample_stem = input_image.stem or "upload"
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
        "output_dir": str(out_glb.parent),
        "mesh_path": str(out_glb),
        "stl_path": str(out_stl),
        "artifact_formats": ["glb", "stl"],
        "category": "user_upload",
        "group": 1,
        "bbox": [0.0, 0.0, float(image.width), float(image.height)],
        "area": float(mask_pixels),
        "width": int(image.width),
        "height": int(image.height),
        "image_pixels": int(image.width * image.height),
        "seed": seed,
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
        "mesh_size_bytes": out_glb.stat().st_size if out_glb.exists() else None,
        "stl_size_bytes": out_stl.stat().st_size if out_stl.exists() else None,
    }
    return record


class CadrilleOptions:
    """Plain config object for the Cadrille half (mirrors the GUI job args)."""

    def __init__(
        self,
        *,
        cadrille_root: Path,
        cadrille_mode: str = "pc",
        cadrille_n_samples: int = 5,
        cadrille_batch_size: int = 64,
        cadrille_runtime: str = "docker",
        cadrille_python: str = "python",
        cadrille_docker_image: str = "cadrille:latest",
        cadrille_docker_python: str = "python",
        cadrille_docker_gpus: str = "device=0",
        cadrille_docker_extra_args: str = "--ipc=host --shm-size=16g",
        cadrille_checkpoint: str = "ckpt/cadrille_rl",
        cadrille_processor_path: str = "ckpt/Qwen2-VL-2B-Instruct",
        selection_mode: str = "evaluate",
        selected_candidate_index: int = 0,
        allow_selection_fallback: bool = False,
        brep_ext: str = "step",
        convert_timeout_sec: float = 5.0,
        normalize_stl: bool = True,
        export_brep: bool = True,
        cleanup: bool = True,
    ) -> None:
        self.cadrille_root = cadrille_root
        self.cadrille_mode = cadrille_mode
        self.cadrille_n_samples = cadrille_n_samples
        self.cadrille_batch_size = cadrille_batch_size
        self.cadrille_runtime = cadrille_runtime
        self.cadrille_python = cadrille_python
        self.cadrille_docker_image = cadrille_docker_image
        self.cadrille_docker_python = cadrille_docker_python
        self.cadrille_docker_gpus = cadrille_docker_gpus
        self.cadrille_docker_extra_args = cadrille_docker_extra_args
        self.cadrille_checkpoint = cadrille_checkpoint
        self.cadrille_processor_path = cadrille_processor_path
        self.selection_mode = selection_mode
        self.selected_candidate_index = selected_candidate_index
        self.allow_selection_fallback = allow_selection_fallback
        self.brep_ext = brep_ext
        self.convert_timeout_sec = convert_timeout_sec
        self.normalize_stl = normalize_stl
        self.export_brep = export_brep
        self.cleanup = cleanup


def run_cadrille_pipeline(
    *,
    repo_root: Path,
    job_root: Path,
    record: dict[str, Any],
    sam3d_mesh_glb: Path,
    sam3d_mesh_stl: Path,
    opts: CadrilleOptions,
) -> dict[str, Any]:
    """Run the Cadrille half: bridge the SAM3D record into a split, run Cadrille
    via run_cadrille_on_split.py, reselect, assemble result_paths, then the
    optional body-cleanup and stage-metrics stages.

    Mirrors the Cadrille section of simple_reconstruct_job.main(). Returns the
    result_paths dict (stable filenames under <job_root>/results/).
    """
    repo_root = repo_root.resolve()
    job_root = job_root.resolve()
    cadrille_root = opts.cadrille_root.resolve()

    srj = _import_gui_helpers(repo_root)
    sys.path.insert(0, str(repo_root / "scripts"))
    from sam3d_cadrille_bridge import (  # type: ignore
        ensure_clean_dir,
        prepare_cadrille_split,
        write_manifest_jsonl,
    )

    sam3d_output_root = job_root / "sam3d"
    bridge_root = job_root / "bridge"
    cadrille_output_root = job_root / "cadrille"
    split_name = "gui_single_upload"
    split_dir = bridge_root / "data" / split_name
    manifest_jsonl = bridge_root / "input_manifest.jsonl"

    ensure_clean_dir(cadrille_output_root, force=True, dry_run=False, label="cadrille-output-root")
    bridge_root.mkdir(parents=True, exist_ok=True)
    ensure_clean_dir(split_dir, force=True, dry_run=False, label="bridge split directory")
    prepared_rows = prepare_cadrille_split(
        [record],
        split_dir=split_dir,
        normalize_stl=bool(opts.normalize_stl),
        dry_run=False,
    )
    write_manifest_jsonl(manifest_jsonl, prepared_rows, dry_run=False)

    runner_script = repo_root / "scripts" / "run_cadrille_on_split.py"
    cmd = [
        sys.executable,
        str(runner_script),
        "--prepared-split-name", split_name,
        "--prepared-split-dir", str(split_dir),
        "--bridge-manifest-jsonl", str(manifest_jsonl),
        "--cadrille-root", str(cadrille_root),
        "--cadrille-output-root", str(cadrille_output_root),
        "--cadrille-mode", opts.cadrille_mode,
        "--cadrille-input-source", "mesh",
        "--cadrille-runtime", opts.cadrille_runtime,
        "--cadrille-python", opts.cadrille_python,
        "--cadrille-docker-image", opts.cadrille_docker_image,
        "--cadrille-docker-python", opts.cadrille_docker_python,
        "--cadrille-docker-gpus", opts.cadrille_docker_gpus,
        f"--cadrille-docker-extra-args={opts.cadrille_docker_extra_args}",
        "--cadrille-checkpoint", opts.cadrille_checkpoint,
        "--cadrille-processor-path", opts.cadrille_processor_path,
        "--cadrille-n-samples",
        str(1 if opts.cadrille_mode == "img" else opts.cadrille_n_samples),
        "--cadrille-batch-size", str(opts.cadrille_batch_size),
        "--selection-mode", opts.selection_mode,
        "--selected-candidate-index", str(opts.selected_candidate_index),
        "--brep-ext", opts.brep_ext,
        "--convert-timeout-sec", str(opts.convert_timeout_sec),
        "--sam3d-output-root", str(sam3d_output_root),
        "--records-found-ok", "1",
        "--records-selected-for-bridge", "1",
    ]
    if opts.allow_selection_fallback:
        cmd.append("--allow-selection-fallback")
    cmd.append("--export-brep" if opts.export_brep else "--no-export-brep")
    srj.run_cmd(cmd)

    selected_mesh = srj.first_match(cadrille_output_root, "selected_mesh/*.stl")
    selected_py = srj.first_match(cadrille_output_root, "selected_py/*.py")
    selected_brep = srj.first_match(cadrille_output_root, f"selected_brep/*.{opts.brep_ext}")

    try:
        reselected = srj.run_reselect_stage(
            repo_root=repo_root,
            job_root=job_root,
            cadrille_output_root=cadrille_output_root,
            brep_ext=opts.brep_ext,
            docker_image=opts.cadrille_docker_image,
        )
        if reselected:
            if reselected.get("brep"):
                selected_brep = reselected["brep"]
            if reselected.get("mesh"):
                selected_mesh = reselected["mesh"]
            if reselected.get("py"):
                selected_py = reselected["py"]
    except Exception as exc:  # noqa: BLE001
        print(f"[reselect] skipped (keeping min-CD pick): {exc!r}", flush=True)

    result_paths = srj.build_simple_result_paths(
        job_root=job_root,
        sam3d_mesh_glb=sam3d_mesh_glb,
        sam3d_mesh_stl=sam3d_mesh_stl,
        cadrille_output_root=cadrille_output_root,
        selected_mesh=selected_mesh,
        selected_py=selected_py,
        selected_brep=selected_brep,
        cadrille_mode=opts.cadrille_mode,
    )

    if opts.cleanup and result_paths.get("selected_brep"):
        try:
            cleanup_results = srj.run_body_cleanup_stage(
                repo_root=repo_root,
                job_root=job_root,
                selected_brep_host=Path(result_paths["selected_brep"]),
                docker_image=opts.cadrille_docker_image,
                reference_mesh_host=srj.find_bridge_gt_stl(job_root),
            )
            result_paths.update(cleanup_results)
        except Exception as exc:  # noqa: BLE001
            print(f"[body-cleanup] stage skipped: {exc!r}", flush=True)

    try:
        metrics_results = srj.run_stage_metrics_stage(
            repo_root=repo_root,
            job_root=job_root,
            result_paths=result_paths,
            docker_image=opts.cadrille_docker_image,
        )
        result_paths.update(metrics_results)
    except Exception as exc:  # noqa: BLE001
        print(f"[stage-metrics] stage skipped: {exc!r}", flush=True)

    return result_paths
