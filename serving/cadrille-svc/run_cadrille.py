#!/usr/bin/env python3
"""cadrille-svc adapter CLI: SAM3D mesh -> Cadrille CAD bundle.

Contract (matches serving/cadrille-svc/app/inference.py):
    run_cadrille.py --mesh P --mode pc|img --n-candidates N --seed N \
                    --ckpt P --device D --out-dir P [--cleanup]

Emits the canonical bundle layout under --out-dir:
    cad/model.py, cad/model.step, preview/model.stl, mesh/sam3d_mesh.stl,
    metrics.json
(render.png is not produced by this stage and is intentionally omitted; the
gateway bundle skips missing artifacts.)

This adapter shells out to scripts/run_cadrille_on_split.py which runs Cadrille
inside the existing cadrille:latest Docker image. Docker must be available on
the host. Validate end to end on a GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Cadrille adapter: mesh -> CAD bundle")
    ap.add_argument("--mesh", type=Path, required=True)
    ap.add_argument("--mode", choices=("pc", "img"), default="pc")
    ap.add_argument("--n-candidates", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", default="ckpt/cadrille_rl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cleanup", action="store_true")
    ap.add_argument("--repo-root", type=Path,
                    default=Path(os.environ.get("AIWS_REPO_ROOT", "/repo")))
    ap.add_argument("--cadrille-root", type=Path, default=None)
    ap.add_argument("--docker-image", default="cadrille:latest")
    ap.add_argument("--docker-gpus", default=None)
    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    cadrille_root = (args.cadrille_root or (repo_root / "repos" / "cadrille")).resolve()

    out_dir = args.out_dir.resolve()
    mesh = args.mesh.resolve()
    stem = mesh.stem or "upload"

    # GPU selection: map cuda:N -> "device=N" for docker --gpus
    gpus = args.docker_gpus or ("device=" + args.device.split(":")[-1])

    # Build a SAM3D-style output layout that run_cadrille_on_split.py expects
    job_root = out_dir / "_job"
    sam3d_out = job_root / "sam3d" / "GUI" / "user_upload" / f"{stem}__obj01"
    sam3d_out.mkdir(parents=True, exist_ok=True)

    stl_path = sam3d_out / "mesh.stl"
    glb_path = sam3d_out / "mesh.glb"
    shutil.copyfile(mesh, stl_path)
    # cadrille also needs a .glb sidecar; use .stl if no .glb is present
    src_glb = mesh.parent / "sam3d_mesh.glb"
    shutil.copyfile(src_glb if src_glb.exists() else mesh, glb_path)

    # Minimal SAM3D record needed by the bridge
    record = {
        "task_id": f"GUI/user_upload/{stem}__obj01",
        "split": "all",
        "subset": "GUI",
        "workpiece": "user_upload",
        "stem": stem,
        "object_index": 1,
        "object_count_in_image": 1,
        "output_dir": str(sam3d_out),
        "mesh_path": str(glb_path),
        "stl_path": str(stl_path),
        "artifact_formats": ["glb", "stl"],
        "status": "ok",
        "dataset_layout": "gui_upload",
        "seed": args.seed,
    }

    # Write a results.jsonl so run_cadrille_on_split.py can find the record
    results_jsonl = job_root / "sam3d" / "results.jsonl"
    results_jsonl.parent.mkdir(parents=True, exist_ok=True)
    results_jsonl.write_text(json.dumps(record) + "\n", encoding="utf-8")

    # Prepare the bridge split (same as simple_reconstruct_job.py does)
    bridge_root = job_root / "bridge"
    split_name = "gui_single_upload"
    split_dir = bridge_root / "data" / split_name
    manifest_jsonl = bridge_root / "input_manifest.jsonl"

    sys.path.insert(0, str(repo_root / "scripts"))
    from sam3d_cadrille_bridge import (  # type: ignore
        ensure_clean_dir, prepare_cadrille_split, write_manifest_jsonl,
    )

    bridge_root.mkdir(parents=True, exist_ok=True)
    ensure_clean_dir(split_dir, force=True, dry_run=False, label="bridge-split")
    prepared_rows = prepare_cadrille_split(
        [record],
        split_dir=split_dir,
        normalize_stl=True,
        dry_run=False,
    )
    write_manifest_jsonl(manifest_jsonl, prepared_rows, dry_run=False)

    # Run Cadrille via run_cadrille_on_split.py
    cadrille_output_root = job_root / "cadrille"
    runner_script = repo_root / "scripts" / "run_cadrille_on_split.py"

    cmd = [
        sys.executable,
        str(runner_script),
        "--prepared-split-name", split_name,
        "--prepared-split-dir", str(split_dir),
        "--bridge-manifest-jsonl", str(manifest_jsonl),
        "--cadrille-root", str(cadrille_root),
        "--cadrille-output-root", str(cadrille_output_root),
        "--cadrille-mode", args.mode,
        "--cadrille-input-source", "mesh",
        "--cadrille-runtime", "docker",
        "--cadrille-docker-image", args.docker_image,
        "--cadrille-docker-gpus", gpus,
        "--cadrille-docker-extra-args=--ipc=host --shm-size=16g",
        "--cadrille-checkpoint", args.ckpt,
        "--cadrille-n-samples", str(1 if args.mode == "img" else args.n_candidates),
        "--selection-mode", "evaluate",
        "--allow-selection-fallback",
        "--brep-ext", "step",
        "--sam3d-output-root", str(job_root / "sam3d"),
        "--records-found-ok", "1",
        "--records-selected-for-bridge", "1",
        "--export-brep",
    ]

    print(f"[run_cadrille] invoking: {' '.join(cmd[:6])} ...", flush=True)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"run_cadrille_on_split.py failed with exit code {e.returncode}")

    # Collect outputs from the Cadrille output root
    def first_match(*patterns):
        for pat in patterns:
            hits = sorted(cadrille_output_root.glob(pat))
            if hits:
                return hits[0]
        return None

    selected_py = first_match("selected_py/*.py")
    selected_step = first_match("selected_brep/*.step", "tmp_brep/*.step")
    selected_stl = first_match("selected_mesh/*.stl", "tmp_mesh/*.stl")

    cad_dir = out_dir / "cad"
    preview_dir = out_dir / "preview"
    mesh_dir = out_dir / "mesh"
    for d in (cad_dir, preview_dir, mesh_dir):
        d.mkdir(parents=True, exist_ok=True)

    if selected_py and selected_py.exists():
        shutil.copyfile(selected_py, cad_dir / "model.py")
    if selected_step and selected_step.exists():
        shutil.copyfile(selected_step, cad_dir / "model.step")
    if selected_stl and selected_stl.exists():
        shutil.copyfile(selected_stl, preview_dir / "model.stl")
    shutil.copyfile(stl_path, mesh_dir / "sam3d_mesh.stl")

    # Read Cadrille metrics if present
    metrics: dict = {}
    metrics_path = cadrille_output_root / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            metrics = {}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[run_cadrille] bundle written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
