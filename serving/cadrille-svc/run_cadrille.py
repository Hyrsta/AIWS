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

cadrille-svc receives only the mesh path, so this adapter reconstructs the
minimal SAM3D `record` and the sam3d_output_root layout the bridge expects.
Runs only on the RXL host (needs cadrille:latest). Validate end to end on a GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
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

    sys.path.insert(0, str(repo_root / "scripts"))
    from aiws_pipeline_core import CadrilleOptions, run_cadrille_pipeline  # type: ignore

    out_dir = args.out_dir.resolve()
    job_root = out_dir / "_job"
    job_root.mkdir(parents=True, exist_ok=True)

    mesh = args.mesh.resolve()
    stem = mesh.stem or "upload"
    sam3d_out = job_root / "sam3d" / "GUI" / "user_upload" / f"{stem}__obj01"
    sam3d_out.mkdir(parents=True, exist_ok=True)
    stl_path = sam3d_out / "mesh.stl"
    glb_path = sam3d_out / "mesh.glb"
    shutil.copyfile(mesh, stl_path)
    src_glb = mesh.parent / "sam3d_mesh.glb"
    shutil.copyfile(src_glb if src_glb.exists() else mesh, glb_path)

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

    gpus = args.docker_gpus or ("device=" + args.device.split(":")[-1])
    opts = CadrilleOptions(
        cadrille_root=cadrille_root,
        cadrille_mode=args.mode,
        cadrille_n_samples=args.n_candidates,
        cadrille_docker_image=args.docker_image,
        cadrille_docker_gpus=gpus,
        cadrille_checkpoint=args.ckpt,
        cleanup=args.cleanup,
    )
    result_paths = run_cadrille_pipeline(
        repo_root=repo_root,
        job_root=job_root,
        record=record,
        sam3d_mesh_glb=glb_path,
        sam3d_mesh_stl=stl_path,
        opts=opts,
    )

    cad_dir = out_dir / "cad"
    preview_dir = out_dir / "preview"
    mesh_dir = out_dir / "mesh"
    for d in (cad_dir, preview_dir, mesh_dir):
        d.mkdir(parents=True, exist_ok=True)

    def pick(*keys):
        for k in keys:
            v = result_paths.get(k)
            if v and Path(v).exists():
                return Path(v)
        return None

    code = pick("scaled_py", "selected_py")
    step = pick("scaled_brep_step", "cleaned_brep_step", "selected_brep")
    prev = pick("scaled_mesh_stl", "cleaned_mesh_stl", "selected_mesh")
    if code:
        shutil.copyfile(code, cad_dir / "model.py")
    if step:
        shutil.copyfile(step, cad_dir / "model.step")
    if prev:
        shutil.copyfile(prev, preview_dir / "model.stl")
    shutil.copyfile(stl_path, mesh_dir / "sam3d_mesh.stl")

    metrics: dict = {}
    sm = result_paths.get("stage_metrics")
    if sm and Path(sm).exists():
        try:
            metrics = json.loads(Path(sm).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            metrics = {}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[run_cadrille] bundle written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
