#!/usr/bin/env python3
"""sam3d-svc adapter CLI: image + mask -> SAM3D mesh.

Contract (matches serving/sam3d-svc/app/inference.py):
    run_sam3d.py --input-image P --input-mask P --seed N --out-mesh P

Writes the reconstructed mesh to --out-mesh and a sidecar
`sam3d_mesh.glb` + `sam3d_record.json` next to it (consumed by the cadrille
adapter). Runs only on the RXL GPU host: it imports the SAM3D repo and the
shared pipeline core, neither of which is available on a CPU box. Validate end
to end on a GPU before relying on it.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="SAM3D adapter: image+mask -> mesh")
    ap.add_argument("--input-image", type=Path, required=True)
    ap.add_argument("--input-mask", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-mesh", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path,
                    default=Path(os.environ.get("AIWS_REPO_ROOT", "/repo")))
    ap.add_argument("--sam3d-repo-root", type=Path, default=None)
    ap.add_argument("--gpu-index", type=int, default=None)
    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    sam3d_repo_root = (args.sam3d_repo_root
                       or (repo_root / "repos" / "sam-3d-objects")).resolve()

    sys.path.insert(0, str(repo_root / "scripts"))
    from aiws_pipeline_core import run_sam3d_inference  # type: ignore

    out_mesh = args.out_mesh.resolve()
    work = out_mesh.parent
    work.mkdir(parents=True, exist_ok=True)
    out_glb = work / "sam3d_mesh.glb"
    out_stl = work / "sam3d_mesh.stl"

    record = run_sam3d_inference(
        repo_root=repo_root,
        sam3d_repo_root=sam3d_repo_root,
        input_image=args.input_image,
        input_mask=args.input_mask,
        seed=args.seed,
        out_glb=out_glb,
        out_stl=out_stl,
        gpu_index=args.gpu_index,
    )

    # Provide the mesh at the requested path (cadrille-svc reads this path).
    if out_mesh != out_stl:
        shutil.copyfile(out_stl, out_mesh)
    (work / "sam3d_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"[run_sam3d] wrote {out_mesh}", flush=True)


if __name__ == "__main__":
    main()
