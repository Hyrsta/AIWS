#!/usr/bin/env python3
"""sam3d-svc adapter CLI: image + mask -> SAM3D mesh.

Contract (matches serving/sam3d-svc/app/inference.py):
    run_sam3d.py --input-image P --input-mask P --seed N --out-mesh P

Writes the reconstructed mesh to --out-mesh (a .ply path; the actual mesh
is STL, which is copied to that path) and a sidecar
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
import types
from pathlib import Path


def _mock_gradio() -> None:
    """SAM3D notebook/inference.py imports gradio for its demo UI.

    The demo section is guarded by `if __name__ == '__main__'` but the import
    is module-level. Mock it out so we can import Inference without a working
    gradio installation.
    """
    if "gradio" not in sys.modules:
        gr = types.ModuleType("gradio")
        # Minimal stubs that inference.py or its transitive imports may touch
        gr.Blocks = object
        gr.Markdown = object
        gr.Model3D = object
        sys.modules["gradio"] = gr


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

    # Set CONDA_PREFIX / CUDA_HOME before importing from the SAM3D notebook
    # (inference.py sets os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]).
    conda_prefix = os.environ.get("CONDA_PREFIX") or str(Path(sys.executable).resolve().parents[1])
    os.environ.setdefault("CONDA_PREFIX", conda_prefix)
    os.environ.setdefault("CUDA_HOME", os.environ["CONDA_PREFIX"])

    # SAM3D notebook/inference.py sets LIDRA_SKIP_INIT so we don't need to.
    os.environ.setdefault("ATTN_BACKEND", "flash_attn")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "flash_attn")

    if args.gpu_index is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    # Mock gradio before any SAM3D import happens.
    _mock_gradio()

    sys.path.insert(0, str(repo_root / "scripts"))
    sys.path.insert(0, str(sam3d_repo_root))
    sys.path.insert(0, str(sam3d_repo_root / "notebook"))

    import random

    import numpy as np
    from PIL import Image

    from inference import Inference  # type: ignore  # noqa: E402

    import torch  # type: ignore

    # ---- seed determinism (mirrors simple_reconstruct_job.py) ----
    seed_int = int(args.seed) % (2 ** 32)
    os.environ["PYTHONHASHSEED"] = str(seed_int)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed_int)
    np.random.seed(seed_int)
    torch.manual_seed(seed_int)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_int)
        torch.cuda.manual_seed_all(seed_int)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    image = Image.open(args.input_image).convert("RGB")
    mask_image = Image.open(args.input_mask).convert("L")
    if image.size != mask_image.size:
        raise RuntimeError(f"Image/mask size mismatch: {image.size} vs {mask_image.size}")

    image_np = np.array(image)
    mask_np = (np.array(mask_image) > 0).astype(np.uint8)
    if int(mask_np.sum()) <= 0:
        raise RuntimeError("Uploaded mask is empty")

    config_path = sam3d_repo_root / "checkpoints" / "hf" / "pipeline.yaml"
    os.chdir(sam3d_repo_root)
    inference = Inference(str(config_path), compile=False)

    # Re-seed after model construction (mirrors simple_reconstruct_job.py).
    torch.manual_seed(seed_int)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_int)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    for name in ("ss_generator", "slat_generator"):
        try:
            pipeline = getattr(inference, "_pipeline", None)
            models = getattr(pipeline, "models", None)
            if models and name in models:
                models[name].seed = seed_int
        except Exception:
            pass

    output = inference(image_np, mask_np, seed=args.seed)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    mesh_glb = output.get("glb")
    if mesh_glb is None:
        raise RuntimeError("SAM3D output did not include a GLB mesh")

    out_mesh = args.out_mesh.resolve()
    work = out_mesh.parent
    work.mkdir(parents=True, exist_ok=True)
    out_glb = work / "sam3d_mesh.glb"
    out_stl = work / "sam3d_mesh.stl"

    mesh_glb.export(str(out_glb))
    mesh_glb.export(str(out_stl))

    # Provide the mesh at the requested path (cadrille-svc reads this path).
    if out_mesh != out_stl:
        shutil.copyfile(out_stl, out_mesh)

    stem = args.input_image.stem or "upload"
    record = {
        "task_id": f"GUI/user_upload/{stem}__obj01",
        "split": "all",
        "subset": "GUI",
        "workpiece": "user_upload",
        "stem": stem,
        "object_index": 1,
        "object_count_in_image": 1,
        "output_dir": str(work),
        "mesh_path": str(out_glb),
        "stl_path": str(out_stl),
        "artifact_formats": ["glb", "stl"],
        "status": "ok",
        "dataset_layout": "gui_upload",
        "seed": args.seed,
    }
    (work / "sam3d_record.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(f"[run_sam3d] wrote {out_mesh}", flush=True)


if __name__ == "__main__":
    main()
