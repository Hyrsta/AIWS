#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from sam3d_cadrille_bridge import (
    ensure_clean_dir,
    load_sam3d_ok_records,
    prepare_cadrille_split,
    sanitize_stem,
    write_manifest_jsonl,
)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end pipeline: RGB input -> SAM3D -> bridged Cadrille split -> "
            "Cadrille inference -> selected CadQuery/CAD outputs."
        )
    )

    sam = parser.add_argument_group("SAM3D stage")
    sam.add_argument("--skip-sam3d", action="store_true", help="Skip SAM3D and reuse existing results.jsonl under --sam3d-output-root")
    sam.add_argument("--sam3d-python", default=sys.executable, help="Python executable for SAM3D stage")
    sam.add_argument(
        "--sam3d-script",
        type=Path,
        default=Path("/ssd1/rxl/zhankaiming/AIWS/scripts/sam3d_batch.py"),
        help="Path to sam3d_batch.py",
    )
    sam.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/ssd1/rxl/zhankaiming/AIWS/data/aiws5.2-usable"),
        help="Dataset root for SAM3D",
    )
    sam.add_argument(
        "--dataset-layout",
        choices=("auto", "split", "subset"),
        default="subset",
        help="Dataset layout passed to SAM3D batch runner",
    )
    sam.add_argument(
        "--sam3d-repo-root",
        type=Path,
        default=Path("/ssd1/rxl/zhankaiming/AIWS/repos/sam-3d-objects"),
        help="SAM3D repo root",
    )
    sam.add_argument(
        "--sam3d-output-root",
        type=Path,
        required=True,
        help="SAM3D output root (single run root or one shard output root)",
    )
    sam.add_argument("--seed", type=int, default=42, help="SAM3D seed")
    sam.add_argument("--limit", type=int, default=None, help="Optional SAM3D limit")
    sam.add_argument("--num-shards", type=int, default=1, help="SAM3D num-shards")
    sam.add_argument("--shard-index", type=int, default=0, help="SAM3D shard-index")
    sam.add_argument("--exclude-stems-file", type=Path, default=None, help="Optional exclude-stems-file for SAM3D")
    sam.add_argument("--resume", dest="resume", action="store_true", default=True, help="Enable SAM3D --resume (default on)")
    sam.add_argument("--no-resume", dest="resume", action="store_false", help="Disable SAM3D --resume")

    bridge = parser.add_argument_group("Bridge stage")
    bridge.add_argument(
        "--normalize-stl",
        dest="normalize_stl",
        action="store_true",
        default=True,
        help="Normalize SAM3D STL to unit cube [0,1] for Cadrille (default on)",
    )
    bridge.add_argument(
        "--no-normalize-stl",
        dest="normalize_stl",
        action="store_false",
        help="Use SAM3D STL as-is (not recommended)",
    )
    bridge.add_argument(
        "--bridge-split-name",
        default=None,
        help="Optional name for the bridged split stored under <cadrille-output-root>/bridge/data",
    )
    bridge.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing e2e output root / bridge directory",
    )

    cad = parser.add_argument_group("Cadrille stage")
    cad.add_argument("--cadrille-python", default=sys.executable, help="Python executable for Cadrille host runtime")
    cad.add_argument(
        "--cadrille-runtime",
        choices=("auto", "docker", "host"),
        default="auto",
        help="Run Cadrille stage on host Python or inside Docker (default: auto, prefer docker if available)",
    )
    cad.add_argument("--cadrille-docker-image", default="cadrille:latest", help="Docker image for Cadrille runtime")
    cad.add_argument("--cadrille-docker-python", default="python", help="Python executable inside Cadrille Docker image")
    cad.add_argument("--cadrille-docker-gpus", default="device=0", help="Value for docker --gpus")
    cad.add_argument(
        "--cadrille-docker-extra-args",
        default="",
        help="Extra raw args appended to docker run",
    )
    cad.add_argument(
        "--cadrille-root",
        type=Path,
        default=Path("/ssd1/rxl/zhankaiming/AIWS/repos/cadrille"),
        help="Cadrille repository root",
    )
    cad.add_argument("--cadrille-checkpoint", default="ckpt/cadrille_sft", help="Cadrille checkpoint path")
    cad.add_argument("--cadrille-processor-path", default="ckpt/Qwen2-VL-2B-Instruct", help="Processor / tokenizer path for Cadrille")
    cad.add_argument("--cadrille-mode", choices=("pc", "img"), default="pc", help="Cadrille modality: point-cloud-conditioned (`pc`) or image-conditioned (`img`)")
    cad.add_argument("--cadrille-n-samples", type=int, default=None, help="Number of Cadrille candidates per sample (defaults: pc=5, img=1 in the downstream runner)")
    cad.add_argument("--cadrille-batch-size", type=int, default=64, help="Batch size used during Cadrille inference")
    cad.add_argument("--cadrille-output-root", type=Path, required=True, help="Output root for this end-to-end run, including bridge artifacts and selected CAD outputs")
    cad.add_argument("--export-brep", dest="export_brep", action="store_true", default=True, help="Export STEP/BRep outputs alongside selected code and mesh outputs")
    cad.add_argument("--no-export-brep", dest="export_brep", action="store_false", help="Disable STEP/BRep export")
    cad.add_argument("--brep-ext", default="step", help="BRep export extension")
    cad.add_argument("--convert-timeout-sec", type=float, default=5.0, help="Timeout for per-candidate CAD conversion during evaluation")

    select = parser.add_argument_group("Candidate selection")
    select.add_argument("--selection-mode", choices=("evaluate", "index"), default="evaluate", help="How to choose the final candidate per sample")
    select.add_argument("--selected-candidate-index", type=int, default=0, help="Candidate index used when --selection-mode index is selected")
    select.add_argument("--allow-selection-fallback", action="store_true", help="Allow fallback to --selected-candidate-index when evaluation cannot choose a best candidate")
    select.add_argument("--eval-gt-path", type=Path, default=None, help="Optional evaluation ground-truth path (defaults to the bridged split directory)")
    select.add_argument("--eval-gt-mesh-ext", default=None, help="Ground-truth mesh extension for evaluation (defaults to stl)")
    select.add_argument("--eval-n-points", type=int, default=8192, help="Number of sampled points used during geometric evaluation")
    select.add_argument("--dry-run", action="store_true", help="Print planned actions without running SAM3D or Cadrille")

    return parser.parse_args()



def run_cmd(cmd: list[str], cwd: Path | None = None, dry_run: bool = False) -> None:
    prefix = "[DRY-RUN]" if dry_run else "[RUN]"
    print(prefix, " ".join(str(x) for x in cmd) + (f" (cwd={cwd})" if cwd else ""))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)



def default_bridge_split_name(cadrille_output_root: Path) -> str:
    return f"sam3d_bridge_{sanitize_stem(cadrille_output_root.name)}"



def main() -> None:
    args = parse_args()

    if args.cadrille_n_samples is not None and args.cadrille_n_samples <= 0:
        raise RuntimeError("--cadrille-n-samples must be > 0")
    if args.cadrille_batch_size <= 0:
        raise RuntimeError("--cadrille-batch-size must be > 0")
    if args.eval_n_points <= 0:
        raise RuntimeError("--eval-n-points must be > 0")

    sam3d_output_root = args.sam3d_output_root.resolve()
    cadrille_root = args.cadrille_root.resolve()
    cadrille_output_root = args.cadrille_output_root.resolve()

    if not args.skip_sam3d:
        sam_cmd = [
            args.sam3d_python,
            str(args.sam3d_script),
            "--dataset-root",
            str(args.dataset_root),
            "--dataset-layout",
            args.dataset_layout,
            "--repo-root",
            str(args.sam3d_repo_root),
            "--output-root",
            str(sam3d_output_root),
            "--seed",
            str(args.seed),
            "--num-shards",
            str(args.num_shards),
            "--shard-index",
            str(args.shard_index),
        ]
        if args.resume:
            sam_cmd.append("--resume")
        if args.limit is not None:
            sam_cmd.extend(["--limit", str(args.limit)])
        if args.exclude_stems_file is not None:
            sam_cmd.extend(["--exclude-stems-file", str(args.exclude_stems_file)])
        run_cmd(sam_cmd, dry_run=args.dry_run)

    ensure_clean_dir(cadrille_output_root, force=args.force, dry_run=args.dry_run, label="cadrille-output-root")
    bridge_root = cadrille_output_root / "bridge"
    if not args.dry_run:
        bridge_root.mkdir(parents=True, exist_ok=True)

    split_name = args.bridge_split_name or default_bridge_split_name(cadrille_output_root)
    split_dir = bridge_root / "data" / split_name
    manifest_jsonl = bridge_root / "input_manifest.jsonl"

    records = load_sam3d_ok_records(sam3d_output_root)
    records_found_ok = len(records)
    records_selected_for_bridge = len(records)
    if not records:
        raise RuntimeError("No SAM3D OK records selected for Cadrille input")

    ensure_clean_dir(split_dir, force=args.force, dry_run=args.dry_run, label="e2e bridge split directory")
    prepared_rows = prepare_cadrille_split(
        records,
        split_dir=split_dir,
        normalize_stl=bool(args.normalize_stl),
        dry_run=args.dry_run,
    )
    write_manifest_jsonl(manifest_jsonl, prepared_rows, dry_run=args.dry_run)
    print(f"[INFO] Prepared {len(prepared_rows)} STL inputs for bridged split: {split_dir}")

    runner_script = Path(__file__).resolve().parent / "run_cadrille_on_split.py"
    run_cmd_obj = [
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
        "--cadrille-batch-size",
        str(args.cadrille_batch_size),
        "--brep-ext",
        args.brep_ext,
        "--convert-timeout-sec",
        str(args.convert_timeout_sec),
        "--selection-mode",
        args.selection_mode,
        "--selected-candidate-index",
        str(args.selected_candidate_index),
        "--eval-gt-format",
        "mesh",
        "--eval-n-points",
        str(args.eval_n_points),
        "--sam3d-output-root",
        str(sam3d_output_root),
        "--records-found-ok",
        str(records_found_ok),
        "--records-selected-for-bridge",
        str(records_selected_for_bridge),
    ]
    if args.cadrille_n_samples is not None:
        run_cmd_obj.extend(["--cadrille-n-samples", str(args.cadrille_n_samples)])
    if args.export_brep:
        run_cmd_obj.append("--export-brep")
    else:
        run_cmd_obj.append("--no-export-brep")
    if args.allow_selection_fallback:
        run_cmd_obj.append("--allow-selection-fallback")
    if args.eval_gt_path is not None:
        run_cmd_obj.extend(["--eval-gt-path", str(args.eval_gt_path)])
    if args.eval_gt_mesh_ext is not None:
        run_cmd_obj.extend(["--eval-gt-mesh-ext", args.eval_gt_mesh_ext])
    if args.normalize_stl:
        run_cmd_obj.append("--bridge-normalized")
    if args.dry_run:
        run_cmd_obj.append("--dry-run")

    run_cmd(run_cmd_obj, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
