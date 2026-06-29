#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from sam3d_cadrille_bridge import (
    compute_chunks,
    ensure_clean_dir,
    load_sam3d_ok_records,
    prepare_cadrille_split,
    write_manifest_jsonl,
)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch sharded Cadrille runs across multiple GPUs using shared prepared SAM3D bridge splits. "
            "The batch script owns split partitioning; e2e remains full-pipeline oriented."
        )
    )
    parser.add_argument(
        "--python",
        default="/home/rxl/anaconda3/envs/sam3d-objects/bin/python",
        help="Python executable used to run the Cadrille split runner",
    )
    parser.add_argument(
        "--runner-script",
        type=Path,
        default=Path("/ssd1/rxl/zhankaiming/AIWS/scripts/run_cadrille_on_split.py"),
        help="Path to run_cadrille_on_split.py",
    )
    parser.add_argument(
        "--sam3d-output-root",
        type=Path,
        default=Path("/ssd1/rxl/zhankaiming/AIWS/outputs/sam3d-aiws52-clean-mesh-stl-20260410-193527"),
        help="SAM3D run root containing results.jsonl (root or shard-*/results.jsonl)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Base output root for modality/shard runs, logs, manifests, and prepared shared splits",
    )
    parser.add_argument(
        "--split-prefix",
        default="sam3d_bridge_full",
        help="Prefix used to generate shared per-shard split names",
    )
    parser.add_argument(
        "--shared-splits-root",
        type=Path,
        default=None,
        help="Directory that will hold prepared per-shard split folders outside the Cadrille repo (default: <output-root>/shared_splits)",
    )
    parser.add_argument(
        "--modalities",
        default="pc,img",
        help="Comma-separated modalities to run (subset of: pc,img)",
    )
    parser.add_argument(
        "--gpus",
        default="0,1,2,3",
        help="Comma-separated GPU ids; one shard process per GPU",
    )
    parser.add_argument(
        "--normalize-stl",
        dest="normalize_stl",
        action="store_true",
        default=True,
        help="Normalize bridged STL files to unit cube [0,1] before Cadrille (default on)",
    )
    parser.add_argument(
        "--no-normalize-stl",
        dest="normalize_stl",
        action="store_false",
        help="Use SAM3D STL as-is when preparing shared shard splits",
    )
    parser.add_argument(
        "--cadrille-runtime",
        choices=("auto", "docker", "host"),
        default="docker",
        help="Runtime passed to the split runner",
    )
    parser.add_argument(
        "--cadrille-docker-image",
        default="cadrille:latest",
        help="Docker image passed to the split runner when runtime=docker/auto",
    )
    parser.add_argument(
        "--cadrille-docker-extra-args",
        default="",
        help="Extra raw args passed through to docker run in the split runner",
    )
    parser.add_argument(
        "--cadrille-root",
        type=Path,
        default=Path("/ssd1/rxl/zhankaiming/AIWS/repos/cadrille"),
        help="Cadrille repo root passed through to the split runner",
    )
    parser.add_argument(
        "--cadrille-checkpoint",
        default="ckpt/cadrille_sft",
        help="Checkpoint path passed through to the split runner",
    )
    parser.add_argument(
        "--cadrille-processor-path",
        default="ckpt/Qwen2-VL-2B-Instruct",
        help="Processor path passed through to the split runner",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("evaluate", "index"),
        default="evaluate",
        help="Selection mode passed to the split runner",
    )
    parser.add_argument(
        "--allow-selection-fallback",
        action="store_true",
        help="Allow evaluate -> index fallback in the split runner",
    )
    parser.add_argument(
        "--pc-n-samples",
        type=int,
        default=5,
        help="Cadrille n_samples for pc mode",
    )
    parser.add_argument(
        "--img-n-samples",
        type=int,
        default=1,
        help="Cadrille n_samples for img mode",
    )
    parser.add_argument(
        "--cadrille-batch-size",
        type=int,
        default=64,
        help="Batch size passed to Cadrille for each shard",
    )
    parser.add_argument(
        "--export-brep",
        dest="export_brep",
        action="store_true",
        default=True,
        help="Enable STEP/BRep export",
    )
    parser.add_argument(
        "--no-export-brep",
        dest="export_brep",
        action="store_false",
        help="Disable STEP/BRep export",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing shared splits and shard output folders",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands only",
    )
    return parser.parse_args()



def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def resolve_shared_splits_root(args: argparse.Namespace) -> Path:
    if args.shared_splits_root is not None:
        return args.shared_splits_root.resolve()
    return (args.output_root / "shared_splits").resolve()



def prepare_shared_splits(
    *,
    records: list[dict[str, Any]],
    chunks: list[tuple[int, int]],
    args: argparse.Namespace,
    shared_splits_root: Path,
) -> list[dict[str, Any]]:
    manifests_root = args.output_root / "bridge_manifests"
    if not args.dry_run:
        manifests_root.mkdir(parents=True, exist_ok=True)

    split_plans: list[dict[str, Any]] = []
    for shard_idx, (sample_offset, max_samples) in enumerate(chunks):
        if max_samples <= 0:
            continue
        shard_records = records[sample_offset: sample_offset + max_samples]
        split_name = f"{args.split_prefix}_s{shard_idx}"
        split_dir = shared_splits_root / split_name
        manifest_jsonl = manifests_root / f"{split_name}.jsonl"

        ensure_clean_dir(split_dir, force=args.force, dry_run=args.dry_run, label=f"shared split shard-{shard_idx}")
        prepared_rows = prepare_cadrille_split(
            shard_records,
            split_dir=split_dir,
            normalize_stl=bool(args.normalize_stl),
            dry_run=args.dry_run,
        )
        write_manifest_jsonl(manifest_jsonl, prepared_rows, dry_run=args.dry_run)
        print(f"[INFO] Prepared shared split shard={shard_idx} count={len(prepared_rows)} split={split_name} dir={split_dir}")

        split_plans.append(
            {
                "shard": shard_idx,
                "sample_offset": sample_offset,
                "max_samples": max_samples,
                "split_name": split_name,
                "split_dir": str(split_dir),
                "manifest_jsonl": str(manifest_jsonl),
                "prepared_count": len(prepared_rows),
            }
        )

    if not split_plans:
        raise RuntimeError("No shared splits were prepared")
    return split_plans



def run_modality(
    *,
    modality: str,
    n_samples: int,
    gpus: list[str],
    split_plans: list[dict[str, Any]],
    args: argparse.Namespace,
    logs_dir: Path,
    total_records: int,
) -> None:
    mode_root = args.output_root / modality
    ensure_dir(mode_root)

    procs: list[tuple[int, str, Path, subprocess.Popen[Any]]] = []
    launch_records: list[dict[str, Any]] = []

    for plan, gpu in zip(split_plans, gpus):
        shard_idx = int(plan["shard"])
        shard_out = mode_root / f"shard-{shard_idx}"
        log_path = logs_dir / f"{modality}-shard-{shard_idx}.log"

        ensure_clean_dir(shard_out, force=args.force, dry_run=args.dry_run, label=f"{modality} shard-{shard_idx} output")

        cmd = [
            args.python,
            str(args.runner_script),
            "--prepared-split-name",
            str(plan["split_name"]),
            "--prepared-split-dir",
            str(plan["split_dir"]),
            "--bridge-manifest-jsonl",
            str(plan["manifest_jsonl"]),
            "--sam3d-output-root",
            str(args.sam3d_output_root),
            "--records-found-ok",
            str(total_records),
            "--records-selected-for-bridge",
            str(plan["max_samples"]),
            "--cadrille-output-root",
            str(shard_out),
            "--cadrille-root",
            str(args.cadrille_root),
            "--cadrille-mode",
            modality,
            "--cadrille-runtime",
            args.cadrille_runtime,
            "--cadrille-docker-image",
            args.cadrille_docker_image,
            f"--cadrille-docker-extra-args={args.cadrille_docker_extra_args}",
            "--cadrille-checkpoint",
            args.cadrille_checkpoint,
            "--cadrille-processor-path",
            args.cadrille_processor_path,
            "--cadrille-n-samples",
            str(n_samples),
            "--cadrille-batch-size",
            str(args.cadrille_batch_size),
            "--cadrille-docker-gpus",
            f"device={gpu}",
            "--selection-mode",
            args.selection_mode,
        ]
        if args.allow_selection_fallback:
            cmd.append("--allow-selection-fallback")
        if args.export_brep:
            cmd.append("--export-brep")
        else:
            cmd.append("--no-export-brep")
        if args.normalize_stl:
            cmd.append("--bridge-normalized")

        launch_records.append(
            {
                "modality": modality,
                "shard": shard_idx,
                "gpu": gpu,
                "sample_offset": plan["sample_offset"],
                "max_samples": plan["max_samples"],
                "prepared_count": plan["prepared_count"],
                "output_root": str(shard_out),
                "split_name": plan["split_name"],
                "split_dir": plan["split_dir"],
                "manifest_jsonl": plan["manifest_jsonl"],
                "log_path": str(log_path),
                "cmd": cmd,
            }
        )

        print(
            f"[PLAN] mode={modality} shard={shard_idx} gpu={gpu} "
            f"offset={plan['sample_offset']} max={plan['max_samples']} split={plan['split_name']}"
        )
        print(f"[CMD] {' '.join(cmd)}")

        if args.dry_run:
            continue

        with log_path.open("w", encoding="utf-8") as logf:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        procs.append((shard_idx, gpu, log_path, proc))

    launch_manifest_path = mode_root / "launch_manifest.json"
    launch_manifest_path.write_text(json.dumps(launch_records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Wrote launch manifest: {launch_manifest_path}")

    if args.dry_run:
        return

    failed = False
    for shard_idx, gpu, log_path, proc in procs:
        ret = proc.wait()
        if ret != 0:
            failed = True
            print(f"[ERROR] mode={modality} shard={shard_idx} gpu={gpu} exited with {ret}. log={log_path}")
        else:
            print(f"[OK] mode={modality} shard={shard_idx} gpu={gpu} completed. log={log_path}")

    if failed:
        raise RuntimeError(f"One or more shard runs failed for modality={modality}")



def main() -> None:
    args = parse_args()

    args.runner_script = args.runner_script.resolve()
    args.sam3d_output_root = args.sam3d_output_root.resolve()
    args.output_root = args.output_root.resolve()
    args.cadrille_root = args.cadrille_root.resolve()

    modalities = [m.strip().lower() for m in args.modalities.split(",") if m.strip()]
    allowed = {"pc", "img"}
    if not modalities or any(m not in allowed for m in modalities):
        raise RuntimeError(f"--modalities must be comma-separated subset of {sorted(allowed)}")

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise RuntimeError("--gpus must provide at least one GPU id")

    if args.pc_n_samples <= 0 or args.img_n_samples <= 0:
        raise RuntimeError("--pc-n-samples and --img-n-samples must be > 0")
    if args.cadrille_batch_size <= 0:
        raise RuntimeError("--cadrille-batch-size must be > 0")

    records = load_sam3d_ok_records(args.sam3d_output_root)
    total = len(records)
    if total == 0:
        raise RuntimeError("No SAM3D ok records found")

    chunks = compute_chunks(total, len(gpus))
    shared_splits_root = resolve_shared_splits_root(args)

    ensure_dir(args.output_root)
    logs_dir = args.output_root / "logs"
    ensure_dir(logs_dir)

    plan_path = args.output_root / "run_plan.json"
    plan = {
        "created_at_epoch": time.time(),
        "sam3d_output_root": str(args.sam3d_output_root),
        "total_records": total,
        "gpus": gpus,
        "chunks": [{"shard": i, "sample_offset": off, "max_samples": sz} for i, (off, sz) in enumerate(chunks)],
        "modalities": modalities,
        "pc_n_samples": args.pc_n_samples,
        "img_n_samples": args.img_n_samples,
        "cadrille_batch_size": args.cadrille_batch_size,
        "normalize_stl": bool(args.normalize_stl),
        "shared_splits_root": str(shared_splits_root),
    }
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Wrote run plan: {plan_path}")

    split_plans = prepare_shared_splits(records=records, chunks=chunks, args=args, shared_splits_root=shared_splits_root)
    split_plan_path = args.output_root / "shared_split_plan.json"
    split_plan_path.write_text(json.dumps(split_plans, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Wrote shared split plan: {split_plan_path}")

    for mode in modalities:
        n_samples = args.pc_n_samples if mode == "pc" else args.img_n_samples
        print(f"\n[INFO] Starting modality={mode} with n_samples={n_samples} across {len(split_plans)} shared splits")
        run_modality(
            modality=mode,
            n_samples=n_samples,
            gpus=gpus,
            split_plans=split_plans,
            args=args,
            logs_dir=logs_dir,
            total_records=total,
        )

    print("\n[DONE] All requested modalities completed successfully.")


if __name__ == "__main__":
    main()
