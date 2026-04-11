#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch full-dataset Cadrille e2e runs for both modalities (pc/img) "
            "across multiple GPUs by partitioning SAM3D records into disjoint shards."
        )
    )
    parser.add_argument(
        "--python",
        default="/home/rxl/anaconda3/envs/sam3d-objects/bin/python",
        help="Python executable used to run the e2e script",
    )
    parser.add_argument(
        "--e2e-script",
        type=Path,
        default=Path("/ssd1/rxl/zhankaiming/AIWS/scripts/run_sam3d_to_cadrille_e2e.py"),
        help="Path to run_sam3d_to_cadrille_e2e.py",
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
        help="Base output root for modality/shard runs",
    )
    parser.add_argument(
        "--split-prefix",
        default="sam3d_bridge_full",
        help="Prefix used to generate per-shard Cadrille split names",
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
        "--cadrille-runtime",
        choices=("auto", "docker", "host"),
        default="docker",
        help="Runtime passed to e2e script",
    )
    parser.add_argument(
        "--cadrille-docker-image",
        default="cadrille:latest",
        help="Docker image passed to e2e script when runtime=docker/auto",
    )
    parser.add_argument(
        "--cadrille-docker-extra-args",
        default="",
        help="Extra raw args passed through to docker run in the e2e script",
    )
    parser.add_argument(
        "--cadrille-root",
        type=Path,
        default=Path("/ssd1/rxl/zhankaiming/cadrille-official-col14m"),
        help="Cadrille repo root passed through to the e2e script",
    )
    parser.add_argument(
        "--cadrille-checkpoint",
        default="ckpt/cadrille_sft",
        help="Checkpoint path passed through to the e2e script",
    )
    parser.add_argument(
        "--cadrille-processor-path",
        default="ckpt/Qwen2-VL-2B-Instruct",
        help="Processor path passed through to the e2e script",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("evaluate", "index"),
        default="evaluate",
        help="Selection mode passed to e2e script",
    )
    parser.add_argument(
        "--allow-selection-fallback",
        action="store_true",
        help="Allow evaluate -> index fallback in e2e script",
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
        help="Batch size passed to Cadrille test.py for each shard",
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
        help="Pass --force to e2e shard jobs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands only",
    )
    return parser.parse_args()


def load_sam3d_ok_records(run_root: Path) -> list[dict[str, Any]]:
    files: list[Path] = []
    root_results = run_root / "results.jsonl"
    if root_results.exists():
        files.append(root_results)
    files.extend(sorted(run_root.glob("shard-*/results.jsonl")))

    if not files:
        raise RuntimeError(f"No results.jsonl found under {run_root}")

    raw: list[dict[str, Any]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("status") != "ok":
                    continue
                if not row.get("stl_path"):
                    continue
                row["_source_results_file"] = str(path)
                row["_source_line"] = line_no
                raw.append(row)

    by_task: dict[str, tuple[float, int, dict[str, Any]]] = {}
    for idx, row in enumerate(raw):
        task_id = str(row.get("task_id") or row.get("stl_path") or f"row-{idx}")
        ts = row.get("ended_at_epoch")
        if ts is None:
            ts = row.get("started_at_epoch")
        try:
            ts_f = float(ts)
        except (TypeError, ValueError):
            ts_f = float(idx)
        prev = by_task.get(task_id)
        if prev is None or ts_f >= prev[0]:
            by_task[task_id] = (ts_f, idx, row)

    deduped = [triple[2] for triple in sorted(by_task.values(), key=lambda t: t[1])]
    deduped.sort(key=lambda r: str(r.get("task_id") or r.get("stl_path")))
    return deduped


def compute_chunks(total: int, n_shards: int) -> list[tuple[int, int]]:
    if n_shards <= 0:
        raise RuntimeError("n_shards must be > 0")
    base = total // n_shards
    rem = total % n_shards
    chunks: list[tuple[int, int]] = []
    offset = 0
    for i in range(n_shards):
        size = base + (1 if i < rem else 0)
        chunks.append((offset, size))
        offset += size
    return chunks


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_modality(
    *,
    modality: str,
    n_samples: int,
    gpus: list[str],
    chunks: list[tuple[int, int]],
    args: argparse.Namespace,
    logs_dir: Path,
) -> None:
    mode_root = args.output_root / modality
    ensure_dir(mode_root)

    procs: list[tuple[int, str, Path, subprocess.Popen[Any]]] = []
    launch_records: list[dict[str, Any]] = []

    for shard_idx, gpu in enumerate(gpus):
        sample_offset, max_samples = chunks[shard_idx]
        if max_samples <= 0:
            continue

        shard_out = mode_root / f"shard-{shard_idx}"
        split_name = f"{args.split_prefix}_{modality}_s{shard_idx}"
        log_path = logs_dir / f"{modality}-shard-{shard_idx}.log"

        cmd = [
            args.python,
            str(args.e2e_script),
            "--skip-sam3d",
            "--sam3d-output-root",
            str(args.sam3d_output_root),
            "--cadrille-output-root",
            str(shard_out),
            "--cadrille-split-name",
            split_name,
            "--cadrille-runtime",
            args.cadrille_runtime,
            "--cadrille-docker-image",
            args.cadrille_docker_image,
            f"--cadrille-docker-extra-args={args.cadrille_docker_extra_args}",
            "--cadrille-root",
            str(args.cadrille_root),
            "--cadrille-checkpoint",
            args.cadrille_checkpoint,
            "--cadrille-processor-path",
            args.cadrille_processor_path,
            "--cadrille-mode",
            modality,
            "--cadrille-input-source",
            "mesh",
            "--cadrille-n-samples",
            str(n_samples),
            "--cadrille-batch-size",
            str(args.cadrille_batch_size),
            "--cadrille-docker-gpus",
            f"device={gpu}",
            "--selection-mode",
            args.selection_mode,
            "--sample-offset",
            str(sample_offset),
            "--max-samples",
            str(max_samples),
        ]
        if args.allow_selection_fallback:
            cmd.append("--allow-selection-fallback")
        if args.export_brep:
            cmd.append("--export-brep")
        else:
            cmd.append("--no-export-brep")
        if args.force:
            cmd.append("--force")

        launch_records.append(
            {
                "modality": modality,
                "shard": shard_idx,
                "gpu": gpu,
                "sample_offset": sample_offset,
                "max_samples": max_samples,
                "output_root": str(shard_out),
                "split_name": split_name,
                "log_path": str(log_path),
                "cmd": cmd,
            }
        )

        print(f"[PLAN] mode={modality} shard={shard_idx} gpu={gpu} offset={sample_offset} max={max_samples}")
        print(f"[CMD] {' '.join(cmd)}")

        if args.dry_run:
            continue

        ensure_dir(shard_out)
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

    args.e2e_script = args.e2e_script.resolve()
    args.sam3d_output_root = args.sam3d_output_root.resolve()
    args.output_root = args.output_root.resolve()

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
    }
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Wrote run plan: {plan_path}")

    for mode in modalities:
        n_samples = args.pc_n_samples if mode == "pc" else args.img_n_samples
        print(f"\n[INFO] Starting modality={mode} with n_samples={n_samples} across {len(gpus)} shards")
        run_modality(
            modality=mode,
            n_samples=n_samples,
            gpus=gpus,
            chunks=chunks,
            args=args,
            logs_dir=logs_dir,
        )

    print("\n[DONE] All requested modalities completed successfully.")


if __name__ == "__main__":
    main()
