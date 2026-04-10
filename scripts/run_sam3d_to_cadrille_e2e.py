#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end pipeline: run SAM3D batch inference, normalize SAM3D STL outputs "
            "to Cadrille-compatible unit cube [0,1], run Cadrille inference, and export CAD outputs."
        )
    )

    # SAM3D stage
    sam = parser.add_argument_group("SAM3D stage")
    sam.add_argument("--skip-sam3d", action="store_true", help="Skip SAM3D and reuse existing results.jsonl under --sam3d-output-root")
    sam.add_argument("--sam3d-python", default=sys.executable, help="Python executable for SAM3D stage")
    sam.add_argument(
        "--sam3d-script",
        type=Path,
        default=Path("/ssd1/rxl/zhankaiming/AIWS/scripts/run_sam3d_aiws52_batch.py"),
        help="Path to run_sam3d_aiws52_batch.py",
    )
    sam.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/ssd1/rxl/zhankaiming/AIWS/data/aiws5.2-usable-materialized"),
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

    # Bridge stage
    bridge = parser.add_argument_group("SAM3D -> Cadrille bridge")
    bridge.add_argument("--normalize-stl", dest="normalize_stl", action="store_true", default=True,
                        help="Normalize SAM3D STL to unit cube [0,1] for Cadrille (default on)")
    bridge.add_argument("--no-normalize-stl", dest="normalize_stl", action="store_false",
                        help="Use SAM3D STL as-is (not recommended)")
    bridge.add_argument("--max-samples", type=int, default=None, help="Max number of SAM3D samples to pass into Cadrille")
    bridge.add_argument("--sample-offset", type=int, default=0, help="Start offset after sorting SAM3D task_id")

    # Cadrille stage
    cad = parser.add_argument_group("Cadrille stage")
    cad.add_argument("--cadrille-python", default=sys.executable, help="Python executable for Cadrille stage")
    cad.add_argument(
        "--cadrille-root",
        type=Path,
        default=Path("/ssd1/rxl/zhankaiming/AIWS/repos/cadrille"),
        help="Cadrille repository root",
    )
    cad.add_argument(
        "--cadrille-data-root",
        type=Path,
        default=None,
        help="Cadrille data root (default: <cadrille-root>/data)",
    )
    cad.add_argument("--cadrille-checkpoint", default="ckpt/cadrille_sft", help="Checkpoint path passed to Cadrille test.py")
    cad.add_argument("--cadrille-mode", choices=("pc", "img"), default="pc", help="Cadrille mode")
    cad.add_argument("--cadrille-split-name", default=None,
                     help="Split name created under Cadrille data root (default auto-generated)")
    cad.add_argument(
        "--cadrille-output-root",
        type=Path,
        required=True,
        help="Output root for Cadrille tmp/selected outputs and pipeline summary",
    )
    cad.add_argument("--mesh-ext", default="stl", help="Mesh extension passed to Cadrille test.py")
    cad.add_argument("--point-cloud-exts", default="ply,pcd,xyz,txt,npz,npy", help="Pass-through to Cadrille test.py")
    cad.add_argument("--image-exts", default="png,jpg,jpeg,bmp", help="Pass-through to Cadrille test.py")
    cad.add_argument("--export-brep", dest="export_brep", action="store_true", default=True,
                     help="Export STEP/BRep via convert_cadquery.py (default on)")
    cad.add_argument("--no-export-brep", dest="export_brep", action="store_false", help="Skip STEP/BRep export")
    cad.add_argument("--brep-ext", default="step", help="BRep extension for convert_cadquery.py")
    cad.add_argument("--convert-timeout-sec", type=float, default=5.0, help="Timeout per CadQuery file conversion")

    # Selection / safety
    misc = parser.add_argument_group("Selection and safety")
    misc.add_argument("--selected-candidate-index", type=int, default=0,
                      help="Preferred generated candidate index (+k suffix) copied to selected outputs")
    misc.add_argument("--prepare-input-only", action="store_true",
                      help="Stop after preparing normalized STL split for Cadrille")
    misc.add_argument("--force", action="store_true",
                      help="Allow deleting existing split/output folders created by this script")
    misc.add_argument("--dry-run", action="store_true", help="Print planned commands/actions without executing")

    return parser.parse_args()


def shell_join(cmd: list[str]) -> str:
    return shlex.join(cmd)


def run_cmd(cmd: list[str], cwd: Path | None = None, dry_run: bool = False) -> None:
    prefix = "[DRY-RUN]" if dry_run else "[RUN]"
    cwd_txt = f" (cwd={cwd})" if cwd else ""
    print(f"{prefix} {shell_join(cmd)}{cwd_txt}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ensure_clean_dir(path: Path, force: bool, dry_run: bool, label: str) -> None:
    if path.exists():
        existing = list(path.iterdir())
        if existing and not force:
            raise RuntimeError(
                f"{label} already exists and is not empty: {path}. "
                "Use --force to overwrite."
            )
        if existing and force:
            print(f"[INFO] Removing existing {label}: {path}")
            if not dry_run:
                shutil.rmtree(path)
    if not dry_run:
        path.mkdir(parents=True, exist_ok=True)


def sanitize_stem(name: str) -> str:
    stem = name.replace("/", "__").replace("\\", "__")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    stem = stem.strip("._-")
    return stem or "sample"


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
                stl_path = row.get("stl_path")
                if not stl_path:
                    continue
                row["_source_results_file"] = str(path)
                row["_source_line"] = line_no
                raw.append(row)

    # dedupe by task_id, keep latest record
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


def normalize_stl_to_unit_cube(src: Path, dst: Path) -> dict[str, Any]:
    mesh = trimesh.load_mesh(str(src), process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if g is not None]
        if not geoms:
            raise RuntimeError(f"No geometry found in scene mesh: {src}")
        mesh = trimesh.util.concatenate(geoms)

    bounds = mesh.bounds
    mins = bounds[0]
    maxs = bounds[1]
    extents = maxs - mins
    scale = float(extents.max())
    if not math.isfinite(scale) or scale <= 1e-12:
        raise RuntimeError(f"Invalid mesh scale for {src}: {scale}")

    mesh.apply_translation(-mins)
    mesh.apply_scale(1.0 / scale)

    dst.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(dst))

    new_bounds = mesh.bounds
    return {
        "src_bounds_min": [float(v) for v in mins],
        "src_bounds_max": [float(v) for v in maxs],
        "src_extent_max": float(scale),
        "dst_bounds_min": [float(v) for v in new_bounds[0]],
        "dst_bounds_max": [float(v) for v in new_bounds[1]],
    }


def pick_candidate_stem(tmp_py_dir: Path, base_stem: str, preferred_idx: int) -> str | None:
    preferred = tmp_py_dir / f"{base_stem}+{preferred_idx}.py"
    if preferred.exists():
        return preferred.stem

    candidates = sorted(tmp_py_dir.glob(f"{base_stem}+*.py"))
    if candidates:
        return candidates[0].stem
    return None


def main() -> None:
    args = parse_args()

    sam3d_output_root = args.sam3d_output_root.resolve()
    cadrille_root = args.cadrille_root.resolve()
    cadrille_data_root = (args.cadrille_data_root.resolve() if args.cadrille_data_root else (cadrille_root / "data").resolve())
    cadrille_output_root = args.cadrille_output_root.resolve()

    split_name = args.cadrille_split_name or f"sam3d_bridge_{int(time.time())}"
    split_dir = cadrille_data_root / split_name

    bridge_dir = cadrille_output_root / "bridge"
    tmp_py_dir = cadrille_output_root / "tmp_py"
    tmp_mesh_dir = cadrille_output_root / "tmp_mesh"
    tmp_brep_dir = cadrille_output_root / "tmp_brep"
    selected_py_dir = cadrille_output_root / "selected_py"
    selected_mesh_dir = cadrille_output_root / "selected_mesh"
    selected_brep_dir = cadrille_output_root / "selected_brep"

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

    records = load_sam3d_ok_records(sam3d_output_root)
    if args.sample_offset < 0:
        raise RuntimeError("--sample-offset must be >= 0")
    if args.sample_offset >= len(records):
        raise RuntimeError(f"--sample-offset {args.sample_offset} exceeds available records {len(records)}")

    selected = records[args.sample_offset :]
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise RuntimeError("--max-samples must be > 0")
        selected = selected[: args.max_samples]

    if not selected:
        raise RuntimeError("No SAM3D OK records selected for Cadrille input")

    ensure_clean_dir(cadrille_output_root, force=args.force, dry_run=args.dry_run, label="cadrille-output-root")
    ensure_clean_dir(split_dir, force=args.force, dry_run=args.dry_run, label="cadrille split directory")
    if not args.dry_run:
        bridge_dir.mkdir(parents=True, exist_ok=True)

    # Prepare normalized STL split for Cadrille
    prepared_rows: list[dict[str, Any]] = []
    used_stems: dict[str, int] = {}

    for row in selected:
        src = Path(str(row.get("stl_path"))).resolve()
        if not src.exists():
            continue

        task_id = str(row.get("task_id") or src.stem)
        stem_base = sanitize_stem(task_id)
        dup_idx = used_stems.get(stem_base, 0)
        used_stems[stem_base] = dup_idx + 1
        stem = stem_base if dup_idx == 0 else f"{stem_base}__dup{dup_idx:02d}"

        dst = split_dir / f"{stem}.stl"
        norm_meta: dict[str, Any] | None = None

        if args.dry_run:
            print(f"[DRY-RUN] prepare {src} -> {dst}")
        else:
            if args.normalize_stl:
                norm_meta = normalize_stl_to_unit_cube(src, dst)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

        prepared_rows.append(
            {
                "task_id": task_id,
                "subset": row.get("subset"),
                "workpiece": row.get("workpiece"),
                "stem": row.get("stem"),
                "object_index": row.get("object_index"),
                "sam3d_stl_path": str(src),
                "cadrille_stl_path": str(dst),
                "cadrille_stem": stem,
                "normalized": bool(args.normalize_stl),
                "normalization": norm_meta,
            }
        )

    if not prepared_rows:
        raise RuntimeError("No STL files were prepared for Cadrille")

    manifest_jsonl = bridge_dir / "input_manifest.jsonl"
    if not args.dry_run:
        with manifest_jsonl.open("w", encoding="utf-8") as f:
            for row in prepared_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[INFO] Prepared {len(prepared_rows)} STL inputs for Cadrille split: {split_name}")

    if args.prepare_input_only:
        print("[INFO] --prepare-input-only set, stopping before Cadrille inference.")
        return

    # Run Cadrille inference
    test_cmd = [
        args.cadrille_python,
        "test.py",
        "--data-path",
        str(cadrille_data_root),
        "--split",
        split_name,
        "--mode",
        args.cadrille_mode,
        "--checkpoint-path",
        args.cadrille_checkpoint,
        "--py-path",
        str(tmp_py_dir),
        "--input-source",
        "mesh",
        "--mesh-ext",
        args.mesh_ext,
        "--point-cloud-exts",
        args.point_cloud_exts,
        "--image-exts",
        args.image_exts,
    ]
    run_cmd(test_cmd, cwd=cadrille_root, dry_run=args.dry_run)

    # Convert CadQuery outputs to CAD meshes/BRep
    convert_cmd = [
        args.cadrille_python,
        "convert_cadquery.py",
        "--src",
        str(tmp_py_dir),
        "--mesh-out",
        str(tmp_mesh_dir),
        "--timeout",
        str(args.convert_timeout_sec),
    ]
    if args.export_brep:
        convert_cmd.extend([
            "--export-brep",
            "--brep-out",
            str(tmp_brep_dir),
            "--brep-ext",
            args.brep_ext,
        ])
    run_cmd(convert_cmd, cwd=cadrille_root, dry_run=args.dry_run)

    # Select preferred candidate per input sample
    selected_rows: list[dict[str, Any]] = []
    if not args.dry_run:
        selected_py_dir.mkdir(parents=True, exist_ok=True)
        selected_mesh_dir.mkdir(parents=True, exist_ok=True)
        if args.export_brep:
            selected_brep_dir.mkdir(parents=True, exist_ok=True)

        for item in prepared_rows:
            base = item["cadrille_stem"]
            candidate_stem = pick_candidate_stem(tmp_py_dir, base, args.selected_candidate_index)
            if candidate_stem is None:
                selected_rows.append({
                    "cadrille_stem": base,
                    "status": "missing_candidate",
                })
                continue

            py_src = tmp_py_dir / f"{candidate_stem}.py"
            mesh_src = tmp_mesh_dir / f"{candidate_stem}.stl"
            brep_src = tmp_brep_dir / f"{candidate_stem}.{args.brep_ext}"

            py_dst = selected_py_dir / f"{base}.py"
            mesh_dst = selected_mesh_dir / f"{base}.stl"
            brep_dst = selected_brep_dir / f"{base}.{args.brep_ext}"

            shutil.copy2(py_src, py_dst)
            if mesh_src.exists():
                shutil.copy2(mesh_src, mesh_dst)
            if args.export_brep and brep_src.exists():
                shutil.copy2(brep_src, brep_dst)

            selected_rows.append(
                {
                    "cadrille_stem": base,
                    "candidate_stem": candidate_stem,
                    "selected_py": str(py_dst),
                    "selected_mesh": str(mesh_dst) if mesh_src.exists() else None,
                    "selected_brep": str(brep_dst) if (args.export_brep and brep_src.exists()) else None,
                    "status": "ok",
                }
            )

    summary = {
        "sam3d": {
            "skip_sam3d": bool(args.skip_sam3d),
            "sam3d_output_root": str(sam3d_output_root),
            "records_found_ok": len(records),
            "records_selected_for_bridge": len(selected),
        },
        "bridge": {
            "normalize_stl": bool(args.normalize_stl),
            "sample_offset": args.sample_offset,
            "max_samples": args.max_samples,
            "prepared_count": len(prepared_rows),
            "cadrille_split_name": split_name,
            "cadrille_split_dir": str(split_dir),
            "manifest_jsonl": str(manifest_jsonl),
        },
        "cadrille": {
            "cadrille_root": str(cadrille_root),
            "cadrille_data_root": str(cadrille_data_root),
            "cadrille_mode": args.cadrille_mode,
            "checkpoint": args.cadrille_checkpoint,
            "tmp_py_dir": str(tmp_py_dir),
            "tmp_mesh_dir": str(tmp_mesh_dir),
            "tmp_brep_dir": str(tmp_brep_dir) if args.export_brep else None,
            "selected_candidate_index": args.selected_candidate_index,
            "selected_outputs": {
                "selected_py_dir": str(selected_py_dir),
                "selected_mesh_dir": str(selected_mesh_dir),
                "selected_brep_dir": str(selected_brep_dir) if args.export_brep else None,
            },
        },
        "selected_rows": selected_rows,
    }

    summary_path = cadrille_output_root / "pipeline_summary.json"
    if args.dry_run:
        print("[DRY-RUN] Summary preview:")
        print(json.dumps(summary, ensure_ascii=False, indent=2)[:2500])
    else:
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote summary: {summary_path}")
        print(f"[INFO] Selected CAD outputs: {selected_py_dir}, {selected_mesh_dir}" + (f", {selected_brep_dir}" if args.export_brep else ""))


if __name__ == "__main__":
    main()
