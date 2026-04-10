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
    cad.add_argument("--cadrille-python", default=sys.executable, help="Python executable for Cadrille host runtime")
    cad.add_argument(
        "--cadrille-runtime",
        choices=("auto", "docker", "host"),
        default="auto",
        help="Run Cadrille stage on host Python or inside Docker (default: auto, prefer docker if available)",
    )
    cad.add_argument("--cadrille-docker-image", default="cadrille:latest", help="Docker image for Cadrille runtime")
    cad.add_argument("--cadrille-docker-python", default="python", help="Python executable inside Cadrille Docker image")
    cad.add_argument("--cadrille-docker-gpus", default="all", help="Value for docker --gpus (for example all, 0, \"device=0\")")
    cad.add_argument(
        "--cadrille-docker-extra-args",
        default="",
        help="Extra raw args appended to docker run (for example '--ipc=host --ulimit memlock=-1')",
    )
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
    cad.add_argument(
        "--cadrille-input-source",
        choices=("mesh", "point_cloud", "multi_view"),
        default="mesh",
        help=(
            "Input source passed to Cadrille test.py. "
            "This SAM3D bridge currently materializes mesh (.stl) inputs."
        ),
    )
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
    misc.add_argument(
        "--selection-mode",
        choices=("evaluate", "index"),
        default="evaluate",
        help=(
            "Candidate selection strategy: evaluate.py best_names (paper-aligned) "
            "or fixed index fallback"
        ),
    )
    misc.add_argument("--selected-candidate-index", type=int, default=0,
                      help="Preferred candidate index (+k suffix), used by selection-mode=index or evaluate fallback")
    misc.add_argument(
        "--allow-selection-fallback",
        action="store_true",
        help="When selection-mode=evaluate and best_names is missing for a sample, fallback to --selected-candidate-index",
    )
    misc.add_argument("--eval-gt-path", type=Path, default=None,
                      help="Ground-truth path for evaluate.py (default: prepared Cadrille split)")
    misc.add_argument("--eval-gt-format", choices=("mesh", "point_cloud"), default="mesh",
                      help="Ground-truth format for evaluate.py")
    misc.add_argument("--eval-gt-mesh-ext", default=None,
                      help="Ground-truth mesh extension for evaluate.py (default: --mesh-ext)")
    misc.add_argument("--eval-gt-point-cloud-exts", default=None,
                      help="Ground-truth point-cloud extensions for evaluate.py (default: --point-cloud-exts)")
    misc.add_argument("--eval-n-points", type=int, default=8192,
                      help="Number of sampled points for Chamfer in evaluate.py")
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


def is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def docker_image_exists(image: str) -> bool:
    if not command_exists("docker"):
        return False
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def choose_cadrille_runtime(args: argparse.Namespace) -> str:
    if args.cadrille_runtime in ("docker", "host"):
        if args.cadrille_runtime == "docker":
            if not command_exists("docker"):
                raise RuntimeError("--cadrille-runtime docker requested but 'docker' command is not available")
            if not docker_image_exists(args.cadrille_docker_image):
                raise RuntimeError(
                    f"--cadrille-runtime docker requested but image not found: {args.cadrille_docker_image}"
                )
        return args.cadrille_runtime

    # auto mode
    if command_exists("docker") and docker_image_exists(args.cadrille_docker_image):
        print(f"[INFO] Cadrille runtime auto-selected: docker ({args.cadrille_docker_image})")
        return "docker"

    print("[INFO] Cadrille runtime auto-selected: host (docker image unavailable)")
    return "host"


def build_docker_mounts(
    cadrille_root: Path,
    cadrille_data_root: Path,
    cadrille_output_root: Path,
) -> tuple[list[tuple[Path, Path]], Path, Path, Path]:
    container_cadrille_root = Path("/workspace/cadrille")
    mounts: list[tuple[Path, Path]] = [(cadrille_root, container_cadrille_root)]

    if is_relative_to(cadrille_data_root, cadrille_root):
        container_data_root = container_cadrille_root / cadrille_data_root.relative_to(cadrille_root)
    else:
        container_data_root = Path("/workspace/cadrille_data")
        mounts.append((cadrille_data_root, container_data_root))

    if is_relative_to(cadrille_output_root, cadrille_root):
        container_output_root = container_cadrille_root / cadrille_output_root.relative_to(cadrille_root)
    elif is_relative_to(cadrille_output_root, cadrille_data_root):
        container_output_root = container_data_root / cadrille_output_root.relative_to(cadrille_data_root)
    else:
        container_output_root = Path("/workspace/cadrille_output")
        mounts.append((cadrille_output_root, container_output_root))

    # dedupe mounts while preserving order
    deduped: list[tuple[Path, Path]] = []
    seen: set[tuple[str, str]] = set()
    for host, container in mounts:
        key = (str(host), str(container))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((host, container))

    return deduped, container_cadrille_root, container_data_root, container_output_root


def map_host_to_container(path: Path, mounts: list[tuple[Path, Path]]) -> Path:
    for host_root, container_root in mounts:
        if is_relative_to(path, host_root):
            return container_root / path.relative_to(host_root)
    raise RuntimeError(f"Path {path} is not covered by docker mounts")


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


def split_candidate_stem(candidate_stem: str) -> tuple[str, str] | None:
    if "+" not in candidate_stem:
        return None
    base, idx = candidate_stem.rsplit("+", 1)
    if not base or not idx:
        return None
    return base, idx


def load_best_candidate_map(metrics_path: Path, best_names_path: Path) -> tuple[dict[str, str], dict[str, Any] | None]:
    best_names: list[str] = []
    eval_summary: dict[str, Any] | None = None

    if metrics_path.exists():
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if isinstance(data.get("summary"), dict):
                eval_summary = data["summary"]
            if isinstance(data.get("best_names"), list):
                best_names = [str(v).strip() for v in data["best_names"] if str(v).strip()]

    if not best_names and best_names_path.exists():
        best_names = [line.strip() for line in best_names_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    best_map: dict[str, str] = {}
    for name in best_names:
        stem = Path(name).stem
        parts = split_candidate_stem(stem)
        if parts is None:
            continue
        base, _idx = parts
        best_map[base] = stem

    return best_map, eval_summary


def main() -> None:
    args = parse_args()

    if args.cadrille_mode == "pc" and args.cadrille_input_source == "multi_view":
        raise RuntimeError("Cadrille mode=pc is incompatible with input-source=multi_view")
    if args.cadrille_mode == "img" and args.cadrille_input_source == "point_cloud":
        raise RuntimeError("Cadrille mode=img is incompatible with input-source=point_cloud")
    if args.cadrille_input_source != "mesh":
        raise RuntimeError(
            "Current SAM3D bridge in this script materializes only mesh (.stl) inputs. "
            "Use --cadrille-input-source mesh for e2e, or run Cadrille directly for point_cloud/multi_view datasets."
        )
    if args.eval_n_points <= 0:
        raise RuntimeError("--eval-n-points must be > 0")

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

    cadrille_runtime = choose_cadrille_runtime(args)
    docker_mounts: list[tuple[Path, Path]] = []
    container_cadrille_root: Path | None = None
    container_cadrille_data_root: Path | None = None
    container_cadrille_output_root: Path | None = None
    if cadrille_runtime == "docker":
        (
            docker_mounts,
            container_cadrille_root,
            container_cadrille_data_root,
            container_cadrille_output_root,
        ) = build_docker_mounts(cadrille_root, cadrille_data_root, cadrille_output_root)

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

    eval_gt_host = args.eval_gt_path.resolve() if args.eval_gt_path else split_dir
    eval_gt_mesh_ext = (args.eval_gt_mesh_ext or args.mesh_ext).lower()
    eval_gt_point_cloud_exts = args.eval_gt_point_cloud_exts or args.point_cloud_exts

    # Prepare runtime-specific paths/commands for Cadrille stage
    if cadrille_runtime == "docker":
        assert container_cadrille_root is not None
        assert container_cadrille_data_root is not None
        assert container_cadrille_output_root is not None

        cadrille_data_arg = str(map_host_to_container(cadrille_data_root, docker_mounts))
        tmp_py_arg = str(map_host_to_container(tmp_py_dir, docker_mounts))
        tmp_mesh_arg = str(map_host_to_container(tmp_mesh_dir, docker_mounts))
        tmp_brep_arg = str(map_host_to_container(tmp_brep_dir, docker_mounts))
        eval_gt_arg = str(map_host_to_container(eval_gt_host, docker_mounts))

        checkpoint_arg = args.cadrille_checkpoint
        checkpoint_path = Path(args.cadrille_checkpoint)
        if checkpoint_path.is_absolute():
            checkpoint_arg = str(map_host_to_container(checkpoint_path.resolve(), docker_mounts))

        py_exec = args.cadrille_docker_python

        def run_cadrille_inner(inner_cmd: list[str]) -> None:
            docker_cmd = ["docker", "run", "--rm", "--gpus", args.cadrille_docker_gpus]
            for host_path, container_path in docker_mounts:
                docker_cmd.extend(["-v", f"{host_path}:{container_path}"])
            if args.cadrille_docker_extra_args.strip():
                docker_cmd.extend(shlex.split(args.cadrille_docker_extra_args))
            docker_cmd.extend(["-w", str(container_cadrille_root), args.cadrille_docker_image])
            docker_cmd.extend(inner_cmd)
            run_cmd(docker_cmd, dry_run=args.dry_run)

    else:
        cadrille_data_arg = str(cadrille_data_root)
        tmp_py_arg = str(tmp_py_dir)
        tmp_mesh_arg = str(tmp_mesh_dir)
        tmp_brep_arg = str(tmp_brep_dir)
        eval_gt_arg = str(eval_gt_host)
        checkpoint_arg = args.cadrille_checkpoint
        py_exec = args.cadrille_python

        def run_cadrille_inner(inner_cmd: list[str]) -> None:
            run_cmd(inner_cmd, cwd=cadrille_root, dry_run=args.dry_run)

    # Run Cadrille inference
    test_cmd = [
        py_exec,
        "test.py",
        "--data-path",
        cadrille_data_arg,
        "--split",
        split_name,
        "--mode",
        args.cadrille_mode,
        "--checkpoint-path",
        checkpoint_arg,
        "--py-path",
        tmp_py_arg,
        "--input-source",
        args.cadrille_input_source,
        "--mesh-ext",
        args.mesh_ext,
        "--point-cloud-exts",
        args.point_cloud_exts,
        "--image-exts",
        args.image_exts,
    ]
    run_cadrille_inner(test_cmd)

    # Convert CadQuery outputs to CAD meshes/BRep
    convert_cmd = [
        py_exec,
        "convert_cadquery.py",
        "--src",
        tmp_py_arg,
        "--mesh-out",
        tmp_mesh_arg,
        "--timeout",
        str(args.convert_timeout_sec),
    ]
    if args.export_brep:
        convert_cmd.extend([
            "--export-brep",
            "--brep-out",
            tmp_brep_arg,
            "--brep-ext",
            args.brep_ext,
        ])
    run_cadrille_inner(convert_cmd)

    metrics_path = cadrille_output_root / "metrics.json"
    best_names_path = cadrille_output_root / "tmp.txt"
    best_candidate_map: dict[str, str] = {}
    evaluate_summary: dict[str, Any] | None = None
    if args.selection_mode == "evaluate":
        evaluate_cmd = [
            py_exec,
            "evaluate.py",
            "--gt-path",
            eval_gt_arg,
            "--gt-format",
            args.eval_gt_format,
            "--gt-point-cloud-exts",
            eval_gt_point_cloud_exts,
            "--gt-mesh-ext",
            eval_gt_mesh_ext,
            "--pred-py-path",
            tmp_py_arg,
            "--n-points",
            str(args.eval_n_points),
        ]
        run_cadrille_inner(evaluate_cmd)
        if not args.dry_run:
            best_candidate_map, evaluate_summary = load_best_candidate_map(metrics_path, best_names_path)
            print(f"[INFO] evaluate.py selected best candidates for {len(best_candidate_map)} samples")

    # Select one candidate per input sample
    selected_rows: list[dict[str, Any]] = []
    if not args.dry_run:
        selected_py_dir.mkdir(parents=True, exist_ok=True)
        selected_mesh_dir.mkdir(parents=True, exist_ok=True)
        if args.export_brep:
            selected_brep_dir.mkdir(parents=True, exist_ok=True)

        for item in prepared_rows:
            base = item["cadrille_stem"]
            candidate_stem: str | None = None
            selection_reason: str | None = None

            if args.selection_mode == "evaluate":
                candidate_stem = best_candidate_map.get(base)
                if candidate_stem is not None:
                    selection_reason = "evaluate_best"
                elif args.allow_selection_fallback:
                    candidate_stem = pick_candidate_stem(tmp_py_dir, base, args.selected_candidate_index)
                    selection_reason = "evaluate_fallback_index"
            else:
                candidate_stem = pick_candidate_stem(tmp_py_dir, base, args.selected_candidate_index)
                if candidate_stem is not None:
                    selection_reason = "fixed_index"

            if candidate_stem is None:
                selected_rows.append({
                    "cadrille_stem": base,
                    "status": "missing_candidate",
                    "selection_reason": selection_reason,
                })
                continue

            py_src = tmp_py_dir / f"{candidate_stem}.py"
            if not py_src.exists():
                selected_rows.append({
                    "cadrille_stem": base,
                    "candidate_stem": candidate_stem,
                    "status": "missing_py",
                    "selection_reason": selection_reason,
                })
                continue

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
                    "selection_reason": selection_reason,
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
            "runtime": cadrille_runtime,
            "cadrille_root": str(cadrille_root),
            "cadrille_data_root": str(cadrille_data_root),
            "cadrille_mode": args.cadrille_mode,
            "cadrille_input_source": args.cadrille_input_source,
            "checkpoint": args.cadrille_checkpoint,
            "host_python": args.cadrille_python,
            "docker_python": args.cadrille_docker_python,
            "docker_image": args.cadrille_docker_image if cadrille_runtime == "docker" else None,
            "docker_gpus": args.cadrille_docker_gpus if cadrille_runtime == "docker" else None,
            "docker_mounts": (
                [{"host": str(h), "container": str(c)} for h, c in docker_mounts]
                if cadrille_runtime == "docker"
                else None
            ),
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
        "selection": {
            "selection_mode": args.selection_mode,
            "selected_candidate_index": args.selected_candidate_index,
            "allow_selection_fallback": bool(args.allow_selection_fallback),
            "best_candidate_count": len(best_candidate_map),
            "evaluate": {
                "gt_path": str(eval_gt_host),
                "gt_format": args.eval_gt_format,
                "gt_mesh_ext": eval_gt_mesh_ext,
                "gt_point_cloud_exts": eval_gt_point_cloud_exts,
                "n_points": args.eval_n_points,
                "metrics_path": str(metrics_path),
                "best_names_path": str(best_names_path),
                "summary": evaluate_summary,
            } if args.selection_mode == "evaluate" else None,
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
