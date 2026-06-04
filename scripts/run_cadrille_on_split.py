#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Cadrille inference, evaluation, and candidate selection on an already-prepared split. "
            "This is the prepared-split runner used by the e2e and batch entrypoints."
        )
    )

    split = parser.add_argument_group("Input split")
    split.add_argument("--prepared-split-name", required=True, help="Prepared split name")
    split.add_argument("--prepared-split-dir", type=Path, default=None, help="Optional explicit directory for the prepared split")
    split.add_argument("--bridge-manifest-jsonl", type=Path, default=None, help="Optional manifest describing bridged rows")
    split.add_argument("--bridge-normalized", action="store_true", help="Record that bridge STL normalization was applied")

    cad = parser.add_argument_group("Cadrille stage")
    cad.add_argument("--cadrille-python", default="python", help="Python executable for Cadrille host runtime")
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
    cad.add_argument(
        "--cadrille-data-root",
        type=Path,
        default=None,
        help="Explicit Cadrille data root. Usually inferred automatically from --prepared-split-dir.",
    )
    cad.add_argument("--cadrille-checkpoint", default="ckpt/cadrille_sft", help="Cadrille checkpoint path")
    cad.add_argument("--cadrille-processor-path", default="ckpt/Qwen2-VL-2B-Instruct", help="Processor / tokenizer path for Cadrille")
    cad.add_argument("--cadrille-mode", choices=("pc", "img"), required=True, help="Cadrille modality to run")
    cad.add_argument("--cadrille-input-source", choices=("mesh", "point_cloud", "multi_view"), default="mesh", help="Input representation expected by the prepared split")
    cad.add_argument("--cadrille-n-samples", type=int, default=None, help="Number of candidates per sample (defaults: pc=5, img=1)")
    cad.add_argument("--cadrille-batch-size", type=int, default=64, help="Batch size used during Cadrille inference")
    cad.add_argument("--cadrille-output-root", type=Path, required=True, help="Output root for inference, evaluation, and selected CAD outputs")
    cad.add_argument("--mesh-ext", default="stl", help="Mesh extension used for prepared split files and mesh evaluation")
    cad.add_argument("--point-cloud-exts", default="ply,pcd,xyz,txt,npz,npy", help="Comma-separated point-cloud extensions for evaluation fallback")
    cad.add_argument("--image-exts", default="png,jpg,jpeg,bmp", help="Comma-separated image extensions reserved for image-based datasets")
    cad.add_argument("--export-brep", dest="export_brep", action="store_true", default=True, help="Export STEP/BRep outputs alongside selected code and mesh outputs")
    cad.add_argument("--no-export-brep", dest="export_brep", action="store_false", help="Disable STEP/BRep export")
    cad.add_argument("--brep-ext", default="step", help="BRep export extension")
    cad.add_argument("--convert-timeout-sec", type=float, default=5.0, help="Timeout for per-candidate CAD conversion during evaluation")

    misc = parser.add_argument_group("Selection and evaluation")
    misc.add_argument("--selection-mode", choices=("evaluate", "index"), default="evaluate", help="How to choose the final candidate per sample")
    misc.add_argument("--selected-candidate-index", type=int, default=0, help="Candidate index used when --selection-mode index is selected")
    misc.add_argument("--allow-selection-fallback", action="store_true", help="Allow fallback to --selected-candidate-index when evaluation cannot choose a best candidate")
    misc.add_argument("--eval-gt-path", type=Path, default=None, help="Optional evaluation ground-truth path (defaults to the prepared split directory)")
    misc.add_argument("--eval-gt-format", choices=("mesh", "point_cloud"), default="mesh", help="Ground-truth format used by evaluation")
    misc.add_argument("--eval-gt-mesh-ext", default=None, help="Ground-truth mesh extension for evaluation (defaults to --mesh-ext)")
    misc.add_argument("--eval-gt-point-cloud-exts", default=None, help="Ground-truth point-cloud extensions for evaluation")
    misc.add_argument("--eval-n-points", type=int, default=8192, help="Number of sampled points used during geometric evaluation")
    misc.add_argument("--dry-run", action="store_true", help="Print planned actions without running inference or evaluation")
    misc.add_argument("--skip-cadrille-inference", action="store_true", help="Reuse existing tmp_py outputs in --cadrille-output-root and skip the inference stage")

    meta = parser.add_argument_group("Internal summary metadata")
    meta.add_argument("--sam3d-output-root", type=Path, default=None, help=argparse.SUPPRESS)
    meta.add_argument("--records-found-ok", type=int, default=None, help=argparse.SUPPRESS)
    meta.add_argument("--records-selected-for-bridge", type=int, default=None, help=argparse.SUPPRESS)
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



def load_prepared_rows(split_dir: Path, manifest_jsonl: Path | None) -> list[dict[str, Any]]:
    if manifest_jsonl is not None and manifest_jsonl.exists():
        rows: list[dict[str, Any]] = []
        with manifest_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        if rows:
            return rows

    rows = []
    for path in sorted(split_dir.glob("*.stl")):
        rows.append(
            {
                "cadrille_stem": path.stem,
                "cadrille_stl_path": str(path),
            }
        )
    return rows


def resolve_split_paths(args: argparse.Namespace) -> tuple[Path, Path, str]:
    if args.prepared_split_dir is not None:
        split_dir = args.prepared_split_dir.resolve()
        split_name = split_dir.name
        if args.prepared_split_name != split_name:
            raise RuntimeError(
                "--prepared-split-name must match the basename of --prepared-split-dir "
                f"({args.prepared_split_name!r} != {split_name!r})"
            )
        if args.cadrille_data_root is None:
            cadrille_data_root = split_dir.parent
        else:
            cadrille_data_root = args.cadrille_data_root.resolve()
            expected = (cadrille_data_root / args.prepared_split_name).resolve()
            if expected != split_dir:
                raise RuntimeError(
                    "Prepared split path is inconsistent with --cadrille-data-root and --prepared-split-name: "
                    f"expected {expected}, got {split_dir}"
                )
        return split_dir, cadrille_data_root, split_name

    cadrille_root = args.cadrille_root.resolve()
    cadrille_data_root = (
        args.cadrille_data_root.resolve()
        if args.cadrille_data_root is not None
        else (cadrille_root / "data").resolve()
    )
    split_dir = (cadrille_data_root / args.prepared_split_name).resolve()
    return split_dir, cadrille_data_root, args.prepared_split_name



def main() -> None:
    args = parse_args()

    if args.cadrille_mode == "pc" and args.cadrille_input_source == "multi_view":
        raise RuntimeError("Cadrille mode=pc is incompatible with input-source=multi_view")
    if args.cadrille_mode == "img" and args.cadrille_input_source == "point_cloud":
        raise RuntimeError("Cadrille mode=img is incompatible with input-source=point_cloud")
    if args.cadrille_input_source != "mesh":
        raise RuntimeError(
            "Current SAM3D bridge in this pipeline materializes only mesh (.stl) inputs. "
            "Use --cadrille-input-source mesh for prepared splits."
        )
    if args.cadrille_n_samples is not None and args.cadrille_n_samples <= 0:
        raise RuntimeError("--cadrille-n-samples must be > 0")
    if args.cadrille_batch_size <= 0:
        raise RuntimeError("--cadrille-batch-size must be > 0")
    if args.eval_n_points <= 0:
        raise RuntimeError("--eval-n-points must be > 0")

    cadrille_n_samples = args.cadrille_n_samples if args.cadrille_n_samples is not None else (1 if args.cadrille_mode == "img" else 5)

    cadrille_root = args.cadrille_root.resolve()
    split_dir, cadrille_data_root, split_name = resolve_split_paths(args)
    cadrille_output_root = args.cadrille_output_root.resolve()
    if not args.dry_run and not split_dir.exists():
        raise RuntimeError(f"Prepared split directory does not exist: {split_dir}")

    bridge_manifest_jsonl = args.bridge_manifest_jsonl.resolve() if args.bridge_manifest_jsonl else None
    prepared_rows = load_prepared_rows(split_dir, bridge_manifest_jsonl)
    if not prepared_rows:
        raise RuntimeError(f"No prepared rows found for split: {split_dir}")

    cadrille_output_root.mkdir(parents=True, exist_ok=True)
    tmp_py_dir = cadrille_output_root / "tmp_py"
    tmp_mesh_dir = cadrille_output_root / "tmp_mesh"
    tmp_brep_dir = cadrille_output_root / "tmp_brep"
    input_points_dir = cadrille_output_root / "input_points"
    input_renders_dir = cadrille_output_root / "input_renders"
    selected_py_dir = cadrille_output_root / "selected_py"
    selected_mesh_dir = cadrille_output_root / "selected_mesh"
    selected_brep_dir = cadrille_output_root / "selected_brep"
    selected_input_points_dir = cadrille_output_root / "selected_input_points"
    selected_input_renders_dir = cadrille_output_root / "selected_input_renders"

    wrapper_scripts_root = Path(__file__).resolve().parent
    cadrille_infer_wrapper_script = wrapper_scripts_root / "cadrille_infer_wrapper.py"
    cadrille_evaluate_wrapper_script = wrapper_scripts_root / "cadrille_evaluate_wrapper.py"
    container_wrapper_scripts_root = Path("/workspace/integration_scripts")

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
        docker_mounts.append((wrapper_scripts_root, container_wrapper_scripts_root))

    eval_gt_host = args.eval_gt_path.resolve() if args.eval_gt_path else split_dir
    eval_gt_mesh_ext = (args.eval_gt_mesh_ext or args.mesh_ext).lower()
    eval_gt_point_cloud_exts = args.eval_gt_point_cloud_exts or args.point_cloud_exts

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

        processor_arg = args.cadrille_processor_path
        processor_path = Path(args.cadrille_processor_path)
        if processor_path.is_absolute():
            processor_arg = str(map_host_to_container(processor_path.resolve(), docker_mounts))

        infer_wrapper_script_arg = str(container_wrapper_scripts_root / cadrille_infer_wrapper_script.name)
        evaluate_wrapper_script_arg = str(container_wrapper_scripts_root / cadrille_evaluate_wrapper_script.name)
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
        processor_arg = args.cadrille_processor_path
        infer_wrapper_script_arg = str(cadrille_infer_wrapper_script)
        evaluate_wrapper_script_arg = str(cadrille_evaluate_wrapper_script)
        py_exec = args.cadrille_python

        def run_cadrille_inner(inner_cmd: list[str]) -> None:
            run_cmd(inner_cmd, cwd=cadrille_root, dry_run=args.dry_run)

    test_cmd = [
        py_exec,
        infer_wrapper_script_arg,
        "--cadrille-root",
        str(container_cadrille_root) if cadrille_runtime == "docker" else str(cadrille_root),
        "--data-path",
        cadrille_data_arg,
        "--split",
        split_name,
        "--mode",
        args.cadrille_mode,
        "--checkpoint-path",
        checkpoint_arg,
        "--processor-path",
        processor_arg,
        "--py-path",
        tmp_py_arg,
        "--n-samples",
        str(cadrille_n_samples),
        "--batch-size",
        str(args.cadrille_batch_size),
    ]
    if args.skip_cadrille_inference:
        if not args.dry_run and not tmp_py_dir.exists():
            raise FileNotFoundError(f"--skip-cadrille-inference requested but tmp_py directory does not exist: {tmp_py_dir}")
        print(f"[INFO] Reusing existing Cadrille tmp_py outputs: {tmp_py_dir}")
    else:
        run_cadrille_inner(test_cmd)

    gpu_memory_path = cadrille_output_root / "gpu_memory.json"
    gpu_memory_summary: dict[str, Any] | None = None
    if not args.dry_run and gpu_memory_path.exists():
        gpu_memory_summary = json.loads(gpu_memory_path.read_text(encoding="utf-8"))

    metrics_path = cadrille_output_root / "metrics.json"
    best_names_path = cadrille_output_root / "tmp.txt"
    best_candidate_map: dict[str, str] = {}
    evaluate_summary: dict[str, Any] | None = None

    evaluate_cmd = [
        py_exec,
        evaluate_wrapper_script_arg,
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
        "--brep-ext",
        args.brep_ext,
        "--convert-timeout-sec",
        str(args.convert_timeout_sec),
    ]
    if args.export_brep:
        evaluate_cmd.append("--export-brep")
    else:
        evaluate_cmd.append("--no-export-brep")
    run_cadrille_inner(evaluate_cmd)
    if not args.dry_run and args.selection_mode == "evaluate":
        best_candidate_map, evaluate_summary = load_best_candidate_map(metrics_path, best_names_path)
        print(f"[INFO] evaluate.py selected best candidates for {len(best_candidate_map)} samples")

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
                selected_rows.append(
                    {
                        "cadrille_stem": base,
                        "status": "missing_candidate",
                        "selection_reason": selection_reason,
                    }
                )
                continue

            py_src = tmp_py_dir / f"{candidate_stem}.py"
            if not py_src.exists():
                selected_rows.append(
                    {
                        "cadrille_stem": base,
                        "candidate_stem": candidate_stem,
                        "status": "missing_py",
                        "selection_reason": selection_reason,
                    }
                )
                continue

            mesh_src = tmp_mesh_dir / f"{candidate_stem}.stl"
            brep_src = tmp_brep_dir / f"{candidate_stem}.{args.brep_ext}"
            input_points_src = input_points_dir / f"{candidate_stem}.json"
            input_render_src = input_renders_dir / f"{candidate_stem}.png"
            input_render_meta_src = input_renders_dir / f"{candidate_stem}.json"

            py_dst = selected_py_dir / f"{base}.py"
            mesh_dst = selected_mesh_dir / f"{base}.stl"
            brep_dst = selected_brep_dir / f"{base}.{args.brep_ext}"
            input_points_dst = selected_input_points_dir / f"{base}.json"
            input_render_dst = selected_input_renders_dir / f"{base}.png"
            input_render_meta_dst = selected_input_renders_dir / f"{base}.json"

            shutil.copy2(py_src, py_dst)
            if mesh_src.exists():
                shutil.copy2(mesh_src, mesh_dst)
            if args.export_brep and brep_src.exists():
                shutil.copy2(brep_src, brep_dst)
            if input_points_src.exists():
                selected_input_points_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(input_points_src, input_points_dst)
            if input_render_src.exists():
                selected_input_renders_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(input_render_src, input_render_dst)
                if input_render_meta_src.exists():
                    shutil.copy2(input_render_meta_src, input_render_meta_dst)

            selected_rows.append(
                {
                    "cadrille_stem": base,
                    "candidate_stem": candidate_stem,
                    "selection_reason": selection_reason,
                    "selected_py": str(py_dst),
                    "selected_mesh": str(mesh_dst) if mesh_src.exists() else None,
                    "selected_brep": str(brep_dst) if (args.export_brep and brep_src.exists()) else None,
                    "selected_input_points": str(input_points_dst) if input_points_src.exists() else None,
                    "selected_input_render": str(input_render_dst) if input_render_src.exists() else None,
                    "status": "ok",
                }
            )

    summary = {
        "sam3d": {
            "skip_sam3d": True,
            "sam3d_output_root": str(args.sam3d_output_root) if args.sam3d_output_root else None,
            "records_found_ok": args.records_found_ok,
            "records_selected_for_bridge": args.records_selected_for_bridge,
        },
        "bridge": {
            "normalize_stl": bool(args.bridge_normalized),
            "prepared_count": len(prepared_rows),
            "cadrille_split_name": args.prepared_split_name,
            "resolved_split_name": split_name,
            "cadrille_split_dir": str(split_dir),
            "manifest_jsonl": str(bridge_manifest_jsonl) if bridge_manifest_jsonl else None,
        },
        "cadrille": {
            "runtime": cadrille_runtime,
            "cadrille_root": str(cadrille_root),
            "cadrille_data_root": str(cadrille_data_root),
            "cadrille_mode": args.cadrille_mode,
            "cadrille_input_source": args.cadrille_input_source,
            "cadrille_n_samples": cadrille_n_samples,
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
            "input_points_dir": str(input_points_dir),
            "input_renders_dir": str(input_renders_dir),
            "selected_candidate_index": args.selected_candidate_index,
            "gpu_memory_path": str(gpu_memory_path),
            "gpu_memory": gpu_memory_summary,
            "selected_outputs": {
                "selected_py_dir": str(selected_py_dir),
                "selected_mesh_dir": str(selected_mesh_dir),
                "selected_brep_dir": str(selected_brep_dir) if args.export_brep else None,
                "selected_input_points_dir": str(selected_input_points_dir),
                "selected_input_renders_dir": str(selected_input_renders_dir),
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
