#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


SUBSETS = ("V1", "V2", "NEW")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM3D on all AIWS5.2 split-materialized samples")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root of aiws5.2-usable-split-materialized")
    parser.add_argument("--repo-root", type=Path, required=True, help="Root of sam-3d-objects repo")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory for batch outputs and logs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of instances")
    parser.add_argument("--resume", action="store_true", help="Skip instances with existing mesh.glb and mesh.stl outputs")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards for parallel multi-GPU runs")
    parser.add_argument("--shard-index", type=int, default=0, help="0-based shard index for this worker")
    parser.add_argument(
        "--dataset-layout",
        choices=("auto", "split", "subset"),
        default="auto",
        help="Dataset directory layout: split=train/val/subset/workpiece or subset=V1/V2/NEW/workpiece",
    )
    parser.add_argument(
        "--exclude-stems-file",
        type=Path,
        default=None,
        help="Optional newline-delimited file of image stems to exclude entirely (for example multi-instance images)",
    )
    return parser.parse_args()


@dataclass
class Task:
    global_index: int
    split: str
    subset: str
    workpiece: str
    stem: str
    image_path: str
    annotation_path: str
    object_index: int
    object_count_in_image: int
    category: str
    group: Any
    bbox: list[float]
    area: float | None
    width: int
    height: int
    polygon: list[list[float]]

    @property
    def task_id(self) -> str:
        if self.split == "all":
            return f"{self.subset}/{self.workpiece}/{self.stem}__obj{self.object_index:02d}"
        return f"{self.split}/{self.subset}/{self.workpiece}/{self.stem}__obj{self.object_index:02d}"


def detect_dataset_layout(dataset_root: Path) -> str:
    if (dataset_root / "train").is_dir() or (dataset_root / "val").is_dir():
        return "split"
    if any((dataset_root / subset).is_dir() for subset in SUBSETS):
        return "subset"
    raise ValueError(f"Could not infer dataset layout under {dataset_root}")


def iter_annotation_paths(dataset_root: Path, dataset_layout: str):
    if dataset_layout == "split":
        for split in ("train", "val"):
            for ann_path in sorted(dataset_root.glob(f"{split}/*/*/annotations/*.json")):
                yield split, ann_path
        return

    if dataset_layout == "subset":
        for subset in SUBSETS:
            for ann_path in sorted((dataset_root / subset).glob("*/annotations/*.json")):
                yield "all", ann_path
        return

    raise ValueError(f"Unsupported dataset layout: {dataset_layout}")


def load_tasks(dataset_root: Path, dataset_layout: str = "auto", exclude_stems: set[str] | None = None) -> list[Task]:
    exclude_stems = exclude_stems or set()
    if dataset_layout == "auto":
        dataset_layout = detect_dataset_layout(dataset_root)

    tasks: list[Task] = []
    for split, ann_path in iter_annotation_paths(dataset_root, dataset_layout):
        subset = ann_path.parts[-4]
        workpiece = ann_path.parts[-3]
        stem = ann_path.stem
        if stem in exclude_stems:
            continue
        image_path = ann_path.parent.parent / "images" / f"{stem}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for annotation: {ann_path}")

        data = json.loads(ann_path.read_text(encoding="utf-8"))
        info = data.get("info", {})
        width = int(info["width"])
        height = int(info["height"])
        objects = data.get("objects", [])
        if not objects:
            raise ValueError(f"No objects found in {ann_path}")

        for idx, obj in enumerate(objects, start=1):
            polygon = obj.get("segmentation") or []
            if not polygon:
                raise ValueError(f"Missing segmentation for {ann_path} object {idx}")
            tasks.append(
                Task(
                    global_index=len(tasks),
                    split=split,
                    subset=subset,
                    workpiece=workpiece,
                    stem=stem,
                    image_path=str(image_path),
                    annotation_path=str(ann_path),
                    object_index=idx,
                    object_count_in_image=len(objects),
                    category=str(obj.get("category", "")),
                    group=obj.get("group"),
                    bbox=[float(x) for x in (obj.get("bbox") or [])],
                    area=float(obj["area"]) if obj.get("area") is not None else None,
                    width=width,
                    height=height,
                    polygon=[[float(x), float(y)] for x, y in polygon],
                )
            )
    return tasks


def write_manifest(tasks: list[Task], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "global_index",
                "task_id",
                "split",
                "subset",
                "workpiece",
                "stem",
                "image_path",
                "annotation_path",
                "object_index",
                "object_count_in_image",
                "category",
                "group",
                "bbox",
                "area",
                "width",
                "height",
            ],
        )
        writer.writeheader()
        for task in tasks:
            row = asdict(task)
            row.pop("polygon", None)
            row["task_id"] = task.task_id
            row["bbox"] = json.dumps(task.bbox, ensure_ascii=False)
            writer.writerow(row)


def select_shard(tasks: list[Task], num_shards: int, shard_index: int) -> list[Task]:
    if num_shards <= 1:
        return tasks
    return [task for task in tasks if task.global_index % num_shards == shard_index]


def build_mask(task: Task) -> np.ndarray:
    mask = Image.new("L", (task.width, task.height), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon([(float(x), float(y)) for x, y in task.polygon], fill=1, outline=1)
    return np.array(mask, dtype=np.uint8)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def patch_torch_hub_for_local_dinov2(torch: Any) -> None:
    local_repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    if not local_repo.exists():
        return

    original_load = torch.hub.load
    announced = {"done": False}

    def wrapped_load(repo_or_dir: str, model: str, *load_args: Any, **load_kwargs: Any):
        source = load_kwargs.get("source", "github")
        if repo_or_dir == "facebookresearch/dinov2" and source == "github":
            load_kwargs["source"] = "local"
            repo_or_dir = str(local_repo)
            if not announced["done"]:
                print(f"[torch.hub] redirecting facebookresearch/dinov2 to local cache: {local_repo}", flush=True)
                announced["done"] = True
        return original_load(repo_or_dir, model, *load_args, **load_kwargs)

    torch.hub.load = wrapped_load


def get_device_info(torch: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        info.update(
            {
                "torch_cuda_device_index": current_device,
                "gpu_name": torch.cuda.get_device_name(current_device),
            }
        )
    return info


def load_exclude_stems(path: Path | None) -> set[str]:
    if path is None:
        return set()
    text = path.read_text(encoding="utf-8")
    return {line.strip() for line in text.splitlines() if line.strip()}


def task_output_dir(output_root: Path, task: Task) -> Path:
    if task.split == "all":
        return output_root / task.subset / task.workpiece / f"{task.stem}__obj{task.object_index:02d}"
    return output_root / task.split / task.subset / task.workpiece / f"{task.stem}__obj{task.object_index:02d}"


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= shard-index < num-shards")

    dataset_root = args.dataset_root.resolve()
    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    exclude_stems_file = args.exclude_stems_file.resolve() if args.exclude_stems_file else None
    exclude_stems = load_exclude_stems(exclude_stems_file)
    dataset_layout = detect_dataset_layout(dataset_root) if args.dataset_layout == "auto" else args.dataset_layout

    tasks_all = load_tasks(dataset_root, dataset_layout=dataset_layout, exclude_stems=exclude_stems)
    tasks = select_shard(tasks_all, args.num_shards, args.shard_index)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    manifest_path = output_root / "manifest.csv"
    results_path = output_root / "results.jsonl"
    summary_path = output_root / "summary.json"
    write_manifest(tasks, manifest_path)

    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "notebook"))
    os.chdir(repo_root)
    from inference import Inference  # type: ignore
    import torch

    patch_torch_hub_for_local_dinov2(torch)

    config_path = repo_root / "checkpoints" / "hf" / "pipeline.yaml"
    model_init_started = time.time()
    inference = Inference(str(config_path), compile=False)
    model_init_sec = time.time() - model_init_started
    device_info = get_device_info(torch)

    total = len(tasks)
    total_global = len(tasks_all)
    done = 0
    skipped = 0
    failed = 0
    started_at = time.time()
    sum_ok_duration = 0.0

    for index, task in enumerate(tasks, start=1):
        task_out_dir = task_output_dir(output_root, task)
        mesh_path = task_out_dir / "mesh.glb"
        tmp_mesh_path = task_out_dir / "mesh.partial.glb"
        stl_path = task_out_dir / "mesh.stl"
        tmp_stl_path = task_out_dir / "mesh.partial.stl"
        meta_path = task_out_dir / "meta.json"

        if args.resume and mesh_path.exists() and mesh_path.stat().st_size > 0 and stl_path.exists() and stl_path.stat().st_size > 0:
            skipped += 1
            print(f"[{index}/{total}] skip {task.task_id}", flush=True)
            continue

        task_out_dir.mkdir(parents=True, exist_ok=True)
        image_pixels = int(task.width * task.height)
        task_started = time.time()
        record = {
            "global_index": task.global_index,
            "task_index_in_shard": index,
            "total_tasks_in_shard": total,
            "total_tasks_global": total_global,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "task_id": task.task_id,
            "split": task.split,
            "subset": task.subset,
            "workpiece": task.workpiece,
            "stem": task.stem,
            "object_index": task.object_index,
            "object_count_in_image": task.object_count_in_image,
            "image_path": task.image_path,
            "annotation_path": task.annotation_path,
            "output_dir": str(task_out_dir),
            "mesh_path": str(mesh_path),
            "stl_path": str(stl_path),
            "artifact_formats": ["glb", "stl"],
            "category": task.category,
            "group": task.group,
            "bbox": task.bbox,
            "area": task.area,
            "width": task.width,
            "height": task.height,
            "image_pixels": image_pixels,
            "seed": args.seed,
            "dataset_layout": dataset_layout,
            "exclude_stems_file": str(exclude_stems_file) if exclude_stems_file else None,
            "exclude_stems_count": len(exclude_stems),
            "started_at_epoch": task_started,
            "model_init_sec": round(model_init_sec, 3),
            "status": "started",
            **device_info,
        }

        try:
            image = Image.open(task.image_path).convert("RGB")
            image_np = np.array(image)
            mask_np = build_mask(task)
            mask_pixels = int(mask_np.sum())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            output = inference(image_np, mask_np, seed=args.seed)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            mesh = output.get("glb")
            if mesh is None:
                raise ValueError("SAM3D output did not include a mesh/glb artifact")
            if tmp_mesh_path.exists():
                tmp_mesh_path.unlink()
            if tmp_stl_path.exists():
                tmp_stl_path.unlink()
            mesh.export(str(tmp_mesh_path))
            mesh.export(str(tmp_stl_path))
            tmp_mesh_path.replace(mesh_path)
            tmp_stl_path.replace(stl_path)
            duration = time.time() - task_started
            peak_allocated_mb = None
            peak_reserved_mb = None
            if torch.cuda.is_available():
                peak_allocated_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 2)
                peak_reserved_mb = round(torch.cuda.max_memory_reserved() / (1024**2), 2)
            record.update(
                {
                    "status": "ok",
                    "duration_sec": round(duration, 3),
                    "ended_at_epoch": round(time.time(), 3),
                    "mask_pixels": mask_pixels,
                    "mask_fraction": round(mask_pixels / image_pixels, 6) if image_pixels else None,
                    "sec_per_megapixel": round(duration / (image_pixels / 1_000_000), 6) if image_pixels else None,
                    "instances_per_hour": round(3600.0 / duration, 3) if duration > 0 else None,
                    "peak_memory_allocated_mb": peak_allocated_mb,
                    "peak_memory_reserved_mb": peak_reserved_mb,
                    "mesh_size_bytes": mesh_path.stat().st_size if mesh_path.exists() else None,
                    "stl_size_bytes": stl_path.stat().st_size if stl_path.exists() else None,
                }
            )
            meta_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            append_jsonl(results_path, record)
            done += 1
            sum_ok_duration += duration
            print(f"[{index}/{total}] ok {task.task_id} ({duration:.1f}s)", flush=True)
            del output
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            duration = time.time() - task_started
            record.update(
                {
                    "status": "error",
                    "duration_sec": round(duration, 3),
                    "ended_at_epoch": round(time.time(), 3),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            if tmp_mesh_path.exists():
                tmp_mesh_path.unlink()
            if tmp_stl_path.exists():
                tmp_stl_path.unlink()
            meta_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            append_jsonl(results_path, record)
            failed += 1
            print(f"[{index}/{total}] error {task.task_id}: {exc}", flush=True)
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

        elapsed = time.time() - started_at
        processed = done + failed
        avg_ok_duration_sec = round(sum_ok_duration / done, 3) if done else None
        tasks_per_hour = round(done * 3600.0 / elapsed, 3) if done and elapsed > 0 else None
        remaining = total - (done + failed + skipped)
        eta_sec = round((elapsed / processed) * remaining, 3) if processed and remaining > 0 else None
        summary = {
            "dataset_root": str(dataset_root),
            "repo_root": str(repo_root),
            "output_root": str(output_root),
            "manifest_path": str(manifest_path),
            "results_path": str(results_path),
            "total_tasks_in_shard": total,
            "total_tasks_global": total_global,
            "completed_ok": done,
            "skipped": skipped,
            "failed": failed,
            "processed": processed,
            "started_at_epoch": started_at,
            "elapsed_sec": round(elapsed, 3),
            "last_task": task.task_id,
            "resume": bool(args.resume),
            "seed": args.seed,
            "dataset_layout": dataset_layout,
            "exclude_stems_file": str(exclude_stems_file) if exclude_stems_file else None,
            "exclude_stems_count": len(exclude_stems),
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "model_init_sec": round(model_init_sec, 3),
            "avg_ok_duration_sec": avg_ok_duration_sec,
            "ok_instances_per_hour": tasks_per_hour,
            "eta_sec": eta_sec,
            **device_info,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
