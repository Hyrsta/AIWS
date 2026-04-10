#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM3D on all AIWS5.2 split-materialized samples")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root of aiws5.2-usable-split-materialized")
    parser.add_argument("--repo-root", type=Path, required=True, help="Root of sam-3d-objects repo")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory for batch outputs and logs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of instances")
    parser.add_argument("--resume", action="store_true", help="Skip instances with an existing splat.ply")
    return parser.parse_args()


@dataclass
class Task:
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
        return f"{self.split}/{self.subset}/{self.workpiece}/{self.stem}__obj{self.object_index:02d}"


def load_tasks(dataset_root: Path) -> list[Task]:
    tasks: list[Task] = []
    for split in ("train", "val"):
        for ann_path in sorted(dataset_root.glob(f"{split}/*/*/annotations/*.json")):
            subset = ann_path.parts[-4]
            workpiece = ann_path.parts[-3]
            stem = ann_path.stem
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


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(dataset_root)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    manifest_path = output_root / "manifest.csv"
    results_path = output_root / "results.jsonl"
    summary_path = output_root / "summary.json"
    write_manifest(tasks, manifest_path)

    sys.path.insert(0, str(repo_root / "notebook"))
    os.chdir(repo_root)
    from inference import Inference  # type: ignore
    import torch

    patch_torch_hub_for_local_dinov2(torch)

    config_path = repo_root / "checkpoints" / "hf" / "pipeline.yaml"
    inference = Inference(str(config_path), compile=False)

    total = len(tasks)
    done = 0
    skipped = 0
    failed = 0
    started_at = time.time()

    for index, task in enumerate(tasks, start=1):
        task_out_dir = output_root / task.split / task.subset / task.workpiece / f"{task.stem}__obj{task.object_index:02d}"
        ply_path = task_out_dir / "splat.ply"
        tmp_ply_path = task_out_dir / "splat.partial.ply"
        meta_path = task_out_dir / "meta.json"

        if args.resume and ply_path.exists() and ply_path.stat().st_size > 0:
            skipped += 1
            print(f"[{index}/{total}] skip {task.task_id}", flush=True)
            continue

        task_out_dir.mkdir(parents=True, exist_ok=True)
        task_started = time.time()
        record = {
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
            "ply_path": str(ply_path),
            "category": task.category,
            "group": task.group,
            "bbox": task.bbox,
            "area": task.area,
            "width": task.width,
            "height": task.height,
            "seed": args.seed,
            "status": "started",
        }

        try:
            image = Image.open(task.image_path).convert("RGB")
            image_np = np.array(image)
            mask_np = build_mask(task)
            output = inference(image_np, mask_np, seed=args.seed)
            if tmp_ply_path.exists():
                tmp_ply_path.unlink()
            output["gs"].save_ply(str(tmp_ply_path))
            tmp_ply_path.replace(ply_path)
            duration = time.time() - task_started
            record.update(
                {
                    "status": "ok",
                    "duration_sec": round(duration, 3),
                    "ply_size_bytes": ply_path.stat().st_size if ply_path.exists() else None,
                }
            )
            meta_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            append_jsonl(results_path, record)
            done += 1
            print(f"[{index}/{total}] ok {task.task_id} ({duration:.1f}s)", flush=True)
            del output
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            duration = time.time() - task_started
            record.update(
                {
                    "status": "error",
                    "duration_sec": round(duration, 3),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            if tmp_ply_path.exists():
                tmp_ply_path.unlink()
            meta_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            append_jsonl(results_path, record)
            failed += 1
            print(f"[{index}/{total}] error {task.task_id}: {exc}", flush=True)
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

        summary = {
            "dataset_root": str(dataset_root),
            "repo_root": str(repo_root),
            "output_root": str(output_root),
            "manifest_path": str(manifest_path),
            "results_path": str(results_path),
            "total_tasks": total,
            "completed_ok": done,
            "skipped": skipped,
            "failed": failed,
            "started_at_epoch": started_at,
            "elapsed_sec": round(time.time() - started_at, 3),
            "last_task": task.task_id,
            "resume": bool(args.resume),
            "seed": args.seed,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
