#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import sys
import time
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from cadrille_seeding import SeededDataset  # sibling module in scripts/


def cuda_synchronize_all() -> None:
    if not torch.cuda.is_available():
        return
    for device_idx in range(torch.cuda.device_count()):
        try:
            torch.cuda.synchronize(device_idx)
        except Exception:
            pass


def reset_cuda_peak_stats() -> None:
    if not torch.cuda.is_available():
        return
    for device_idx in range(torch.cuda.device_count()):
        try:
            torch.cuda.reset_peak_memory_stats(device_idx)
        except Exception:
            pass


def collect_cuda_memory_stats() -> list[dict[str, object]]:
    stats: list[dict[str, object]] = []
    if not torch.cuda.is_available():
        return stats

    for device_idx in range(torch.cuda.device_count()):
        try:
            torch.cuda.synchronize(device_idx)
        except Exception:
            pass
        props = torch.cuda.get_device_properties(device_idx)
        stats.append(
            {
                "device_index": device_idx,
                "device_name": props.name,
                "total_memory_mb": round(props.total_memory / (1024**2), 2),
                "peak_memory_allocated_mb": round(torch.cuda.max_memory_allocated(device_idx) / (1024**2), 2),
                "peak_memory_reserved_mb": round(torch.cuda.max_memory_reserved(device_idx) / (1024**2), 2),
                "current_memory_allocated_mb": round(torch.cuda.memory_allocated(device_idx) / (1024**2), 2),
                "current_memory_reserved_mb": round(torch.cuda.memory_reserved(device_idx) / (1024**2), 2),
            }
        )
    return stats


def _force_open3d_allow_arbitrary_camera() -> None:
    """open3d 0.18's legacy ViewControl.convert_from_pinhole_camera_parameters
    silently DROPS an off-axis camera extrinsic unless allow_arbitrary=True,
    collapsing cadrille's 4 multi-view (img-mode) renders to a single viewpoint.
    Force the flag on at the open3d layer so upstream cadrille/dataset.py's
    mesh_to_image() stays untouched (no submodule edit). Idempotent."""
    import open3d  # already imported by dataset.py; this just grabs the module

    view_control = open3d.visualization.ViewControl
    original = view_control.convert_from_pinhole_camera_parameters
    if getattr(original, "_aiws_allow_arbitrary", False):
        return

    def convert_from_pinhole_camera_parameters(self, parameters, allow_arbitrary=True):
        return original(self, parameters, allow_arbitrary)

    convert_from_pinhole_camera_parameters._aiws_allow_arbitrary = True
    view_control.convert_from_pinhole_camera_parameters = convert_from_pinhole_camera_parameters


def _percentile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


def summarize_values(values: list[float], *, digits: int) -> dict[str, object]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "std": None,
        }
    vals = sorted(float(v) for v in values)
    n = len(vals)
    avg = sum(vals) / n
    var = sum((x - avg) ** 2 for x in vals) / n
    return {
        "count": n,
        "min": round(vals[0], digits),
        "max": round(vals[-1], digits),
        "mean": round(avg, digits),
        "median": round(_percentile(vals, 0.5), digits),
        "p90": round(_percentile(vals, 0.9), digits),
        "p95": round(_percentile(vals, 0.95), digits),
        "p99": round(_percentile(vals, 0.99), digits),
        "std": round(math.sqrt(var), digits),
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _point_tensor_to_rows(value: Any) -> list[list[float]] | None:
    if not torch.is_tensor(value):
        return None
    points = value.detach().cpu().float()
    if points.ndim != 2:
        return None
    if points.shape[-1] == 3:
        rows = points.tolist()
    elif points.shape[0] == 3:
        rows = points.transpose(0, 1).tolist()
    else:
        return None
    return [[float(x), float(y), float(z)] for x, y, z in rows]


def write_input_points_artifact(
    output_root: Path,
    *,
    mode: str,
    source_stem: str,
    output_file_name: str,
    generation_id: int,
    points: Any,
) -> str | None:
    if mode != "pc":
        return None
    rows = _point_tensor_to_rows(points)
    if not rows:
        return None
    out_dir = output_root / "input_points"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(output_file_name).stem}.json"
    artifact = {
        "mode": mode,
        "source_stem": source_stem,
        "source_candidate": Path(output_file_name).stem,
        "output_file_name": output_file_name,
        "generation_id": generation_id,
        "n_points": len(rows),
        "points": rows,
        "note": "Saved from batch['point_clouds'] immediately before Cadrille generate().",
    }
    out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)


def write_input_render_artifact(
    output_root: Path,
    *,
    mode: str,
    source_stem: str,
    output_file_name: str,
    generation_id: int,
    video: Any,
) -> str | None:
    if mode != "img":
        return None
    if isinstance(video, (list, tuple)):
        image = video[0] if video else None
        n_frames = len(video)
    else:
        image = video
        n_frames = 1 if video is not None else 0
    if image is None or not hasattr(image, "save"):
        return None

    out_dir = output_root / "input_renders"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(output_file_name).stem
    out_path = out_dir / f"{stem}.png"
    meta_path = out_dir / f"{stem}.json"
    image.save(out_path)
    artifact = {
        "mode": mode,
        "source_stem": source_stem,
        "source_candidate": stem,
        "output_file_name": output_file_name,
        "generation_id": generation_id,
        "n_frames": n_frames,
        "image_size": list(getattr(image, "size", ())),
        "image_mode": getattr(image, "mode", None),
        "note": "Saved from batch['input_videos'] immediately before Cadrille generate().",
    }
    meta_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)


def collate_with_input_artifacts(
    batch: list[dict[str, Any]],
    *,
    upstream_collate: Any,
    processor: Any,
    n_points: int,
    eval: bool = False,
) -> Any:
    inputs = upstream_collate(batch, processor=processor, n_points=n_points, eval=eval)
    inputs["input_videos"] = [m.get("video") for m in batch]
    return inputs


def write_gpu_memory_report(
    output_root: Path,
    *,
    split: str,
    mode: str,
    checkpoint_path: str,
    processor_path: str,
    batch_size: int | None,
    n_samples: int,
    seed: int,
    num_workers: int | None,
    dataset_size: int | None,
    batches_processed: int,
    generated_file_count: int,
    started_at_epoch: float,
    finished_at_epoch: float,
    batch_trace_rows: list[dict[str, object]],
    sample_trace_rows: list[dict[str, object]],
    error: dict[str, object] | None = None,
) -> None:
    output_path = output_root / "gpu_memory.json"
    batch_trace_path = output_root / "gpu_memory_batches.jsonl"
    sample_trace_path = output_root / "gpu_memory_samples.jsonl"

    device_stats = collect_cuda_memory_stats()
    per_batch_runtime_sec = [float(row["batch_duration_sec"]) for row in batch_trace_rows if row.get("batch_duration_sec") is not None]
    per_sample_runtime_sec = [float(row["estimated_runtime_sec"]) for row in sample_trace_rows if row.get("estimated_runtime_sec") is not None]
    per_sample_peak_allocated_mb = [float(row["batch_peak_memory_allocated_mb"]) for row in sample_trace_rows if row.get("batch_peak_memory_allocated_mb") is not None]
    per_sample_peak_reserved_mb = [float(row["batch_peak_memory_reserved_mb"]) for row in sample_trace_rows if row.get("batch_peak_memory_reserved_mb") is not None]

    if batch_trace_rows:
        write_jsonl(batch_trace_path, batch_trace_rows)
    if sample_trace_rows:
        write_jsonl(sample_trace_path, sample_trace_rows)

    report: dict[str, object] = {
        "split": split,
        "mode": mode,
        "checkpoint_path": checkpoint_path,
        "processor_path": processor_path,
        "batch_size": batch_size,
        "n_samples": n_samples,
        "seed": seed,
        "num_workers": num_workers,
        "dataset_size": dataset_size,
        "batches_processed": batches_processed,
        "generated_file_count": generated_file_count,
        "started_at_epoch": started_at_epoch,
        "finished_at_epoch": finished_at_epoch,
        "duration_sec": round(finished_at_epoch - started_at_epoch, 3),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "devices": device_stats,
        "measurement_note": (
            "Per-sample runtime and memory summaries are batch-derived: "
            "batch wall-clock generation time is divided evenly across samples in that batch, "
            "and each sample inherits that batch peak GPU memory."
        ),
        "batch_trace_jsonl": str(batch_trace_path),
        "sample_trace_jsonl": str(sample_trace_path),
        "per_batch_runtime_sec": summarize_values(per_batch_runtime_sec, digits=6),
        "per_sample_runtime_sec": summarize_values(per_sample_runtime_sec, digits=6),
        "per_sample_peak_memory_allocated_mb": summarize_values(per_sample_peak_allocated_mb, digits=2),
        "per_sample_peak_memory_reserved_mb": summarize_values(per_sample_peak_reserved_mb, digits=2),
    }
    if device_stats:
        report["peak_memory_allocated_mb_max"] = round(
            max(d["peak_memory_allocated_mb"] for d in device_stats), 2
        )
        report["peak_memory_reserved_mb_max"] = round(
            max(d["peak_memory_reserved_mb"] for d in device_stats), 2
        )
    if error is not None:
        report["error"] = error
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[gpu-memory] wrote {output_path}")
    if batch_trace_rows:
        print(f"[gpu-memory] wrote {batch_trace_path}")
    if sample_trace_rows:
        print(f"[gpu-memory] wrote {sample_trace_path}")


def _load_kwargs(path_or_id: str) -> dict[str, Any]:
    return {"local_files_only": True} if Path(path_or_id).exists() else {}


def run(
    *,
    cadrille_root: Path,
    data_path: Path,
    split: str,
    mode: str,
    checkpoint_path: str,
    processor_path: str,
    py_path: Path,
    n_samples: int,
    batch_size_override: int | None,
    seed: int,
) -> None:
    if mode not in {"pc", "img"}:
        raise ValueError("AIWS Cadrille wrapper supports only mode=pc or mode=img")

    cadrille_root = cadrille_root.resolve()
    if str(cadrille_root) not in sys.path:
        sys.path.insert(0, str(cadrille_root))

    from cadrille import Cadrille, collate  # noqa: E402
    from dataset import CadRecodeDataset  # noqa: E402

    # img mode renders multi-view sheets via open3d; force the camera fix without
    # editing the pinned upstream cadrille/dataset.py (see helper docstring).
    if mode == "img":
        _force_open3d_allow_arbitrary_camera()

    py_path = py_path.resolve()
    py_path.mkdir(parents=True, exist_ok=True)
    if any(py_path.iterdir()):
        raise RuntimeError(f"py-path must be empty before inference: {py_path}")

    output_root = py_path.parent
    dataset = None
    batch_size = None
    num_workers = None
    generated_count = 0
    batches_processed = 0
    error = None
    batch_trace_rows: list[dict[str, object]] = []
    sample_trace_rows: list[dict[str, object]] = []
    started_at_epoch = time.time()

    reset_cuda_peak_stats()

    try:
        model = Cadrille.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            **_load_kwargs(checkpoint_path),
        )

        processor = AutoProcessor.from_pretrained(
            processor_path,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
            padding_side="left",
            **_load_kwargs(processor_path),
        )

        dataset = CadRecodeDataset(
            root_dir=str(data_path),
            split=split,
            n_points=256,
            normalize_std_pc=100,
            noise_scale_pc=None,
            img_size=128,
            normalize_std_img=200,
            noise_scale_img=-1,
            num_imgs=4,
            mode=mode,
        )
        batch_size = 256

        if n_samples <= 0:
            raise ValueError("n_samples must be > 0")
        if batch_size_override is not None:
            batch_size = int(batch_size_override)
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        num_workers = 16
        dataloader = DataLoader(
            dataset=SeededDataset(ConcatDataset([dataset] * n_samples), seed=seed),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=partial(
                collate_with_input_artifacts,
                upstream_collate=collate,
                processor=processor,
                n_points=256,
                eval=True,
            ),
        )

        for batch_idx, batch in enumerate(tqdm(dataloader), start=1):
            batch_file_names = [str(v) for v in batch["file_name"]]
            batch_item_count = len(batch_file_names)
            batch_point_clouds = batch.get("point_clouds")
            batch_input_videos = batch.get("input_videos") or []
            cuda_synchronize_all()
            reset_cuda_peak_stats()
            batch_started = time.perf_counter()
            generated_ids = model.generate(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                point_clouds=batch["point_clouds"].to(model.device),
                is_pc=batch["is_pc"].to(model.device),
                is_img=batch["is_img"].to(model.device),
                pixel_values_videos=batch["pixel_values_videos"].to(model.device)
                if batch.get("pixel_values_videos") is not None
                else None,
                video_grid_thw=batch["video_grid_thw"].to(model.device)
                if batch.get("video_grid_thw") is not None
                else None,
                max_new_tokens=768,
            )
            cuda_synchronize_all()
            batch_finished = time.perf_counter()
            batch_duration_sec = batch_finished - batch_started
            batch_device_stats = collect_cuda_memory_stats()
            batch_peak_allocated_mb = (
                round(max(d["peak_memory_allocated_mb"] for d in batch_device_stats), 2)
                if batch_device_stats
                else None
            )
            batch_peak_reserved_mb = (
                round(max(d["peak_memory_reserved_mb"] for d in batch_device_stats), 2)
                if batch_device_stats
                else None
            )
            estimated_runtime_sec = (batch_duration_sec / batch_item_count) if batch_item_count else None

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch.input_ids, generated_ids)
            ]
            py_strings = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            batch_trace_rows.append(
                {
                    "batch_index": batch_idx,
                    "batch_size_actual": batch_item_count,
                    "batch_duration_sec": round(batch_duration_sec, 6),
                    "estimated_runtime_sec_per_sample": round(estimated_runtime_sec, 6) if estimated_runtime_sec is not None else None,
                    "batch_peak_memory_allocated_mb": batch_peak_allocated_mb,
                    "batch_peak_memory_reserved_mb": batch_peak_reserved_mb,
                    "generated_count_before_batch": generated_count,
                    "file_names": batch_file_names,
                    "device_stats": batch_device_stats,
                }
            )

            for sample_idx, (stem, py_string) in enumerate(zip(batch_file_names, py_strings)):
                generation_id = generated_count // len(dataset)
                file_name = f"{stem}+{generation_id}.py"
                (py_path / file_name).write_text(py_string, encoding="utf-8")
                input_points_path = None
                input_render_path = None
                if torch.is_tensor(batch_point_clouds) and sample_idx < batch_point_clouds.shape[0]:
                    input_points_path = write_input_points_artifact(
                        output_root,
                        mode=mode,
                        source_stem=stem,
                        output_file_name=file_name,
                        generation_id=generation_id,
                        points=batch_point_clouds[sample_idx],
                    )
                if sample_idx < len(batch_input_videos):
                    input_render_path = write_input_render_artifact(
                        output_root,
                        mode=mode,
                        source_stem=stem,
                        output_file_name=file_name,
                        generation_id=generation_id,
                        video=batch_input_videos[sample_idx],
                    )
                sample_trace_rows.append(
                    {
                        "batch_index": batch_idx,
                        "source_stem": stem,
                        "output_file_name": file_name,
                        "generation_id": generation_id,
                        "input_points_path": input_points_path,
                        "input_render_path": input_render_path,
                        "estimated_runtime_sec": round(estimated_runtime_sec, 6) if estimated_runtime_sec is not None else None,
                        "batch_peak_memory_allocated_mb": batch_peak_allocated_mb,
                        "batch_peak_memory_reserved_mb": batch_peak_reserved_mb,
                    }
                )
                generated_count += 1
            batches_processed += 1
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        write_gpu_memory_report(
            output_root,
            split=split,
            mode=mode,
            checkpoint_path=checkpoint_path,
            processor_path=processor_path,
            batch_size=batch_size,
            n_samples=n_samples,
            seed=seed,
            num_workers=num_workers,
            dataset_size=(len(dataset) if dataset is not None else None),
            batches_processed=batches_processed,
            generated_file_count=generated_count,
            started_at_epoch=started_at_epoch,
            finished_at_epoch=time.time(),
            batch_trace_rows=batch_trace_rows,
            sample_trace_rows=sample_trace_rows,
            error=error,
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Thin wrapper for running official Cadrille with path/HF-id override and GPU-memory logging")
    parser.add_argument("--cadrille-root", type=Path, default=Path("."), help="Official Cadrille repo root")
    parser.add_argument("--data-path", type=Path, default=Path("./data"))
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["pc", "img"], required=True)
    parser.add_argument("--checkpoint-path", type=str, default="ckpt/cadrille_sft")
    parser.add_argument("--processor-path", type=str, default="ckpt/Qwen2-VL-2B-Instruct")
    parser.add_argument("--py-path", type=Path, default=Path("./work_dirs/tmp_py"))
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Base seed for deterministic point-cloud sampling")
    args = parser.parse_args()

    run(
        cadrille_root=args.cadrille_root,
        data_path=args.data_path,
        split=args.split,
        mode=args.mode,
        checkpoint_path=args.checkpoint_path,
        processor_path=args.processor_path,
        py_path=args.py_path,
        n_samples=args.n_samples,
        batch_size_override=args.batch_size,
        seed=args.seed,
    )
