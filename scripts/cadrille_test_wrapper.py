#!/usr/bin/env python3
from __future__ import annotations

import json
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


def write_gpu_memory_report(
    output_root: Path,
    *,
    split: str,
    mode: str,
    checkpoint_path: str,
    processor_path: str,
    batch_size: int | None,
    n_samples: int,
    num_workers: int | None,
    dataset_size: int | None,
    batches_processed: int,
    generated_file_count: int,
    started_at_epoch: float,
    finished_at_epoch: float,
    error: dict[str, object] | None = None,
) -> None:
    output_path = output_root / "gpu_memory.json"
    device_stats = collect_cuda_memory_stats()
    report: dict[str, object] = {
        "split": split,
        "mode": mode,
        "checkpoint_path": checkpoint_path,
        "processor_path": processor_path,
        "batch_size": batch_size,
        "n_samples": n_samples,
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
) -> None:
    if mode not in {"pc", "img"}:
        raise ValueError("AIWS Cadrille wrapper supports only mode=pc or mode=img")

    cadrille_root = cadrille_root.resolve()
    if str(cadrille_root) not in sys.path:
        sys.path.insert(0, str(cadrille_root))

    from cadrille import Cadrille, collate  # noqa: E402
    from dataset import CadRecodeDataset  # noqa: E402

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
            dataset=ConcatDataset([dataset] * n_samples),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=partial(collate, processor=processor, n_points=256, eval=True),
        )

        for batch in tqdm(dataloader):
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
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch.input_ids, generated_ids)
            ]
            py_strings = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for stem, py_string in zip(batch["file_name"], py_strings):
                generation_id = generated_count // len(dataset)
                file_name = f"{stem}+{generation_id}.py"
                (py_path / file_name).write_text(py_string, encoding="utf-8")
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
            num_workers=num_workers,
            dataset_size=(len(dataset) if dataset is not None else None),
            batches_processed=batches_processed,
            generated_file_count=generated_count,
            started_at_epoch=started_at_epoch,
            finished_at_epoch=time.time(),
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
    )
