#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import trimesh


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



def prepare_cadrille_split(
    records: list[dict[str, Any]],
    *,
    split_dir: Path,
    normalize_stl: bool,
    dry_run: bool,
) -> list[dict[str, Any]]:
    prepared_rows: list[dict[str, Any]] = []
    used_stems: dict[str, int] = {}

    for row in records:
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

        if dry_run:
            print(f"[DRY-RUN] prepare {src} -> {dst}")
        else:
            if normalize_stl:
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
                "normalized": bool(normalize_stl),
                "normalization": norm_meta,
            }
        )

    if not prepared_rows:
        raise RuntimeError("No STL files were prepared for Cadrille")
    return prepared_rows



def write_manifest_jsonl(path: Path, rows: list[dict[str, Any]], dry_run: bool = False) -> None:
    if dry_run:
        print(f"[DRY-RUN] write manifest: {path} ({len(rows)} rows)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
