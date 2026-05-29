#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_GLOB = "**/pipeline_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Cadrille batch-run metrics (selection success, quality, GPU memory, "
            "and per-group coverage) from pipeline_summary.json/metrics.json files."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True, help="Run root that contains pipeline_summary.json files")
    parser.add_argument(
        "--summary-glob",
        default=DEFAULT_SUMMARY_GLOB,
        help=f"Glob for pipeline summaries under run root (default: {DEFAULT_SUMMARY_GLOB})",
    )
    parser.add_argument("--top-n", type=int, default=15, help="Top-N rows for worst/best sample tables")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to write full analysis JSON")
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory to write CSV tables "
            "(by_modality/by_shard/by_subset/by_workpiece/by_subset_workpiece/top_worst_cd/top_lowest_iou/top_failures)"
        ),
    )
    return parser.parse_args()


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if q <= 0:
        return values[0]
    if q >= 1:
        return values[-1]
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    weight = pos - lo
    return values[lo] * (1 - weight) + values[hi] * weight


def mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def describe(values: list[float]) -> dict[str, Any]:
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
    vals = sorted(values)
    n = len(vals)
    avg = sum(vals) / n
    var = sum((x - avg) ** 2 for x in vals) / n
    return {
        "count": n,
        "min": vals[0],
        "max": vals[-1],
        "mean": avg,
        "median": percentile(vals, 0.5),
        "p90": percentile(vals, 0.9),
        "p95": percentile(vals, 0.95),
        "p99": percentile(vals, 0.99),
        "std": math.sqrt(var),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_source_file"] = str(path)
            row["_source_line"] = line_no
            rows.append(row)
    return rows


def find_summary_files(run_root: Path, summary_glob: str) -> list[Path]:
    return sorted(p for p in run_root.glob(summary_glob) if p.is_file())


def infer_shard(summary_path: Path) -> str:
    parent = summary_path.parent.name
    if parent.startswith("shard-"):
        return parent
    return "single"


def pick_metrics_path(summary_obj: dict[str, Any], summary_path: Path) -> Path | None:
    selection = summary_obj.get("selection") or {}
    evaluate = selection.get("evaluate") or {}
    metrics_path = evaluate.get("metrics_path")
    if metrics_path:
        p = Path(metrics_path)
        if p.exists():
            return p
    fallback = summary_path.parent / "metrics.json"
    return fallback if fallback.exists() else None


def manifest_union_keys(
    manifest_rows: list[dict[str, Any]],
    selected_map: dict[str, dict[str, Any]],
    metrics_map: dict[str, dict[str, Any]],
) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for row in manifest_rows:
        stem = str(row.get("cadrille_stem") or "").strip()
        if stem and stem not in seen:
            keys.append(stem)
            seen.add(stem)
    for stem in list(selected_map.keys()) + list(metrics_map.keys()):
        stem = str(stem).strip()
        if stem and stem not in seen:
            keys.append(stem)
            seen.add(stem)
    return keys


def build_sample_rows(summary_path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    summary_obj = load_json(summary_path)
    modality = str(((summary_obj.get("cadrille") or {}).get("cadrille_mode")) or "unknown")
    shard = infer_shard(summary_path)

    bridge = summary_obj.get("bridge") or {}
    manifest_path = Path(bridge["manifest_jsonl"]) if bridge.get("manifest_jsonl") else None
    manifest_rows = load_jsonl(manifest_path) if manifest_path and manifest_path.exists() else []
    manifest_map = {str(row.get("cadrille_stem")): row for row in manifest_rows if row.get("cadrille_stem")}

    selected_rows = summary_obj.get("selected_rows") or []
    selected_map = {str(row.get("cadrille_stem")): row for row in selected_rows if row.get("cadrille_stem")}

    metrics_path = pick_metrics_path(summary_obj, summary_path)
    metrics_obj = load_json(metrics_path) if metrics_path and metrics_path.exists() else {}
    metrics_map = metrics_obj.get("metrics") or {}
    metrics_summary = metrics_obj.get("summary") or (((summary_obj.get("selection") or {}).get("evaluate") or {}).get("summary")) or {}

    sample_rows: list[dict[str, Any]] = []
    for stem in manifest_union_keys(manifest_rows, selected_map, metrics_map):
        manifest_row = manifest_map.get(stem, {})
        selected_row = selected_map.get(stem, {})
        metric_row = metrics_map.get(stem, {})

        cd_values = [float(v) for v in metric_row.get("cd", []) if to_float(v) is not None]
        iou_values = [float(v) for v in metric_row.get("iou", []) if to_float(v) is not None]
        metric_ids = [str(v) for v in metric_row.get("id", [])]

        status = str(selected_row.get("status") or ("missing_selected_row" if stem not in selected_map else "unknown"))
        best_cd = min(cd_values) if cd_values else None
        best_iou = max(iou_values) if iou_values else None

        sample_rows.append(
            {
                "modality": modality,
                "shard": shard,
                "task_id": manifest_row.get("task_id"),
                "subset": manifest_row.get("subset"),
                "workpiece": manifest_row.get("workpiece"),
                "cadrille_stem": stem,
                "selection_status": status,
                "selection_reason": selected_row.get("selection_reason"),
                "selected_ok": int(status == "ok"),
                "candidate_stem": selected_row.get("candidate_stem"),
                "selected_py": selected_row.get("selected_py"),
                "selected_mesh": selected_row.get("selected_mesh"),
                "selected_brep": selected_row.get("selected_brep"),
                "best_cd": best_cd,
                "best_iou": best_iou,
                "valid_cd_predictions": len(cd_values),
                "valid_iou_predictions": len(iou_values),
                "metric_candidate_ids": ",".join(metric_ids) if metric_ids else None,
                "sam3d_stl_path": manifest_row.get("sam3d_stl_path"),
                "cadrille_stl_path": manifest_row.get("cadrille_stl_path"),
            }
        )

    status_counts = Counter(row["selection_status"] for row in sample_rows)
    ok_count = status_counts.get("ok", 0)
    gpu_memory = (summary_obj.get("cadrille") or {}).get("gpu_memory") or {}

    shard_row = {
        "modality": modality,
        "shard": shard,
        "pipeline_summary_path": str(summary_path),
        "manifest_jsonl": str(manifest_path) if manifest_path else None,
        "metrics_json": str(metrics_path) if metrics_path else None,
        "prepared_count": bridge.get("prepared_count"),
        "records_selected_for_bridge": ((summary_obj.get("sam3d") or {}).get("records_selected_for_bridge")),
        "sample_rows": len(sample_rows),
        "selected_ok": ok_count,
        "selected_non_ok": len(sample_rows) - ok_count,
        "selection_success_rate": (ok_count / len(sample_rows)) if sample_rows else None,
        "best_candidate_count": ((summary_obj.get("selection") or {}).get("best_candidate_count")),
        "invalid_cd": metrics_summary.get("invalid_cd"),
        "invalid_iou": metrics_summary.get("invalid_iou"),
        "mean_iou": metrics_summary.get("mean_iou"),
        "median_cd": metrics_summary.get("median_cd"),
        "gpu_duration_sec": gpu_memory.get("duration_sec"),
        "generated_file_count": gpu_memory.get("generated_file_count"),
        "dataset_size": gpu_memory.get("dataset_size"),
        "batches_processed": gpu_memory.get("batches_processed"),
        "batch_size": gpu_memory.get("batch_size"),
        "peak_memory_allocated_mb_max": gpu_memory.get("peak_memory_allocated_mb_max"),
        "peak_memory_reserved_mb_max": gpu_memory.get("peak_memory_reserved_mb_max"),
        "device_name": ((gpu_memory.get("devices") or [{}])[0]).get("device_name") if gpu_memory.get("devices") else None,
    }

    return summary_obj, shard_row, sample_rows


def summarize_sample_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(row.get("selection_status") or "unknown") for row in rows)
    ok_count = status_counts.get("ok", 0)
    cds = [float(v) for row in rows if (v := row.get("best_cd")) is not None]
    ious = [float(v) for row in rows if (v := row.get("best_iou")) is not None]

    return {
        "records": len(rows),
        "selected_ok": ok_count,
        "selected_non_ok": len(rows) - ok_count,
        "selection_success_rate": (ok_count / len(rows)) if rows else None,
        "missing_candidate": status_counts.get("missing_candidate", 0),
        "missing_py": status_counts.get("missing_py", 0),
        "missing_selected_row": status_counts.get("missing_selected_row", 0),
        "other_non_ok": sum(v for k, v in status_counts.items() if k not in {"ok", "missing_candidate", "missing_py", "missing_selected_row"}),
        "best_cd_valid": len(cds),
        "best_cd_mean": mean(cds),
        "best_cd_median": percentile(sorted(cds), 0.5) if cds else None,
        "best_cd_p90": percentile(sorted(cds), 0.9) if cds else None,
        "best_iou_valid": len(ious),
        "best_iou_mean": mean(ious),
        "best_iou_median": percentile(sorted(ious), 0.5) if ious else None,
        "best_iou_p10": percentile(sorted(ious), 0.1) if ious else None,
    }


def sample_group_breakdown(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(k) for k in keys)].append(row)

    out: list[dict[str, Any]] = []
    for group_key, grp_rows in sorted(grouped.items(), key=lambda kv: tuple("" if v is None else str(v) for v in kv[0])):
        item = {k: v for k, v in zip(keys, group_key)}
        item.update(summarize_sample_group(grp_rows))
        out.append(item)
    return out


def modality_breakdown(sample_rows: list[dict[str, Any]], shard_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sample_groups = sample_group_breakdown(sample_rows, ["modality"])
    shard_by_modality: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in shard_rows:
        shard_by_modality[str(row.get("modality"))].append(row)

    out: list[dict[str, Any]] = []
    for item in sample_groups:
        modality = str(item.get("modality"))
        shards = shard_by_modality.get(modality, [])
        gpu_durations = [float(v) for row in shards if (v := to_float(row.get("gpu_duration_sec"))) is not None]
        peak_alloc = [float(v) for row in shards if (v := to_float(row.get("peak_memory_allocated_mb_max"))) is not None]
        generated = [float(v) for row in shards if (v := to_float(row.get("generated_file_count"))) is not None]
        item.update(
            {
                "shard_runs": len(shards),
                "gpu_duration_total_sec": sum(gpu_durations) if gpu_durations else None,
                "gpu_duration_mean_sec": mean(gpu_durations),
                "peak_memory_allocated_mb_max": max(peak_alloc) if peak_alloc else None,
                "generated_file_count_total": int(sum(generated)) if generated else None,
            }
        )
        out.append(item)
    return out


def top_rows(rows: list[dict[str, Any]], *, key: str, n: int, reverse: bool) -> list[dict[str, Any]]:
    filtered = [row for row in rows if row.get(key) is not None]
    filtered.sort(key=lambda row: row[key], reverse=reverse)
    return filtered[:n]


def round_floats(obj: Any, ndigits: int = 6) -> Any:
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, list):
        return [round_floats(v, ndigits=ndigits) for v in obj]
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits=ndigits) for k, v in obj.items()}
    return obj


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.run_root = args.run_root.resolve()

    summary_files = find_summary_files(args.run_root, args.summary_glob)
    if not summary_files:
        raise RuntimeError(f"No pipeline_summary.json files found under {args.run_root} with glob={args.summary_glob!r}")

    shard_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for summary_path in summary_files:
        _summary_obj, shard_row, shard_sample_rows = build_sample_rows(summary_path)
        shard_rows.append(shard_row)
        sample_rows.extend(shard_sample_rows)

    overall = summarize_sample_group(sample_rows)
    gpu_durations = [float(v) for row in shard_rows if (v := to_float(row.get("gpu_duration_sec"))) is not None]
    peak_alloc = [float(v) for row in shard_rows if (v := to_float(row.get("peak_memory_allocated_mb_max"))) is not None]
    peak_reserved = [float(v) for row in shard_rows if (v := to_float(row.get("peak_memory_reserved_mb_max"))) is not None]

    by_modality = modality_breakdown(sample_rows, shard_rows)
    by_shard = sorted(shard_rows, key=lambda row: (str(row.get("modality")), str(row.get("shard"))))
    by_subset = sample_group_breakdown(sample_rows, ["subset"])
    by_workpiece = sample_group_breakdown(sample_rows, ["workpiece"])
    by_subset_workpiece = sample_group_breakdown(sample_rows, ["subset", "workpiece"])

    top_worst_cd = top_rows(sample_rows, key="best_cd", n=args.top_n, reverse=True)
    top_lowest_iou = top_rows(sample_rows, key="best_iou", n=args.top_n, reverse=False)
    top_failures = [row for row in sample_rows if row.get("selection_status") != "ok"][: args.top_n]

    analysis = {
        "run_root": str(args.run_root),
        "pipeline_summaries_found": len(summary_files),
        "shard_metrics": {
            "gpu_duration_sec": describe(gpu_durations),
            "peak_memory_allocated_mb_max": describe(peak_alloc),
            "peak_memory_reserved_mb_max": describe(peak_reserved),
        },
        "overall": overall,
        "by_modality": by_modality,
        "by_shard": by_shard,
        "by_subset": by_subset,
        "by_workpiece": by_workpiece,
        "by_subset_workpiece": by_subset_workpiece,
        "top_worst_cd_samples": top_worst_cd,
        "top_lowest_iou_samples": top_lowest_iou,
        "top_failures": top_failures,
    }

    print(json.dumps(round_floats({
        "run_root": analysis["run_root"],
        "pipeline_summaries_found": analysis["pipeline_summaries_found"],
        "overall": analysis["overall"],
        "by_modality": analysis["by_modality"],
        "shard_metrics": analysis["shard_metrics"],
    }, ndigits=4), indent=2, ensure_ascii=False))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(round_floats(analysis), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote JSON: {args.json_out}")

    if args.csv_dir:
        args.csv_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.csv_dir / "by_modality.csv", round_floats(by_modality, ndigits=4))
        write_csv(args.csv_dir / "by_shard.csv", round_floats(by_shard, ndigits=4))
        write_csv(args.csv_dir / "by_subset.csv", round_floats(by_subset, ndigits=4))
        write_csv(args.csv_dir / "by_workpiece.csv", round_floats(by_workpiece, ndigits=4))
        write_csv(args.csv_dir / "by_subset_workpiece.csv", round_floats(by_subset_workpiece, ndigits=4))
        write_csv(args.csv_dir / "top_worst_cd_samples.csv", round_floats(top_worst_cd, ndigits=4))
        write_csv(args.csv_dir / "top_lowest_iou_samples.csv", round_floats(top_lowest_iou, ndigits=4))
        write_csv(args.csv_dir / "top_failures.csv", round_floats(top_failures, ndigits=4))
        print(f"Wrote CSV directory: {args.csv_dir}")


if __name__ == "__main__":
    main()
