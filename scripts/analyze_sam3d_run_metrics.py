#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze SAM3D batch-run metrics (time, memory, throughput, failures) from results.jsonl/summary.json files."
    )
    parser.add_argument("--run-root", type=Path, required=True, help="Run root that contains shard-* folders")
    parser.add_argument("--shard-glob", default="shard-*", help="Shard folder glob under run root (default: shard-*)")
    parser.add_argument("--top-n", type=int, default=15, help="Top-N slow tasks to display/export")
    parser.add_argument(
        "--no-dedupe-latest",
        action="store_true",
        help="Disable dedupe by task_id (default dedupes and keeps latest record per task)",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to write full analysis JSON")
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help="Optional directory to write CSV tables (shard/subset/workpiece/top_slowest)",
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


def to_int(value: Any) -> int | None:
    f = to_float(value)
    if f is None:
        return None
    return int(f)


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
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / n
    return {
        "count": n,
        "min": vals[0],
        "max": vals[-1],
        "mean": mean,
        "median": percentile(vals, 0.5),
        "p90": percentile(vals, 0.9),
        "p95": percentile(vals, 0.95),
        "p99": percentile(vals, 0.99),
        "std": math.sqrt(var),
    }


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    sxx = sum(v * v for v in dx)
    syy = sum(v * v for v in dy)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum(a * b for a, b in zip(dx, dy))
    return sxy / math.sqrt(sxx * syy)


def find_files(run_root: Path, shard_glob: str) -> tuple[list[Path], list[Path]]:
    results_files: list[Path] = []
    summary_files: list[Path] = []

    root_results = run_root / "results.jsonl"
    root_summary = run_root / "summary.json"
    if root_results.exists():
        results_files.append(root_results)
    if root_summary.exists():
        summary_files.append(root_summary)

    for shard_dir in sorted(p for p in run_root.glob(shard_glob) if p.is_dir()):
        rp = shard_dir / "results.jsonl"
        sp = shard_dir / "summary.json"
        if rp.exists():
            results_files.append(rp)
        if sp.exists():
            summary_files.append(sp)

    dedup_results = sorted(set(results_files))
    dedup_summaries = sorted(set(summary_files))
    return dedup_results, dedup_summaries


def infer_shard(record: dict[str, Any], source_file: Path) -> str:
    shard_index = to_int(record.get("shard_index"))
    if shard_index is not None:
        return f"shard-{shard_index}"

    parent = source_file.parent.name
    if parent.startswith("shard-"):
        return parent

    return "single"


def load_records(results_files: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in results_files:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_source_file"] = str(path)
                row["_source_line"] = line_no
                row["_shard"] = infer_shard(row, path)
                records.append(row)
    return records


def dedupe_latest(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    if not records:
        return [], 0

    by_key: dict[str, tuple[float, int, dict[str, Any]]] = {}
    for idx, row in enumerate(records):
        task_id = str(row.get("task_id") or f"__row_{idx}")
        ts = to_float(row.get("ended_at_epoch"))
        if ts is None:
            ts = to_float(row.get("started_at_epoch"))
        if ts is None:
            ts = float(idx)

        prev = by_key.get(task_id)
        if prev is None or ts >= prev[0]:
            by_key[task_id] = (ts, idx, row)

    deduped = [triple[2] for triple in sorted(by_key.values(), key=lambda t: t[1])]
    duplicates = len(records) - len(deduped)
    return deduped, duplicates


def load_summaries(summary_files: list[Path]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in summary_files:
        row = json.loads(path.read_text(encoding="utf-8"))
        shard_name = path.parent.name if path.parent.name.startswith("shard-") else "single"
        row["_source_file"] = str(path)
        row["_shard"] = shard_name
        summaries.append(row)
    return summaries


def stats_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [v for r in rows if (v := to_float(r.get("duration_sec"))) is not None]
    mem_alloc = [v for r in rows if (v := to_float(r.get("peak_memory_allocated_mb"))) is not None]
    mem_res = [v for r in rows if (v := to_float(r.get("peak_memory_reserved_mb"))) is not None]
    sec_per_mp = [v for r in rows if (v := to_float(r.get("sec_per_megapixel"))) is not None]
    iph = [v for r in rows if (v := to_float(r.get("instances_per_hour"))) is not None]
    mask_frac = [v for r in rows if (v := to_float(r.get("mask_fraction"))) is not None]
    mesh_bytes = [v for r in rows if (v := to_float(r.get("mesh_size_bytes"))) is not None]
    stl_bytes = [v for r in rows if (v := to_float(r.get("stl_size_bytes"))) is not None]

    duration_vs_mask_x: list[float] = []
    duration_vs_mask_y: list[float] = []
    duration_vs_pixels_x: list[float] = []
    duration_vs_pixels_y: list[float] = []

    for r in rows:
        d = to_float(r.get("duration_sec"))
        mf = to_float(r.get("mask_fraction"))
        px = to_float(r.get("image_pixels"))
        if d is not None and mf is not None:
            duration_vs_mask_x.append(d)
            duration_vs_mask_y.append(mf)
        if d is not None and px is not None:
            duration_vs_pixels_x.append(d)
            duration_vs_pixels_y.append(px)

    return {
        "duration_sec": describe(durations),
        "peak_memory_allocated_mb": describe(mem_alloc),
        "peak_memory_reserved_mb": describe(mem_res),
        "sec_per_megapixel": describe(sec_per_mp),
        "instances_per_hour": describe(iph),
        "mask_fraction": describe(mask_frac),
        "mesh_size_bytes": describe(mesh_bytes),
        "stl_size_bytes": describe(stl_bytes),
        "correlations": {
            "duration_vs_mask_fraction": pearson_correlation(duration_vs_mask_x, duration_vs_mask_y),
            "duration_vs_image_pixels": pearson_correlation(duration_vs_pixels_x, duration_vs_pixels_y),
        },
    }


def group_breakdown(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        group_val = r.get(key)
        label = str(group_val) if group_val not in (None, "") else "<unknown>"
        groups[label].append(r)

    output: list[dict[str, Any]] = []
    for label, grp_rows in groups.items():
        ok_rows = [r for r in grp_rows if r.get("status") == "ok"]
        err_rows = [r for r in grp_rows if r.get("status") == "error"]
        duration = [v for r in ok_rows if (v := to_float(r.get("duration_sec"))) is not None]
        mem_alloc = [v for r in ok_rows if (v := to_float(r.get("peak_memory_allocated_mb"))) is not None]
        sec_per_mp = [v for r in ok_rows if (v := to_float(r.get("sec_per_megapixel"))) is not None]
        output.append(
            {
                "group": label,
                "records": len(grp_rows),
                "ok": len(ok_rows),
                "error": len(err_rows),
                "duration_mean_sec": (sum(duration) / len(duration)) if duration else None,
                "duration_p50_sec": percentile(sorted(duration), 0.5) if duration else None,
                "duration_p90_sec": percentile(sorted(duration), 0.9) if duration else None,
                "mem_alloc_mean_mb": (sum(mem_alloc) / len(mem_alloc)) if mem_alloc else None,
                "mem_alloc_p90_mb": percentile(sorted(mem_alloc), 0.9) if mem_alloc else None,
                "mem_alloc_max_mb": max(mem_alloc) if mem_alloc else None,
                "sec_per_megapixel_mean": (sum(sec_per_mp) / len(sec_per_mp)) if sec_per_mp else None,
            }
        )

    output.sort(key=lambda r: (-r["ok"], r["group"]))
    return output


def top_slowest(ok_rows: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    rows = [r for r in ok_rows if to_float(r.get("duration_sec")) is not None]
    rows.sort(key=lambda r: to_float(r.get("duration_sec")) or 0.0, reverse=True)
    top = rows[:top_n]
    out: list[dict[str, Any]] = []
    for r in top:
        out.append(
            {
                "task_id": r.get("task_id"),
                "shard": r.get("_shard"),
                "subset": r.get("subset"),
                "workpiece": r.get("workpiece"),
                "duration_sec": to_float(r.get("duration_sec")),
                "sec_per_megapixel": to_float(r.get("sec_per_megapixel")),
                "mask_fraction": to_float(r.get("mask_fraction")),
                "peak_memory_allocated_mb": to_float(r.get("peak_memory_allocated_mb")),
                "peak_memory_reserved_mb": to_float(r.get("peak_memory_reserved_mb")),
                "mesh_size_bytes": to_int(r.get("mesh_size_bytes")),
                "stl_size_bytes": to_int(r.get("stl_size_bytes")),
            }
        )
    return out


def round_floats(obj: Any, ndigits: int = 4) -> Any:
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, list):
        return [round_floats(x, ndigits=ndigits) for x in obj]
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits=ndigits) for k, v in obj.items()}
    return obj


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_stats_block(title: str, stats: dict[str, Any], unit: str = "") -> None:
    if stats.get("count", 0) == 0:
        print(f"- {title}: no data")
        return
    u = f" {unit}" if unit else ""
    print(
        f"- {title}: n={stats['count']}, mean={stats['mean']:.3f}{u}, "
        f"p50={stats['median']:.3f}{u}, p90={stats['p90']:.3f}{u}, p95={stats['p95']:.3f}{u}, max={stats['max']:.3f}{u}"
    )


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()

    if not run_root.exists():
        raise SystemExit(f"Run root does not exist: {run_root}")

    results_files, summary_files = find_files(run_root, args.shard_glob)
    if not results_files:
        raise SystemExit(f"No results.jsonl files found under: {run_root}")

    raw_records = load_records(results_files)
    use_dedupe = not args.no_dedupe_latest
    if use_dedupe:
        records, duplicate_count = dedupe_latest(raw_records)
    else:
        records = raw_records
        duplicate_count = 0

    summaries = load_summaries(summary_files)

    ok_rows = [r for r in records if r.get("status") == "ok"]
    err_rows = [r for r in records if r.get("status") == "error"]

    summary_totals = {
        "total_tasks": None,
        "completed_ok": None,
        "failed": None,
        "skipped": None,
        "processed": None,
        "remaining": None,
        "completion_pct": None,
    }

    if summaries:
        total_tasks = sum(to_int(s.get("total_tasks_in_shard")) or 0 for s in summaries)
        completed_ok = sum(to_int(s.get("completed_ok")) or 0 for s in summaries)
        failed = sum(to_int(s.get("failed")) or 0 for s in summaries)
        skipped = sum(to_int(s.get("skipped")) or 0 for s in summaries)
        processed = sum(to_int(s.get("processed")) or 0 for s in summaries)
        remaining = total_tasks - (completed_ok + failed + skipped)
        completion_pct = (completed_ok + failed + skipped) * 100.0 / total_tasks if total_tasks else None
        summary_totals = {
            "total_tasks": total_tasks,
            "completed_ok": completed_ok,
            "failed": failed,
            "skipped": skipped,
            "processed": processed,
            "remaining": remaining,
            "completion_pct": completion_pct,
        }

    perf = stats_for_rows(ok_rows)
    shard_breakdown = group_breakdown(records, "_shard")
    subset_breakdown = group_breakdown(records, "subset")
    workpiece_breakdown = group_breakdown(records, "workpiece")
    top_slowest_rows = top_slowest(ok_rows, args.top_n)

    error_counts = Counter(str(r.get("error_type") or "UnknownError") for r in err_rows)
    gpu_counts = Counter(str(r.get("gpu_name") or "unknown") for r in ok_rows)

    analysis: dict[str, Any] = {
        "run_root": str(run_root),
        "files": {
            "results_files": [str(p) for p in results_files],
            "summary_files": [str(p) for p in summary_files],
        },
        "record_counts": {
            "raw_records": len(raw_records),
            "deduped_records": len(records),
            "duplicates_removed": duplicate_count,
            "ok_records": len(ok_rows),
            "error_records": len(err_rows),
        },
        "task_totals_from_summary": summary_totals,
        "performance_ok": perf,
        "error_type_counts": dict(error_counts),
        "gpu_name_counts_ok": dict(gpu_counts),
        "breakdown": {
            "by_shard": shard_breakdown,
            "by_subset": subset_breakdown,
            "by_workpiece": workpiece_breakdown,
        },
        "top_slowest_ok": top_slowest_rows,
    }

    rounded = round_floats(analysis, ndigits=4)

    print("=== SAM3D Run Metrics Analysis ===")
    print(f"run_root: {rounded['run_root']}")
    print(f"results files: {len(results_files)}, summary files: {len(summary_files)}")
    print(
        "records: "
        f"raw={rounded['record_counts']['raw_records']}, "
        f"deduped={rounded['record_counts']['deduped_records']}, "
        f"ok={rounded['record_counts']['ok_records']}, "
        f"error={rounded['record_counts']['error_records']}, "
        f"duplicates_removed={rounded['record_counts']['duplicates_removed']}"
    )

    totals = rounded["task_totals_from_summary"]
    if totals["total_tasks"] is not None:
        print(
            "task totals (from summary.json): "
            f"total={totals['total_tasks']}, ok={totals['completed_ok']}, failed={totals['failed']}, "
            f"skipped={totals['skipped']}, remaining={totals['remaining']}, "
            f"completion={totals['completion_pct']:.2f}%"
        )

    print("\n--- Performance (ok tasks) ---")
    print_stats_block("duration_sec", rounded["performance_ok"]["duration_sec"], "s")
    print_stats_block("peak_memory_allocated_mb", rounded["performance_ok"]["peak_memory_allocated_mb"], "MB")
    print_stats_block("peak_memory_reserved_mb", rounded["performance_ok"]["peak_memory_reserved_mb"], "MB")
    print_stats_block("sec_per_megapixel", rounded["performance_ok"]["sec_per_megapixel"], "s/MP")
    print_stats_block("instances_per_hour", rounded["performance_ok"]["instances_per_hour"], "inst/h")

    corr = rounded["performance_ok"]["correlations"]
    print("\n--- Correlation hints (ok tasks) ---")
    print(f"- duration vs mask_fraction: {corr['duration_vs_mask_fraction']}")
    print(f"- duration vs image_pixels: {corr['duration_vs_image_pixels']}")

    if rounded["error_type_counts"]:
        print("\n--- Error types ---")
        for k, v in sorted(rounded["error_type_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"- {k}: {v}")

    print("\n--- Slowest tasks ---")
    for i, row in enumerate(rounded["top_slowest_ok"], start=1):
        print(
            f"{i:2d}. {row['duration_sec']}s | {row['task_id']} | {row['shard']} | "
            f"mem_alloc={row['peak_memory_allocated_mb']}MB"
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rounded, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote JSON: {args.json_out}")

    if args.csv_dir:
        args.csv_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.csv_dir / "by_shard.csv", round_floats(shard_breakdown, ndigits=4))
        write_csv(args.csv_dir / "by_subset.csv", round_floats(subset_breakdown, ndigits=4))
        write_csv(args.csv_dir / "by_workpiece.csv", round_floats(workpiece_breakdown, ndigits=4))
        write_csv(args.csv_dir / "top_slowest_ok.csv", round_floats(top_slowest_rows, ndigits=4))
        print(f"Wrote CSV directory: {args.csv_dir}")


if __name__ == "__main__":
    main()
