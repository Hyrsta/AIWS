#!/usr/bin/env python3
"""Deterministic, geometry-only body cleanup for Cadrille CAD output.

Loads a .step, enumerates solids, clusters them by proximity (exact minimum
surface gap normalized by the overall bbox diagonal), keeps the largest-volume
cluster, removes the rest, and writes a cleaned .step (+ optional .stl) and
cleanup_metadata.json. Runs inside cadrille:latest (CadQuery 2.5 + OCP).

See docs/superpowers/specs/2026-05-29-cad-body-cleanup-design.md
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cadquery as cq
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from body_cleanup_core import assess_confidence, cluster_bodies  # noqa: E402

try:
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    _HAVE_OCP = True
except Exception:  # noqa: BLE001
    _HAVE_OCP = False


def load_solids(in_step: str) -> list:
    shape = cq.importers.importStep(in_step).val()
    return list(shape.Solids())


def _bbox_gap(a, b) -> float:
    ba, bb = a.BoundingBox(), b.BoundingBox()

    def axis(amin, amax, bmin, bmax):
        if amax < bmin:
            return bmin - amax
        if bmax < amin:
            return amin - bmax
        return 0.0

    dx = axis(ba.xmin, ba.xmax, bb.xmin, bb.xmax)
    dy = axis(ba.ymin, ba.ymax, bb.ymin, bb.ymax)
    dz = axis(ba.zmin, ba.zmax, bb.zmin, bb.zmax)
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def solid_min_gap(a, b) -> float:
    """Exact minimum surface distance via OCP; bbox-gap fallback on error.
    The fallback over-connects (bbox gap <= surface gap), which is the
    fidelity-safe direction: we prefer keeping over deleting."""
    if _HAVE_OCP:
        try:
            ext = BRepExtrema_DistShapeShape(a.wrapped, b.wrapped)
            if ext.IsDone():
                return float(ext.Value())
        except Exception:  # noqa: BLE001
            pass
    return _bbox_gap(a, b)


def overall_bbox_diagonal(solids: list) -> float:
    if not solids:
        return 0.0
    xmin = ymin = zmin = float("inf")
    xmax = ymax = zmax = float("-inf")
    for s in solids:
        bb = s.BoundingBox()
        xmin, ymin, zmin = min(xmin, bb.xmin), min(ymin, bb.ymin), min(zmin, bb.zmin)
        xmax, ymax, zmax = max(xmax, bb.xmax), max(ymax, bb.ymax), max(zmax, bb.zmax)
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def compound_to_mesh(compound, linear_deflection: float, angular_deflection: float):
    vertices, faces = compound.tessellate(linear_deflection, angular_deflection)
    return trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)


def run_cleanup(args) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_stem or Path(args.in_step).stem
    out_step = out_dir / f"{stem}__cleaned.step"
    out_stl = out_dir / f"{stem}__cleaned.stl"
    out_meta = out_dir / f"{stem}__cleanup_metadata.json"

    solids = load_solids(args.in_step)
    n = len(solids)
    volumes = [float(s.Volume()) for s in solids]
    diag = overall_bbox_diagonal(solids)

    gap = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = solid_min_gap(solids[i], solids[j])
            gap[i][j] = gap[j][i] = g

    if n == 0:
        shutil.copy2(args.in_step, out_step)
        meta = {
            "input_step": str(args.in_step), "epsilon_rel": args.epsilon_rel,
            "bbox_diagonal": diag, "n_bodies_before": 0, "n_clusters": 0,
            "n_bodies_after": 0, "n_bodies_removed": 0, "removed_volume_fraction": 0.0,
            "kept_cluster_index": -1, "noop": True, "confidence_flag": False,
            "confidence_reasons": [], "clusters": [], "bodies": [],
        }
        out_meta.write_text(json.dumps(meta, indent=2))
        if args.export_stl:
            try:
                compound_to_mesh(cq.importers.importStep(str(out_step)).val(),
                                 args.stl_linear_deflection, args.stl_angular_deflection).export(str(out_stl))
            except Exception:  # noqa: BLE001
                pass
        return meta

    clustering = cluster_bodies(volumes, gap, diag, args.epsilon_rel)
    kept_idx = clustering["kept_body_indices"]
    kept_set = set(kept_idx)
    kept_solids = [solids[i] for i in kept_idx]

    total_vol = sum(volumes) or 0.0
    removed_vol = sum(volumes[i] for i in range(n) if i not in kept_set)
    removed_vf = (removed_vol / total_vol) if total_vol > 0 else 0.0
    conf_flag, conf_reasons = assess_confidence(
        clustering["clusters"], clustering["kept_cluster_index"], removed_vf,
        removed_vol_frac_thresh=args.confidence_removed_vol_frac,
        runnerup_ratio_thresh=args.confidence_runnerup_ratio,
    )

    comp = cq.Compound.makeCompound(kept_solids)
    cq.exporters.export(comp, str(out_step))
    if args.export_stl:
        compound_to_mesh(comp, args.stl_linear_deflection, args.stl_angular_deflection).export(str(out_stl))

    meta = {
        "input_step": str(args.in_step),
        "epsilon_rel": args.epsilon_rel,
        "bbox_diagonal": diag,
        "n_bodies_before": n,
        "n_clusters": clustering["n_clusters"],
        "n_bodies_after": len(kept_idx),
        "n_bodies_removed": n - len(kept_idx),
        "removed_volume_fraction": removed_vf,
        "kept_cluster_index": clustering["kept_cluster_index"],
        "noop": clustering["noop"],
        "confidence_flag": conf_flag,
        "confidence_reasons": conf_reasons,
        "clusters": [
            {"index": c["index"], "n_bodies": len(c["bodies"]), "volume": c["volume"],
             "volume_fraction": c["volume_fraction"], "kept": c["kept"]}
            for c in clustering["clusters"]
        ],
        "bodies": clustering["per_body"],
    }
    out_meta.write_text(json.dumps(meta, indent=2))
    return meta


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Geometry-only CAD body cleanup.")
    p.add_argument("--in-step", dest="in_step", required=True)
    p.add_argument("--out-dir", dest="out_dir", required=True)
    p.add_argument("--out-stem", dest="out_stem", default=None)
    p.add_argument("--epsilon-rel", dest="epsilon_rel", type=float, default=0.07)
    p.add_argument("--export-stl", dest="export_stl", action="store_true")
    p.add_argument("--stl-linear-deflection", dest="stl_linear_deflection", type=float, default=0.001)
    p.add_argument("--stl-angular-deflection", dest="stl_angular_deflection", type=float, default=0.1)
    p.add_argument("--confidence-removed-vol-frac", dest="confidence_removed_vol_frac", type=float, default=0.05)
    p.add_argument("--confidence-runnerup-ratio", dest="confidence_runnerup_ratio", type=float, default=0.30)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not os.path.isfile(args.in_step):
        print(f"[body-cleanup] input not found: {args.in_step}", file=sys.stderr)
        return 2
    try:
        meta = run_cleanup(args)
    except Exception as exc:  # noqa: BLE001
        print(f"[body-cleanup] failed: {exc!r}", file=sys.stderr)
        return 1
    print(json.dumps({k: meta[k] for k in
                      ("n_bodies_before", "n_bodies_after", "n_bodies_removed", "noop", "confidence_flag")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
