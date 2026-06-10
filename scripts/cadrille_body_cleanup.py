#!/usr/bin/env python3
"""Deterministic, geometry-only body cleanup for Cadrille CAD output.

Loads a .step, enumerates solids, clusters them by proximity (exact minimum
surface gap normalized by the overall bbox diagonal), keeps the largest-volume
cluster, removes the rest, and writes a cleaned .step (+ optional .stl) and
cleanup_metadata.json. Runs inside cadrille:latest (CadQuery 2.5 + OCP).

See the 2026-05-29 cad-body-cleanup design spec (local dev notes, not tracked in this repo).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cadquery as cq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from body_cleanup_core import (  # noqa: E402
    apply_runnerup_guard,
    cleanup_hypotheses,
    cluster_bodies,
)
# assess_confidence remains in body_cleanup_core for external callers; the
# confidence computation here is outcome-aware (see run_cleanup).
from cad_mesh_utils import compound_to_mesh  # noqa: E402

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


_REF_CACHE: dict = {}


def _reference_context(path: str, n_points: int):
    """Load + decimate + center-normalize the reference mesh once per path."""
    key = (path, n_points)
    if key in _REF_CACHE:
        return _REF_CACHE[key]
    import trimesh  # local import: base CLI stays usable without the dep
    import cadrille_stage_metrics as sm
    gtn = sm.normalize(sm.decimate(trimesh.load_mesh(path)))
    ctx = {"sm": sm, "gtn": gtn, "n_points": n_points}
    _REF_CACHE[key] = ctx
    return ctx


def _score_hypotheses(solids: list, hyps: list, ctx: dict, lin_defl: float, ang_defl: float) -> list:
    """Mesh + score each hypothesis's kept set vs the normalized reference."""
    sm, gtn, n_points = ctx["sm"], ctx["gtn"], ctx["n_points"]
    scored = []
    for h in hyps:
        iou = cd = None
        try:
            comp = cq.Compound.makeCompound([solids[i] for i in h["kept_body_indices"]])
            m = sm.normalize(compound_to_mesh(comp, lin_defl, ang_defl))
            try:
                cd = sm.compute_cd(gtn, m, n_points)
            except Exception:  # noqa: BLE001
                cd = None
            try:
                iou = sm.compute_iou(gtn, m)
            except Exception:  # noqa: BLE001
                iou = None
        except Exception:  # noqa: BLE001
            pass
        scored.append(dict(h, iou=iou, cd=cd))
    return scored


def _hypothesis_key(score: dict):
    # Watertight-first (valid IoU beats none), then max IoU, then min CD.
    # Hypotheses are pre-sorted most-conservative-first and max() keeps the
    # FIRST maximal element, so exact ties resolve toward removing less.
    return (
        score["iou"] if score.get("iou") is not None else -1.0,
        -(score["cd"] if score.get("cd") is not None else 1e9),
    )


def run_cleanup(args) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_stem or Path(args.in_step).stem
    out_step = out_dir / f"{stem}__cleaned.step"
    out_stl = out_dir / f"{stem}__cleaned.stl"
    out_meta = out_dir / f"{stem}__cleanup_metadata.json"

    solids_all = load_solids(args.in_step)
    volumes_all = []
    for s in solids_all:
        try:
            volumes_all.append(float(s.Volume()))
        except Exception:  # noqa: BLE001
            volumes_all.append(0.0)
    # Hygiene: drop zero/degenerate-volume slivers so they can neither win the
    # keep rule nor distort the bbox; if EVERYTHING is degenerate, keep as-is.
    keepable = [i for i, v in enumerate(volumes_all) if v > 0.0]
    if keepable and len(keepable) < len(solids_all):
        solids = [solids_all[i] for i in keepable]
        volumes = [volumes_all[i] for i in keepable]
        n_degenerate = len(solids_all) - len(keepable)
    else:
        solids = solids_all
        volumes = volumes_all
        n_degenerate = 0
    n = len(solids)
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
            "confidence_reasons": [], "selection_mode": "epsilon",
            "guard_runnerup_ratio": float(getattr(args, "guard_runnerup_ratio", 0.30) or 0.0),
            "reference_mesh": None, "epsilon_kept_body_indices": [],
            "hypotheses": None, "n_degenerate_dropped": 0,
            "clusters": [], "bodies": [],
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
    epsilon_kept = list(clustering["kept_body_indices"])

    # ── Decision: reference rerank > guarded epsilon rule > plain epsilon ──
    reference_mesh = getattr(args, "reference_mesh", None)
    guard_ratio = float(getattr(args, "guard_runnerup_ratio", 0.30) or 0.0)
    selection_mode = "epsilon"
    hypotheses_meta = None
    chosen = epsilon_kept
    if reference_mesh and n > 1:
        try:
            ctx = _reference_context(str(reference_mesh), int(getattr(args, "reference_n_points", 8192)))
            hyps = cleanup_hypotheses(volumes, gap, diag, epsilon_rel=args.epsilon_rel)
            scored = _score_hypotheses(
                solids, hyps, ctx, args.stl_linear_deflection, args.stl_angular_deflection)
            best = max(scored, key=_hypothesis_key)
            if best["iou"] is not None or best["cd"] is not None:
                chosen = list(best["kept_body_indices"])
                selection_mode = "reference_rerank"
            else:
                selection_mode = "epsilon_guarded_fallback"
                if guard_ratio > 0:
                    chosen = apply_runnerup_guard(clustering, guard_ratio=guard_ratio)
            hypotheses_meta = scored
        except Exception as exc:  # noqa: BLE001
            print(f"[body-cleanup] reference rerank failed, falling back: {exc!r}", file=sys.stderr)
            selection_mode = "epsilon_guarded_fallback"
            if guard_ratio > 0:
                chosen = apply_runnerup_guard(clustering, guard_ratio=guard_ratio)
    elif guard_ratio > 0 and n > 1:
        guarded = apply_runnerup_guard(clustering, guard_ratio=guard_ratio)
        if guarded != chosen:
            selection_mode = "epsilon_guarded"
            chosen = guarded

    kept_set = set(chosen)
    kept_solids = [solids[i] for i in chosen]

    total_vol = sum(volumes) or 0.0
    removed_vol = sum(volumes[i] for i in range(n) if i not in kept_set)
    removed_vf = (removed_vol / total_vol) if total_vol > 0 else 0.0

    # Confidence over the FINAL outcome: volume actually removed, and the
    # largest fully-removed cluster vs the kept volume.
    conf_reasons: list[str] = []
    if removed_vf > args.confidence_removed_vol_frac:
        conf_reasons.append(
            f"removed volume {removed_vf:.3f} > {args.confidence_removed_vol_frac:.3f}")
    kept_total = sum(volumes[i] for i in kept_set)
    removed_cluster_vols = [
        c["volume"] for c in clustering["clusters"]
        if not any(b in kept_set for b in c["bodies"])
    ]
    runnerup = max(removed_cluster_vols) if removed_cluster_vols else 0.0
    if kept_total > 0 and runnerup / kept_total >= args.confidence_runnerup_ratio:
        conf_reasons.append(
            f"runner-up cluster {runnerup / kept_total:.2f}x kept (>= {args.confidence_runnerup_ratio:.2f})")
    conf_flag = len(conf_reasons) > 0
    if conf_flag and selection_mode == "reference_rerank":
        conf_reasons.append("note: decision verified against reference mesh (rerank)")

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
        "n_bodies_after": len(chosen),
        "n_bodies_removed": n - len(chosen),
        "removed_volume_fraction": removed_vf,
        "kept_cluster_index": clustering["kept_cluster_index"],
        "noop": clustering["noop"] and len(chosen) == n,
        "confidence_flag": conf_flag,
        "confidence_reasons": conf_reasons,
        "selection_mode": selection_mode,
        "guard_runnerup_ratio": guard_ratio,
        "reference_mesh": str(reference_mesh) if reference_mesh else None,
        "epsilon_kept_body_indices": epsilon_kept,
        "hypotheses": hypotheses_meta,
        "n_degenerate_dropped": n_degenerate,
        "clusters": [
            {"index": c["index"], "n_bodies": len(c["bodies"]), "volume": c["volume"],
             "volume_fraction": c["volume_fraction"],
             "kept": all(b in kept_set for b in c["bodies"])}
            for c in clustering["clusters"]
        ],
        "bodies": [dict(b, kept=(b["index"] in kept_set)) for b in clustering["per_body"]],
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
    p.add_argument("--reference-mesh", dest="reference_mesh", default=None,
                   help="Reference mesh (SAM3D source); when set, every achievable cleanup "
                        "hypothesis is scored by centered IoU vs it and the best is kept")
    p.add_argument("--reference-n-points", dest="reference_n_points", type=int, default=8192)
    p.add_argument("--guard-runnerup-ratio", dest="guard_runnerup_ratio", type=float, default=0.30,
                   help="Without a reference: never delete a cluster whose volume is >= this "
                        "fraction of the kept volume (0 disables the guard)")
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
                      ("n_bodies_before", "n_bodies_after", "n_bodies_removed", "noop",
                       "confidence_flag", "selection_mode")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
