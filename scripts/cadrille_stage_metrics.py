#!/usr/bin/env python3
"""Per-stage IoU + Chamfer Distance for the offline CAD pipeline.

Scores each available stage mesh — Cadrille canonical, after body-cleanup, and
after cleanup+metric-alignment — against the SAM3D GT mesh, using Cadrille's own
evaluate.py metrics (compute_iou / compute_chamfer_distance) but with CONSISTENT
normalization: BOTH the GT and the prediction are centered and scaled to unit
max-extent, so the three stages are directly comparable (a position/scale-
invariant shape comparison). Runs inside cadrille:latest (needs the trimesh
boolean backend for IoU). Best-effort: a missing/failing stage yields nulls.
"""
import argparse, json, os
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def compute_cd(gt, pred, n):
    gp, _ = trimesh.sample.sample_surface(gt, n)
    pp, _ = trimesh.sample.sample_surface(pred, n)
    gd, _ = cKDTree(gp).query(pp, k=1)
    pd, _ = cKDTree(pp).query(gp, k=1)
    return float(np.mean(np.square(gd)) + np.mean(np.square(pd)))


def compute_iou(gt, pred):
    # Whole-mesh boolean (NOT per-connected-component .split()): numerically
    # identical to Cadrille's split IoU where both succeed (|delta|<1e-10) but far
    # more robust on fragmented SAM3D GT (~93% vs ~15% coverage). null => the
    # boolean backend rejected the geometry (non-watertight) => treated as invalid
    # (excluded from aggregates, reported as invalidity rate — never zero-filled).
    try:
        x = gt.intersection(pred)
        inter = x.volume if x is not None else 0.0
        u = gt.volume + pred.volume - inter
        return float(inter / u) if u > 0 else None
    except Exception:
        return None


DECIM_BUDGET = 60000  # == gui SAM3D_PREVIEW_MAX_FACES


def decimate(m, budget=DECIM_BUDGET):
    """Quadric-decimate a dense GT to the SAM3D viewer budget so the boolean is
    fast; lossless for volume IoU since the SAM3D GT is ~94% watertight."""
    if len(m.faces) <= budget:
        return m
    try:
        import open3d as o3d
        om = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(m.vertices)),
            o3d.utility.Vector3iVector(np.asarray(m.faces)))
        dm = om.simplify_quadric_decimation(int(budget))
        return trimesh.Trimesh(np.asarray(dm.vertices), np.asarray(dm.triangles), process=False)
    except Exception:
        return m


def normalize(m):
    m = m.copy()
    c = (m.bounds[0] + m.bounds[1]) / 2.0
    m.apply_translation(-c)
    e = float(np.max(m.extents))
    if e > 1e-7:
        m.apply_scale(1.0 / e)
    m.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
    return m


def n_corner(m):
    # Matched corner convention: min-corner at origin, scaled to unit max-extent
    # (no centering). Bottom-aligned; robust to spurious bodies above the main body.
    m = m.copy()
    m.apply_translation(-m.bounds[0])
    e = float(np.max(m.extents))
    if e > 1e-7:
        m.apply_scale(1.0 / e)
    return m


def score(gtn, gtk, path, n):
    if not path or not os.path.exists(path):
        return None
    try:
        raw = trimesh.load_mesh(path)
        mc, mk = normalize(raw), n_corner(raw)
        return {"iou": compute_iou(gtn, mc), "cd": compute_cd(gtn, mc, n),
                "iou_corner": compute_iou(gtk, mk), "cd_corner": compute_cd(gtk, mk, n)}
    except Exception as e:  # noqa: BLE001
        return {"iou": None, "cd": None, "iou_corner": None, "cd_corner": None, "error": repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-mesh", required=True)
    ap.add_argument("--canonical-mesh")
    ap.add_argument("--cleaned-mesh")
    ap.add_argument("--scaled-mesh")
    ap.add_argument("--cadrille-metrics")
    ap.add_argument("--reselect-json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-points", type=int, default=8192)
    a = ap.parse_args()
    np.random.seed(0)

    out = {
        "n_points": a.n_points,
        "normalization": ("two conventions reported per stage: centered (GT+pred centred in the "
                          "unit cube) and corner (GT+pred min-corner at origin); both scaled to unit max-extent"),
        "gt_mesh": a.gt_mesh,
        "stages": {},
    }
    if a.cadrille_metrics and os.path.exists(a.cadrille_metrics):
        try:
            s = json.load(open(a.cadrille_metrics)).get("summary", {})
            out["cadrille_selection_metric"] = {
                "mean_iou": s.get("mean_iou"), "median_cd": s.get("median_cd"),
                "note": "canonical only, Cadrille's selection convention (GT as-prepared, not re-normalized)",
            }
        except Exception:
            pass
    if a.reselect_json and os.path.exists(a.reselect_json):
        try:
            rs = json.load(open(a.reselect_json))
            summary = rs.get("summary") or {}
            out["cadrille_reselect"] = {
                "best": rs.get("best"),
                "selection_protocol": summary.get("selection_protocol") or "valid_code_then_cleaned_centered_iou",
                "candidate_count": summary.get("candidate_count"),
                "code_valid_count": summary.get("code_valid_count"),
                "code_invalid_count": summary.get("code_invalid_count"),
                "boolean_iou_invalid_count": summary.get("boolean_iou_invalid_count"),
                "metric_invalid_count": summary.get("metric_invalid_count"),
                "selectable_count": summary.get("selectable_count"),
            }
        except Exception:
            pass

    gt_decim = decimate(trimesh.load_mesh(a.gt_mesh))
    gtn = normalize(gt_decim)
    gtk = n_corner(gt_decim)
    for name, p in [("canonical", a.canonical_mesh),
                    ("cleaned", a.cleaned_mesh),
                    ("scaled", a.scaled_mesh)]:
        r = score(gtn, gtk, p, a.n_points)
        if r is not None:
            out["stages"][name] = r

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("[stage-metrics] " + a.out + ": " + ", ".join(
        f"{k}(IoU={v.get('iou')}, CD={v.get('cd')})" for k, v in out["stages"].items()), flush=True)


if __name__ == "__main__":
    main()
