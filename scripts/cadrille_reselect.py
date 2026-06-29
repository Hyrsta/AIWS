#!/usr/bin/env python3
"""Live-pipeline candidate reselection (PC).

Protocol: first filter to candidates whose generated code materialized into
usable CAD outputs, then pick the valid-code candidate whose BODY-CLEANED mesh
best matches the SAM3D GT under the centered convention (max IoU, CD tiebreak).
CD remains usable when Boolean IoU is unavailable, so non-watertight/Boolean
failures are reported separately from generated-code invalidity.

Runs in cadrille:latest. Writes {best, scores, summary} JSON; simple_reconstruct_job
then points selected_brep/mesh/py at <best> before the existing cleanup/postscale/
metrics flow.

Args: --candidates-dir <dir of <stem>+<id>.step>  --gt-mesh <SAM3D stl>  --out <json>
      [--py-dir <dir of <stem>+<id>.py>]  [--mesh-dir <dir of <stem>+<id>.stl>]
      [--brep-ext step]  [--n-points 8192]
"""
import sys, os, glob, json, argparse, tempfile
from types import SimpleNamespace
import trimesh

sys.path.insert(0, "/repo/scripts")
import cadrille_body_cleanup as bc          # noqa: E402
import cadrille_stage_metrics as sm          # decimate / normalize(centered) / compute_iou(whole) / compute_cd


def clean(step, od, reference_mesh=None):
    # reference_mesh enables multi-hypothesis cleanup reranked by centered IoU
    # vs the SAM3D mesh (the same reference this script selects candidates by).
    a = SimpleNamespace(in_step=step, out_dir=od, out_stem="c", epsilon_rel=0.07,
                        export_stl=True, stl_linear_deflection=0.001, stl_angular_deflection=0.1,
                        confidence_removed_vol_frac=0.05, confidence_runnerup_ratio=0.30,
                        reference_mesh=reference_mesh, reference_n_points=8192,
                        guard_runnerup_ratio=0.30)
    bc.run_cleanup(a)
    return trimesh.load_mesh(os.path.join(od, "c__cleaned.stl"))


def candidate_name(path, ext):
    name = os.path.basename(path)[: -(len(ext) + 1)]
    return name if "+" in name else None


def collect_candidates(candidates_dir, brep_ext, py_dir=None, mesh_dir=None):
    step_paths = {
        name: p
        for p in glob.glob(os.path.join(candidates_dir, f"*.{brep_ext}"))
        if (name := candidate_name(p, brep_ext))
    }
    if mesh_dir is None:
        mesh_dir = os.path.join(os.path.dirname(candidates_dir), "tmp_mesh")
    mesh_paths = {
        name: p
        for p in glob.glob(os.path.join(mesh_dir, "*.stl"))
        if (name := candidate_name(p, "stl"))
    } if mesh_dir and os.path.isdir(mesh_dir) else {}
    py_paths = {
        name: p
        for p in glob.glob(os.path.join(py_dir, "*.py"))
        if (name := candidate_name(p, "py"))
    } if py_dir and os.path.isdir(py_dir) else {}

    names = sorted(py_paths or (set(step_paths) | set(mesh_paths)))
    rows = []
    for name in names:
        step = step_paths.get(name)
        mesh = mesh_paths.get(name)
        py = py_paths.get(name)
        missing = []
        if py_paths and not py:
            missing.append("py")
        if not step:
            missing.append(brep_ext)
        if not mesh:
            missing.append("stl")
        rows.append({
            "name": name,
            "step": step,
            "mesh": mesh,
            "py": py,
            "code_valid": bool(step and mesh and (py or not py_paths)),
            "missing_outputs": missing,
        })
    return rows


def selection_key(score):
    # Watertight-first selection. The first key element is the cleaned-centered
    # Boolean IoU: any code-valid candidate whose cleaned mesh is watertight (iou is
    # not None, hence >= 0) outranks every non-watertight candidate (mapped to -1.0).
    # Among watertight candidates the higher IoU wins; CD (second element) only
    # breaks ties, or orders candidates when NO candidate is watertight. Hence the
    # Boolean-IoU coverage gap = samples whose every candidate is non-watertight,
    # not a side effect of selection.
    return (
        score["iou"] if score.get("iou") is not None else -1.0,
        -(score["cd"] if score.get("cd") is not None else 1e9),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates-dir", required=True)
    ap.add_argument("--gt-mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--brep-ext", default="step")
    ap.add_argument("--py-dir")
    ap.add_argument("--mesh-dir")
    ap.add_argument("--n-points", type=int, default=8192)
    a = ap.parse_args()

    gt = sm.normalize(sm.decimate(trimesh.load_mesh(a.gt_mesh)))
    cands = collect_candidates(a.candidates_dir, a.brep_ext, a.py_dir, a.mesh_dir)
    scores = []
    for cand in cands:
        name = cand["name"]
        iou = cd = None
        invalidity = None
        if not cand["code_valid"]:
            scores.append({
                "name": name,
                "code_valid": False,
                "invalidity": "code_invalid",
                "missing_outputs": cand["missing_outputs"],
                "iou": None,
                "cd": None,
            })
            continue
        try:
            with tempfile.TemporaryDirectory() as td:
                m = sm.normalize(clean(cand["step"], td, a.gt_mesh))
        except Exception as e:  # noqa: BLE001
            print(f"[reselect] {name} cleanup failed: {e!r}", flush=True)
            scores.append({
                "name": name,
                "code_valid": False,
                "invalidity": "code_invalid",
                "missing_outputs": ["cleanup_failed"],
                "iou": None,
                "cd": None,
            })
            continue
        try:
            cd = sm.compute_cd(gt, m, a.n_points)
        except Exception as e:  # noqa: BLE001
            print(f"[reselect] {name} CD failed: {e!r}", flush=True)
        try:
            iou = sm.compute_iou(gt, m)
        except Exception as e:  # noqa: BLE001
            print(f"[reselect] {name} IoU failed: {e!r}", flush=True)
        if cd is None:
            invalidity = "metric_invalid"
        elif iou is None:
            invalidity = "boolean_iou_invalid"
        scores.append({
            "name": name,
            "code_valid": True,
            "invalidity": invalidity,
            "missing_outputs": [],
            "iou": iou,
            "cd": cd,
        })

    # Candidate must be code-valid AND have a computable CD (a materialized mesh).
    # Watertightness is enforced by selection_key (watertight-first), not by this
    # filter, so a non-watertight candidate is selected only when the sample has no
    # watertight candidate at all.
    selectable = [s for s in scores if s.get("code_valid") and s.get("cd") is not None]
    best = max(selectable, key=selection_key) if selectable else None
    summary = {
        "selection_protocol": "valid_code_then_cleaned_centered_iou",
        "cleanup": "reference_reranked_hypotheses",
        "candidate_count": len(scores),
        "code_valid_count": sum(1 for s in scores if s.get("code_valid")),
        "code_invalid_count": sum(1 for s in scores if s.get("invalidity") == "code_invalid"),
        "boolean_iou_invalid_count": sum(1 for s in scores if s.get("invalidity") == "boolean_iou_invalid"),
        "metric_invalid_count": sum(1 for s in scores if s.get("invalidity") == "metric_invalid"),
        "selectable_count": len(selectable),
    }
    json.dump({"best": best["name"] if best else None, "scores": scores, "summary": summary},
              open(a.out, "w"), indent=2)
    print(f"[reselect] {len(scores)} candidates, {summary['code_valid_count']} code-valid "
          f"-> best={best['name'] if best else None} "
          f"(iou={best['iou'] if best else None}, cd={best['cd'] if best else None})", flush=True)


if __name__ == "__main__":
    main()
