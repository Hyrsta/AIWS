#!/usr/bin/env python3
"""Select the top-N Cadrille eval results by IoU for each (model, modality) config.

The batched SAM3D->Cadrille evaluation (1418 samples) lives under:
    outputs/cadrille-{rl,sft}-gpu-mem-rerun-*/{img,pc}/shard-{0..3}/
Each shard's metrics.json (cadrille evaluate.py format) holds:
    {"summary": {...}, "best_names": [...],
     "metrics": {"<subset>__<class>__<stem>__obj01": {"cd":[...], "iou":[...], "id":[...]}}}
Predicted candidates: shard-*/tmp_{py,mesh,brep}/<fn>+<id>.{py,stl,step}
Source mesh (IoU comparison target, = SAM3D bridge mesh): shared_splits/*/<fn>.stl
Post-scaled CAD: {img,pc}/postscale-axiswise/<fn>__scaled.{step,py}, <fn>__metadata.json
Input photo + mask (to upload into the GUI):
    data/aiws5.2-usable/<subset>/<class>/images/<stem>.<ext>
    data/aiws5.2-usable/<subset>/<class>/masks/<stem>.<ext>

For each of the 4 configs (RL/SFT x IMG/PC) it ranks samples by their best IoU
(highest = best), takes the top-N, and writes INDEPENDENT file copies of:
  input.png, mask.png, annotation.json, pred.{stl,step,py}, source_mesh.stl,
  scaled.{step,py}, scaled_metadata.json, metrics.json
plus a markdown summary. Lives in the AIWS folder on RXL.

Usage:  python3 scripts/select_best_results.py [--top 5] [--out outputs/best_results_selection]
"""
from __future__ import annotations
import argparse, json, os, shutil, glob
from pathlib import Path

AIWS = Path("/ssd1/rxl/zhankaiming/AIWS")
RUNS = {
    "rl":  AIWS / "outputs/cadrille-rl-gpu-mem-rerun-20260412-211348",
    "sft": AIWS / "outputs/cadrille-sft-gpu-mem-rerun-20260412-172238",
}
CONFIGS = [("rl", "img"), ("rl", "pc"), ("sft", "img"), ("sft", "pc")]
DATASET = AIWS / "data/aiws5.2-usable"
IMG_EXTS = ("png", "jpg", "jpeg", "bmp", "webp")


def gather(run_dir: Path, mod: str):
    """Best-IoU candidate per file_name across all shards of a config."""
    best = {}
    for mj in sorted((run_dir / mod).glob("shard-*/metrics.json")):
        shard = str(mj.parent)
        try:
            data = json.load(open(mj))
        except Exception:
            continue
        for fn, v in data.get("metrics", {}).items():
            ious = v.get("iou") or []
            if not ious:
                continue
            ids = v.get("id") or []
            cds = v.get("cd") or []
            bi = max(range(len(ious)), key=lambda i: ious[i])
            rec = {
                "fn": fn, "iou": float(ious[bi]),
                "id": ids[bi] if bi < len(ids) else (ids[0] if ids else "0"),
                "cd": float(cds[bi]) if bi < len(cds) and cds[bi] is not None else None,
                "shard": shard,
            }
            if fn not in best or rec["iou"] > best[fn]["iou"]:
                best[fn] = rec
    return list(best.values())


def first(*globs):
    for g in globs:
        hits = sorted(glob.glob(g, recursive=True))
        if hits:
            return hits[0]
    return None


def find_pred(shard: str, sub: str, fn: str, cid, ext: str):
    return first(os.path.join(shard, f"tmp_{sub}", f"{fn}+{cid}.{ext}"),
                 os.path.join(shard, "**", f"tmp_{sub}", f"{fn}+{cid}.{ext}"))


def find_source_mesh(run_dir: Path, mod: str, fn: str):
    return first(os.path.join(str(run_dir), "shared_splits", "*", fn + ".stl"),
                 os.path.join(str(AIWS), "repos/cadrille/data", f"sam3d_bridge_*_{mod}_*", fn + ".stl"),
                 os.path.join(str(AIWS), "repos/cadrille/data", "sam3d_bridge_*", fn + ".stl"))


def find_input(subset: str, cls: str, stem: str, kind: str):
    """kind = 'images' or 'masks'."""
    pats = [str(DATASET / subset / cls / kind / f"{stem}.{e}") for e in IMG_EXTS]
    pats += [str(DATASET / "**" / kind / f"{stem}.{e}") for e in IMG_EXTS]
    return first(*pats)


def find_annotation(subset: str, cls: str, stem: str):
    return first(str(DATASET / subset / cls / "annotations" / f"{stem}.json"),
                 str(DATASET / "**" / "annotations" / f"{stem}.json"))


def find_scaled(run_dir: Path, mod: str, fn: str):
    base = run_dir / mod / "postscale-axiswise"
    out = {}
    for suf, name in (("__scaled.step", "scaled.step"), ("__scaled.py", "scaled.py"),
                      ("__metadata.json", "scaled_metadata.json")):
        p = base / (fn + suf)
        if p.is_file():
            out[name] = str(p)
    return out


def cp(src, dst, got, label):
    if src and os.path.isfile(src):
        shutil.copy2(src, dst)
        got.append(label)
        return True
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out", default=str(AIWS / "outputs" / "best_results_selection"))
    args = ap.parse_args(argv)

    out_root = Path(args.out)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    lines = [f"# Best Cadrille results by IoU — top {args.top} per config", "",
             f"AIWS root: `{AIWS}`", "",
             "Criterion: **highest IoU** (Cadrille predicted CAD vs the SAM3D source mesh it was "
             "reconstructed from). 4 configs = model (RL / SFT) x modality (IMG / PC).", "",
             "Each `rank_NN_<sample>/` bundle is split by pipeline stage:", "",
             "- **Input Data/** — `input.png`, `mask.png`, `annotation.json` (upload the photo + mask into the GUI to reproduce)",
             "- **SAM3D Output/** — `source_mesh.stl` (the SAM3D mesh; the IoU comparison target)",
             "- **Cadrille Output/** — `pred.{stl,step,py}` (the predicted CAD) + `scaled.{step,py}` / `scaled_metadata.json` (post-scaled to mm, when available)",
             "- `metrics.json` at the bundle root (IoU, chamfer, candidate id, source shard)", ""]
    grand = 0
    for model, mod in CONFIGS:
        cfg = f"{model}_{mod}"
        run_dir = RUNS[model]
        recs = gather(run_dir, mod)
        recs.sort(key=lambda r: r["iou"], reverse=True)
        top = recs[: args.top]
        cdir = out_root / f"top{args.top}_iou_{cfg}"
        cdir.mkdir(parents=True, exist_ok=True)
        lines += [f"## {cfg}  —  {len(recs)} samples with valid IoU", "",
                  "| rank | sample | IoU | chamfer | cand | photo+mask | pred | scaled |",
                  "|---|---|---|---|---|---|---|---|"]
        for rank, r in enumerate(top, 1):
            fn, cid = r["fn"], r["id"]
            parts = fn.split("__")
            subset, cls, stem = (parts + ["", "", ""])[:3]
            rd = cdir / f"rank_{rank:02d}_{fn}"
            # pipeline-stage subfolders
            inp = rd / "Input Data"; sam = rd / "SAM3D Output"; cad = rd / "Cadrille Output"
            for s in (inp, sam, cad):
                s.mkdir(parents=True, exist_ok=True)
            got = []
            has_photo = cp(find_input(subset, cls, stem, "images"), inp / "input.png", got, "input.png")
            has_mask = cp(find_input(subset, cls, stem, "masks"), inp / "mask.png", got, "mask.png")
            cp(find_annotation(subset, cls, stem), inp / "annotation.json", got, "annotation.json")
            cp(find_source_mesh(run_dir, mod, fn), sam / "source_mesh.stl", got, "source_mesh.stl")
            cp(find_pred(r["shard"], "mesh", fn, cid, "stl"), cad / "pred.stl", got, "pred.stl")
            cp(find_pred(r["shard"], "brep", fn, cid, "step"), cad / "pred.step", got, "pred.step")
            cp(find_pred(r["shard"], "py", fn, cid, "py"), cad / "pred.py", got, "pred.py")
            has_scaled = False
            for name, p in find_scaled(run_dir, mod, fn).items():
                cp(p, cad / name, got, name); has_scaled = True
            # drop any empty stage dir (e.g. some sft picks have no scaled.*)
            for s in (inp, sam, cad):
                if not any(s.iterdir()):
                    s.rmdir()
            json.dump({"file_name": fn, "config": cfg, "subset": subset, "class": cls, "stem": stem,
                       "iou": r["iou"], "chamfer": r["cd"], "candidate_id": cid,
                       "source_shard": os.path.relpath(r["shard"], AIWS)},
                      open(rd / "metrics.json", "w"), indent=2)
            got.append("metrics.json")
            grand += 1
            ch = f"{r['cd']:.3e}" if isinstance(r["cd"], (int, float)) else "—"
            lines.append(f"| {rank} | `{fn}` | {r['iou']:.4f} | {ch} | +{cid} | "
                         f"{'yes' if has_photo and has_mask else ('photo' if has_photo else 'NO')} | "
                         f"{'yes' if (cad/'pred.stl').exists() else 'no'} | "
                         f"{'yes' if has_scaled else 'no'} |")
        lines.append("")
        if top:
            print(f"[{cfg}] {len(recs)} valid-IoU samples -> top {len(top)} "
                  f"(best {top[0]['iou']:.4f}, #{len(top)} {top[-1]['iou']:.4f})")
        else:
            print(f"[{cfg}] NO valid-IoU samples under {run_dir/mod}")
    (out_root / "best_results_summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nSelected {grand} samples into {out_root}")
    print("=" * 70)
    print((out_root / "best_results_summary.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
