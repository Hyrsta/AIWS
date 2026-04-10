#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
IMG_DIR = BASE / "aiws5.2-dataset" / "images"
DEPTH_DIR = BASE / "aiws5.2-dataset" / "depth"
ANN_DIR = BASE / "isat_annotations"
TRAIN_JSON = BASE / "train.json"
VAL_JSON = BASE / "val.json"
OUT_DIR = BASE / "aiws5.2-usable-split"

WORKPIECE_MAP = {
    "盖板": "cover_plate",
    "方管": "square_tube",
    "H型钢": "h_beam",
    "槽钢": "channel_steel",
    "喇叭口": "bellmouth",
}

ALL_WORKPIECES = [
    "cover_plate",
    "square_tube",
    "h_beam",
    "channel_steel",
    "bellmouth",
]

SUBSETS = ["V1", "V2", "NEW"]
SPLITS = ["train", "val"]


def infer_subset(stem: str) -> str:
    if stem.startswith("V1-"):
        return "V1"
    if stem.startswith("G90-v2-"):
        return "V2"
    if stem.startswith("NEW-"):
        return "NEW"
    return "UNKNOWN"


def expected_depth_path(stem: str, subset: str) -> Path | None:
    if subset == "V2":
        p = DEPTH_DIR / f"{stem}.png"
        return p if p.exists() else None
    if subset == "NEW":
        p = DEPTH_DIR / f"{stem}.exr"
        return p if p.exists() else None
    return None


def safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    rel = Path(os.path.relpath(src.resolve(), start=dst.parent.resolve()))
    try:
        dst.symlink_to(rel)
    except FileExistsError:
        return


def load_split_membership() -> dict[str, str]:
    split_membership: dict[str, str] = {}
    for split, path in [("train", TRAIN_JSON), ("val", VAL_JSON)]:
        data = json.loads(path.read_text(encoding="utf-8"))
        for image in data.get("images", []):
            stem = Path(image["file_name"]).stem
            if stem in split_membership and split_membership[stem] != split:
                raise SystemExit(f"Image assigned to multiple splits: {stem}")
            split_membership[stem] = split
    return split_membership


def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {OUT_DIR}")
    if not TRAIN_JSON.exists() or not VAL_JSON.exists():
        raise SystemExit("train.json / val.json not found")

    split_membership = load_split_membership()
    metadata_dir = OUT_DIR / "metadata"
    misc_dir = OUT_DIR / "misc" / "unannotated_images"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    misc_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        for subset in SUBSETS:
            for workpiece in ALL_WORKPIECES:
                base = OUT_DIR / split / subset / workpiece
                (base / "images").mkdir(parents=True, exist_ok=True)
                (base / "annotations").mkdir(parents=True, exist_ok=True)
                if subset == "V2":
                    (base / "depth_png").mkdir(parents=True, exist_ok=True)
                elif subset == "NEW":
                    (base / "depth_exr").mkdir(parents=True, exist_ok=True)

    ann_files = sorted(ANN_DIR.glob("*.json"))
    image_stems = {p.stem for p in IMG_DIR.glob("*.png")}
    ann_stems = {p.stem for p in ann_files}

    rows = []
    unique_image_counts = Counter()
    object_counts = Counter()
    warnings = {
        "unknown_subset": [],
        "missing_image": [],
        "missing_split": [],
        "missing_depth": [],
        "unmapped_labels": defaultdict(set),
        "unannotated_images": sorted(image_stems - ann_stems),
    }

    for ann_path in ann_files:
        stem = ann_path.stem
        subset = infer_subset(stem)
        if subset == "UNKNOWN":
            warnings["unknown_subset"].append(stem)
            continue

        split = split_membership.get(stem)
        if split is None:
            warnings["missing_split"].append(stem)
            continue

        image_path = IMG_DIR / f"{stem}.png"
        if not image_path.exists():
            warnings["missing_image"].append(stem)
            continue

        data = json.loads(ann_path.read_text(encoding="utf-8"))
        info = data.get("info", {})
        objects = data.get("objects", [])
        label_counter = Counter(obj.get("category") for obj in objects if obj.get("category"))
        mapped_labels_zh = sorted([lab for lab in label_counter if lab in WORKPIECE_MAP])
        mapped_labels_en = sorted(WORKPIECE_MAP[lab] for lab in mapped_labels_zh)
        unmapped = sorted([lab for lab in label_counter if lab not in WORKPIECE_MAP])
        for label in unmapped:
            warnings["unmapped_labels"][label].add(stem)

        depth_path = expected_depth_path(stem, subset)
        if subset in {"V2", "NEW"} and depth_path is None:
            warnings["missing_depth"].append(stem)

        for label_zh, count in label_counter.items():
            if label_zh not in WORKPIECE_MAP:
                continue
            workpiece = WORKPIECE_MAP[label_zh]
            base = OUT_DIR / split / subset / workpiece
            safe_symlink(image_path, base / "images" / image_path.name)
            safe_symlink(ann_path, base / "annotations" / ann_path.name)
            if depth_path is not None:
                depth_folder = "depth_png" if depth_path.suffix.lower() == ".png" else "depth_exr"
                safe_symlink(depth_path, base / depth_folder / depth_path.name)
            unique_image_counts[(split, subset, workpiece)] += 1
            object_counts[(split, subset, workpiece)] += count

        rows.append(
            {
                "stem": stem,
                "split": split,
                "subset": subset,
                "image_name": image_path.name,
                "annotation_name": ann_path.name,
                "depth_name": depth_path.name if depth_path else "",
                "depth_type": depth_path.suffix.lower().lstrip(".") if depth_path else "none",
                "width": info.get("width", ""),
                "height": info.get("height", ""),
                "num_objects": len(objects),
                "labels_zh": ";".join(mapped_labels_zh),
                "labels_en": ";".join(mapped_labels_en),
                "image_path": str(image_path.relative_to(BASE)),
                "annotation_path": str(ann_path.relative_to(BASE)),
                "depth_path": str(depth_path.relative_to(BASE)) if depth_path else "",
            }
        )

    for stem in warnings["unannotated_images"]:
        image_path = IMG_DIR / f"{stem}.png"
        if image_path.exists():
            safe_symlink(image_path, misc_dir / image_path.name)

    with (metadata_dir / "samples.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stem",
                "split",
                "subset",
                "image_name",
                "annotation_name",
                "depth_name",
                "depth_type",
                "width",
                "height",
                "num_objects",
                "labels_zh",
                "labels_en",
                "image_path",
                "annotation_path",
                "depth_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source_of_truth": "isat_annotations",
        "split_source": "train.json + val.json (membership only)",
        "output_type": "split_symlink_view",
        "splits": {},
        "unannotated_images": warnings["unannotated_images"],
        "warnings": {
            "unknown_subset": warnings["unknown_subset"],
            "missing_image": warnings["missing_image"],
            "missing_split": warnings["missing_split"],
            "missing_depth": warnings["missing_depth"],
            "unmapped_labels": {k: sorted(v) for k, v in warnings["unmapped_labels"].items()},
        },
    }

    for split in SPLITS:
        summary["splits"][split] = {}
        for subset in SUBSETS:
            summary["splits"][split][subset] = {
                "num_samples": sum(1 for r in rows if r["split"] == split and r["subset"] == subset),
                "workpieces": {},
            }
            for workpiece in ALL_WORKPIECES:
                summary["splits"][split][subset]["workpieces"][workpiece] = {
                    "num_linked_images": unique_image_counts[(split, subset, workpiece)],
                    "num_objects": object_counts[(split, subset, workpiece)],
                }

    (metadata_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readme = """# AIWS5.2 split-aware usable view

This is a non-destructive symlink view.

Rules used:
- split membership comes from `train.json` and `val.json`
- annotation truth comes from `isat_annotations/`
- depth linking follows subset rules:
  - `V1`: no depth
  - `V2`: `.png`
  - `NEW`: `.exr`

Layout:
- `train/` and `val/`
- under each split: `V1/`, `V2/`, `NEW/`
- under each subset: five workpiece folders

This avoids the dropped-instance bug in the COCO export while still preserving the train/val split.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    print(f"Created: {OUT_DIR}")
    print(f"Samples indexed: {len(rows)}")


if __name__ == "__main__":
    main()
