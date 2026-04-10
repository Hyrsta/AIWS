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
OUT_DIR = BASE / "aiws5.2-usable"

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
    dst.symlink_to(rel)


def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {OUT_DIR}")

    if not IMG_DIR.exists() or not ANN_DIR.exists():
        raise SystemExit("Source dataset folders not found")

    metadata_dir = OUT_DIR / "metadata"
    unannotated_dir = OUT_DIR / "misc" / "unannotated_images"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    unannotated_dir.mkdir(parents=True, exist_ok=True)

    for subset in SUBSETS:
        for workpiece in ALL_WORKPIECES:
            base = OUT_DIR / subset / workpiece
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
    subset_counts = Counter()
    subset_workpiece_unique = defaultdict(set)
    subset_workpiece_objects = Counter()
    warnings = {
        "unknown_subset": [],
        "missing_image": [],
        "unmapped_labels": defaultdict(set),
        "missing_depth": [],
        "unannotated_images": sorted(image_stems - ann_stems),
    }

    for ann_path in ann_files:
        stem = ann_path.stem
        subset = infer_subset(stem)
        if subset == "UNKNOWN":
            warnings["unknown_subset"].append(stem)
            continue

        image_path = IMG_DIR / f"{stem}.png"
        if not image_path.exists():
            warnings["missing_image"].append(stem)
            continue

        data = json.loads(ann_path.read_text(encoding="utf-8"))
        info = data.get("info", {})
        objects = data.get("objects", [])
        labels_zh = [obj.get("category") for obj in objects if obj.get("category")]
        label_counter = Counter(labels_zh)
        mapped_labels_en = sorted({WORKPIECE_MAP[label] for label in label_counter if label in WORKPIECE_MAP})
        mapped_labels_zh = sorted({label for label in label_counter if label in WORKPIECE_MAP})
        unmapped = sorted({label for label in label_counter if label not in WORKPIECE_MAP})
        for label in unmapped:
            warnings["unmapped_labels"][label].add(stem)

        depth_path = expected_depth_path(stem, subset)
        if subset in {"V2", "NEW"} and depth_path is None:
            warnings["missing_depth"].append(stem)

        subset_counts[subset] += 1
        for label_zh, count in label_counter.items():
            if label_zh not in WORKPIECE_MAP:
                continue
            workpiece = WORKPIECE_MAP[label_zh]
            subset_workpiece_unique[(subset, workpiece)].add(stem)
            subset_workpiece_objects[(subset, workpiece)] += count

            base = OUT_DIR / subset / workpiece
            safe_symlink(image_path, base / "images" / image_path.name)
            safe_symlink(ann_path, base / "annotations" / ann_path.name)
            if depth_path is not None:
                depth_folder = "depth_png" if depth_path.suffix.lower() == ".png" else "depth_exr"
                safe_symlink(depth_path, base / depth_folder / depth_path.name)

        rows.append(
            {
                "stem": stem,
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
                "all_labels_zh": ";".join(sorted(set(labels_zh))),
                "unmapped_labels_zh": ";".join(unmapped),
                "image_path": str(image_path.relative_to(BASE)),
                "annotation_path": str(ann_path.relative_to(BASE)),
                "depth_path": str(depth_path.relative_to(BASE)) if depth_path else "",
            }
        )

    for stem in warnings["unannotated_images"]:
        image_path = IMG_DIR / f"{stem}.png"
        if image_path.exists():
            safe_symlink(image_path, unannotated_dir / image_path.name)

    csv_path = metadata_dir / "samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stem",
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
                "all_labels_zh",
                "unmapped_labels_zh",
                "image_path",
                "annotation_path",
                "depth_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source_of_truth": "isat_annotations",
        "output_type": "symlink_view",
        "subsets": {},
        "unannotated_images": warnings["unannotated_images"],
        "warnings": {
            "unknown_subset": warnings["unknown_subset"],
            "missing_image": warnings["missing_image"],
            "missing_depth": warnings["missing_depth"],
            "unmapped_labels": {k: sorted(v) for k, v in warnings["unmapped_labels"].items()},
        },
    }

    for subset in SUBSETS:
        subset_rows = [r for r in rows if r["subset"] == subset]
        summary["subsets"][subset] = {
            "num_samples": len(subset_rows),
            "num_with_depth": sum(1 for r in subset_rows if r["depth_type"] != "none"),
            "depth_type": "none" if subset == "V1" else ("png" if subset == "V2" else "exr"),
            "workpieces": {},
        }
        for workpiece in ALL_WORKPIECES:
            summary["subsets"][subset]["workpieces"][workpiece] = {
                "num_unique_images": len(subset_workpiece_unique[(subset, workpiece)]),
                "num_objects": subset_workpiece_objects[(subset, workpiece)],
            }

    (metadata_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readme = f"""# AIWS5.2 可用视图

这个目录是 **非破坏性的符号链接视图**，不会复制原始大文件，也不会改动原始数据。

## 目录组织

- `V1/`：无深度
- `V2/`：带 PNG 深度
- `NEW/`：带 EXR 深度

每个子集下面再按 5 类工件分组：

- `cover_plate`（盖板）
- `square_tube`（方管）
- `h_beam`（H型钢）
- `channel_steel`（槽钢）
- `bellmouth`（喇叭口）

每个工件目录中包含：

- `images/`
- `annotations/`
- `depth_png/` 或 `depth_exr/`（如果该子集有深度）

## 重要说明

1. 这里使用的标注真值来源是 **`isat_annotations/`**，不是 `train.json` / `val.json`。
2. 如果一张图里有多个工件类别，这张图会同时出现在多个工件目录里，**因为这里是符号链接，不会重复占用存储**。
3. `misc/unannotated_images/` 中放的是当前发现的有图像但没有标注的样本。
4. `metadata/samples.csv` 和 `metadata/summary.json` 提供机器可读的汇总信息。

## 当前已知情况

- `V1`：无深度
- `V2`：深度为 `.png`
- `NEW`：深度为 `.exr`
- 当前数据中 `channel_steel` 目录可能为空，因为现有标注里没有对应实例。
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    print(f"Created: {OUT_DIR}")
    print(f"Samples indexed: {len(rows)}")


if __name__ == "__main__":
    main()
