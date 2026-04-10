#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parents[1]
VIEW_DIR = BASE / "aiws5.2-usable"
CSV_PATH = VIEW_DIR / "metadata" / "samples.csv"
OUT_CSV = VIEW_DIR / "metadata" / "masks.csv"


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing samples CSV: {CSV_PATH}")

    rows_out = []
    generated = 0

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not as_bool(row.get("include_in_main_view", "False")):
                continue

            subset = row["subset"]
            workpiece = row["labels_en"]
            stem = row["stem"]
            ann_path = BASE / row["annotation_path"]
            img_path = BASE / row["image_path"]

            data = json.loads(ann_path.read_text(encoding="utf-8"))
            objects = data.get("objects", [])
            if len(objects) != 1:
                raise SystemExit(f"Expected exactly 1 object for kept sample: {stem}")

            info = data.get("info", {})
            width = int(info["width"])
            height = int(info["height"])
            obj = objects[0]
            pts = obj.get("segmentation") or []
            if not pts:
                raise SystemExit(f"Missing segmentation for sample: {stem}")

            mask_dir = VIEW_DIR / subset / workpiece / "masks"
            mask_dir.mkdir(parents=True, exist_ok=True)
            mask_path = mask_dir / f"{stem}.png"

            mask = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(mask)
            polygon = [(float(x), float(y)) for x, y in pts]
            draw.polygon(polygon, fill=255, outline=255)
            mask.save(mask_path)

            rows_out.append(
                {
                    "stem": stem,
                    "subset": subset,
                    "workpiece": workpiece,
                    "image_path": row["image_path"],
                    "annotation_path": row["annotation_path"],
                    "mask_path": str(mask_path.relative_to(BASE)),
                    "width": width,
                    "height": height,
                }
            )
            generated += 1

    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stem",
                "subset",
                "workpiece",
                "image_path",
                "annotation_path",
                "mask_path",
                "width",
                "height",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Generated masks: {generated}")
    print(f"Mask manifest: {OUT_CSV}")


if __name__ == "__main__":
    main()
