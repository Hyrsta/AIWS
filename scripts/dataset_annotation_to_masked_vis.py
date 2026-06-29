#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageColor, ImageDraw

BASE = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = BASE / "data" / "aiws5.2-usable"
DEFAULT_OUTPUT_ROOT = BASE / "data" / "aiws5.2-usable-visualizations" / "masked_images"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert AIWS dataset polygon annotations into visualization images. "
            "By default, writes masked RGB images with black background while preserving dataset structure."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Dataset root containing subset/workpiece/(images,annotations) (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root for visualization images (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--mode",
        choices=("masked", "rgba", "overlay"),
        default="masked",
        help=(
            "masked: keep object, black elsewhere; "
            "rgba: keep object with transparent background; "
            "overlay: blend a colored mask onto the source image"
        ),
    )
    parser.add_argument(
        "--overlay-color",
        default="#00ff00",
        help="Overlay color for --mode overlay (default: #00ff00)",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Overlay alpha in [0,1] for --mode overlay (default: 0.45)",
    )
    parser.add_argument(
        "--subset",
        nargs="*",
        default=None,
        help="Optional subset filter, e.g. V1 V2 NEW",
    )
    parser.add_argument(
        "--workpiece",
        nargs="*",
        default=None,
        help="Optional workpiece filter, e.g. cover_plate bellmouth",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing visualization files",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=None,
        help="Optional CSV path. Defaults to <output-root>/manifest.csv",
    )
    return parser.parse_args()


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def find_image_path(images_dir: Path, stem: str) -> Path:
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing image for stem={stem} under {images_dir}")


def iter_annotation_paths(dataset_root: Path) -> Iterable[Path]:
    return sorted(dataset_root.glob("*/*/annotations/*.json"))


def load_annotation(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_mask(width: int, height: int, objects: list[dict]) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for obj in objects:
        pts = obj.get("segmentation") or []
        if not pts:
            continue
        polygon = [(float(x), float(y)) for x, y in pts]
        if len(polygon) >= 3:
            draw.polygon(polygon, fill=255, outline=255)
    return mask


def render_masked(rgb: Image.Image, mask: Image.Image) -> Image.Image:
    out = Image.new("RGB", rgb.size, (0, 0, 0))
    out.paste(rgb, mask=mask)
    return out


def render_rgba(rgb: Image.Image, mask: Image.Image) -> Image.Image:
    out = rgb.convert("RGBA")
    out.putalpha(mask)
    return out


def render_overlay(rgb: Image.Image, mask: Image.Image, color: tuple[int, int, int], alpha: float) -> Image.Image:
    alpha = clamp01(alpha)
    base = rgb.convert("RGBA")
    overlay = Image.new("RGBA", rgb.size, color + (0,))
    overlay_alpha = mask.point(lambda x: int(x * alpha))
    overlay.putalpha(overlay_alpha)
    return Image.alpha_composite(base, overlay).convert("RGB")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    manifest_csv = args.manifest_csv.resolve() if args.manifest_csv else output_root / "manifest.csv"

    if not dataset_root.exists():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")

    subset_filter = set(args.subset or [])
    workpiece_filter = set(args.workpiece or [])
    overlay_color = ImageColor.getrgb(args.overlay_color)

    rows: list[dict[str, object]] = []
    written = 0
    skipped = 0

    for ann_path in iter_annotation_paths(dataset_root):
        workpiece = ann_path.parents[1].name
        subset = ann_path.parents[2].name
        stem = ann_path.stem

        if subset_filter and subset not in subset_filter:
            continue
        if workpiece_filter and workpiece not in workpiece_filter:
            continue

        images_dir = ann_path.parents[1] / "images"
        image_path = find_image_path(images_dir, stem)
        rel_dir = ann_path.relative_to(dataset_root).parents[1]
        out_path = output_root / rel_dir / f"{stem}.png"

        if out_path.exists() and not args.overwrite:
            skipped += 1
            rows.append(
                {
                    "subset": subset,
                    "workpiece": workpiece,
                    "stem": stem,
                    "image_path": str(image_path.relative_to(BASE)),
                    "annotation_path": str(ann_path.relative_to(BASE)),
                    "output_path": str(out_path.relative_to(BASE)),
                    "mode": args.mode,
                    "object_count": "existing",
                    "status": "skipped_exists",
                }
            )
            continue

        data = load_annotation(ann_path)
        info = data.get("info") or {}
        width = int(info.get("width"))
        height = int(info.get("height"))
        objects = data.get("objects") or []

        if width <= 0 or height <= 0:
            raise SystemExit(f"Invalid image size in annotation: {ann_path}")
        if not objects:
            raise SystemExit(f"No annotated objects found: {ann_path}")

        mask = build_mask(width, height, objects)
        if mask.getbbox() is None:
            raise SystemExit(f"Annotation produced empty mask: {ann_path}")

        rgb = Image.open(image_path).convert("RGB")
        if rgb.size != (width, height):
            raise SystemExit(
                f"Image size mismatch for {image_path}: image={rgb.size}, ann={(width, height)}"
            )

        if args.mode == "masked":
            vis = render_masked(rgb, mask)
        elif args.mode == "rgba":
            vis = render_rgba(rgb, mask)
        else:
            vis = render_overlay(rgb, mask, overlay_color, args.overlay_alpha)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        vis.save(out_path)
        written += 1

        rows.append(
            {
                "subset": subset,
                "workpiece": workpiece,
                "stem": stem,
                "image_path": str(image_path.relative_to(BASE)),
                "annotation_path": str(ann_path.relative_to(BASE)),
                "output_path": str(out_path.relative_to(BASE)),
                "mode": args.mode,
                "object_count": len(objects),
                "status": "written",
            }
        )

    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with manifest_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subset",
                "workpiece",
                "stem",
                "image_path",
                "annotation_path",
                "output_path",
                "mode",
                "object_count",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Dataset root: {dataset_root}")
    print(f"Output root: {output_root}")
    print(f"Manifest CSV: {manifest_csv}")
    print(f"Mode: {args.mode}")
    print(f"Written: {written}")
    print(f"Skipped existing: {skipped}")


if __name__ == "__main__":
    main()
