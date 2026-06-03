#!/usr/bin/env python3
"""Apply metric post-scaling to a Cadrille tmp_py output.

Implements "Method 1: Metric post-scaling + ICP + code rewrite" from
docs/Cadrille Physical Scaling.docx, simplified for the AIWS workspace:

  - Scale source: catalog dimensions from docs/workpiece-dimensions.md
    (per workpiece class + model code), instead of depth/ICP.
  - Code rewrite: wrapper transform only. Appends a transform on the
    final `r` solid. No AST constant rewriting.
  - Integration: standalone. Does not modify cadrille_evaluate_wrapper.py.

Two rewrite modes (--rewrite-mode):

  axiswise (default)  Path A: anisotropic per-axis scale + center-at-origin.
                      Pairs sorted canonical axes with sorted catalog axes
                      so each axis matches its catalog target exactly.
                      Zero bbox residual on every axis, at the cost of
                      distorting Cadrille's intrinsic cross-section shape.

  uniform             Original POC: single scalar s = max(catalog)/max(canon).
                      Matches the catalog max axis exactly; non-max axes
                      inherit Cadrille's reconstruction proportions.

Per .py file:

    Cadrille filename → parse → (subset, class, stem, obj, cand)
    stem ─▶ model_code (e.g. G140) ─▶ catalog bbox_mm = [tx, ty, tz]
    exec(original .py) ─▶ r ─▶ canonical bbox (cx, cy, cz)

    axiswise:                                uniform:
      sort can.ext → sort cat.ext              s = max(cat)/max(can)
      pair by rank → per-axis (sx, sy, sz)
      center at origin, apply diagonal scale

    append wrapper transform to .py
    re-exec, measure after-scale bbox
    export <stem>__scaled.step / .stl, write metadata.json

  --dry-run    Parse + catalog lookup only. Does not import cadquery.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("cadrille_metric_postscale")


# ─────────────────────────────────────────────────────────────────────────────
# Catalog loader

DEFAULT_DIMENSIONS_DOC = Path(__file__).resolve().parents[1] / "docs" / "workpiece-dimensions.md"

# Per-class lookup heuristics, tried in order. The first one to find a
# catalog-matching model code wins. Two strategies are encoded:
#
#   1. NEW-style explicit code in the stem, e.g. "NEW-G140-52"
#      → (G|F|L)\d+ taken verbatim.
#   2. V1-style pinyin tag with a numeric suffix, e.g. "labakou1-75-14"
#      → numeric suffix prefixed with the class letter (G/F/L).
#      Note: the V1 number is only a *candidate* model code; we still
#      verify it exists in the catalog before accepting it.
MODEL_CODE_PATTERNS: dict[str, list[tuple[re.Pattern[str], str]]] = {
    "cover_plate": [
        (re.compile(r"(G\d+)"), "{0}"),                  # NEW-G140-52
        (re.compile(r"gaiban\d*-(\d+)-"), "G{0}"),       # V1-NO-DEPTH-gaiban1-65-07
    ],
    "square_tube": [
        (re.compile(r"(F\d+)"), "{0}"),
        # V1 square_tube samples use "diban\d+-…" with no F-code in the stem;
        # the numeric suffix is uncertain (not validated against catalog), so
        # there is no V1-pinyin fallback here yet. Pass --model-code to map.
    ],
    "bellmouth": [
        (re.compile(r"(L\d+)"), "{0}"),
        (re.compile(r"labakou\d*-(\d+)-"), "L{0}"),
    ],
    # h_beam: no per-model code in the catalog. Resolved via the 'default' entry.
    # V1 stems use "xinggang\d+-…", but the catalog has only one entry, so the
    # class-level fallback in resolve_catalog_entry handles them all.
}


@dataclasses.dataclass
class CatalogEntry:
    workpiece_class: str
    model_code: str
    bbox_m: tuple[float, float, float]   # as printed in the source image, in meters

    @property
    def bbox_mm(self) -> tuple[float, float, float]:
        return tuple(v * 1000.0 for v in self.bbox_m)  # type: ignore[return-value]


def load_catalog(dimensions_md: Path) -> dict[str, dict[str, CatalogEntry]]:
    """Read the YAML block embedded in docs/workpiece-dimensions.md.

    Returns: {workpiece_class: {model_code: CatalogEntry}}.
    """
    text = dimensions_md.read_text(encoding="utf-8")
    # Extract the first ```yaml ... ``` fence.
    m = re.search(r"```yaml\s*\n(.*?)\n```", text, flags=re.DOTALL)
    if not m:
        raise RuntimeError(f"No ```yaml ... ``` block found in {dimensions_md}")
    yaml_text = m.group(1)
    data = _parse_simple_yaml(yaml_text)

    catalog: dict[str, dict[str, CatalogEntry]] = {}
    for cls, models in data.items():
        catalog[cls] = {}
        for model_code, bbox in models.items():
            if not (isinstance(bbox, list) and len(bbox) == 3):
                raise RuntimeError(f"Bad bbox for {cls}/{model_code}: {bbox!r}")
            catalog[cls][model_code] = CatalogEntry(
                workpiece_class=cls,
                model_code=model_code,
                bbox_m=tuple(float(v) for v in bbox),  # type: ignore[arg-type]
            )
    return catalog


def _parse_simple_yaml(text: str) -> dict[str, dict[str, list[float]]]:
    """Tiny YAML reader for the fixed schema we control. Avoids a PyYAML dep.

    Accepts:
        class_name:
          model_code: [x, y, z]
          ...
    Comments after `#` and blank lines are skipped.
    """
    result: dict[str, dict[str, list[float]]] = {}
    current_cls: str | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" "):
            # top-level: class name
            key = line.rstrip(":").strip()
            current_cls = key
            result[current_cls] = {}
        else:
            if current_cls is None:
                raise RuntimeError("Found indented entry before any top-level class")
            inner = line.strip()
            if ":" not in inner:
                raise RuntimeError(f"Bad line in catalog YAML: {raw!r}")
            model_code, _, rhs = inner.partition(":")
            model_code = model_code.strip()
            try:
                bbox = ast.literal_eval(rhs.strip())
            except Exception as exc:
                raise RuntimeError(f"Bad bbox literal in catalog YAML: {raw!r}") from exc
            result[current_cls][model_code] = [float(v) for v in bbox]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Filename parser

# Two Cadrille filename conventions are supported:
#   tmp_py/ candidates : {subset}__{class}__{stem}__obj{NN}+{cand}.py
#   selected_py/ picks : {subset}__{class}__{stem}__obj{NN}.py     (no `+{cand}`)
# Workpiece class names contain underscores (cover_plate, square_tube, h_beam).
# Anchor on `__obj{N}[+{M}].py` from the right, then split the prefix on `__`.
FNAME_TAIL_RE = re.compile(
    r"^(?P<prefix>.+)__obj(?P<obj>\d+)(?:\+(?P<cand>\d+))?\.py$"
)


@dataclasses.dataclass
class SampleId:
    subset: str
    workpiece_class: str
    sample_stem: str
    obj_idx: int
    candidate_idx: int

    @classmethod
    def from_filename(cls, path: Path) -> "SampleId":
        m = FNAME_TAIL_RE.match(path.name)
        if not m:
            raise ValueError(
                f"Filename does not match Cadrille tmp_py convention: {path.name}"
            )
        prefix = m.group("prefix")
        parts = prefix.split("__", 2)
        if len(parts) != 3:
            raise ValueError(
                f"Filename prefix should split into 3 fields on '__': {prefix!r}"
            )
        cand_raw = m.group("cand")
        return cls(
            subset=parts[0],
            workpiece_class=parts[1],
            sample_stem=parts[2],
            obj_idx=int(m.group("obj")),
            candidate_idx=int(cand_raw) if cand_raw is not None else -1,
        )


def resolve_catalog_entry(
    sample: SampleId,
    catalog: dict[str, dict[str, CatalogEntry]],
    override_model_code: str | None = None,
    override_workpiece_class: str | None = None,
) -> CatalogEntry:
    """Pick the right catalog entry for this sample, falling back if needed.

    If `override_workpiece_class` is provided, it bypasses parsing-from-stem
    (used by the GUI flow where the user picks the class explicitly).
    """
    wclass = override_workpiece_class or sample.workpiece_class
    if wclass not in catalog:
        raise LookupError(
            f"Workpiece class {wclass!r} not in catalog "
            f"(known: {sorted(catalog)})"
        )

    if override_model_code:
        if override_model_code not in catalog[wclass]:
            raise LookupError(
                f"Override model code {override_model_code!r} not in catalog "
                f"for {wclass} (known: {sorted(catalog[wclass])})"
            )
        return catalog[wclass][override_model_code]

    # H-beam: single 'default' entry in the catalog, no per-model code.
    if wclass == "h_beam":
        if "default" not in catalog[wclass]:
            raise LookupError("h_beam catalog has no 'default' entry")
        return catalog[wclass]["default"]

    # Try each per-class lookup heuristic in order; accept the first that
    # produces a model code present in the catalog. Track near-misses so
    # the error message is actionable.
    near_misses: list[str] = []
    for regex, code_template in MODEL_CODE_PATTERNS.get(wclass, []):
        m = regex.search(sample.sample_stem)
        if not m:
            continue
        candidate_code = code_template.format(*m.groups())
        if candidate_code in catalog[wclass]:
            return catalog[wclass][candidate_code]
        near_misses.append(candidate_code)

    msg = f"Could not resolve a {wclass} model code from stem {sample.sample_stem!r}. "
    if near_misses:
        msg += (
            f"Tried: {near_misses} (none in catalog: {sorted(catalog[wclass])}). "
        )
    msg += "Use --model-code to override or extend docs/workpiece-dimensions.md."
    raise LookupError(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Cadrille code execution and bbox extraction

@dataclasses.dataclass
class CanonicalBBox:
    xlen: float
    ylen: float
    zlen: float
    xmin: float
    ymin: float
    zmin: float
    xmax: float
    ymax: float
    zmax: float

    @property
    def extents(self) -> tuple[float, float, float]:
        return (self.xlen, self.ylen, self.zlen)


def execute_cadrille_code(py_path: Path):
    """Execute the Cadrille .py and return the resulting cadquery object `r`.

    The Cadrille convention is that the last assignment binds the final solid
    to the name `r`. We exec the file in a fresh namespace and return that.
    """
    import cadquery as cq  # noqa: F401  # imported by the script under exec

    code = py_path.read_text(encoding="utf-8")
    ns: dict[str, Any] = {"__name__": "__cadrille_postscale__"}
    exec(compile(code, str(py_path), "exec"), ns, ns)
    if "r" not in ns:
        raise RuntimeError(f"{py_path} did not produce a variable named `r`")
    return ns["r"]


def measure_canonical_bbox(r) -> CanonicalBBox:
    """BoundingBox of the Cadrille-produced solid in its native (canonical) units."""
    shape = r.val()
    bb = shape.BoundingBox()
    return CanonicalBBox(
        xlen=bb.xlen, ylen=bb.ylen, zlen=bb.zlen,
        xmin=bb.xmin, ymin=bb.ymin, zmin=bb.zmin,
        xmax=bb.xmax, ymax=bb.ymax, zmax=bb.zmax,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scale + rewrite

@dataclasses.dataclass
class UniformScalePlan:
    scale: float
    canonical: CanonicalBBox


@dataclasses.dataclass
class AxiswiseScalePlan:
    """Anisotropic per-axis scale plan with axis-aligning rotation.

    Combines two operations into a single 3x3 affine matrix M:

      1. Axis permutation: the k-th-longest canonical axis is rotated to
         land on the k-th-longest catalog axis. This makes the output's
         literal X/Y/Z line up with the catalog's literal X/Y/Z.
      2. Per-axis scale: after permutation, each axis is scaled so its
         extent matches the catalog extent on that axis exactly.

    The resulting 3x3 has exactly one non-zero entry per row and column:
    M[cat_i, can_i] = catalog_extent[cat_i] / canonical_extent[can_i] where
    (can_i, cat_i) is the rank-paired axis correspondence.

    If the raw permutation is a reflection (det < 0), the smallest-catalog
    axis is flipped (its row negated). This adds a 180° rotation about
    that axis and yields a proper rotation (det > 0) — preserves the
    output's right-handed coordinate frame.

    Translation: before applying M, the solid is translated by -center_canonical
    so the bbox center sits at the origin. After M, the bbox is centered
    at the origin in the metric frame.
    """
    matrix_3x3: list[list[float]]
    center_canonical: tuple[float, float, float]
    axis_map: list[tuple[int, int]]
    det_note: str
    canonical: CanonicalBBox

    @property
    def per_axis_scale(self) -> tuple[float, float, float]:
        """The scale magnitude applied to each catalog axis (row max-abs)."""
        return tuple(max(abs(v) for v in row) for row in self.matrix_3x3)  # type: ignore[return-value]


def compute_uniform_scale(canonical: CanonicalBBox, bbox_mm: tuple[float, float, float]) -> UniformScalePlan:
    """Single uniform scale: max-extent matches max-catalog dimension."""
    max_canonical = max(canonical.extents)
    max_metric = max(bbox_mm)
    if max_canonical <= 0:
        raise ValueError(f"Canonical bbox has non-positive max extent: {canonical}")
    return UniformScalePlan(scale=max_metric / max_canonical, canonical=canonical)


def _det3(M: list[list[float]]) -> float:
    """Determinant of a 3x3 matrix. Hand-written to avoid a numpy dependency."""
    return (
        M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
      - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
      + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])
    )


def compute_axiswise_scale(canonical: CanonicalBBox, bbox_mm: tuple[float, float, float]) -> AxiswiseScalePlan:
    """Build a 3x3 matrix that rotates + scales canonical → catalog axes.

    Algorithm:
      1. Sort canonical extents and catalog extents by descending magnitude.
      2. Pair them by rank: the k-th-longest canonical axis maps to the
         k-th-longest catalog axis.
      3. Build M such that the canonical can_i unit vector is sent to
         catalog cat_i with magnitude catalog_extent[cat_i]/canonical_extent[can_i].
         i.e. M[cat_i, can_i] = scale_for_that_pair, others zero.
      4. If det(M) < 0 (reflection), flip the row corresponding to the
         smallest catalog axis to recover det > 0 (proper rotation).

    For ties in either side (e.g. G140 catalog X=Z=150), Python's stable
    sort keeps original-index order. The numeric result is independent of
    which way ties resolve because the paired targets are equal.
    """
    extents = canonical.extents
    for i, v in enumerate(extents):
        if v <= 0:
            raise ValueError(f"Canonical extent {'XYZ'[i]} is non-positive: {v} (bbox={canonical})")

    canonical_order = sorted(range(3), key=lambda i: -extents[i])
    catalog_order = sorted(range(3), key=lambda i: -bbox_mm[i])

    # Build the 3x3 matrix: M[cat_i, can_i] = scale_pair.
    M: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(3)]
    axis_map: list[tuple[int, int]] = []
    for k in range(3):
        can_axis = canonical_order[k]
        cat_axis = catalog_order[k]
        M[cat_axis][can_axis] = bbox_mm[cat_axis] / extents[can_axis]
        axis_map.append((can_axis, cat_axis))

    # Ensure proper rotation: if det < 0, flip the smallest-catalog axis row.
    det = _det3(M)
    if det < 0:
        smallest_cat_axis = min(range(3), key=lambda i: bbox_mm[i])
        for j in range(3):
            M[smallest_cat_axis][j] *= -1.0
        det_note = f"raw det = -1 (reflection); flipped row {'XYZ'[smallest_cat_axis]} to recover proper rotation"
    else:
        det_note = "raw det = +1 (proper rotation, no flip needed)"

    center = (
        (canonical.xmin + canonical.xmax) / 2.0,
        (canonical.ymin + canonical.ymax) / 2.0,
        (canonical.zmin + canonical.zmax) / 2.0,
    )

    return AxiswiseScalePlan(
        matrix_3x3=M,
        center_canonical=center,
        axis_map=axis_map,
        det_note=det_note,
        canonical=canonical,
    )


WRAPPER_TEMPLATE_UNIFORM = """

# ─── AIWS metric post-scaling (wrapper transform; uniform scale) ───
# Applied by scripts/cadrille_metric_postscale.py  (mode: uniform).
# Source of scale: docs/workpiece-dimensions.md  ({workpiece_class}/{model_code})
# Canonical extents (xlen,ylen,zlen): {canonical_extents}
# Target metric bbox (mm)            : {bbox_mm}
# Scale = max(metric) / max(canonical) = {scale:.6f} mm/code-unit
_AIWS_SCALE_MM = {scale:.6f}
_aiws_scaled_solid = r.val().scale(_AIWS_SCALE_MM)
r = cq.Workplane(obj=_aiws_scaled_solid)
"""


WRAPPER_TEMPLATE_AXISWISE = """

# ─── AIWS metric post-scaling (axiswise: rotation + per-axis scale) ───
# Applied by scripts/cadrille_metric_postscale.py  (mode: axiswise).
# Source of scale: docs/workpiece-dimensions.md  ({workpiece_class}/{model_code})
# Canonical extents (xlen,ylen,zlen): {canonical_extents}
# Target metric bbox (mm)            : {bbox_mm}
# Axis pairing (canonical → catalog, by descending extent):
#   {axis_map_str}
# Per-axis scale magnitudes on catalog X/Y/Z: ({s_cat_x:.9f}, {s_cat_y:.9f}, {s_cat_z:.9f})
# Combined matrix note: {det_note}
# Canonical bbox center pre-transform (xc,yc,zc): ({cx:.9f}, {cy:.9f}, {cz:.9f})
#
# Operations applied in order:
#   1. translate solid by -center_canonical (so the bbox center reaches origin)
#   2. apply combined 4x4 affine M (rotation + diagonal scale)
#   3. result: solid is centered at origin; literal X/Y/Z bbox extents
#      equal the catalog's literal X/Y/Z extents
_AIWS_CX, _AIWS_CY, _AIWS_CZ = ({cx:.9f}, {cy:.9f}, {cz:.9f})
_aiws_centered = r.val().translate((-_AIWS_CX, -_AIWS_CY, -_AIWS_CZ))
_aiws_matrix = cq.Matrix([
    [{m00:.12f}, {m01:.12f}, {m02:.12f}, 0.0],
    [{m10:.12f}, {m11:.12f}, {m12:.12f}, 0.0],
    [{m20:.12f}, {m21:.12f}, {m22:.12f}, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
_aiws_scaled_solid = _aiws_centered.transformGeometry(_aiws_matrix)
r = cq.Workplane(obj=_aiws_scaled_solid)
"""


def _format_axis_map(axis_map: list[tuple[int, int]]) -> str:
    names = "XYZ"
    parts = []
    for rank, (can_i, cat_i) in enumerate(axis_map):
        parts.append(f"rank{rank}: canonical_{names[can_i]} → catalog_{names[cat_i]}")
    return "; ".join(parts)


def _per_axis_scale_on_catalog_axes(M: list[list[float]]) -> tuple[float, float, float]:
    """For a row-permuted diagonal matrix, the scale on catalog axis i is the
    magnitude of the single non-zero entry in row i."""
    return tuple(max(abs(v) for v in row) for row in M)  # type: ignore[return-value]


def rewrite_code(
    original_py: Path,
    plan: "UniformScalePlan | AxiswiseScalePlan",
    entry: CatalogEntry,
) -> str:
    body = original_py.read_text(encoding="utf-8")
    if not body.endswith("\n"):
        body += "\n"

    if isinstance(plan, UniformScalePlan):
        footer = WRAPPER_TEMPLATE_UNIFORM.format(
            workpiece_class=entry.workpiece_class,
            model_code=entry.model_code,
            canonical_extents=tuple(round(v, 6) for v in plan.canonical.extents),
            bbox_mm=tuple(round(v, 6) for v in entry.bbox_mm),
            scale=plan.scale,
        )
    elif isinstance(plan, AxiswiseScalePlan):
        cx, cy, cz = plan.center_canonical
        M = plan.matrix_3x3
        s_cat = _per_axis_scale_on_catalog_axes(M)
        footer = WRAPPER_TEMPLATE_AXISWISE.format(
            workpiece_class=entry.workpiece_class,
            model_code=entry.model_code,
            canonical_extents=tuple(round(v, 6) for v in plan.canonical.extents),
            bbox_mm=tuple(round(v, 6) for v in entry.bbox_mm),
            axis_map_str=_format_axis_map(plan.axis_map),
            s_cat_x=s_cat[0], s_cat_y=s_cat[1], s_cat_z=s_cat[2],
            det_note=plan.det_note,
            cx=cx, cy=cy, cz=cz,
            m00=M[0][0], m01=M[0][1], m02=M[0][2],
            m10=M[1][0], m11=M[1][1], m12=M[1][2],
            m20=M[2][0], m21=M[2][1], m22=M[2][2],
        )
    else:
        raise TypeError(f"Unknown plan type: {type(plan).__name__}")
    return body + footer


# ─────────────────────────────────────────────────────────────────────────────
# Export + metadata

def export_step_and_stl(r_scaled, out_step: Path, out_stl: Path | None) -> None:
    import cadquery as cq
    cq.exporters.export(r_scaled, str(out_step), exportType="STEP")
    if out_stl is not None:
        cq.exporters.export(r_scaled, str(out_stl), exportType="STL")


def measure_after_scale(r_scaled) -> CanonicalBBox:
    # Same shape as CanonicalBBox; values are now in mm because the scale was in mm.
    return measure_canonical_bbox(r_scaled)


# ─────────────────────────────────────────────────────────────────────────────
# CLI

def process_one(
    py_path: Path,
    catalog: dict[str, dict[str, CatalogEntry]],
    out_dir: Path,
    override_model_code: str | None,
    dry_run: bool,
    export_stl: bool,
    rewrite_mode: str,
    override_workpiece_class: str | None = None,
) -> dict[str, Any]:
    sample = SampleId.from_filename(py_path)
    entry = resolve_catalog_entry(sample, catalog, override_model_code, override_workpiece_class)
    bbox_mm = entry.bbox_mm

    record: dict[str, Any] = {
        "input_py": str(py_path),
        "sample": dataclasses.asdict(sample),
        "catalog": {
            "workpiece_class": entry.workpiece_class,
            "model_code": entry.model_code,
            "bbox_m": list(entry.bbox_m),
            "bbox_mm": list(bbox_mm),
        },
        "rewrite_mode": rewrite_mode,
        "dry_run": dry_run,
    }

    if dry_run:
        log.info(
            "[dry-run] %s → class=%s model=%s target_bbox_mm=%s mode=%s",
            py_path.name, entry.workpiece_class, entry.model_code, bbox_mm, rewrite_mode,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = out_dir / f"{py_path.stem}__metadata.json"
        meta_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        record["output_metadata"] = str(meta_path)
        return record

    # Full run: execute, measure, scale, rewrite, export.
    log.info("[exec] %s", py_path.name)
    r_original = execute_cadrille_code(py_path)
    canonical = measure_canonical_bbox(r_original)

    if rewrite_mode == "uniform":
        plan: "UniformScalePlan | AxiswiseScalePlan" = compute_uniform_scale(canonical, bbox_mm)
        log.info(
            "[scale uniform] canonical_extents=%s  target_bbox_mm=%s  s=%.6f mm/unit",
            tuple(round(v, 3) for v in canonical.extents), bbox_mm, plan.scale,
        )
    elif rewrite_mode == "axiswise":
        plan = compute_axiswise_scale(canonical, bbox_mm)
        s_cat = _per_axis_scale_on_catalog_axes(plan.matrix_3x3)
        log.info(
            "[scale axiswise] canonical_extents=%s  target_bbox_mm=%s  s_on_catalog_axes=(%.4f, %.4f, %.4f)  %s",
            tuple(round(v, 3) for v in canonical.extents), bbox_mm,
            s_cat[0], s_cat[1], s_cat[2], plan.det_note,
        )
    else:
        raise ValueError(f"Unknown rewrite_mode: {rewrite_mode!r}")

    rewritten = rewrite_code(py_path, plan, entry)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_py = out_dir / f"{py_path.stem}__scaled.py"
    out_py.write_text(rewritten, encoding="utf-8")

    # Re-execute the rewritten file (cleanest verification path).
    r_scaled = execute_cadrille_code(out_py)
    after = measure_after_scale(r_scaled)

    out_step = out_dir / f"{py_path.stem}__scaled.step"
    out_stl = out_dir / f"{py_path.stem}__scaled.stl" if export_stl else None
    export_step_and_stl(r_scaled, out_step, out_stl)

    if isinstance(plan, UniformScalePlan):
        scale_record: dict[str, Any] = {"mode": "uniform", "scale_mm_per_unit": plan.scale}
    else:
        s_cat = _per_axis_scale_on_catalog_axes(plan.matrix_3x3)
        scale_record = {
            "mode": "axiswise",
            "matrix_3x3": [list(row) for row in plan.matrix_3x3],
            "scale_on_catalog_axes": {"X": s_cat[0], "Y": s_cat[1], "Z": s_cat[2]},
            "center_canonical": list(plan.center_canonical),
            "axis_map": [
                {"rank": k, "canonical_axis": "XYZ"[can_i], "catalog_axis": "XYZ"[cat_i]}
                for k, (can_i, cat_i) in enumerate(plan.axis_map)
            ],
            "det_note": plan.det_note,
        }

    record.update({
        "canonical_bbox": dataclasses.asdict(canonical),
        "scale": scale_record,
        "after_scale_bbox_mm": dataclasses.asdict(after),
        "output_py": str(out_py),
        "output_step": str(out_step),
        "output_stl": str(out_stl) if out_stl else None,
    })
    meta_path = out_dir / f"{py_path.stem}__metadata.json"
    meta_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    record["output_metadata"] = str(meta_path)

    # Sanity check.
    after_ext = after.extents
    if rewrite_mode == "uniform":
        rel_err = abs(max(after_ext) - max(bbox_mm)) / max(bbox_mm)
        log.info("[check uniform] after-scale max extent = %.3f mm  (target max = %.3f mm, rel err = %.2e)",
                 max(after_ext), max(bbox_mm), rel_err)
        if rel_err > 1e-3:
            log.warning("uniform: max-extent mismatch > 0.1%% (rel err = %.2e)", rel_err)
    else:
        # axiswise (with rotation): expect literal X/Y/Z bbox to match catalog X/Y/Z.
        rel_errs = [abs(a - t) / t for a, t in zip(after_ext, bbox_mm)]
        max_rel = max(rel_errs)
        log.info("[check axiswise] after=%s  target=%s  per-axis rel err = (%.2e, %.2e, %.2e)",
                 tuple(round(v, 3) for v in after_ext),
                 tuple(round(v, 3) for v in bbox_mm),
                 rel_errs[0], rel_errs[1], rel_errs[2])
        if max_rel > 1e-4:
            log.warning("axiswise: literal X/Y/Z mismatch > 0.01%% (max rel err = %.2e)", max_rel)
    return record


def load_step_solid(step_path: Path):
    """Load a STEP into a cadquery Workplane bound to `r` (same object the .py
    path yields). Used for the --in-step flow (e.g. body-cleanup output)."""
    import cadquery as cq
    r = cq.importers.importStep(str(step_path))
    if r is None or r.val() is None:
        raise RuntimeError(f"Could not import STEP: {step_path}")
    vals = r.vals()
    if len(vals) > 1:
        # Combine multiple solids into one compound so the bbox + transforms
        # cover the whole (cleaned) part, not just the first solid.
        r = cq.Workplane(obj=cq.Compound.makeCompound(vals))
    return r


def apply_plan_in_process(r, plan: "UniformScalePlan | AxiswiseScalePlan"):
    """Apply a scale plan directly to a loaded solid, mirroring the wrapper
    templates used by the .py-rewrite path (no source rewrite needed)."""
    import cadquery as cq
    if isinstance(plan, UniformScalePlan):
        return cq.Workplane(obj=r.val().scale(plan.scale))
    if isinstance(plan, AxiswiseScalePlan):
        cx, cy, cz = plan.center_canonical
        centered = r.val().translate((-cx, -cy, -cz))
        M = plan.matrix_3x3
        mat = cq.Matrix([
            [M[0][0], M[0][1], M[0][2], 0.0],
            [M[1][0], M[1][1], M[1][2], 0.0],
            [M[2][0], M[2][1], M[2][2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        return cq.Workplane(obj=centered.transformGeometry(mat))
    raise TypeError(f"Unknown plan type: {type(plan).__name__}")


def process_one_step(
    step_path: Path,
    catalog: dict[str, dict[str, CatalogEntry]],
    out_dir: Path,
    override_model_code: str | None,
    export_stl: bool,
    rewrite_mode: str,
    override_workpiece_class: str | None,
) -> dict[str, Any]:
    """Post-scale a STEP file directly (no source .py to rewrite). Requires
    --workpiece-class since a step filename has no Cadrille tmp_py convention."""
    if not override_workpiece_class:
        raise ValueError("--in-step requires --workpiece-class")
    sample = SampleId(
        subset="GUI", workpiece_class=override_workpiece_class,
        sample_stem=step_path.stem, obj_idx=1, candidate_idx=-1,
    )
    entry = resolve_catalog_entry(sample, catalog, override_model_code, override_workpiece_class)
    bbox_mm = entry.bbox_mm

    record: dict[str, Any] = {
        "input_step": str(step_path),
        "sample": dataclasses.asdict(sample),
        "catalog": {
            "workpiece_class": entry.workpiece_class,
            "model_code": entry.model_code,
            "bbox_m": list(entry.bbox_m),
            "bbox_mm": list(bbox_mm),
        },
        "rewrite_mode": rewrite_mode,
        "dry_run": False,
    }

    log.info("[exec step] %s", step_path.name)
    r_original = load_step_solid(step_path)
    canonical = measure_canonical_bbox(r_original)

    if rewrite_mode == "uniform":
        plan: "UniformScalePlan | AxiswiseScalePlan" = compute_uniform_scale(canonical, bbox_mm)
    elif rewrite_mode == "axiswise":
        plan = compute_axiswise_scale(canonical, bbox_mm)
    else:
        raise ValueError(f"Unknown rewrite_mode: {rewrite_mode!r}")

    r_scaled = apply_plan_in_process(r_original, plan)
    after = measure_after_scale(r_scaled)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_step = out_dir / f"{step_path.stem}__scaled.step"
    out_stl = out_dir / f"{step_path.stem}__scaled.stl" if export_stl else None
    export_step_and_stl(r_scaled, out_step, out_stl)

    if isinstance(plan, UniformScalePlan):
        scale_record: dict[str, Any] = {"mode": "uniform", "scale_mm_per_unit": plan.scale}
    else:
        s_cat = _per_axis_scale_on_catalog_axes(plan.matrix_3x3)
        scale_record = {
            "mode": "axiswise",
            "matrix_3x3": [list(row) for row in plan.matrix_3x3],
            "scale_on_catalog_axes": {"X": s_cat[0], "Y": s_cat[1], "Z": s_cat[2]},
            "center_canonical": list(plan.center_canonical),
            "axis_map": [
                {"rank": k, "canonical_axis": "XYZ"[can_i], "catalog_axis": "XYZ"[cat_i]}
                for k, (can_i, cat_i) in enumerate(plan.axis_map)
            ],
            "det_note": plan.det_note,
        }

    record.update({
        "canonical_bbox": dataclasses.asdict(canonical),
        "scale": scale_record,
        "after_scale_bbox_mm": dataclasses.asdict(after),
        "output_step": str(out_step),
        "output_stl": str(out_stl) if out_stl else None,
    })
    meta_path = out_dir / f"{step_path.stem}__metadata.json"
    meta_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    record["output_metadata"] = str(meta_path)

    # Sanity check (axiswise): literal X/Y/Z should match the catalog.
    after_ext = after.extents
    rel_errs = [abs(a - t) / t for a, t in zip(after_ext, bbox_mm)]
    log.info("[check in-step %s] after=%s target=%s max_rel_err=%.2e",
             rewrite_mode, tuple(round(v, 3) for v in after_ext),
             tuple(bbox_mm), max(rel_errs))
    return record


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--py", type=Path, help="A single Cadrille tmp_py .py file.")
    p.add_argument("--py-dir", type=Path, help="Directory of Cadrille tmp_py .py files (processes all).")
    p.add_argument("--in-step", type=Path, help="A single .step CAD file (e.g. body-cleanup output). Scales the loaded solid directly; requires --workpiece-class.")
    p.add_argument("--dimensions", type=Path, default=DEFAULT_DIMENSIONS_DOC,
                   help=f"Path to workpiece-dimensions.md (default: {DEFAULT_DIMENSIONS_DOC}).")
    p.add_argument("--out-dir", type=Path, required=True, help="Where to write outputs.")
    p.add_argument("--model-code", type=str, default=None,
                   help="Override the catalog model code (e.g. G140). Required for V1 samples whose pinyin stem does not match the catalog.")
    p.add_argument("--workpiece-class", type=str, default=None,
                   choices=["cover_plate", "square_tube", "bellmouth", "h_beam"],
                   help="Override the workpiece class (e.g. cover_plate). Required for GUI uploads whose filename does not parse to a real catalog class.")
    p.add_argument("--dry-run", action="store_true",
                   help="Parse filename + catalog lookup only. Does not import cadquery.")
    p.add_argument("--rewrite-mode", choices=["axiswise", "uniform"], default="axiswise",
                   help="axiswise (default) = anisotropic per-axis scale + center-at-origin (Path A); "
                        "uniform = single scalar scale via max-extent matching.")
    p.add_argument("--export-stl", action="store_true", help="Also export STL (default: STEP only).")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    n_inputs = sum(bool(x) for x in (args.py, args.py_dir, args.in_step))
    if n_inputs == 0:
        p.error("Provide one of --py, --py-dir, or --in-step.")
    if n_inputs > 1:
        p.error("Provide only one of --py, --py-dir, or --in-step.")

    catalog = load_catalog(args.dimensions)
    log.info("Loaded catalog from %s: %s",
             args.dimensions, {k: sorted(v) for k, v in catalog.items()})

    if args.in_step:
        out = process_one_step(
            step_path=args.in_step, catalog=catalog, out_dir=args.out_dir,
            override_model_code=args.model_code, export_stl=args.export_stl,
            rewrite_mode=args.rewrite_mode, override_workpiece_class=args.workpiece_class,
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "_postscale_summary.json").write_text(
            json.dumps({"results": [out], "n_failed": 0}, indent=2), encoding="utf-8")
        return 0

    if args.py:
        targets = [args.py]
    else:
        targets = sorted(args.py_dir.glob("*.py"))
        if not targets:
            p.error(f"No .py files found in {args.py_dir}")

    results: list[dict[str, Any]] = []
    failed = 0
    for path in targets:
        try:
            r = process_one(
                py_path=path,
                catalog=catalog,
                out_dir=args.out_dir,
                override_model_code=args.model_code,
                dry_run=args.dry_run,
                export_stl=args.export_stl,
                rewrite_mode=args.rewrite_mode,
                override_workpiece_class=args.workpiece_class,
            )
            results.append(r)
        except Exception as exc:
            failed += 1
            log.error("FAILED %s: %s", path.name, exc)
            results.append({"input_py": str(path), "error": str(exc)})

    summary_path = args.out_dir / "_postscale_summary.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"results": results, "n_failed": failed}, indent=2),
        encoding="utf-8",
    )
    log.info("Wrote summary: %s (failed=%d / total=%d)",
             summary_path, failed, len(targets))
    return 1 if failed and len(targets) == 1 else 0


if __name__ == "__main__":
    sys.exit(main())
