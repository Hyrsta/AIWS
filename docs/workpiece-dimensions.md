# Workpiece Dimensions (catalog ground truth)

Source: `docs/工件尺寸.png`. Units in the source are **meters**; values are reproduced verbatim in meters and also given in millimeters for downstream CAD use.

Lookup convention (from the source note):
> 可根据样本的文件名或者是数据采集中的目录名定位具体型号的工件
> The specific workpiece model can be identified from the sample filename or the data-collection directory name.

So the model code (`G90`, `F101`, `L75`, …) is extracted from the sample stem / directory and used as the key into the table below.

The bbox order reproduced from the source is `[X, Y, Z]`. The semantic meaning of each axis is **not** uniform across workpiece classes (see the per-class bbox shapes), so axis correspondence between Cadrille's canonical output and this catalog must be resolved per class — see the open question at the bottom of this file.

---

## YAML lookup (machine-readable)

```yaml
# All bboxes in METERS, in [X, Y, Z] order, as printed in 工件尺寸.png
cover_plate:    # 盖板
  G90:  [0.100000, 0.057590, 0.100000]
  G93:  [0.103000, 0.223000, 0.103000]
  G113: [0.124000, 0.115000, 0.124000]
  G140: [0.150000, 0.404000, 0.150000]

square_tube:    # 方管
  F101: [0.220000, 0.209000, 0.220000]
  F120: [0.220000, 0.209000, 0.220000]
  F150: [0.220000, 0.408000, 0.220000]

bellmouth:      # 喇叭口
  L75:  [0.150000, 0.087000, 0.300000]
  L148: [0.198000, 0.168000, 0.300000]

h_beam:         # H 型钢 — single entry in source, no model code
  default: [0.210000, 0.128000, 0.100000]
```

---

## Human-readable summary

### 盖板 / cover_plate

| Model | X (m) | Y (m) | Z (m) | X (mm) | Y (mm) | Z (mm) |
|---|---:|---:|---:|---:|---:|---:|
| G90  | 0.100000 | 0.057590 | 0.100000 | 100.00 |  57.59 | 100.00 |
| G93  | 0.103000 | 0.223000 | 0.103000 | 103.00 | 223.00 | 103.00 |
| G113 | 0.124000 | 0.115000 | 0.124000 | 124.00 | 115.00 | 124.00 |
| G140 | 0.150000 | 0.404000 | 0.150000 | 150.00 | 404.00 | 150.00 |

### 方管 / square_tube

| Model | X (m) | Y (m) | Z (m) | X (mm) | Y (mm) | Z (mm) |
|---|---:|---:|---:|---:|---:|---:|
| F101 | 0.220000 | 0.209000 | 0.220000 | 220.00 | 209.00 | 220.00 |
| F120 | 0.220000 | 0.209000 | 0.220000 | 220.00 | 209.00 | 220.00 |
| F150 | 0.220000 | 0.408000 | 0.220000 | 220.00 | 408.00 | 220.00 |

Note: F101 and F120 share identical bboxes in the source — verify whether this is intentional (same outer envelope, different wall thickness or other internal feature) or a transcription detail in the source image.

### 喇叭口 / bellmouth

| Model | X (m) | Y (m) | Z (m) | X (mm) | Y (mm) | Z (mm) |
|---|---:|---:|---:|---:|---:|---:|
| L75  | 0.150000 | 0.087000 | 0.300000 | 150.00 |  87.00 | 300.00 |
| L148 | 0.198000 | 0.168000 | 0.300000 | 198.00 | 168.00 | 300.00 |

### H 型钢 / h_beam

| Model | X (m) | Y (m) | Z (m) | X (mm) | Y (mm) | Z (mm) |
|---|---:|---:|---:|---:|---:|---:|
| default | 0.210000 | 0.128000 | 0.100000 | 210.00 | 128.00 | 100.00 |

H-beam has a single entry in the source with no model code, implying all 100 V1 H-beam samples share these dimensions.

---

## Coverage vs. current dataset

Current usable-view counts (from `docs/offline-pipeline-report.en.md` §2.2.2):

| Subset | cover_plate | square_tube | h_beam | bellmouth | channel_steel |
|---|---:|---:|---:|---:|---:|
| V1   | 200 |  99 | 100 | 194 | 0 |
| V2   | 524 |   0 |   0 |   0 | 0 |
| NEW  | 301 |   0 |   0 |   0 | 0 |

- **Covered by this table:** all current samples (cover_plate, square_tube, h_beam, bellmouth).
- **Not covered:** `channel_steel` — currently 0 usable samples, so no immediate gap, but the catalog will need to be extended if channel_steel data lands later.

---

## Open questions (must be resolved before per-axis metric scaling)

1. **Axis correspondence.** Cadrille emits CadQuery code in canonical units; its X/Y/Z need to be aligned with the catalog X/Y/Z before any anisotropic per-axis rewrite. For an isotropic single-scalar `s` rewrite (the Method-1 first-cut wrapper transform), this can be deferred — we just match max-extent → max-extent and accept that aspect-ratio mistakes from Cadrille survive. For a per-axis scale, we need either a fixed convention per class or an axis-sorting heuristic (sort canonical-bbox axes by extent and match to catalog axes sorted the same way).

2. **Sample-stem → model-code parser.** Confirm the exact filename pattern. The single known example so far (`NEW-G90-BLACK-24`) suggests the model code appears as a token between hyphens. A simple regex over `^[A-Z]+-([A-Z]\d+)-` should work for cover_plate (`G\d+`), square_tube (`F\d+`), and bellmouth (`L\d+`). H-beam needs a class-level fallback rather than a code lookup.

3. **F101 vs F120 identical envelopes.** Verify whether these really share the same bbox or whether the source image conflates them.
