# CAD Body-Cleanup Post-Process Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, geometry-only post-process that removes *unrelated* hallucinated bodies from Cadrille's CAD output (while never breaking a legitimately-split object), surfaced in the GUI with a before/after comparison.

**Architecture:** A pure-python clustering core (`body_cleanup_core.py`, no CAD deps, fast unit tests on the Mac) is wrapped by a CadQuery/OCP I/O script (`cadrille_body_cleanup.py`, runs in `cadrille:latest`) that loads the `.step`, enumerates solids, computes exact pairwise surface gaps, clusters by proximity (gap ≤ ε·bbox-diagonal), keeps the largest-volume cluster, and exports a cleaned `.step`/`.stl` + metadata. The GUI job (`simple_reconstruct_job.py`) runs it as an always-on Docker stage after Cadrille selection; the frontend (`streamlit_app.py`) renders before/after STL previews.

**Tech Stack:** Python 3, CadQuery 2.5 + OCP (`BRepExtrema_DistShapeShape`), trimesh, FastAPI + Streamlit + Plotly, Docker (`cadrille:latest`), pytest.

**Spec:** `docs/superpowers/specs/2026-05-29-cad-body-cleanup-design.md`

---

## Environment, base, and the edit→test loop (read first)

- **Branch:** work on `feature/cad-body-cleanup` (already checked out locally, based on `dc01160`; PR targets `sync/rxl-runtime-20260529`).
- **The real runtime is RXL.** CadQuery/OCP and the `cadrille:latest` image live only on RXL (`/ssd1/rxl/zhankaiming/AIWS`). The Mac has no CAD stack. So:
  - **Pure core tests** (Task 2) run on the **Mac** (plain python, no deps beyond stdlib).
  - **CadQuery integration + real-case tests** (Tasks 3–4) and the **GUI E2E** (Tasks 5–7) run on **RXL in Docker**, reached by syncing changed files over SSH.
- **Sync snippet (Mac → RXL).** Run from the repo root. Only ever syncs *our* files; never touches RXL's uncommitted `cadrille_batch.py` / `e2e_sam3d_to_cadrille.py` (we don't list them).

  ```bash
  REPO=/Users/hyrsta/.openclaw/workspaces/welding-algorithm/.claude/worktrees/trusting-johnson-c8c655
  cd "$REPO"
  rsync -avR \
    scripts/body_cleanup_core.py \
    scripts/cadrille_body_cleanup.py \
    tests/test_body_cleanup_core.py \
    tests/test_body_cleanup.py \
    gui/backend/simple_reconstruct_job.py \
    gui/streamlit_app.py \
    RXL:/ssd1/rxl/zhankaiming/AIWS/
  ```
  (Sync only the files that exist at each step; `-R` preserves the `scripts/`, `tests/`, `gui/` layout. `rsync` over the `RXL` ssh alias works; if `rsync` is unavailable on RXL, fall back to `scp` per file.)
- **Docker test invocation (on RXL)** — CadQuery is available; pytest may not be, so our Docker test file has a plain `__main__` runner and is invoked with `python` (no pytest dependency):

  ```bash
  ssh RXL 'cd /ssd1/rxl/zhankaiming/AIWS && \
    docker run --rm --user $(id -u):$(id -g) \
      -v /ssd1/rxl/zhankaiming/AIWS:/repo:ro \
      cadrille:latest python /repo/tests/test_body_cleanup.py'
  ```
- **GUI is already running on RXL** (uvicorn `:18000`, Streamlit `:18501`, tmux `aiws-gui`), tunneled to the Mac at `http://localhost:18501`. `simple_reconstruct_job.py` is spawned fresh per reconstruction (picks up edits automatically); `streamlit_app.py` is hot-reloaded by Streamlit's watcher (use the "Rerun" prompt). `app.py` is **not** modified, so no uvicorn restart is needed.

## File structure

| File | Action | Responsibility |
|---|---|---|
| `scripts/body_cleanup_core.py` | **create** | Pure clustering + decision logic on plain numbers (volumes, gap matrix). No CAD deps. |
| `scripts/cadrille_body_cleanup.py` | **create** | CadQuery/OCP I/O: load `.step`, enumerate solids, compute gaps, call core, export cleaned `.step`/`.stl` + metadata. CLI. |
| `tests/test_body_cleanup_core.py` | **create** | Fast pure unit tests for the core (Mac). |
| `tests/test_body_cleanup.py` | **create** | CadQuery integration tests on synthetic solids (Docker, RXL). |
| `gui/backend/simple_reconstruct_job.py` | **modify** | New always-on "Body cleanup" Docker stage + result_paths keys + stage label. |
| `gui/streamlit_app.py` | **modify** | Before/after side-by-side previews, body-count caption, confidence warning, download buttons. |

The pure-core/I-O split is a deliberate refinement of the spec's single-script design: it makes the clustering logic testable on the Mac without Docker, keeping TDD fast. The spec's CLI, outputs, and `cleanup_metadata.json` schema (§4.1) are preserved exactly by `cadrille_body_cleanup.py`.

---

## Task 1: Verify environment & fixtures on RXL (no code)

**Goal:** De-risk everything by confirming CadQuery/OCP/BRepExtrema work in Docker, that real `.step` smoke cases exist, and that the sync path is correct.

- [ ] **Step 1: Confirm CadQuery + OCP BRepExtrema in the image**

Run:
```bash
ssh RXL 'docker run --rm cadrille:latest python -c "
import cadquery as cq
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
a = cq.Solid.makeBox(1,1,1)
b = cq.Solid.makeBox(1,1,1).moved(cq.Location(cq.Vector(3,0,0)))
e = BRepExtrema_DistShapeShape(a.wrapped, b.wrapped)
print(\"cq\", cq.__version__, \"done\", e.IsDone(), \"gap\", round(e.Value(),3))
print(\"solids\", len(cq.Compound.makeCompound([a,b]).Solids()))
"'
```
Expected: prints a CadQuery version, `done True gap 2.0`, `solids 2`.

- [ ] **Step 2: Locate the real-case smoke `.step` files**

Run:
```bash
ssh RXL 'cd /ssd1/rxl/zhankaiming/AIWS && \
  ls outputs/cadrille-rl-gpu-mem-rerun-20260412-211348/pc/shard-*/selected_brep/NEW-G90-WHITE-59.step \
     outputs/cadrille-rl-gpu-mem-rerun-20260412-211348/pc/shard-*/selected_brep/NEW-G140-47.step \
     outputs/cadrille-rl-gpu-mem-rerun-20260412-211348/pc/shard-*/selected_brep/NEW-G140-56.step 2>&1'
```
Expected: at least the three named `.step` files resolve (note their absolute paths for Task 4). If a name is missing, pick any `selected_brep/*.step` with ≥2 solids by scanning with the Step-1 snippet; record substitutes.

- [ ] **Step 3: Confirm the sync path & repo root on RXL**

Run:
```bash
ssh RXL 'ls -d /ssd1/rxl/zhankaiming/AIWS/scripts /ssd1/rxl/zhankaiming/AIWS/gui/backend /ssd1/rxl/zhankaiming/AIWS/tests 2>&1 || echo "tests/ absent (will be created on first sync)"'
```
Expected: `scripts/` and `gui/backend/` exist; `tests/` may be absent (rsync creates it).

No commit (verification only).

---

## Task 2: Pure clustering core + unit tests (Mac)

**Files:**
- Create: `scripts/body_cleanup_core.py`
- Test: `tests/test_body_cleanup_core.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_body_cleanup_core.py`:
```python
"""Pure unit tests for body_cleanup_core (no CAD deps; runs on the Mac)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from body_cleanup_core import connected_components, cluster_bodies, assess_confidence

D = 1.0  # bbox diagonal in all cases below
EPS = 0.07


def _gap(n, pairs):
    """Build a symmetric n x n gap matrix; pairs is {(i,j): gap}, default far."""
    m = [[0.0 if i == j else 0.42 for j in range(n)] for i in range(n)]
    for (i, j), g in pairs.items():
        m[i][j] = m[j][i] = g
    return m


def test_connected_components_basic():
    assert connected_components(3, [(0, 1)]) == [[0, 1], [2]]
    assert connected_components(3, []) == [[0], [1], [2]]
    assert connected_components(3, [(0, 1), (1, 2)]) == [[0, 1, 2]]


def test_single_body_is_noop():
    r = cluster_bodies([1.0], [[0.0]], D, EPS)
    assert r["n_clusters"] == 1 and r["noop"] is True
    assert r["kept_body_indices"] == [0]


def test_two_touching_one_cluster_both_kept():
    r = cluster_bodies([0.6, 0.4], _gap(2, {(0, 1): 0.0}), D, EPS)
    assert r["n_clusters"] == 1 and r["noop"] is True
    assert r["kept_body_indices"] == [0, 1]


def test_far_small_speck_removed():
    r = cluster_bodies([0.999, 0.001], _gap(2, {(0, 1): 0.42}), D, EPS)
    assert r["n_clusters"] == 2 and r["noop"] is False
    assert r["kept_body_indices"] == [0]  # larger-volume cluster kept


def test_far_large_body_removed_regardless_of_size():
    # far ⇒ junk even when sizeable; the larger-volume body is kept.
    r = cluster_bodies([0.55, 0.45], _gap(2, {(0, 1): 0.42}), D, EPS)
    assert r["n_clusters"] == 2
    assert r["kept_body_indices"] == [0]
    assert r["per_body"][1]["kept"] is False


def test_three_touching_plus_floating_speck():
    vols = [0.574, 0.243, 0.183, 0.001]
    pairs = {(0, 1): 0.005, (1, 2): 0.005, (0, 2): 0.006}  # 0,1,2 touch; 3 far
    r = cluster_bodies(vols, _gap(4, pairs), D, EPS)
    assert r["kept_body_indices"] == [0, 1, 2]
    assert r["per_body"][3]["kept"] is False
    assert r["n_clusters"] == 2 and r["noop"] is False


def test_epsilon_boundary_is_inclusive():
    # gap exactly at eps*D connects (<=).
    r = cluster_bodies([0.5, 0.5], _gap(2, {(0, 1): EPS * D}), D, EPS)
    assert r["n_clusters"] == 1


def test_assess_confidence_flags_large_runnerup():
    clusters = [{"index": 0, "volume": 0.55, "bodies": [0]},
                {"index": 1, "volume": 0.45, "bodies": [1]}]
    flag, reasons = assess_confidence(clusters, 0, removed_volume_fraction=0.45)
    assert flag is True and len(reasons) >= 1


def test_assess_confidence_quiet_for_tiny_speck():
    clusters = [{"index": 0, "volume": 0.999, "bodies": [0]},
                {"index": 1, "volume": 0.001, "bodies": [1]}]
    flag, reasons = assess_confidence(clusters, 0, removed_volume_fraction=0.001)
    assert flag is False and reasons == []


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} core tests passed.")


if __name__ == "__main__":
    _run_all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "$REPO" && python3 tests/test_body_cleanup_core.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'body_cleanup_core'`.

- [ ] **Step 3: Implement the core**

Create `scripts/body_cleanup_core.py`:
```python
"""Pure-geometry clustering logic for CAD body cleanup.

No CadQuery dependency, so it unit-tests anywhere. The CadQuery/OCP I/O lives in
cadrille_body_cleanup.py, which feeds plain numbers into these functions.
"""
from __future__ import annotations

from typing import Any


def connected_components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """Union-find connected components over n nodes. Returns components as sorted
    index lists, ordered by smallest member ascending."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    groups: dict[int, list[int]] = {}
    for x in range(n):
        groups.setdefault(find(x), []).append(x)
    return sorted((sorted(v) for v in groups.values()), key=lambda c: c[0])


def cluster_bodies(
    volumes: list[float],
    gap_matrix: list[list[float]],
    bbox_diagonal: float,
    epsilon_rel: float,
) -> dict[str, Any]:
    """Cluster solids by proximity and choose the main cluster.

    volumes[i]       : volume of solid i (>= 0)
    gap_matrix[i][j] : exact min surface gap between solids i, j (absolute units),
                       symmetric, diagonal ignored
    bbox_diagonal    : overall bounding-box diagonal D (> 0)
    epsilon_rel      : connect i, j when gap <= epsilon_rel * D
    """
    n = len(volumes)
    total_vol = float(sum(volumes)) if volumes else 0.0

    min_gap_rel: list[float] = []
    for i in range(n):
        others = [gap_matrix[i][j] for j in range(n) if j != i]
        g = min(others) if others else float("inf")
        min_gap_rel.append((g / bbox_diagonal) if bbox_diagonal > 0 else float("inf"))

    threshold = epsilon_rel * bbox_diagonal
    edges = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if gap_matrix[i][j] <= threshold
    ]
    comps = connected_components(n, edges)

    clusters: list[dict[str, Any]] = []
    for idx, bodies in enumerate(comps):
        vol = float(sum(volumes[b] for b in bodies))
        clusters.append({
            "index": idx,
            "bodies": bodies,
            "volume": vol,
            "volume_fraction": (vol / total_vol) if total_vol > 0 else 0.0,
            "kept": False,
        })

    kept_cluster_index = 0
    if clusters:
        # largest total volume; ties broken by smallest index (deterministic)
        kept_cluster_index = max(
            range(len(clusters)),
            key=lambda c: (clusters[c]["volume"], -c),
        )
        clusters[kept_cluster_index]["kept"] = True

    kept_body_indices = sorted(clusters[kept_cluster_index]["bodies"]) if clusters else []
    kept_set = set(kept_body_indices)

    body_cluster: dict[int, int] = {}
    for cl in clusters:
        for b in cl["bodies"]:
            body_cluster[b] = cl["index"]

    per_body: list[dict[str, Any]] = []
    for i in range(n):
        per_body.append({
            "index": i,
            "volume": float(volumes[i]),
            "volume_fraction": (float(volumes[i]) / total_vol) if total_vol > 0 else 0.0,
            "cluster": body_cluster.get(i, -1),
            "kept": i in kept_set,
            "min_gap_rel": min_gap_rel[i],
        })

    return {
        "clusters": clusters,
        "kept_cluster_index": kept_cluster_index,
        "kept_body_indices": kept_body_indices,
        "per_body": per_body,
        "n_clusters": len(clusters),
        "n_bodies_after": len(kept_body_indices),
        "noop": len(clusters) <= 1,
    }


def assess_confidence(
    clusters: list[dict[str, Any]],
    kept_cluster_index: int,
    removed_volume_fraction: float,
    *,
    removed_vol_frac_thresh: float = 0.05,
    runnerup_ratio_thresh: float = 0.30,
) -> tuple[bool, list[str]]:
    """Flag low-confidence deletions for GUI review. Does not change the decision."""
    reasons: list[str] = []
    if removed_volume_fraction > removed_vol_frac_thresh:
        reasons.append(
            f"removed volume {removed_volume_fraction:.3f} > {removed_vol_frac_thresh:.3f}"
        )
    if clusters:
        kept_vol = clusters[kept_cluster_index]["volume"]
        others = [c["volume"] for i, c in enumerate(clusters) if i != kept_cluster_index]
        runnerup = max(others) if others else 0.0
        if kept_vol > 0 and runnerup / kept_vol >= runnerup_ratio_thresh:
            reasons.append(
                f"runner-up cluster {runnerup / kept_vol:.2f}x kept (>= {runnerup_ratio_thresh:.2f})"
            )
    return (len(reasons) > 0, reasons)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "$REPO" && python3 tests/test_body_cleanup_core.py`
Expected: `PASS test_...` for each, then `All 10 core tests passed.`

- [ ] **Step 5: Commit**

```bash
cd "$REPO"
git add scripts/body_cleanup_core.py tests/test_body_cleanup_core.py
git commit -m "feat(cleanup): pure proximity-clustering core + unit tests"
```

---

## Task 3: CadQuery I/O script + Docker integration test (RXL)

**Files:**
- Create: `scripts/cadrille_body_cleanup.py`
- Test: `tests/test_body_cleanup.py`

- [ ] **Step 1: Write the failing integration test**

Create `tests/test_body_cleanup.py`:
```python
"""CadQuery integration tests for cadrille_body_cleanup. Runs in cadrille:latest:
    docker run --rm -v <repo>:/repo:ro cadrille:latest python /repo/tests/test_body_cleanup.py
Uses plain asserts + a __main__ runner so pytest is NOT required in the image.
"""
import json
import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import cadquery as cq
import cadrille_body_cleanup as bc


def _export(solids, path):
    cq.exporters.export(cq.Compound.makeCompound(solids), path)


def _args(in_step, out_dir, **kw):
    base = dict(in_step=in_step, out_dir=out_dir, out_stem=None, epsilon_rel=0.07,
                export_stl=True, stl_linear_deflection=0.001, stl_angular_deflection=0.1,
                confidence_removed_vol_frac=0.05, confidence_runnerup_ratio=0.30)
    base.update(kw)
    return SimpleNamespace(**base)


def _run(solids, **kw):
    d = tempfile.mkdtemp()
    in_step = os.path.join(d, "in.step")
    _export(solids, in_step)
    return bc.run_cleanup(_args(in_step, d, **kw)), d


def test_two_near_boxes_one_cluster_both_kept():
    a = cq.Solid.makeBox(1, 1, 1)
    b = cq.Solid.makeBox(1, 1, 1).moved(cq.Location(cq.Vector(1.05, 0, 0)))  # 0.05 gap
    meta, _ = _run([a, b])
    assert meta["n_bodies_before"] == 2
    assert meta["n_bodies_after"] == 2
    assert meta["noop"] is True


def test_main_plus_far_small_speck_removed():
    main = cq.Solid.makeBox(2, 2, 2)
    speck = cq.Solid.makeBox(0.1, 0.1, 0.1).moved(cq.Location(cq.Vector(8, 0, 0)))
    meta, _ = _run([main, speck])
    assert meta["n_bodies_before"] == 2
    assert meta["n_bodies_after"] == 1
    assert meta["n_bodies_removed"] == 1
    assert meta["noop"] is False


def test_main_plus_far_large_body_removed():
    main = cq.Solid.makeBox(2, 2, 2)                                   # vol 8
    big = cq.Solid.makeBox(1.8, 1.8, 1.8).moved(cq.Location(cq.Vector(9, 0, 0)))  # vol ~5.8, far
    meta, _ = _run([main, big])
    assert meta["n_bodies_after"] == 1  # far ⇒ removed despite size
    assert meta["confidence_flag"] is True  # large run-up should warn


def test_three_near_boxes_all_kept():
    a = cq.Solid.makeBox(1, 1, 1)
    b = cq.Solid.makeBox(1, 1, 1).moved(cq.Location(cq.Vector(1.05, 0, 0)))
    c = cq.Solid.makeBox(1, 1, 1).moved(cq.Location(cq.Vector(2.10, 0, 0)))
    meta, _ = _run([a, b, c])
    assert meta["n_bodies_before"] == 3
    assert meta["n_bodies_after"] == 3 and meta["noop"] is True


def test_single_box_noop_and_outputs_exist():
    meta, d = _run([cq.Solid.makeBox(1, 1, 1)])
    assert meta["noop"] is True and meta["n_bodies_after"] == 1
    assert os.path.exists(os.path.join(d, "in__cleaned.step"))
    assert os.path.exists(os.path.join(d, "in__cleaned.stl"))
    assert os.path.exists(os.path.join(d, "in__cleanup_metadata.json"))


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}: {exc!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} integration tests passed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _run_all()
```

- [ ] **Step 2: Sync and run to verify it fails**

Run (sync only the files that exist now — core + this test):
```bash
cd "$REPO"
rsync -avR scripts/body_cleanup_core.py tests/test_body_cleanup.py RXL:/ssd1/rxl/zhankaiming/AIWS/
ssh RXL 'cd /ssd1/rxl/zhankaiming/AIWS && docker run --rm --user $(id -u):$(id -g) -v /ssd1/rxl/zhankaiming/AIWS:/repo:ro cadrille:latest python /repo/tests/test_body_cleanup.py'
```
Expected: FAIL — `ModuleNotFoundError: No module named 'cadrille_body_cleanup'`.

- [ ] **Step 3: Implement the I/O script**

Create `scripts/cadrille_body_cleanup.py`:
```python
#!/usr/bin/env python3
"""Deterministic, geometry-only body cleanup for Cadrille CAD output.

Loads a .step, enumerates solids, clusters them by proximity (exact minimum
surface gap normalized by the overall bbox diagonal), keeps the largest-volume
cluster, removes the rest, and writes a cleaned .step (+ optional .stl) and
cleanup_metadata.json. Runs inside cadrille:latest (CadQuery 2.5 + OCP).

See docs/superpowers/specs/2026-05-29-cad-body-cleanup-design.md
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cadquery as cq
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from body_cleanup_core import assess_confidence, cluster_bodies  # noqa: E402

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


def compound_to_mesh(compound, linear_deflection: float, angular_deflection: float):
    vertices, faces = compound.tessellate(linear_deflection, angular_deflection)
    return trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)


def run_cleanup(args) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_stem or Path(args.in_step).stem
    out_step = out_dir / f"{stem}__cleaned.step"
    out_stl = out_dir / f"{stem}__cleaned.stl"
    out_meta = out_dir / f"{stem}__cleanup_metadata.json"

    solids = load_solids(args.in_step)
    n = len(solids)
    volumes = [float(s.Volume()) for s in solids]
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
            "confidence_reasons": [], "clusters": [], "bodies": [],
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
    kept_idx = clustering["kept_body_indices"]
    kept_set = set(kept_idx)
    kept_solids = [solids[i] for i in kept_idx]

    total_vol = sum(volumes) or 0.0
    removed_vol = sum(volumes[i] for i in range(n) if i not in kept_set)
    removed_vf = (removed_vol / total_vol) if total_vol > 0 else 0.0
    conf_flag, conf_reasons = assess_confidence(
        clustering["clusters"], clustering["kept_cluster_index"], removed_vf,
        removed_vol_frac_thresh=args.confidence_removed_vol_frac,
        runnerup_ratio_thresh=args.confidence_runnerup_ratio,
    )

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
        "n_bodies_after": len(kept_idx),
        "n_bodies_removed": n - len(kept_idx),
        "removed_volume_fraction": removed_vf,
        "kept_cluster_index": clustering["kept_cluster_index"],
        "noop": clustering["noop"],
        "confidence_flag": conf_flag,
        "confidence_reasons": conf_reasons,
        "clusters": [
            {"index": c["index"], "n_bodies": len(c["bodies"]), "volume": c["volume"],
             "volume_fraction": c["volume_fraction"], "kept": c["kept"]}
            for c in clustering["clusters"]
        ],
        "bodies": clustering["per_body"],
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
                      ("n_bodies_before", "n_bodies_after", "n_bodies_removed", "noop", "confidence_flag")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Sync and run to verify it passes**

Run:
```bash
cd "$REPO"
rsync -avR scripts/body_cleanup_core.py scripts/cadrille_body_cleanup.py tests/test_body_cleanup.py RXL:/ssd1/rxl/zhankaiming/AIWS/
ssh RXL 'cd /ssd1/rxl/zhankaiming/AIWS && docker run --rm --user $(id -u):$(id -g) -v /ssd1/rxl/zhankaiming/AIWS:/repo:ro cadrille:latest python /repo/tests/test_body_cleanup.py'
```
Expected: `PASS` for all five tests, then `5/5 integration tests passed.` (exit 0).

- [ ] **Step 5: Commit**

```bash
cd "$REPO"
git add scripts/cadrille_body_cleanup.py tests/test_body_cleanup.py
git commit -m "feat(cleanup): CadQuery body-cleanup script + Docker integration tests"
```

---

## Task 4: Real-case smoke on RXL (validation, no new code)

**Goal:** Confirm the spec's §5 real cases behave as predicted on actual Cadrille outputs.

- [ ] **Step 1: Run cleanup on the three real `.step` files**

Using the absolute paths recorded in Task 1 Step 2, run each in Docker (mounting the output root read-only and writing metadata to `/tmp`):
```bash
ssh RXL 'cd /ssd1/rxl/zhankaiming/AIWS && for f in \
  outputs/cadrille-rl-gpu-mem-rerun-20260412-211348/pc/shard-*/selected_brep/NEW-G90-WHITE-59.step \
  outputs/cadrille-rl-gpu-mem-rerun-20260412-211348/pc/shard-*/selected_brep/NEW-G140-47.step \
  outputs/cadrille-rl-gpu-mem-rerun-20260412-211348/pc/shard-*/selected_brep/NEW-G140-56.step ; do \
    echo "== $f =="; \
    docker run --rm --user $(id -u):$(id -g) -v /ssd1/rxl/zhankaiming/AIWS:/repo:ro -v /tmp/bc:/out \
      cadrille:latest python /repo/scripts/cadrille_body_cleanup.py \
      --in-step "/repo/$f" --out-dir /out --export-stl ; \
  done'
```
Expected (per spec §5), from the printed JSON line per file:
- `NEW-G90-WHITE-59`: `noop: true` (the two bodies touch → 1 cluster, both kept).
- `NEW-G140-47`: `n_bodies_removed: 1` (far speck removed).
- `NEW-G140-56`: `n_bodies_after: 3`, `n_bodies_removed: 1` (3 touching kept, 1 speck removed).

If any disagree, inspect that case's `cleanup_metadata.json` (`min_gap_rel`, `volume_fraction`, `bbox_diagonal`) under `/tmp/bc/` and reconcile against the spec's regimes (attached ≤ ~0.076, floating ≥ ~0.09). Only adjust `--epsilon-rel` default if a real case clearly straddles 0.07; record the finding in the spec.

No commit (validation only; note results in the PR description later).

---

## Task 5: GUI job — always-on body-cleanup stage

**Files:**
- Modify: `gui/backend/simple_reconstruct_job.py`
- Test: `tests/test_simple_job_cleanup_wiring.py` (create)

- [ ] **Step 1: Add the stage label**

In `gui/backend/simple_reconstruct_job.py`, replace the labels block:
```python
POSTSCALE_STAGE_LABEL = "Post-scaling: Aligning CAD to catalog (mm)"
PIPELINE_STAGE_LABELS = [
    "SAM3D: Loading checkpoints",
    "SAM3D: Generating mesh",
    "Cadrille: Preparing input",
    "Cadrille: Generating CAD result",
    POSTSCALE_STAGE_LABEL,
]
```
with:
```python
POSTSCALE_STAGE_LABEL = "Post-scaling: Aligning CAD to catalog (mm)"
BODY_CLEANUP_STAGE_LABEL = "Body cleanup: Removing hallucinated bodies"
PIPELINE_STAGE_LABELS = [
    "SAM3D: Loading checkpoints",
    "SAM3D: Generating mesh",
    "Cadrille: Preparing input",
    "Cadrille: Generating CAD result",
    BODY_CLEANUP_STAGE_LABEL,
    POSTSCALE_STAGE_LABEL,
]
```

- [ ] **Step 2: Add the new result_paths keys**

In `build_simple_result_paths`, replace:
```python
        "selected_brep": None,
        # Post-scaling slots — populated by run_postscale_stage when applicable.
```
with:
```python
        "selected_brep": None,
        # Body-cleanup slots — populated by run_body_cleanup_stage.
        "cleaned_brep_step": None,
        "cleaned_mesh_stl": None,
        "cleanup_metadata": None,
        "n_bodies_before": None,
        "n_bodies_after": None,
        # Post-scaling slots — populated by run_postscale_stage when applicable.
```

- [ ] **Step 3: Add `run_body_cleanup_stage`**

Immediately after the `run_postscale_stage` function definition, add:
```python
def run_body_cleanup_stage(
    *,
    repo_root: Path,
    job_root: Path,
    selected_brep_host: Path,
    docker_image: str,
) -> dict[str, Any]:
    """Run scripts/cadrille_body_cleanup.py in docker on the selected .step and
    copy cleaned outputs into <job_root>/results/. Returns result_paths keys to
    merge. Raises on hard failure (caller treats cleanup as best-effort)."""
    cleanup_work = (job_root / "cleanup").resolve()
    cleanup_work.mkdir(parents=True, exist_ok=True)
    brep_in_ctr = "/job/" + str(selected_brep_host.resolve().relative_to(job_root))

    docker_cmd = [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{repo_root}:/repo:ro",
        "-v", f"{job_root}:/job",
        docker_image,
        "python",
        "/repo/scripts/cadrille_body_cleanup.py",
        "--in-step", brep_in_ctr,
        "--out-dir", "/job/cleanup",
        "--export-stl",
    ]
    run_cmd(docker_cmd)

    stem = selected_brep_host.stem  # e.g. "cadrille_selected"
    src_step = cleanup_work / f"{stem}__cleaned.step"
    src_stl = cleanup_work / f"{stem}__cleaned.stl"
    src_meta = cleanup_work / f"{stem}__cleanup_metadata.json"

    results_root = job_root / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    populated: dict[str, Any] = {
        "cleaned_brep_step": copy_result_file(str(src_step), results_root / "cadrille_cleaned.step"),
        "cleaned_mesh_stl": copy_result_file(str(src_stl), results_root / "cadrille_cleaned.stl"),
        "cleanup_metadata": copy_result_file(str(src_meta), results_root / "cadrille_cleanup_metadata.json"),
    }
    if src_meta.exists():
        try:
            meta = json.loads(src_meta.read_text(encoding="utf-8"))
            populated["n_bodies_before"] = meta.get("n_bodies_before")
            populated["n_bodies_after"] = meta.get("n_bodies_after")
        except Exception:  # noqa: BLE001
            pass
    return {k: v for k, v in populated.items() if v is not None}
```

- [ ] **Step 4: Call the stage after Cadrille selection (before post-scaling)**

Replace:
```python
            selected_brep=selected_brep,
        )

        # ─── Post-scaling stage (optional) ───
```
with:
```python
            selected_brep=selected_brep,
        )

        # ─── Body cleanup stage (always-on when a BRep exists) ───
        if result_paths.get("selected_brep"):
            current_stage = "body_cleanup"
            write_status(
                status_path,
                status="running",
                stage="body_cleanup",
                stage_label=BODY_CLEANUP_STAGE_LABEL,
                result_paths=result_paths,
            )
            try:
                cleanup_results = run_body_cleanup_stage(
                    repo_root=repo_root,
                    job_root=job_root,
                    selected_brep_host=Path(result_paths["selected_brep"]),
                    docker_image=args.cadrille_docker_image,
                )
                result_paths.update(cleanup_results)
            except Exception as exc:  # noqa: BLE001
                print(f"[body-cleanup] stage skipped: {exc!r}", flush=True)

        # ─── Post-scaling stage (optional) ───
```

- [ ] **Step 5: Write a wiring unit test (Mac)**

Create `tests/test_simple_job_cleanup_wiring.py`:
```python
"""Fast wiring checks for the body-cleanup GUI stage (no Docker, Mac-runnable).
Verifies the stage label position and that result_paths exposes cleanup keys."""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

MOD = os.path.join(os.path.dirname(__file__), "..", "gui", "backend", "simple_reconstruct_job.py")
spec = importlib.util.spec_from_file_location("simple_reconstruct_job", MOD)
srj = importlib.util.module_from_spec(spec)
sys.modules["simple_reconstruct_job"] = srj
spec.loader.exec_module(srj)


def test_stage_label_inserted_before_postscale():
    labels = srj.PIPELINE_STAGE_LABELS
    assert srj.BODY_CLEANUP_STAGE_LABEL in labels
    assert labels.index(srj.BODY_CLEANUP_STAGE_LABEL) == labels.index("Cadrille: Generating CAD result") + 1
    assert labels.index(srj.BODY_CLEANUP_STAGE_LABEL) < labels.index(srj.POSTSCALE_STAGE_LABEL)


def test_result_paths_expose_cleanup_keys():
    with tempfile.TemporaryDirectory() as d:
        job_root = Path(d)
        glb = job_root / "m.glb"; glb.write_bytes(b"glb")
        stl = job_root / "m.stl"; stl.write_bytes(b"stl")
        rp = srj.build_simple_result_paths(
            job_root=job_root, sam3d_mesh_glb=glb, sam3d_mesh_stl=stl,
            cadrille_output_root=job_root, selected_mesh=None, selected_py=None, selected_brep=None,
        )
        for k in ("cleaned_brep_step", "cleaned_mesh_stl", "cleanup_metadata", "n_bodies_before", "n_bodies_after"):
            assert k in rp


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} wiring tests passed.")


if __name__ == "__main__":
    _run_all()
```

- [ ] **Step 6: Run the wiring test**

Run: `cd "$REPO" && python3 tests/test_simple_job_cleanup_wiring.py`
Expected: `PASS test_stage_label_inserted_before_postscale`, `PASS test_result_paths_expose_cleanup_keys`, `All 2 wiring tests passed.`

- [ ] **Step 7: Commit**

```bash
cd "$REPO"
git add gui/backend/simple_reconstruct_job.py tests/test_simple_job_cleanup_wiring.py
git commit -m "feat(cleanup): add always-on body-cleanup stage to GUI reconstruct job"
```

---

## Task 6: GUI frontend — before/after previews

**Files:**
- Modify: `gui/streamlit_app.py`

Streamlit UI is verified visually (Task 7), not by unit test.

- [ ] **Step 1: Add `fetch_cleanup_metadata`**

Immediately after the `fetch_postscale_metadata` function (≈ line 611), add:
```python
def fetch_cleanup_metadata(job: dict[str, Any]) -> dict[str, Any] | None:
    """Load cadrille_cleanup_metadata.json via /jobs/{id}/file for the
    before/after caption (body counts, removed-volume %, confidence flag)."""
    result_paths = job.get("result_paths") or {}
    meta_path = result_paths.get("cleanup_metadata")
    if not meta_path:
        return None
    try:
        response = requests.get(
            f"{BACKEND_URL}/jobs/{job['job_id']}/file",
            params={"path": meta_path},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        return None
```

- [ ] **Step 2: Render the before/after pair**

In `show_completed_result`, read the region around the "Mesh previews" block (≈ lines 1382–1413) and insert the following **immediately after** that block (after the `if scaled_mesh: … else: …` columns close, before the outputs/downloads section):
```python
    # ─── Body cleanup before/after ───
    cleaned_mesh = result_paths.get("cleaned_mesh_stl")
    if cleaned_mesh:
        cleanup_meta = fetch_cleanup_metadata(job) or {}
        nb = result_paths.get("n_bodies_before")
        if nb is None:
            nb = cleanup_meta.get("n_bodies_before")
        na = result_paths.get("n_bodies_after")
        if na is None:
            na = cleanup_meta.get("n_bodies_after")
        removed = cleanup_meta.get("n_bodies_removed")
        render_section_heading("Body cleanup (before / after)")
        bc_col1, bc_col2 = st.columns(2)
        with bc_col1:
            before_title = f"Before — {nb} bodies" if nb is not None else "Before (raw Cadrille)"
            show_mesh_preview(job, title=before_title, path=result_paths.get("selected_mesh"),
                              color="#94a3b8", units="(canonical units)")
        with bc_col2:
            if na is not None and removed:
                after_title = f"After — {na} bodies ({removed} removed)"
            elif na is not None:
                after_title = f"After — {na} bodies (no change)"
            else:
                after_title = "After (cleaned)"
            show_mesh_preview(job, title=after_title, path=cleaned_mesh,
                              color="#22c55e", units="(canonical units)")
        if cleanup_meta.get("confidence_flag"):
            reasons = "; ".join(cleanup_meta.get("confidence_reasons") or [])
            base_msg = ("Multiple comparable bodies were present — please verify the "
                        "cleanup didn't remove a real part.")
            st.warning(f"{base_msg} ({reasons})" if reasons else base_msg)
```

- [ ] **Step 3: Add download buttons for the cleaned outputs**

Read the existing scaled-outputs download block (≈ `streamlit_app.py:1579–1582`, which calls `render_output_group(...)` for `scaled_brep_step` / `scaled_py` / `scaled_metadata`). Mirror that exact call pattern to add a group for the cleaned outputs, using keys `cleaned_brep_step` and `cleaned_mesh_stl` (and `cleanup_metadata`), labelled "Cleaned CAD (post body-cleanup)". Match the surrounding `if result_paths.get(...)` guards used there.

- [ ] **Step 4: Sync to RXL and reload**

Run:
```bash
cd "$REPO"
rsync -avR gui/streamlit_app.py gui/backend/simple_reconstruct_job.py RXL:/ssd1/rxl/zhankaiming/AIWS/
```
Streamlit's file watcher will offer "Rerun" at `http://localhost:18501` — click it (no server restart needed).

- [ ] **Step 5: Commit**

```bash
cd "$REPO"
git add gui/streamlit_app.py
git commit -m "feat(cleanup): before/after body-cleanup previews + downloads in GUI"
```

---

## Task 7: End-to-end verification on RXL (GUI)

**Goal:** Confirm the full path works for a real upload and a known multi-body case.

- [ ] **Step 1: Ensure all changed files are on RXL**

Run the full sync snippet from the top of this plan (all six files).

- [ ] **Step 2: Run a reconstruction from the GUI**

In the browser (`http://localhost:18501`): upload a photo + mask and start a reconstruction (RL · PC). Watch the pipeline stages — confirm **"Body cleanup: Removing hallucinated bodies"** appears between "Cadrille: Generating CAD result" and post-scaling, and the job reaches "Done".

- [ ] **Step 3: Verify the before/after panel**

On the completed page, confirm:
- a "Body cleanup (before / after)" section with two side-by-side previews;
- "Before — N bodies" (raw) and "After — M bodies (K removed)" captions populated;
- when a real multi-body hallucination is present, the floating body is gone on the right while the main object is intact;
- the confidence warning appears only for ambiguous (comparable-volume) cases;
- download buttons produce a valid `cadrille_cleaned.step` / `.stl`.

- [ ] **Step 4: Spot-check artifacts on disk**

Run (substitute the actual job id shown in the GUI):
```bash
ssh RXL 'ls -la /ssd1/rxl/zhankaiming/AIWS/outputs/gui-simple/<job_id>/results/ | grep -E "cadrille_cleaned|cleanup_metadata"; \
  cat /ssd1/rxl/zhankaiming/AIWS/outputs/gui-simple/<job_id>/results/cadrille_cleanup_metadata.json'
```
Expected: `cadrille_cleaned.step`, `cadrille_cleaned.stl`, `cadrille_cleanup_metadata.json` present; metadata matches what the GUI showed.

- [ ] **Step 5: Finalize**

Use `superpowers:requesting-code-review`, then `superpowers:finishing-a-development-branch` to push `feature/cad-body-cleanup` and open a PR against `sync/rxl-runtime-20260529`. Summarize Task 4 real-case results and the E2E check in the PR description.

---

## Notes for the executor

- **Never** sync or commit RXL's `scripts/cadrille_batch.py` or `scripts/e2e_sam3d_to_cadrille.py` (uncommitted RXL work, unrelated).
- The cleanup is **best-effort** in the GUI: a Docker/cleanup failure must not fail the reconstruction job (the `try/except` in Task 5 Step 4 guarantees this) — the raw Cadrille output remains the result.
- Post-scaling still runs on the **raw** selected `.py` (clean-then-scale is out of scope, spec §6).
- Default `--epsilon-rel 0.07` is empirically grounded (spec §2/§3); only change it if Task 4 surfaces a real counter-example, and record the change in the spec.
