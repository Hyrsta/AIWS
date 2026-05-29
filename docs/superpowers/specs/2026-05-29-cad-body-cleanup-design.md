# CAD body-cleanup post-process — design spec

- Date: 2026-05-29
- Status: approved design, pending spec review → implementation plan
- Scope: offline CAD reconstruction pipeline (`SAM3D → Cadrille`), GUI single-reconstruction flow

## 1. Problem & goal

Cadrille frequently emits CAD whose materialized solid contains **more than one
body** when the target workpiece is a single object. Two distinct cases occur:

- **Hallucination** — one (or more) extra body that is *unrelated* to the
  intended object (a floating speck or chunk). This hurts fidelity and should be
  removed.
- **Legitimate split** — the *real* object is reconstructed as multiple discrete
  bodies that, taken together, still represent the intended part. Here we must
  **keep every body**, because deleting one would move the result further from
  ground truth, even though body-count purity would prefer a single body.

Goal: a **deterministic, geometry-only (ground-truth-free)** post-process that
detects how many bodies exist, removes only the unrelated ones, never breaks a
legitimately-split object, and is surfaced in the GUI with a **before/after**
visual comparison.

## 2. Empirical grounding

Profiled 150 real `RL · PC` selected outputs
(`outputs/cadrille-rl-gpu-mem-rerun-20260412-211348/pc/shard-*/selected_brep/*.step`)
with CadQuery 2.5 + OCP `BRepExtrema_DistShapeShape` inside `cadrille:latest`:

- **Multi-body is the norm:** solid-count histogram `{1: 57, 2: 65, 3: 16, 4: 8,
  5: 2, 6: 2}` → **62% of outputs have ≥2 solids.**
- **Junk vs. split separates cleanly** once the per-body nearest-neighbor (NN)
  distance is normalized by the object bbox diagonal:
  - *Floating hallucinations* — tiny volume + far: e.g. `[1.0, 0.0]` with
    gap/diag `0.42`; `[0.999, 0.001, 0, 0]` with gaps up to `0.41`.
  - *Genuinely-split object* — comparable volume + touching: e.g. `[0.6, 0.4]`
    gap `0.0`; `[0.574, 0.243, 0.183, 0.0]` where the three big pieces touch
    (gap `~0.005`) and only a speck floats off at `0.42`.
  - **Regimes:** attached pieces sit at gap/diag ≤ ~0.076; floating junk at
    ≥ ~0.09. A threshold in between cleanly separates them.

This is what makes **proximity clustering** the right policy and fixes the
default threshold (below).

## 3. Approach — proximity clustering (chosen)

Operate at the **B-rep / solid level** on the exported `.step` (so the output is
a genuinely cleaned CAD model, not just a trimmed mesh), purely geometric:

1. Load `.step` → enumerate solids (`cq.importers.importStep(path).val().Solids()`).
2. Compute the overall bbox diagonal `D` and each solid's volume.
3. Build an undirected graph over solids: add an edge between solids `i, j` when
   their **exact minimum surface distance** `BRepExtrema_DistShapeShape(i, j)` is
   `≤ ε · D`.
4. Connected components → **clusters**.
5. **Keep the single cluster with the largest total volume; delete all other
   clusters.**
   - A split-but-touching real object → one cluster → kept whole.
   - A floating speck/chunk → its own cluster → deleted, regardless of its size.
   - Domain assumption (valid for these workpieces — `cover_plate`,
     `square_tube`, `h_beam`, `bellmouth` are single connected parts):
     far-apart ⇒ unrelated.
6. Rebuild a compound from the kept solids; export cleaned `.step` and a
   tessellated `.stl`; write `cleanup_metadata.json`.
7. **No-op** when there is a single cluster (1 body, or all bodies connected):
   cleaned output is a copy of the input, `noop: true`. Outputs are always
   produced so the GUI always has an "after" to show.

### Parameters / defaults

| Parameter | Default | Notes |
|---|---|---|
| `epsilon-rel` (ε) | **0.07** | Gap threshold as fraction of bbox diagonal. Sits above the ≤0.076 "attached" band and below the ≥0.09 "floating" band. CLI-tunable. |
| main-cluster rule | largest total volume | Cluster with max summed solid volume is kept. |
| `confidence-removed-vol-frac` | 0.05 | Raise confidence flag if removed volume > 5% of total. |
| `confidence-runnerup-ratio` | 0.30 | Raise flag if 2nd-largest cluster volume ≥ 30% of the kept cluster (ambiguous which is "main"). |
| STL tessellation | `(0.001, 0.1)` | Matches `repos/cadrille/evaluate.py` linear/angular deflection. |

The confidence flag does not change the deletion decision; it tells the GUI to
prompt the user to eyeball the before/after.

## 4. Components

### 4.1 `scripts/cadrille_body_cleanup.py` (new, runs in `cadrille:latest`)

CLI:

```
--in-step PATH                 (required) input .step
--out-dir DIR                  (required) output directory
--out-stem NAME                (optional) default = input stem
--epsilon-rel FLOAT            default 0.07
--export-stl                   also write tessellated .stl
--stl-linear-deflection FLOAT  default 0.001
--stl-angular-deflection FLOAT default 0.1
--confidence-removed-vol-frac FLOAT  default 0.05
--confidence-runnerup-ratio FLOAT    default 0.30
```

Outputs into `--out-dir`:

- `<stem>__cleaned.step`
- `<stem>__cleaned.stl` (when `--export-stl`)
- `<stem>__cleanup_metadata.json`

`cleanup_metadata.json` schema:

```json
{
  "input_step": "…",
  "epsilon_rel": 0.07,
  "bbox_diagonal": 0.0,
  "n_bodies_before": 6,
  "n_clusters": 3,
  "n_bodies_after": 4,
  "n_bodies_removed": 2,
  "removed_volume_fraction": 0.012,
  "kept_cluster_index": 0,
  "noop": false,
  "confidence_flag": false,
  "confidence_reasons": [],
  "clusters": [
    {"index": 0, "n_bodies": 4, "volume": 0.0, "volume_fraction": 0.99, "kept": true}
  ],
  "bodies": [
    {"index": 0, "volume": 0.0, "volume_fraction": 0.84, "cluster": 0, "kept": true, "min_gap_rel": 0.006}
  ]
}
```

Error handling:

- Missing/garbage input → exit non-zero with a clear message; the GUI stage
  treats this as "cleanup skipped" and leaves the raw output as the result.
- Import yields 0 solids → write `noop: true`, cleaned = copy of input.

### 4.2 GUI job integration — `gui/backend/simple_reconstruct_job.py`

- Add an **always-on** stage label `"Body cleanup: Removing hallucinated
  bodies"` to `PIPELINE_STAGE_LABELS`, inserted **after** `"Cadrille: Generating
  CAD result"` and **before** `POSTSCALE_STAGE_LABEL`.
- New `run_body_cleanup_stage(...)` mirroring `run_postscale_stage`'s Docker
  pattern:

  ```
  docker run --rm --user UID:GID -v repo:/repo:ro -v job_root:/job IMAGE \
    python /repo/scripts/cadrille_body_cleanup.py \
    --in-step /job/results/cadrille_selected.step \
    --out-dir /job/cleanup --export-stl
  ```

- Runs on `results/cadrille_selected.step`; copies outputs into `results/` as
  `cadrille_cleaned.step`, `cadrille_cleaned.stl`,
  `cadrille_cleanup_metadata.json`.
- Adds `result_paths` keys (initialized to `None` in `build_simple_result_paths`):
  `cleaned_brep_step`, `cleaned_mesh_stl`, `cleanup_metadata`,
  `n_bodies_before`, `n_bodies_after` (counts read back from metadata).
- **Independent of post-scaling**: post-scaling continues to run on the raw
  selected `.py`. (Clean-then-scale ordering is explicitly out of scope for v1.)
- Skips gracefully (stage marked done, no cleaned artifacts) if
  `cadrille_selected.step` is absent.

### 4.3 GUI before/after — `gui/streamlit_app.py`

- In `show_completed_result`, render a **side-by-side** pair using `st.columns`
  and the existing `show_mesh_preview(job, title, path, color)`:
  - left: `"Before — {n_bodies_before} bodies"` → `cadrille_selected_mesh.stl`
  - right: `"After — {n_bodies_after} bodies ({removed} removed)"` →
    `cadrille_cleaned.stl`
- Add `fetch_cleanup_metadata(job)` (mirror of `fetch_postscale_metadata`) to
  read removed-volume % and the confidence flag; show `st.warning(...)` when
  `confidence_flag` is true ("multiple comparable bodies — please verify the
  cleanup didn't remove a real part").
- Add download buttons for the cleaned `.step` / `.stl` via `render_output_group`.

## 5. Verification

- **Unit tests** (`tests/test_body_cleanup.py`, run in `cadrille:latest`) on
  synthetic CadQuery solids:
  - two adjacent/touching boxes → 1 cluster, both kept;
  - main box + small far box → main kept, far removed;
  - main box + **large** far box → larger kept, the other removed (far ⇒ junk
    regardless of size);
  - three mutually touching boxes → all kept;
  - single box → `noop: true`.
- **Real-case smoke** on known steps:
  - `NEW-G90-WHITE-59` (`[0.6, 0.4]`, touching) → both kept (1 cluster);
  - `NEW-G140-47` (`[1.0, 0.0]`, far) → speck removed;
  - `NEW-G140-56` (`[0.574,0.243,0.183, 0.0]`) → 3 kept, 1 speck removed.
- **Visual** before/after in the GUI on a few uploads.

## 6. Out of scope (follow-ups)

- Full-dataset (1418) IoU/CD before-vs-after **quantification** of the fidelity
  gain — deferred follow-up.
- Clean-then-scale ordering (feeding cleaned geometry into post-scaling).
- STEP/BRep preview in the GUI (still STL-only).

## 7. Environment, base, and deployment

- **Source of truth / base.** The latest code lived only on RXL (55 commits
  ahead of every remote branch, tip `dc01160`) because RXL cannot reach GitHub
  (TLS failures — China network). On 2026-05-29 it was bridged to GitHub via the
  Mac (`git fetch` from RXL over SSH → `git push origin`) as branch
  **`sync/rxl-runtime-20260529`**. Implementation happens on
  **`feature/cad-body-cleanup`** (off `dc01160`); the PR targets
  `sync/rxl-runtime-20260529`.
- **Edit / test loop.** Edit locally on the Mac (the real code is checked out on
  the feature branch). Deploy to RXL for testing by syncing changed files over
  SSH — **RXL can't pull from GitHub**, so the Mac is the hub. The GUI runs on
  RXL (FastAPI `:18000` + Streamlit `:18501`, conda env `pytorch111`, tmux
  `aiws-gui`), viewed from the Mac via an SSH tunnel on `:18501`.
- The cleanup runs in Docker image `cadrille:latest` (CadQuery 2.5.0.dev0, OCP
  `BRepExtrema` confirmed available).
- ⚠️ Leave RXL's **uncommitted** (`scripts/cadrille_batch.py`,
  `scripts/e2e_sam3d_to_cadrille.py`) and **untracked**
  (`scripts/cadrille_metric_postscale.py`, `docs/workpiece-dimensions.md`) work
  untouched — unrelated to this feature and not yet on GitHub.
