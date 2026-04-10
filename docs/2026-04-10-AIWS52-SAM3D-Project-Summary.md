# 2026-04-10 Project Summary: AIWS5.2 and SAM3D

## 1. Final Position

The project now has a clean official workflow for the AIWS5.2 to SAM3D pipeline.

The standard setup is now:

- official dataset: `aiws5.2-usable-materialized`
- unified project root: `/ssd1/rxl/zhankaiming/AIWS`
- official output format: `mesh.glb + mesh.stl`
- official execution mode: 4-way sharded parallel reconstruction on 4 RTX A6000 GPUs

To keep future writing clean, this summary only keeps the accepted final workflow and does not include discarded intermediate dataset versions.

---

## 2. Dataset Outcome

The official dataset used for the formal experiment is:

- `/ssd1/rxl/zhankaiming/AIWS/data/aiws5.2-usable-materialized`

Its role is to provide a clean, structured dataset for direct batch processing.

Official statistics:

- total formal tasks: **1418**
- multi-instance images: **23**
- multi-label images: **0**
- unannotated images: **1**

For the official narrative, these can simply be described as:

- **1418 cleaned samples**
- **1418 reconstruction tasks**

---

## 3. Environment Outcome

SAM3D is now restored and organized on `RXL` under the unified root:

- SAM3D repo: `/ssd1/rxl/zhankaiming/AIWS/repos/sam-3d-objects`
- Cadrille repo: `/ssd1/rxl/zhankaiming/AIWS/repos/cadrille`
- Conda env: `/home/rxl/anaconda3/envs/sam3d-objects`

Because GitHub and Hugging Face access are unstable on the server, the working solution is based on:

- local clone and sync
- restored checkpoint assets
- local MoGe path
- local DINO cache

---

## 4. `flash_attn` Outcome

`flash_attn` is now working in the real runtime path on RTX A6000.

This required two steps:

1. local source build to solve the binary compatibility issue
2. explicit runtime backend selection through environment variables

The current run uses:

- `ATTN_BACKEND=flash_attn`
- `SPARSE_ATTN_BACKEND=flash_attn`

---

## 5. Batch Runner Outcome

The formal batch runner is:

- `scripts/run_sam3d_aiws52_batch.py`

It now supports:

- direct traversal of `aiws5.2-usable-materialized`
- on-the-fly polygon-to-mask conversion
- `mesh.glb` and `mesh.stl` export
- `manifest.csv`, `results.jsonl`, `summary.json`, and `meta.json`
- resume mode
- 4-way sharding
- per-task metrics
- local DINO cache redirection

After moving the whole project into `AIWS/`, the runner and checkpoint links were also fixed so the official path layout is stable again.

---

## 6. Official Run Status

Current official run root:

- `/ssd1/rxl/zhankaiming/AIWS/outputs/sam3d-aiws52-clean-mesh-stl-20260410-193527`

Run structure:

- 4 GPUs
- 4 workers
- 4 shards
- total tasks: **1418**

At the time of this revision, the run has already started successfully and multiple shards have begun producing successful outputs.

---

## 7. Most Important Achievements Today

The most important outcomes of the day are:

1. a single clean official dataset definition is now in place
2. SAM3D is back to a stable reusable state on `RXL`
3. RTX A6000 is actually running with `flash_attn`
4. project assets are consolidated under `AIWS/`
5. the official 4-GPU reconstruction run on **1418** samples has been launched

---

## 8. Suggested Next Steps

1. wait for the official run to finish
2. merge the shard result files
3. inspect failed samples separately
4. continue writing the methods and experiment sections based on this final clean version only
