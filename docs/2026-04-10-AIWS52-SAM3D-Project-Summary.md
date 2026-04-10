# 2026-04-10 Project Summary: AIWS5.2 and SAM3D

## 1. Final Position

The AIWS5.2 to SAM3D pipeline is now fixed to one reproducible production path:

- official dataset: `aiws5.2-usable-materialized`
- project root: `/ssd1/rxl/zhankaiming/AIWS`
- output format: `mesh.glb + mesh.stl`
- execution mode: 4-way sharded run on 4 RTX A6000 GPUs

All follow-up reporting should use this single narrative only.

---

## 2. Dataset Outcome

Official dataset stats (`metadata/summary.json`):

- total tasks: **1418**
- `V1`: 593
- `V2`: 524
- `NEW`: 301
- multi-instance: 23
- multi-label: 0
- unannotated: 1

Recommended wording for papers/reports:

- “1418 cleaned valid samples”
- “1418 official reconstruction tasks”

---

## 3. Environment Outcome

SAM3D on `RXL` is restored to a stable reusable state:

- repo: `/ssd1/rxl/zhankaiming/AIWS/repos/sam-3d-objects`
- conda env: `/home/rxl/anaconda3/envs/sam3d-objects`
- runtime assets (checkpoints, MoGe, DINO cache) are in place
- migration fixes after unifying under `AIWS/` are complete

---

## 4. Performance Path Outcome (`flash_attn`)

`flash_attn` is now active in the production path:

1. source build solved wheel/runtime compatibility issues
2. explicit backend env vars solved the A6000 auto-selection gap

Dense and sparse attention are both running with `flash_attn`.

---

## 5. Batch Runner Outcome

Formal runner: `scripts/run_sam3d_aiws52_batch.py`

Now supports:

- direct traversal of `subset` layout
- on-the-fly polygon-to-mask conversion
- GLB + STL export
- `manifest.csv` / `results.jsonl` / `summary.json` / `meta.json`
- checkpoint resume (`--resume`)
- sharded parallel run (`--num-shards`, `--shard-index`)
- per-task performance and memory metrics

---

## 6. Run Status

Official run root:

- `/ssd1/rxl/zhankaiming/AIWS/outputs/sam3d-aiws52-clean-mesh-stl-20260410-193527`

Current status:

- all 4 shards launched
- successful outputs are being produced continuously
- shard summaries and per-task outputs are generated correctly

---

## 7. Key Deliverables Today

1. one official dataset definition was finalized
2. SAM3D runtime on `RXL` was restored and stabilized
3. A6000 `flash_attn` path is now operational
4. assets were consolidated under `AIWS/`
5. official 1418-task 4-GPU reconstruction run was launched

---

## 8. Next Steps

1. wait for run completion
2. merge shard `results.jsonl` files
3. analyze failed tasks by error type
4. continue paper/report writing from the current technical baseline
