# 2026-04-10 Technical Record: AIWS5.2 Clean Dataset and SAM3D Runtime

**Author**: Weld  
**Date**: 2026-04-10  
**Status**: Implemented (reproducible baseline)

---

## 1. Purpose and Scope

This document defines the official engineering baseline for the AIWS5.2 to SAM3D pipeline, including:

- the official dataset definition (single source of truth)
- server deployment and dependency recovery
- `flash_attn` enablement on RTX A6000
- batch-runner technical behavior
- 4-GPU parallel runbook

Only the final accepted workflow is kept. Discarded intermediate dataset stories are intentionally omitted.

---

## 2. Environment Baseline

### 2.1 Local workspace

- Workspace: `/Users/hyrsta/.openclaw/workspaces/welding-algorithm`
- Key scripts:
  - `scripts/build_aiws52_usable_view.py`
  - `scripts/generate_aiws52_instance_masks.py`
  - `scripts/run_sam3d_aiws52_batch.py`

### 2.2 Remote server (RXL)

- Project root: `/ssd1/rxl/zhankaiming/AIWS`
- SAM3D repo: `/ssd1/rxl/zhankaiming/AIWS/repos/sam-3d-objects`
- Cadrille repo: `/ssd1/rxl/zhankaiming/AIWS/repos/cadrille`
- Dataset root: `/ssd1/rxl/zhankaiming/AIWS/data/aiws5.2-usable-materialized`
- Batch runner: `/ssd1/rxl/zhankaiming/AIWS/scripts/run_sam3d_aiws52_batch.py`
- Output root: `/ssd1/rxl/zhankaiming/AIWS/outputs`
- Conda env: `/home/rxl/anaconda3/envs/sam3d-objects`

### 2.3 Official run root used in production

- `/ssd1/rxl/zhankaiming/AIWS/outputs/sam3d-aiws52-clean-mesh-stl-20260410-193527`

---

## 3. Official Dataset Definition (single narrative)

### 3.1 Official dataset

- Name: `aiws5.2-usable-materialized`
- Nature: cleaned runnable data view (non-destructive, symlink-organized)
- Annotation source of truth: `isat_annotations`

### 3.2 Directory layout

- Top-level subsets: `V1 / V2 / NEW`
- Samples grouped by workpiece class inside each subset (for example `cover_plate`, `square_tube`, `h_beam`, `channel_steel`, `bellmouth`)
- Each workpiece directory contains:
  - `images/`
  - `annotations/`
  - `depth_png/` or `depth_exr/` (depending on subset type)

### 3.3 Dataset statistics (`metadata/summary.json`)

- Total official tasks: **1418**
- `V1`: 593
- `V2`: 524
- `NEW`: 301
- Multi-instance images: 23
- Multi-label images: 0
- Unannotated images: 1

### 3.4 Writing rule for all follow-up docs

Use this wording consistently in reports/papers:

- “The experiment uses the cleaned `aiws5.2-usable-materialized` dataset.”
- “The official reconstruction task count is 1418.”

Do not reintroduce intermediate path variants.

---

## 4. Batch Runner Technical Specification

Script: `scripts/run_sam3d_aiws52_batch.py`

### 4.1 Key parameters

- `--dataset-root`: dataset root
- `--dataset-layout`: `subset` (official project setting)
- `--repo-root`: SAM3D repository root
- `--output-root`: output directory for this shard
- `--resume`: skip tasks with existing non-empty `mesh.glb` and `mesh.stl`
- `--num-shards` / `--shard-index`: sharded parallel execution
- `--exclude-stems-file`: optional exclusion list

### 4.2 Per-task data flow

For each task, the runner performs:

1. load RGB image
2. load annotation JSON
3. rasterize polygon to binary mask
4. call `Inference(image, mask, seed)`
5. export `mesh.glb` and `mesh.stl`
6. write per-task `meta.json` and append to shard-level `results.jsonl`

### 4.3 Sharding strategy

Tasks are assigned by `global_index % num_shards == shard_index`.

In 4-shard mode:

- shard 0: `% 4 == 0`
- shard 1: `% 4 == 1`
- shard 2: `% 4 == 2`
- shard 3: `% 4 == 3`

This guarantees:

- no duplicate processing
- no missing tasks
- independent shard-level tracking

### 4.4 Artifacts and metrics

Per-shard artifacts:

- `manifest.csv`
- `results.jsonl`
- `summary.json`
- per-task output directories (`mesh.glb`, `mesh.stl`, `meta.json`)

Key metrics include:

- `duration_sec`
- `instances_per_hour`
- `sec_per_megapixel`
- `mask_pixels` / `mask_fraction`
- `peak_memory_allocated_mb` / `peak_memory_reserved_mb`
- `gpu_name`
- `num_shards` / `shard_index`

### 4.5 Failure and recovery semantics

- Failed tasks still write `meta.json` with error type/message/traceback
- Temporary files (`mesh.partial.*`) are cleaned on failure
- `--resume` provides checkpoint restart behavior

---

## 5. Server Deployment and Dependency Recovery

### 5.1 Repository recovery strategy

Given unstable direct GitHub/HF access on the server, recovery used:

1. local clone
2. local verification
3. sync to RXL

Current repo commit:

- `81a82373a3a7f4cbb00bd5b32aaf6b4d0f659ddd`

### 5.2 Runtime assets restored

Stable SAM3D runtime requires:

- `checkpoints/hf/pipeline.yaml`
- SAM3D checkpoints
- local MoGe weights
- local DINOv2 cache + checkpoint

Key paths:

- checkpoints: `/ssd1/rxl/zhankaiming/AIWS/repos/sam-3d-objects/checkpoints/hf`
- MoGe: `/ssd1/rxl/zhankaiming/AIWS/models/moge-vitl/model-real.pt`
- DINO cache: `/home/rxl/.cache/torch/hub/facebookresearch_dinov2_main`
- DINO checkpoint: `/home/rxl/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth`

### 5.3 Path migration fixes after unifying under `AIWS/`

Two fixes were required:

- add both `repo_root` and `repo_root/notebook` to `sys.path`
- repoint old symlinks under `checkpoints/hf` to the new `AIWS/backups/...` paths

---

## 6. `flash_attn` on RTX A6000

### 6.1 Issue 1: prebuilt wheel incompatibility

- Prebuilt wheels required a newer glibc level
- The server runtime could not use them directly

Resolution: local source build of `flash_attn` on the server.

### 6.2 Issue 2: backend auto-selection did not cover A6000

Even after successful import, runtime backend selection did not automatically switch to A6000.

Resolution: explicitly set:

- `ATTN_BACKEND=flash_attn`
- `SPARSE_ATTN_BACKEND=flash_attn`

### 6.3 Final state

In the official run, both dense and sparse attention are using `flash_attn`, and backend logs confirm this.

---

## 7. Runbook

### 7.1 Start one shard (example)

```bash
ATTN_BACKEND=flash_attn \
SPARSE_ATTN_BACKEND=flash_attn \
CUDA_VISIBLE_DEVICES=0 \
/home/rxl/anaconda3/envs/sam3d-objects/bin/python -u \
/ssd1/rxl/zhankaiming/AIWS/scripts/run_sam3d_aiws52_batch.py \
  --dataset-root /ssd1/rxl/zhankaiming/AIWS/data/aiws5.2-usable-materialized \
  --dataset-layout subset \
  --repo-root /ssd1/rxl/zhankaiming/AIWS/repos/sam-3d-objects \
  --output-root /ssd1/rxl/zhankaiming/AIWS/outputs/sam3d-aiws52-clean-mesh-stl-20260410-193527/shard-0 \
  --resume \
  --num-shards 4 \
  --shard-index 0
```

### 7.2 Monitor all shards

```bash
for s in /ssd1/rxl/zhankaiming/AIWS/outputs/sam3d-aiws52-clean-mesh-stl-20260410-193527/shard-*; do
  echo "--- ${s} ---"
  jq '{completed_ok, failed, skipped, processed, total_tasks_in_shard, eta_sec}' "${s}/summary.json"
done
```

### 7.3 Resume behavior

Restart the same shard command with `--resume` using the same run directory.

---

## 8. Confirmed Outcomes

1. Official dataset definition is fixed as `aiws5.2-usable-materialized`
2. SAM3D runtime on `RXL` is restored and reusable
3. `flash_attn` is active on RTX A6000 in the production path
4. 4-shard parallel run is live and producing outputs
5. Output structure, per-task metadata, and failure traces are complete

---

## 9. Recommended Next Steps

1. merge shard `results.jsonl` files after run completion
2. build a failed-sample list and group by failure type
3. write paper method/experiment sections from this baseline only
4. keep this runbook as SOP for future reproductions
