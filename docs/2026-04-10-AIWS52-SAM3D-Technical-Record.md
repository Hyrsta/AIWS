# 2026-04-10 Technical Record: Clean AIWS5.2 Dataset and SAM3D Deployment

## 1. Purpose

This document records the final production workflow adopted on 2026-04-10 for the AIWS5.2 project, including the clean dataset definition, SAM3D deployment on `RXL`, `flash_attn` enablement on RTX A6000 GPUs, batch-runner development, and the launch of the official 4-GPU reconstruction run.

To keep the document readable and reusable, this version only keeps the **final accepted workflow**. Discarded intermediate dataset variants are intentionally omitted.

---

## 2. Environment and Key Paths

### 2.1 Local workspace

- Workspace: `/Users/hyrsta/.openclaw/workspaces/welding-algorithm`
- Relevant scripts:
  - `scripts/build_aiws52_usable_view.py`
  - `scripts/generate_aiws52_instance_masks.py`
  - `scripts/run_sam3d_aiws52_batch.py`

### 2.2 Remote server

- Server alias: `RXL`
- Unified project root: `/ssd1/rxl/zhankaiming/AIWS`
- SAM3D repo: `/ssd1/rxl/zhankaiming/AIWS/repos/sam-3d-objects`
- Cadrille repo: `/ssd1/rxl/zhankaiming/AIWS/repos/cadrille`
- Dataset root: `/ssd1/rxl/zhankaiming/AIWS/data/aiws5.2-usable-materialized`
- Batch runner: `/ssd1/rxl/zhankaiming/AIWS/scripts/run_sam3d_aiws52_batch.py`
- Output root: `/ssd1/rxl/zhankaiming/AIWS/outputs`
- Demo launcher: `/ssd1/rxl/zhankaiming/AIWS/run_sam3d_demo.sh`
- Runtime notes: `/ssd1/rxl/zhankaiming/AIWS/SAM3D_RUNTIME.md`
- Conda environment: `/home/rxl/anaconda3/envs/sam3d-objects`

### 2.3 Current official run

- Official run root: `/ssd1/rxl/zhankaiming/AIWS/outputs/sam3d-aiws52-clean-mesh-stl-20260410-193527`

---

## 3. Official AIWS5.2 Dataset Definition

### 3.1 Goal

The final dataset definition used for the official experiment is intentionally simple:

- use a clean directory layout
- preserve `V1 / V2 / NEW` subset information
- preserve workpiece categories
- keep one formal reconstruction task per accepted sample
- avoid ambiguity caused by multi-instance images in the main experiment narrative

Therefore, the official experiment uses the clean materialized dataset:

- `aiws5.2-usable-materialized`

### 3.2 Directory layout

The dataset is organized as:

- `V1/`
- `V2/`
- `NEW/`

Inside each subset, samples are grouped by workpiece class, for example:

- `cover_plate`
- `square_tube`
- `h_beam`
- `channel_steel`
- `bellmouth`

Each workpiece directory contains:

- `images/`
- `annotations/`
- `depth_png/` or `depth_exr/` when depth exists for that subset

### 3.3 Dataset policy

The official experiment follows these rules:

- annotations come from `isat_annotations/`
- only the cleaned main dataset view is used for the formal run
- multi-instance images are excluded from the official run
- unannotated images are excluded from the official run
- all later writing and reporting should consistently refer to this clean dataset only

### 3.4 Statistics

From `aiws5.2-usable-materialized/metadata/summary.json`:

- official tasks: **1418**
- multi-instance images: **23**
- multi-label images: **0**
- unannotated images: **1**

Subset counts:

- `V1`: 593
- `V2`: 524
- `NEW`: 301

For the official workflow, these numbers can be described simply as:

- **1418 cleaned samples**
- **1418 formal SAM3D reconstruction tasks**

---

## 4. What SAM3D Takes as Input

For each sample, SAM3D receives three core inputs:

1. the RGB image
2. a binary mask rasterized from the annotation polygon
3. a random seed

This means the project does **not** rely on a mandatory pre-generated mask file for the formal run. Instead, the batch runner does the following on the fly:

- load the annotation JSON
- read the polygon
- rasterize the polygon into a binary mask
- call `Inference(image, mask, seed)`

The current batch runner then exports:

- `mesh.glb`
- `mesh.stl`

---

## 5. SAM3D Recovery on the Server

### 5.1 Repository recovery

Because the server is in a weak-network environment for GitHub and Hugging Face access, the repository was restored by:

1. cloning locally
2. verifying locally
3. syncing to `RXL`

Final repo path:

- `/ssd1/rxl/zhankaiming/AIWS/repos/sam-3d-objects`

Commit:

- `81a82373a3a7f4cbb00bd5b32aaf6b4d0f659ddd`

### 5.2 Runtime assets

The restored repo still required runtime assets, including:

- `checkpoints/hf/pipeline.yaml`
- SAM3D checkpoints
- local MoGe weights
- local DINOv2 cache and checkpoint

Key paths:

- checkpoints: `/ssd1/rxl/zhankaiming/AIWS/repos/sam-3d-objects/checkpoints/hf`
- MoGe: `/ssd1/rxl/zhankaiming/AIWS/models/moge-vitl/model-real.pt`
- DINO cache: `/home/rxl/.cache/torch/hub/facebookresearch_dinov2_main`
- DINO checkpoint: `/home/rxl/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth`

### 5.3 AIWS path migration fixes

After moving the project into the unified `AIWS/` root, two additional fixes were required:

1. the batch runner had to add both the repo root and `notebook/` to `sys.path`
2. the checkpoint symlinks under `checkpoints/hf/` had to be repointed from the old pre-`AIWS` backup path to the new `AIWS/backups/...` path

After these fixes, the new unified layout became fully runnable again.

---

## 6. `flash_attn` on RTX A6000

### 6.1 Binary compatibility issue

The first `flash_attn` problem was not CUDA capability, but binary compatibility:

- prebuilt wheels required `GLIBC_2.32`
- the server environment could not use those wheels directly

The practical fix was a local source build on the server.

### 6.2 Runtime backend selection issue

Even after import succeeded, SAM3D did not automatically switch to `flash_attn` on RTX A6000 because the repository only auto-selects it for a restricted GPU whitelist such as:

- `A100`
- `H100`
- `H200`

The final solution was to explicitly export:

- `ATTN_BACKEND=flash_attn`
- `SPARSE_ATTN_BACKEND=flash_attn`

### 6.3 Final state

The current official run on `RXL` confirms:

- dense attention uses `flash_attn`
- sparse attention uses `flash_attn`
- the backend is visible in runtime logs

---

## 7. Batch Runner Capabilities

The current `run_sam3d_aiws52_batch.py` supports:

- direct traversal of the `subset` layout used by `aiws5.2-usable-materialized`
- `--resume`
- multi-GPU sharding with `--num-shards` and `--shard-index`
- structured output files
- instance-level metrics
- automatic local redirection for DINO cache access instead of live GitHub access

Main recorded metrics include:

- `model_init_sec`
- `duration_sec`
- `instances_per_hour`
- `sec_per_megapixel`
- `mask_pixels`
- `mask_fraction`
- `peak_memory_allocated_mb`
- `peak_memory_reserved_mb`
- `gpu_name`
- `cuda_visible_devices`
- `num_shards`
- `shard_index`

---

## 8. Official Run Organization

The official experiment currently runs as:

- 4 RTX A6000 GPUs
- 4 workers
- 4 shards

The sharding rule is:

- `global_index % 4 == 0`
- `global_index % 4 == 1`
- `global_index % 4 == 2`
- `global_index % 4 == 3`

This guarantees:

- no duplicated tasks
- no missing tasks
- independent runtime statistics per task

Current official run root:

- `/ssd1/rxl/zhankaiming/AIWS/outputs/sam3d-aiws52-clean-mesh-stl-20260410-193527`

Each shard contains:

- `manifest.csv`
- `results.jsonl`
- `summary.json`
- `worker.log`
- per-task output folders

---

## 9. Final Outcome of the Day

The main engineering outcomes of the day are:

1. the project now has a single clean official dataset definition
2. SAM3D has been restored to a reusable state on `RXL`
3. `flash_attn` is working on RTX A6000 in the actual runtime path
4. project assets are unified under the `AIWS/` root
5. the batch runner now matches the official clean dataset layout
6. the official 4-GPU run on **1418** cleaned samples has been launched

---

## 10. Recommended Next Steps

1. let the current official run finish
2. merge the `results.jsonl` files from all four shards
3. review failed samples separately
4. use this document as the baseline for the later methods and experiment sections of the paper or technical report
