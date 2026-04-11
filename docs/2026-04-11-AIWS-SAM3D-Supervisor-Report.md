# 2026-04-11 AIWS × SAM3D × Cadrille Supervisor Report

**Author**: Weld  
**Date**: 2026-04-11  
**Audience**: Supervisor / project lead  
**Purpose**: High-level summary of environment readiness, full-dataset experiments, runtime and memory behavior under current hardware constraints, and how inference should be run in future work.

---

## 1. Executive Summary

This round of work established a reusable end-to-end baseline from AIWS5.2 images to 3D reconstruction and downstream CAD generation:

1. **The SAM3D environment is now stable and reproducible** on the `RXL` server.
2. **SAM3D completed full-dataset inference on all 1418 cleaned samples with 100% success**.
3. **Cadrille PC-mode full-dataset inference was validated successfully**, showing that SAM3D STL outputs can be consumed by the downstream CAD stage.
4. **Cadrille IMG-mode required explicit shared-memory tuning**. It failed under the default large-batch setup, but completed successfully after shm and dataloader adjustments.
5. For future large-scale inference, the recommended operational strategy is to **decouple SAM3D and Cadrille**, and to **run Cadrille PC and IMG separately**, because their resource bottlenecks are different.

---

## 2. Dataset and Hardware Constraints

### 2.1 Official dataset scope

- Dataset: `aiws5.2-usable-materialized`
- Total formal tasks: **1418**
- Subset breakdown:
  - `V1`: 593
  - `V2`: 524
  - `NEW`: 301

### 2.2 Dataset restructuring and folder semantics

The dataset did not start in a directly model-friendly layout. The original assets were closer to a **flat resource pool**:

- `aiws5.2-dataset/images/`: RGB images largely stored together in one place
- `aiws5.2-dataset/depth/`: depth files stored together in one place
- `isat_annotations/`: ISAT JSON annotations, used as the final source of annotation truth
- `train.json` / `val.json`: useful for train/val membership, but not sufficient as the only source of instance-level ground truth

That original organization is inconvenient for large-scale reconstruction because it does not directly encode:

1. subset structure (`V1 / V2 / NEW`)
2. workpiece categories
3. separation of clean single-instance samples from problematic samples
4. stable traversal rules for SAM3D and downstream batch scripts

To solve this, the data was reorganized into three progressively more usable views.

#### (1) `aiws5.2-usable-split/`

This is a **split-aware, non-destructive symlink view** with the structure:

- `train/`, `val/`
- under each split: `V1/`, `V2/`, `NEW/`
- under each subset: five workpiece folders

The five workpiece folders are:

- `cover_plate`
- `square_tube`
- `h_beam`
- `channel_steel`
- `bellmouth`

Purpose:

- preserve train/val semantics
- validate split membership
- support split-aware statistics when needed

#### (2) `aiws5.2-usable/`

This is a **cleaned usable view** that no longer emphasizes train/val, but instead emphasizes whether a sample is appropriate for the main reconstruction pipeline.

It is organized as:

- top level: `V1 / V2 / NEW`
- second level: workpiece category
- special cases moved under `misc/`

Inside `misc/`:

- `misc/multi_instance/`: images containing multiple instances
- `misc/multi_label/`: images containing multiple categories
- `misc/unannotated_images/`: images found without usable annotations

Purpose:

- define the clean experimental subset
- separate clean single-instance data from problematic samples

#### (3) `aiws5.2-usable-materialized/`

This is the **final formal dataset root used in the production experiments**, and the one actually consumed by the SAM3D batch runner.

It keeps the clear `subset / workpiece` hierarchy and adds the supporting files needed for formal runs, for example:

- `metadata/samples.csv`
- `metadata/summary.json`
- `metadata/masks.csv`
- per-workpiece `masks/` folders

#### Meaning of the formal experiment folders

Using `aiws5.2-usable-materialized/` as the example:

- `V1/`: no depth
- `V2/`: depth stored under `depth_png/`
- `NEW/`: depth stored under `depth_exr/`

Inside each workpiece folder, the typical contents are:

- `images/`: RGB inputs
- `annotations/`: ISAT annotation JSON files
- `depth_png/` or `depth_exr/`: depth inputs when available
- `masks/`: rasterized instance masks generated from polygon annotations

Supporting folders:

- `metadata/`: machine-readable manifests and summary statistics
- `misc/`: special samples excluded from the main formal pipeline

In short, the dataset evolution is:

**flat raw resource pool → split-aware view → cleaned usable view → formal materialized experiment view.**

### 2.3 Compute server (`RXL`)

- CPU: `2 × Intel Xeon Gold 6326 @ 2.90GHz`
- CPU threads: `64`
- System RAM: `503 GiB`
- GPU: `4 × NVIDIA RTX A6000`
- Per-GPU memory: `49140 MiB` (about `48 GB`)
- Driver: `580.82.09`

### 2.4 Main hardware constraints observed in this work

1. **SAM3D is primarily constrained by GPU VRAM**, but is stable on 48 GB A6000 cards.
2. **Cadrille PC mode** is sensitive to GPU process placement. If multiple shard jobs land on the same GPU, CUDA OOM occurs.
3. **Cadrille IMG mode** is limited first by **Docker / PyTorch shared memory (shm)** rather than by GPU VRAM.

---

## 3. SAM3D Environment Status

Current production baseline:

- Project root: `/ssd1/rxl/zhankaiming/AIWS`
- SAM3D repo: `/ssd1/rxl/zhankaiming/AIWS/repos/sam-3d-objects`
- Conda env: `/home/rxl/anaconda3/envs/sam3d-objects`
- Repo baseline commit: `81a82373a3a7f4cbb00bd5b32aaf6b4d0f659ddd`

### 3.1 Required runtime assets restored

- `checkpoints/hf/pipeline.yaml`
- SAM3D checkpoints
- MoGe weights
- DINOv2 local cache and checkpoint

Key paths:

- checkpoints: `/ssd1/rxl/zhankaiming/AIWS/repos/sam-3d-objects/checkpoints/hf`
- MoGe: `/ssd1/rxl/zhankaiming/AIWS/models/moge-vitl/model-real.pt`
- DINO cache: `/home/rxl/.cache/torch/hub/facebookresearch_dinov2_main`
- DINO checkpoint: `/home/rxl/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth`

### 3.2 Performance-related environment decision

To make `flash_attn` actually take effect on `RTX A6000`, the formal runs use:

- `ATTN_BACKEND=flash_attn`
- `SPARSE_ATTN_BACKEND=flash_attn`

In practice, `flash_attn` had to be **compiled from source on the server**, because the prebuilt wheel was not compatible with the server environment.

> Detailed setup steps and future inference commands are documented in the paired SOP.

---

## 4. Full-Dataset Experiment Summary

## 4.1 SAM3D full-dataset results

Official output root:

- `/ssd1/rxl/zhankaiming/AIWS/outputs/sam3d-aiws52-clean-mesh-stl-20260410-193527`

### 4.1.1 Completion

| Metric | Value |
|---|---:|
| Total tasks | 1418 |
| Successful | 1418 |
| Failed | 0 |
| Completion rate | 100% |

### 4.1.2 Overall runtime

- Earliest task start: `2026-04-10 19:36:09`
- Latest task end: `2026-04-10 21:12:19`
- **Total wall-clock time**: about **1.60 hours**
- Effective wall-clock throughput: about **884.7 samples/hour**

Per-shard summary:

| Shard | GPU | Completed | Avg latency per sample (s) | Shard throughput (samples/hour) |
|---|---:|---:|---:|---:|
| shard-0 | 0 | 355 | 14.074 | 254.337 |
| shard-1 | 1 | 355 | 14.121 | 253.559 |
| shard-2 | 2 | 354 | 14.813 | 241.790 |
| shard-3 | 3 | 354 | 16.191 | 221.272 |

### 4.1.3 Per-sample timing statistics (1418 samples)

| Metric | Mean | P50 | P90 | P95 | Max |
|---|---:|---:|---:|---:|---:|
| `duration_sec` | 14.799 | 13.812 | 17.555 | 18.977 | 73.562 |
| `sec_per_megapixel` | 7.374 | 6.664 | 8.505 | 9.359 | 51.063 |

### 4.1.4 GPU memory statistics (1418 samples)

| Metric | Mean | P50 | P90 | P95 | Max |
|---|---:|---:|---:|---:|---:|
| `peak_memory_allocated_mb` | 18709.201 | 18738.090 | 19428.319 | 19614.863 | 20071.270 |
| `peak_memory_reserved_mb` | 24647.275 | 24936.000 | 26420.000 | 27136.000 | 28076.000 |

Interpretation:

- On `48 GB` A6000 cards, SAM3D peaks at about **28.1 GB reserved memory**.
- This indicates that the current SAM3D configuration is operationally safe on the present hardware, with usable headroom.

### 4.1.5 Main bottleneck observation

The main long-tail came from the `V1 / bellmouth` subset, which should be the first target if runtime optimization becomes important.

---

## 4.2 Cadrille full-dataset results

### 4.2.1 PC mode (full-dataset success)

The statistics come from a mixed full-modal run where **PC completed successfully** but the later IMG stage failed. The top-level folder was archived during cleanup, but the **PC shard outputs remain intact and usable**.

Statistics source:

- `/ssd1/rxl/zhankaiming/AIWS/.trash/outputs-cleanup-20260411-153952/cadrille-full-modalities-20260411-090531-bs64/pc`

Configuration:

- Mode: `pc`
- Input: SAM3D-exported STL
- `n_samples=5`
- `batch_size=64`
- 4-GPU parallel execution
- Selection strategy: `evaluate`

Results:

| Metric | Value |
|---|---:|
| Samples prepared for Cadrille | 1418 |
| Samples with selected best candidate | 1418 |
| Selection success rate | 100.0% |
| Wall-clock time | about 80.1 minutes |
| Effective throughput | about 1061.7 samples/hour |

Additional quality indicators (average across shard summaries):

| Metric | Value |
|---|---:|
| Average `mean_iou` | 0.021192 |
| Average `median_cd` | 0.038003 |

### 4.2.2 IMG mode (full-dataset success after fix)

Final retained successful output root:

- `/ssd1/rxl/zhankaiming/AIWS/outputs/cadrille-img-only-20260411-143505-shmfix`

Stable configuration:

- Mode: `img`
- `n_samples=1`
- `batch_size=32`
- `--ipc=host --shm-size=16g`
- Reduced IMG dataloader worker pressure
- Selection strategy: `evaluate`

Results:

| Metric | Value |
|---|---:|
| Samples prepared for Cadrille | 1418 |
| Samples with selected best candidate | 1213 |
| Selection success rate | 85.54% |
| Selected STL outputs | 1213 |
| Selected STEP outputs | 1211 |
| Wall-clock time | about 25.9 minutes |
| Effective throughput | about 3290.3 samples/hour |

Additional quality indicators (average across shard summaries):

| Metric | Value |
|---|---:|
| Average `mean_iou` | 0.026166 |
| Average `median_cd` | 0.047796 |

### 4.2.3 Memory and shared-memory constraints in Cadrille

**PC mode constraint**:

- An earlier large-scale attempt produced the following representative error:
  - `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.21 GiB`
- Log inspection showed that multiple processes had effectively collided on the same GPU.
- Conclusion: **PC mode is feasible on the current hardware, but only if GPU pinning is strict.**

**IMG mode constraint**:

- Under `batch_size=64`, all four shards showed errors such as:
  - `ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).`
  - `RuntimeError: DataLoader worker ... is killed by signal: Bus error`
  - `RuntimeError: unable to write to file </torch_...>: No space left on device (28)`
- Conclusion: **the primary bottleneck for IMG mode is Docker / PyTorch shared memory, not raw GPU VRAM.**

### 4.2.4 Why there is not yet a formal peak-GPU-memory table for Cadrille

Unlike SAM3D, the current Cadrille launcher did not yet log:

- `torch.cuda.max_memory_allocated()`
- `torch.cuda.max_memory_reserved()`

So for this iteration, the Cadrille memory summary is best described as:

- backed by **real failure evidence** (OOM and shm bus errors)
- backed by a **validated stable configuration**
- but **not yet backed by per-sample peak-memory instrumentation**

This does not change the operational conclusion, but it should be added before paper writing or a formal defense.

---

## 5. Supervisor-Facing Takeaways

The following statements can be reported directly:

1. **SAM3D has already completed 100% full-dataset inference on 1418 formal samples**, with a total wall-clock time of about 1.6 hours on `4 × RTX A6000`.
2. **SAM3D averages about 14.8 seconds per sample**, with an upper observed reserved-memory bound of about **28.1 GB**, showing that it is stable on 48 GB A6000 GPUs.
3. **After converting SAM3D outputs to STL, Cadrille PC mode has been validated at full-dataset scale**, with all 1418 inputs producing selectable CAD results.
4. **Cadrille IMG mode is limited by shared-memory configuration rather than by insufficient compute**, and it completed successfully after `batch=32 + --ipc=host + --shm-size=16g + reduced workers`.
5. The current system now provides a reusable pipeline from **AIWS image data → SAM3D reconstruction → Cadrille CAD generation**.

---

## 6. Recommended Next Steps

### Recommendation 1: Add formal GPU-memory instrumentation to Cadrille

Add logging around `test.py` / `evaluate.py` or the outer launcher to capture:

- mean memory
- P95 memory
- max memory
- modality-specific memory tables

### Recommendation 2: Run future batches in decoupled stages

Recommended order:

1. **Run SAM3D first** and export STL
2. **Run Cadrille-PC separately**
3. **Run Cadrille-IMG separately**

This is safer because the bottlenecks are different:

- PC mode is more sensitive to VRAM and GPU placement
- IMG mode is more sensitive to DataLoader / shm behavior

### Recommendation 3: Analyze the IMG failure subset

The final IMG retention rate is about `85.5%`. The next analysis should focus on:

- which categories fail more often
- whether failures are generation, conversion, or evaluation-selection failures
- whether failures correlate with shape complexity, viewpoint, or occlusion

---

## 7. Key Artifact Paths Used in This Report

### Retained successful outputs

- SAM3D official successful output:
  - `/ssd1/rxl/zhankaiming/AIWS/outputs/sam3d-aiws52-clean-mesh-stl-20260410-193527`
- Cadrille IMG official successful output:
  - `/ssd1/rxl/zhankaiming/AIWS/outputs/cadrille-img-only-20260411-143505-shmfix`

### Archived but recoverable statistics path

- Cadrille PC full-dataset success statistics:
  - `/ssd1/rxl/zhankaiming/AIWS/.trash/outputs-cleanup-20260411-153952/cadrille-full-modalities-20260411-090531-bs64/pc`

---

## 8. One-Sentence Conclusion

**Under `4 × RTX A6000`, SAM3D is already stable for full-dataset reconstruction, and the downstream Cadrille pipeline has been validated; the main optimization target is no longer whether the pipeline can run, but how to run each modality more robustly and with better resource observability.**
