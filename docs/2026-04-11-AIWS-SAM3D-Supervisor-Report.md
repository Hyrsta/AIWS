# 2026-04-11 AIWS × SAM3D × Cadrille Supervisor Report

**Author**: Kaiming Zhan  
**Date**: 2026-04-11  
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

- Dataset: `aiws5.2-usable`
- Total formal tasks: **1418**
- Subset breakdown:
    - `V1`: 593
    - `V2`: 524
    - `NEW`: 301

### 2.2 Current dataset structure and condition

The formal experiments now use `aiws5.2-usable/` as the unified dataset root.

Its core semantics are:

- top level: `V1 / V2 / NEW`
- under each subset: five workpiece folders
    - `cover_plate`
    - `square_tube`
    - `h_beam`
    - `channel_steel`
    - `bellmouth`
- inside each workpiece folder, the typical contents are:
    - `images/`: RGB inputs
    - `annotations/`: ISAT annotation JSON files
    - `depth_png/` or `depth_exr/`: depth inputs when available
- supporting top-level folders:
    - `metadata/`: manifests and summary statistics
    - `misc/`: special samples excluded from the main pipeline, including:
        - `multi_instance/`: images containing multiple instances
        - `multi_label/`: images containing multiple categories
        - `unannotated_images/`: images found without usable annotations

Annotation truth should still be checked against `isat_annotations/`.

The main benchmark uses the cleaner single-instance portion of the dataset, so `misc/` is listed separately.

The current dataset condition is as follows:

- the current main experimental view contains **1418** usable samples
- `V1` contains **593** samples, with **no depth**, and currently covers:
    - `cover_plate`: 200
    - `square_tube`: 99
    - `h_beam`: 100
    - `bellmouth`: 194
    - `channel_steel`: 0
- `V2` contains **524** samples, **all with PNG depth**, and the current main-view labels are all `cover_plate`
- `NEW` contains **301** samples, **all with EXR depth**, and the current main-view labels are all `cover_plate`
- in other words:
    - `V1` is currently the most diverse subset by workpiece type
    - `V2` and `NEW` are currently dominated by depth-enabled `cover_plate` samples
    - `channel_steel` currently has **no populated instances** in the usable main view

At present, the identified special-case counts are:

- **23** `multi_instance` samples
- **0** `multi_label` samples
- **1** unannotated image (`NEW-G90-BLACK-24`)

In the current data, `misc/` is dominated by `multi_instance` samples.

### 2.3 AIWS project structure

AIWS has two connected parts:

- **Online vision pipeline**: used during on-site welding to recognize the real workpiece, estimate its pose, and align it with CAD models for downstream localization and weld path planning.
    - model stack: `YOLOv11-seg + GenPose++ + FoundationPose`
- **Offline CAD reconstruction pipeline**: used before deployment to build the CAD model database.
    - model stack: `SAM3D + Cadrille`
    - `SAM3D`: offline RGB images → mesh reconstruction
    - `Cadrille`: starts from the reconstructed mesh, then either samples point clouds for PC mode or renders 4-view RGB images for IMG mode before CAD reconstruction

**Current repository layout**

- GitHub: `https://github.com/Hyrsta/AIWS`
- `repos/sam-3d-objects`: upstream SAM3D repo, tracked as a submodule
- `repos/cadrille`: upstream Cadrille repo, tracked as a submodule
- `scripts/`: AIWS orchestration, wrappers, dataset preparation, and analysis scripts
- `docs/`: reports, SOPs, and presentation materials
- `gui/`: local GUI for launching runs and inspecting outputs

**Current AIWS GUI**

- location: `gui/`
- architecture: `FastAPI` backend + `Streamlit` frontend
- default execution path: run locally and connect to `RXL` over SSH
- current capabilities:
    - launch full Cadrille runs with `scripts/cadrille_full_modalities_4gpu.py`
    - launch single bridge/e2e runs with `scripts/sam3d_to_cadrille_e2e.py`
    - inspect output roots, shard progress, and job logs
    - preview remote STL meshes from outputs such as `selected_mesh/` and `tmp_mesh/`
- current limitation: STL preview is supported now, while STEP/BRep preview is still not implemented
- detailed setup and usage notes are kept in `gui/README.md`

**Upstream entry points used in the offline pipeline**

- SAM3D (`repos/sam-3d-objects`): `demo.py`, `notebook/inference.py`, `checkpoints/hf/pipeline.yaml`
- Cadrille (`repos/cadrille`): `test.py`, `evaluate.py`, `convert_cadquery.py`

**AIWS wrapper and orchestration scripts**

- `scripts/cadrille_test_wrapper.py`: thin wrapper for processor/checkpoint override, sample-count control, batch-size control, and GPU-memory logging
- `scripts/cadrille_evaluate.py`: evaluation wrapper used by the AIWS e2e pipeline
- `scripts/cadrille_convert_cadquery.py`: CAD conversion wrapper used by the AIWS e2e pipeline
- `scripts/dataset_usable_view_build.py`: builds the cleaned `aiws5.2-usable/` dataset view
- `scripts/dataset_instance_masks_generate.py`: prepares instance-level masks and intermediate data from annotations
- `scripts/sam3d_batch.py`: resumable SAM3D batch runner with sharding, multi-GPU support, and runtime metrics
- `scripts/sam3d_run_metrics_analysis.py`: summarizes and analyzes SAM3D run statistics
- `scripts/sam3d_to_cadrille_e2e.py`: bridges SAM3D mesh outputs into Cadrille by preparing the mesh-derived inputs required by each mode, and writes downstream summaries
- `scripts/cadrille_full_modalities_4gpu.py`: launches full-dataset Cadrille `pc/img` shard jobs across GPUs

### 2.4 Compute server (`RXL`)

- CPU: `2 × Intel Xeon Gold 6326 @ 2.90GHz`
- CPU threads: `64`
- System RAM: `503 GiB`
- GPU: `4 × NVIDIA RTX A6000`
- Per-GPU memory: `49140 MiB` (about `48 GB`)
- Driver: `580.82.09`

### 2.5 Main hardware constraints observed in this work

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

> The paired SOP now includes the `flash_attn` source-build commands, the verification step, and the explicit backend environment settings required on A6000.

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
| shard-0 | 0 | 355 | 14.07 | 254.34 |
| shard-1 | 1 | 355 | 14.12 | 253.56 |
| shard-2 | 2 | 354 | 14.81 | 241.79 |
| shard-3 | 3 | 354 | 16.19 | 221.27 |

### 4.1.3 Per-sample timing statistics (1418 samples)

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Metric</th>
      <th>Mean</th>
      <th>P50</th>
      <th>P90</th>
      <th>P95</th>
      <th>Max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">Runtime</td>
      <td>Per-sample runtime (s)</td>
      <td>14.80</td>
      <td>13.81</td>
      <td>17.56</td>
      <td>18.98</td>
      <td>73.56</td>
    </tr>
    <tr>
      <td>Runtime per megapixel (s/MP)</td>
      <td>7.37</td>
      <td>6.66</td>
      <td>8.51</td>
      <td>9.36</td>
      <td>51.06</td>
    </tr>
  </tbody>
</table>

### 4.1.4 GPU memory statistics (1418 samples)

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Metric</th>
      <th>Mean</th>
      <th>P50</th>
      <th>P90</th>
      <th>P95</th>
      <th>Max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">GPU memory</td>
      <td>Peak allocated GPU memory per sample (GB)</td>
      <td>18.71</td>
      <td>18.74</td>
      <td>19.43</td>
      <td>19.61</td>
      <td>20.07</td>
    </tr>
    <tr>
      <td>Peak reserved-memory upper bound per sample (GB)</td>
      <td>24.65</td>
      <td>24.94</td>
      <td>26.42</td>
      <td>27.14</td>
      <td>28.08</td>
    </tr>
  </tbody>
</table>

Interpretation:

- On `48 GB` A6000 cards, SAM3D peaks at about **28.1 GB reserved memory**.
- This indicates that the current SAM3D configuration is operationally safe on the present hardware, with usable headroom.

### 4.1.5 Runtime and memory by dataset version and workpiece type

To answer the practical question of **how much time and GPU memory each dataset version and workpiece type actually consumes**, the following table summarizes the full-run SAM3D statistics at the `V1 / V2 / NEW × workpiece` level.

Notes:

- the runtime columns report **mean runtime** and **P90 runtime**
- the memory columns report **average allocated GPU memory** and the **peak reserved-memory upper bound**
- `—` means that the current main experimental view has no samples for that combination

<table>
  <thead>
    <tr>
      <th>Dataset version</th>
      <th>Workpiece type</th>
      <th>Samples</th>
      <th>Mean runtime (s)</th>
      <th>P90 runtime (s)</th>
      <th>Avg peak allocated GPU memory (GB)</th>
      <th>Peak reserved-memory upper bound (GB)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4">V1</td>
      <td>cover_plate</td>
      <td>200</td>
      <td>13.93</td>
      <td>16.77</td>
      <td>18.70</td>
      <td>28.02</td>
    </tr>
    <tr>
      <td>square_tube</td>
      <td>99</td>
      <td>16.34</td>
      <td>17.28</td>
      <td>19.11</td>
      <td>27.34</td>
    </tr>
    <tr>
      <td>h_beam</td>
      <td>100</td>
      <td>11.92</td>
      <td>13.27</td>
      <td>18.13</td>
      <td>24.86</td>
    </tr>
    <tr>
      <td>bellmouth</td>
      <td>194</td>
      <td>17.83</td>
      <td>34.94</td>
      <td>18.30</td>
      <td>26.30</td>
    </tr>
    <tr>
      <td>V2</td>
      <td>cover_plate</td>
      <td>524</td>
      <td>14.66</td>
      <td>17.47</td>
      <td>18.90</td>
      <td>28.04</td>
    </tr>
    <tr>
      <td>NEW</td>
      <td>cover_plate</td>
      <td>301</td>
      <td>14.11</td>
      <td>18.00</td>
      <td>18.71</td>
      <td>28.08</td>
    </tr>
  </tbody>
</table>

This table makes the main pattern clear:

1. **`V1 / bellmouth` is the slowest current segment**, with a mean runtime of about `17.83 s` and a P90 runtime of about `34.94 s`.
2. **`V1 / square_tube` is also relatively slow**, but its long-tail behavior is much milder than `bellmouth`.
3. **`V1 / h_beam` is currently the fastest populated V1 category**.
4. **`V2` and `NEW` currently behave mainly as depth-enabled `cover_plate` workloads**, with mean runtime around `14.1–14.7 s`.
5. Across the populated groups, GPU memory remains within the current `48 GB` A6000 safety range; the larger practical difference is **runtime tail behavior**, not memory overflow.

### 4.1.6 Main bottleneck observation

The main long-tail came from the `V1 / bellmouth` subset, which should be the first target if runtime optimization becomes important.

---

## 4.2 Cadrille full-dataset results

Reporting note: in this section, Cadrille runtime, throughput, and end-to-end timing are all reported on the **successful-output basis**. `Samples prepared for Cadrille` is kept only as a completeness note, not as a separate efficiency denominator.

### 4.2.1 PC mode (full-dataset success)

The statistics come from a mixed full-modal run where **PC completed successfully** but the later IMG stage failed. The top-level folder was archived during cleanup, but the **PC shard outputs remain intact and usable**.

Statistics source:

- `/ssd1/rxl/zhankaiming/AIWS/.trash/outputs-cleanup-20260411-153952/cadrille-full-modalities-20260411-090531-bs64/pc`

Configuration:

- Mode: `pc`
- Input: SAM3D-exported STL
- `n_samples=5` (five candidates are generated per input and `evaluate.py` selects the best one, which increases wall-clock time)
- `batch_size=64`
- 4-GPU parallel execution
- Selection strategy: `evaluate`

Results:

| Metric | Value |
|---|---:|
| Samples prepared for Cadrille (completeness note only) | 1418 |
| Final successful outputs | 1418 |
| Success rate | 100.0% |
| Wall-clock time | about 80.1 minutes |
| Effective throughput (successful-output basis) | about 1061.7 successful outputs/hour |

Additional quality indicators (average across shard summaries, reported in the original Cadrille convention):

| Metric | Value |
|---|---:|
| Average IoU (%) | 2.12 |
| Median Chamfer distance (×10³) | 38.00 |

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
| Samples prepared for Cadrille (completeness note only) | 1418 |
| Final successful outputs | 1213 |
| Success rate | 85.54% |
| Selected STL outputs | 1213 |
| Selected STEP outputs | 1211 |
| Wall-clock time | about 25.9 minutes |
| Effective throughput (successful-output basis) | about 2814.6 successful outputs/hour |

Additional quality indicators (average across shard summaries, reported in the original Cadrille convention):

| Metric | Value |
|---|---:|
| Average IoU (%) | 2.62 |
| Median Chamfer distance (×10³) | 47.80 |

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

### 4.3 Average end-to-end inference time (estimated from existing full-run statistics)

Since SAM3D, Cadrille-PC, and Cadrille-IMG were not all executed as one single uninterrupted job under exactly the same settings, the most practical way to report end-to-end timing here is to derive it from the **existing successful full-run wall-clock statistics**.

Here, “average end-to-end time” is defined as:

- **total full-run wall-clock time divided by successful outputs**
- representing the **batch-throughput-equivalent time** under the current 4-GPU setup
- rather than a single-sample serial latency on one GPU
- for SAM3D this still equals the full `1418` formal samples, while for Cadrille it is reported on the final selected-output count

The stage-wise wall-clock equivalents are:

- SAM3D: about `1.60` hours, equivalent to about **4.07 s/sample**
- Cadrille-PC: about `80.1` minutes, equivalent to about **3.39 s/successful output**
- Cadrille-IMG: about `25.9` minutes, equivalent to about **1.28 s/successful output**

This gives the following end-to-end baselines:

| Pipeline | Computation | Average end-to-end time |
|---|---:|---:|
| SAM3D → Cadrille-PC | `4.07 + 3.39` | **7.46 s/successful output** |
| SAM3D → Cadrille-IMG (successful-output basis) | `(SAM3D total wall-clock + IMG total wall-clock) / 1213` | **6.04 s/successful output** |

The interpretation is:

1. **On the current successful-output basis, the IMG pipeline is still faster end-to-end.**
2. **The IMG advantage is smaller than a raw attempted-sample view would suggest**, because the current IMG success rate is about `85.54%`.
3. **The PC pipeline is currently slower in part because it uses `n_samples=5`**, meaning five candidates are generated per input before evaluation selects the best one. That increases compute cost, but the validated full-dataset run also achieved `100%` success.

One caveat is important: this is not a perfectly apples-to-apples modality comparison, because the currently validated stable settings are different:

- PC: `n_samples=5`
- IMG: `n_samples=1`

Therefore, these numbers are best interpreted as **operational baselines under the current production configuration**, rather than as a pure academic comparison with all other variables strictly controlled.

---

## 5. Main Conclusions

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
