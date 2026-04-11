# 2026-04-11 AIWS × SAM3D × Cadrille Presentation Outline

**Author**: Kaiming Zhan  
**Use case**: Group meeting or oral presentation  
**Suggested length**: 6 to 8 slides  
**Suggested style**: One conclusion per slide, minimal code, emphasize structure and numbers

---

## Slide 1. Goal and Main Deliverables

### Suggested title
Progress on single-image 3D reconstruction and CAD generation for AIWS welding data

### Main message
The goal of this work was to establish a complete pipeline from AIWS5.2 data into **SAM3D → Cadrille**.

Target outputs:
1. full-dataset SAM3D reconstruction
2. STL outputs reusable by Cadrille
3. runtime, memory, and hardware-constraint analysis
4. a reusable SOP for future runs

### Keep only 3 bullets on the slide
- dataset cleaning and restructuring completed
- SAM3D full-dataset run completed, 1418 / 1418
- downstream Cadrille path validated

### Suggested spoken note
“This round was not just about making one run succeed, but about turning the whole experimental path into a reusable baseline.”

---

## Slide 2. Current dataset structure and formal experimental view

### Suggested title
Current dataset structure and formal experimental view

### Suggested diagram

```text
aiws5.2-usable/
├── V1/                               no depth
├── V2/                               PNG depth
├── NEW/                              EXR depth
├── metadata/                         manifests and summary statistics
└── misc/                             multi-instance, multi-label, and unannotated cases

Under each subset, samples are grouped by workpiece, and each workpiece folder typically contains:
- images/
- annotations/
- depth_png/ or depth_exr/
```

### Main points to say
- the formal experiments all use the unified `aiws5.2-usable` structure
- `isat_annotations/` remains the annotation source of truth
- `misc/` stores samples outside the main benchmark path
- the benchmark path is the cleaner single-instance portion of the dataset

### Useful numbers to show
- total formal samples: **1418**
- `V1`: 593
- `V2`: 524
- `NEW`: 301
- multi-instance samples: 23
- unannotated images: 1
- in the current data, `misc/` is mainly made up of `multi_instance` samples

### Current-condition points this slide should also state
- `V1` has no depth and is currently the most diverse subset by workpiece type
- `V2` is fully depth-enabled with PNG depth, and its current main-view samples are all `cover_plate`
- `NEW` is fully depth-enabled with EXR depth, and its current main-view samples are all `cover_plate`
- `channel_steel` currently has no populated instances in the main usable view

### Suggested spoken note
“At this stage, we can directly show the current dataset structure: the formal experiments use `aiws5.2-usable`, and the main benchmark uses its cleaner single-instance portion.”

---

## Slide 3. Project structure

### Suggested title
Project structure: AIWS online/offline split and the offline CAD pipeline

### Suggested content
- AIWS repo: `https://github.com/Hyrsta/AIWS`
- online pipeline, used on-site during welding:
  - purpose: identify the real workpiece, estimate pose, and align it against CAD models for downstream localization and weld path planning
  - model stack: `YOLOv11-seg + GenPose++ + FoundationPose`
  - `YOLOv11-seg`: segmentation and recognition
  - `GenPose++`: coarse size / pose estimation
  - `FoundationPose`: precise CAD alignment
- offline pipeline, used before deployment:
  - purpose: build the CAD model database
  - model stack: `SAM3D + Cadrille`
  - `SAM3D`: offline RGB images → mesh reconstruction
  - `Cadrille`: start from the reconstructed mesh, then either sample point clouds from that mesh for PC mode or render 4-view RGB images from that mesh for IMG mode, before CAD reconstruction
- upstream repo, SAM3D: `/ssd1/rxl/zhankaiming/AIWS/repos/sam-3d-objects`
  - default entry points used: `demo.py`, `notebook/inference.py`, `checkpoints/hf/pipeline.yaml`
- upstream repo, Cadrille: `/ssd1/rxl/zhankaiming/AIWS/repos/cadrille`
  - default scripts used: `test.py`, `evaluate.py`, `convert_cadquery.py`
- AIWS wrapper scripts used around the official Cadrille repo:
  - `scripts/cadrille_test_wrapper.py`
  - `scripts/cadrille_evaluate.py`
  - `scripts/cadrille_convert_cadquery.py`
- AIWS offline scripts added in this work:
  - `scripts/build_aiws52_usable_view.py`
  - `scripts/generate_aiws52_instance_masks.py`
  - `scripts/sam3d_aiws52_batch.py`
  - `scripts/sam3d_run_metrics_analysis.py`
  - `scripts/sam3d_to_cadrille_e2e.py`
  - `scripts/cadrille_full_modalities_4gpu.py`

### Main point to say
- explain what each pipeline is used for first, then introduce the model stack under each one
- this work focuses on the offline CAD pipeline, whose outputs are later consumed by the online pipeline

---

## Slide 4. Hardware and environment baseline

### Suggested title
Experimental hardware and environment baseline

### Suggested content
- server: `RXL`
- GPU: `4 × NVIDIA RTX A6000`
- per-GPU memory: `48 GB`
- system RAM: `503 GiB`
- CPU: `2 × Intel Xeon Gold 6326`
- SAM3D env: `/home/rxl/anaconda3/envs/sam3d-objects`
- SAM3D repo: `/ssd1/rxl/zhankaiming/AIWS/repos/sam-3d-objects`

### Key point to emphasize
- `flash_attn` is enabled in the production path
- checkpoints, MoGe, and DINO cache were restored
- the environment is now reproducible, not just temporarily runnable

### Suggested spoken note
“At this stage, environment setup is no longer the main risk. The focus has shifted to scheduling and resource control.”

---

## Slide 5. SAM3D full-dataset results

### Suggested title
Full-dataset SAM3D reconstruction results

### Suggested table

| Metric | Value |
|---|---:|
| Samples | 1418 |
| Successful | 1418 |
| Failed | 0 |
| Completion rate | 100% |
| Total wall-clock time | about 1.60 h |
| Mean per-sample time | 14.799 s |
| P95 per-sample time | 18.977 s |
| Max reserved GPU memory | 28076 MB |

### Main conclusions
- all formal samples completed successfully
- stable under `4 × A6000`
- peak reserved memory is about `28.1 GB`, below the 48 GB limit
- the current SAM3D configuration is production-reusable

### Extra analysis line
- the main long-tail appears in `V1 / bellmouth`

### Suggested spoken note
“For SAM3D, the question is no longer whether it can run, but whether we want to optimize the long-tail categories further.”

---

## Slide 6. Downstream Cadrille validation and bottlenecks

### Suggested title
Validation of SAM3D outputs in the downstream Cadrille stage

### Split the slide into two columns

#### Left, PC mode
- 1418 / 1418 best candidates selected
- 100% success rate
- about 80.1 minutes wall-clock time
- confirms that SAM3D STL outputs can reliably feed the CAD stage

#### Right, IMG mode
- default large-batch setting failed
- the limiting factor was not raw GPU compute, but **shared memory (shm)**
- after fixing:
    - `batch_size=32`
    - `--ipc=host`
    - `--shm-size=16g`
- successful retry produced 1213 STL and 1211 STEP outputs

### Core conclusion of this slide
- PC mode is mainly limited by **GPU pinning and OOM risk**
- IMG mode is mainly limited by **DataLoader / shm behavior**
- therefore the two modalities should not be treated as the same operational workload

### Suggested spoken note
“This shows that future operations should treat PC and IMG as two different resource profiles, not one combined job type.”

---

## Slide 7. How future runs should be executed

### Suggested title
Recommended future inference workflow

### Suggested flow diagram

```text
AIWS dataset
   ↓
aiws5.2-usable
   ↓
SAM3D full-dataset inference
   ↓ STL outputs
Run Cadrille-PC separately
   ↓
Run Cadrille-IMG separately
```

### Recommended operational guidance
- do not launch `pc,img` together in one mixed job for now
- recommended order:
  1. SAM3D
  2. Cadrille-PC
  3. Cadrille-IMG
- always run a `100`-sample smoke test before a new full-dataset launch

### One-line strategy
- **decouple SAM3D and Cadrille**
- **schedule PC and IMG separately**
- **validate on a small subset before full launch**

---

## Slide 8. Final conclusions and next steps

### Suggested title
Conclusion and next-step plan

### Three concise conclusions
1. the formal AIWS5.2 dataset definition is now fixed
2. a stable full-dataset SAM3D baseline has been established
3. the downstream SAM3D → Cadrille path has been validated

### Suggested next steps
- add GPU-memory instrumentation to Cadrille
- analyze the IMG samples that were not successfully selected
- perform targeted analysis on the `bellmouth` long-tail subset

### Closing sentence
**The main achievement is not only that one model ran successfully, but that the full experimental chain is now reproducible and extensible.**

---

## If you need only 5 slides

You can compress the story into:

1. goal and contribution
2. dataset restructuring and formal dataset definition
3. SAM3D full-dataset results
4. downstream Cadrille results and bottlenecks
5. future inference SOP and next steps

---

## Three sentences worth emphasizing in the talk

1. **The formal experiment dataset has been unified as the `aiws5.2-usable` structure.**
2. **SAM3D already achieved 100% reconstruction completion on 1418 formal samples.**
3. **The remaining challenge is no longer whether the pipeline runs, but how to schedule each modality more robustly and measure resources more precisely.**
