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

## Slide 2. Dataset evolution and formal experimental view

### Suggested title
How the dataset evolved from raw storage to a formal experiment-ready view

### Suggested diagram

```text
Raw data
├── aiws5.2-dataset/images/           all RGB images mixed together
├── aiws5.2-dataset/depth/            all depth files mixed together
├── isat_annotations/                 annotation-truth JSON
└── train.json / val.json             split membership only

        ↓ cleaning and restructuring

aiws5.2-usable-split/
└── train|val / V1|V2|NEW / workpiece/
   preserves train/val semantics

aiws5.2-usable/
└── V1|V2|NEW / workpiece/ + misc/
   isolates clean single-instance samples from problematic cases

aiws5.2-usable-materialized/
└── V1|V2|NEW / workpiece /
    ├── images/
    ├── annotations/
    ├── depth_png/ or depth_exr/
    ├── masks/
    └── metadata/
   the formal dataset root actually used in experiments
```

### Main points to say
- the original raw layout was not suitable for stable batch processing
- `isat_annotations/` is the annotation source of truth
- `train.json / val.json` only preserve split membership
- all formal experiments reported here use `aiws5.2-usable-materialized`

### Useful numbers to show
- total formal samples: **1418**
- `V1`: 593
- `V2`: 524
- `NEW`: 301
- multi-instance samples: 23
- unannotated images: 1

### Suggested spoken note
“All results in this report use the same final dataset definition, namely `aiws5.2-usable-materialized`.”

---

## Slide 3. Hardware and environment baseline

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

## Slide 4. SAM3D full-dataset results

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

## Slide 5. Downstream Cadrille validation and bottlenecks

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

## Slide 6. How future runs should be executed

### Suggested title
Recommended future inference workflow

### Suggested flow diagram

```text
AIWS dataset
   ↓
aiws5.2-usable-materialized
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

## Slide 7. Final conclusions and next steps

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

1. **The formal dataset definition has been unified as `aiws5.2-usable-materialized`.**
2. **SAM3D already achieved 100% reconstruction completion on 1418 formal samples.**
3. **The remaining challenge is no longer whether the pipeline runs, but how to schedule each modality more robustly and measure resources more precisely.**
