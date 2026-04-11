# AIWS

AIWS is split into two connected parts:

- an **online vision pipeline** used during on-site welding
- an **offline CAD reconstruction pipeline** used before deployment to build the CAD model database

This repository is the AIWS integration layer. It keeps AIWS-specific scripts, docs, and workflow glue here, while the upstream model repos are tracked as submodules under `repos/`.

## Pipeline overview

```mermaid
graph TD
    A[Real workpiece RGB-D observations] --> B[Online pipeline]
    B --> B1[YOLOv11-seg<br/>recognition and segmentation]
    B1 --> B2[GenPose++<br/>coarse size and pose estimation]
    B2 --> B3[FoundationPose<br/>precise CAD alignment]
    B3 --> B4[Localization and weld path planning]

    C[Offline RGB images] --> D[Offline CAD pipeline]
    D --> D1[SAM3D<br/>mesh reconstruction]
    D1 --> D2[Cadrille PC mode<br/>sample point cloud from mesh]
    D1 --> D3[Cadrille IMG mode<br/>render 4-view RGB images from mesh]
    D2 --> E[CAD reconstruction outputs]
    D3 --> E
    E --> B3
```

## Repository layout

```text
AIWS/
├── repos/
│   ├── sam-3d-objects/   # upstream SAM3D repo (submodule)
│   └── cadrille/         # upstream Cadrille repo (submodule)
├── scripts/             # AIWS orchestration, wrappers, and experiment runners
├── docs/                # reports, SOPs, technical records, and outlines
├── gui/                 # local GUI for launching and inspecting runs
├── outputs/             # local generated artifacts and previews
└── aiws5.2-usable*/     # local dataset views used in experiments
```

## Upstream repos used by AIWS

- `repos/sam-3d-objects`
  - upstream SAM3D codebase
  - main entry points used by this workflow include `demo.py`, `notebook/inference.py`, and `checkpoints/hf/pipeline.yaml`
- `repos/cadrille`
  - upstream Cadrille codebase
  - main entry points used by this workflow include `test.py`, `evaluate.py`, and `convert_cadquery.py`

## AIWS-specific integration code

### Cadrille wrappers around the upstream repo

- `scripts/cadrille_test_wrapper.py`
  - thin wrapper for processor/checkpoint override, sample-count control, batch-size control, and GPU-memory logging
- `scripts/cadrille_evaluate.py`
  - evaluation wrapper used by the AIWS e2e pipeline
- `scripts/cadrille_convert_cadquery.py`
  - CAD conversion wrapper used by the AIWS e2e pipeline

### Main offline pipeline scripts

- `scripts/build_aiws52_usable_view.py`
- `scripts/generate_aiws52_instance_masks.py`
- `scripts/sam3d_aiws52_batch.py`
- `scripts/sam3d_run_metrics_analysis.py`
- `scripts/sam3d_to_cadrille_e2e.py`
- `scripts/cadrille_full_modalities_4gpu.py`

## Clone and initialize

```bash
git clone <your-aiws-repo-url>
cd AIWS
git submodule update --init --recursive
```

If you already cloned the repo before submodules were added:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

## What this repo is responsible for

This repo is meant to contain:

- AIWS project documentation
- AIWS dataset-view builders and experiment runners
- wrapper code that adapts upstream model repos to the AIWS workflow
- local GUI and analysis tooling

This repo is **not** meant to vendor large upstream codebases directly into AIWS when a clean submodule can track them instead.

## Current focus

The current documented workflow focuses on the **offline CAD reconstruction pipeline**:

- offline RGB images
- SAM3D mesh reconstruction
- Cadrille CAD reconstruction from mesh-derived PC or IMG inputs
- downstream analysis, evaluation, and reporting
