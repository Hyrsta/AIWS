# AIWS: Artificial Intelligence Welding System

AIWS is an automated welding system that replaces manual workpiece measurement and CAD modeling with a vision-based pipeline. A welder currently spends hours per workpiece: photographing it, measuring dimensions by hand, drawing up the CAD model, then planning the weld path. AIWS does the same job in under a minute per part.

The system handles 5 standard workpiece types used in structural welding: **cover plate**, **square tube**, **H-beam**, **channel steel**, and **bellmouth**.

> Part of the AIWS project at Fudan University (Oct 2025 - present).

## System overview

AIWS has two parts: an **offline** CAD reconstruction step (this repo) and an **online** 3-stage vision pipeline (developed by collaborators). The offline step runs first to build CAD models for each workpiece type, which the online pipeline then uses for real-time alignment during welding.

### Offline: CAD reconstruction (this repo)

```
Multi-view RGB images of workpiece
        |
        v
  ┌───────────┐
  │  SAM 3D   │  single-image 3D reconstruction
  │  Objects   │  (mesh with pose + shape)
  └─────┬─────┘
        |
        v
   3D Mesh (.ply)
        |
        v
  ┌───────────┐
  │ Cadrille  │  point cloud -> CadQuery script
  │           │  (parametric CAD program)
  └─────┬─────┘
        |
        v
  CadQuery .py  ──>  STL mesh  /  STEP B-Rep
        |
        v
  Stored in CAD model database
```

### Online: real-time welding pipeline (3 stages)

The online pipeline runs on-site with RGB-D cameras and uses the CAD models built offline.

```
Stage 1: Reachability check
  Can the welding torch reach this workpiece in its current position?
  If not, a human repositions it before proceeding.

Stage 2: Recognition + coarse estimation
  YOLOv11-seg identifies the workpiece type and segments it.
  GenPose++ estimates rough dimensions and pose from RGB-D input.
  Retrieves the matching CAD model from the database (built offline).

Stage 3: Precise alignment
  FoundationPose aligns the CAD model to the actual workpiece,
  outputting a 6D pose in camera coordinates.
  This feeds into weld seam mapping and robot path planning.
```

## What this repo does

This repo wraps two models into a single image-to-CAD pipeline for the offline step:

1. **[SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects)**: takes multi-view RGB photographs and reconstructs a 3D mesh with pose and shape.
2. **[Cadrille](https://github.com/col14m/cadrille)**: takes the mesh (as a sampled point cloud) and generates a parametric [CadQuery](https://github.com/CadQuery/cadquery) program, which can be exported as STL or STEP.

## Repository Structure

```
AIWS/
├── pipeline.py            # End-to-end: images -> mesh -> CAD
├── sam3d_wrapper.py       # SAM 3D Objects integration
├── cadrille_wrapper.py    # Cadrille integration
├── convert_cadquery.py    # Batch CadQuery script -> STL/STEP conversion
├── evaluate.py            # IoU, Chamfer Distance, invalidity metrics
├── third_party/           # Git submodules
│   ├── sam-3d-objects/    # Meta's SAM 3D Objects
│   └── cadrille/          # Cadrille CAD reconstruction
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Installation

**Prerequisites:** Linux 64-bit, CUDA GPU (32GB+ VRAM), Python 3.10+, conda.

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/Hyrsta/AIWS.git
cd AIWS

# If you already cloned without submodules:
git submodule update --init --recursive

# Create environment
conda create -n aiws python=3.10 -y
conda activate aiws

# PyTorch (match your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Project dependencies
pip install -r requirements.txt

# Install SAM 3D Objects
cd third_party/sam-3d-objects
pip install -e .
cd ../..

# Install Cadrille dependencies (see their Dockerfile for pinned versions)
cd third_party/cadrille
pip install -e .
cd ../..

# CadQuery (for STL/STEP export)
conda install -c conda-forge cadquery -y
```

### Model checkpoints

- SAM 3D: download per [their setup guide](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md) (requires HuggingFace access request)
- Cadrille: weights pull automatically from HuggingFace (`maksimko123/cadrille-rl`)

## Usage

### Full pipeline (images to CAD)

```bash
python pipeline.py \
    --input-dir data/workpiece_01 \
    --output-dir output/workpiece_01
```

### Use an existing mesh (skip SAM3D)

```bash
python pipeline.py \
    --skip-sam3d \
    --mesh data/workpiece_01/scan.ply \
    --output-dir output/workpiece_01
```

### Mesh reconstruction only (skip Cadrille)

```bash
python pipeline.py \
    --input-dir data/workpiece_01 \
    --output-dir output/workpiece_01 \
    --skip-cadrille
```

### Batch CadQuery conversion

```bash
python convert_cadquery.py --src work_dirs/cad --mesh-out work_dirs/stl

# Also export STEP files
python convert_cadquery.py --src work_dirs/cad --mesh-out work_dirs/stl --export-brep
```

### Evaluation

```bash
python evaluate.py \
    --pred-dir output/workpiece_01/cad \
    --gt-path data/workpiece_01/gt.stl \
    --output results.json
```

## Output

```
output/workpiece_01/
├── mesh/
│   └── reconstructed.ply    # 3D mesh from SAM3D
└── cad/
    ├── reconstructed.py     # CadQuery parametric script
    ├── reconstructed.stl    # Tessellated mesh
    └── reconstructed.step   # B-Rep (if --export-brep)
```

## Tech Stack

Python, PyTorch, Open3D, Transformers, cadquery, trimesh, Git

## Third-party Components

| Component | Purpose | License |
|-----------|---------|---------|
| [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) | Single-image 3D mesh reconstruction | Meta SAM License |
| [Cadrille](https://github.com/col14m/cadrille) | Multi-modal mesh-to-CAD reconstruction | See their LICENSE |
| [CadQuery](https://github.com/CadQuery/cadquery) | Parametric CAD scripting + STEP export | Apache 2.0 |

## License

MIT (this repo). Third-party components retain their own licenses. See [LICENSE](LICENSE).
