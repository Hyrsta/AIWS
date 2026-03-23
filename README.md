# AIWS: Artificial Intelligence Welding System

CAD reconstruction module for an automated welding system. Takes multi-view RGB images of workpieces and produces weld-ready CAD models through a two-stage pipeline:

1. **Images to mesh**: [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) reconstructs a 3D mesh from multi-view photographs.
2. **Mesh to CAD**: [Cadrille](https://github.com/col14m/cadrille) converts the mesh into a parametric [CadQuery](https://github.com/CadQuery/cadquery) program, exportable as STL or STEP for welding path planning.

The system targets 5 workpiece types: cover plate, square tube, H-beam, channel steel, and bellmouth.

> Part of the AIWS project at Fudan University (Oct 2025 - present).

## Pipeline

```
Multi-view RGB images
        |
        v
  ┌───────────┐
  │  SAM 3D   │  single-image 3D reconstruction
  │  Objects   │  (per-view mesh with pose + shape)
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
  Welding path planning & control
```

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

# Install Cadrille dependencies (see their Dockerfile for exact versions)
cd third_party/cadrille
pip install -e .
cd ../..

# CadQuery (for STL/STEP export)
conda install -c conda-forge cadquery -y
```

### Model checkpoints

SAM 3D Objects and Cadrille each need pre-trained weights:
- SAM 3D: download per [their setup guide](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md) (requires HuggingFace access request)
- Cadrille: weights pull automatically from HuggingFace (`maksimko123/cadrille-rl`)

## Usage

### Full pipeline (images to CAD)

```bash
python pipeline.py \
    --input-dir data/workpiece_01 \
    --output-dir output/workpiece_01
```

### Skip SAM3D (use an existing mesh)

```bash
python pipeline.py \
    --skip-sam3d \
    --mesh data/workpiece_01/scan.ply \
    --output-dir output/workpiece_01
```

### Mesh reconstruction only

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
