# 2026-04-11 AIWS × SAM3D Environment and Inference SOP

**Author**: Kaiming Zhan  
**Date**: 2026-04-11  
**Purpose**: Reusable operating procedure for rebuilding the SAM3D environment and running the AIWS offline RGB image → Cadrille pipeline in future work.

---

## 1. Scope

This SOP answers three questions:

1. How to set up the **SAM3D** environment on `RXL`.
2. How to run **future SAM3D inference on AIWS offline RGB images**.
3. How to continue from **AIWS offline RGB images through SAM3D into Cadrille** for downstream CAD generation.

---

## 2. Current Production Paths

### 2.1 Server and directories

- Server alias: `RXL`
- Project root: `/ssd1/rxl/zhankaiming/AIWS`
- SAM3D repo: `/ssd1/rxl/zhankaiming/AIWS/repos/sam-3d-objects`
- Cadrille repo: `/ssd1/rxl/zhankaiming/AIWS/repos/cadrille`
- Dataset root: `/ssd1/rxl/zhankaiming/AIWS/data/aiws5.2-usable`
- SAM3D Python env: `/home/rxl/anaconda3/envs/sam3d-objects`

### 2.2 Current successful outputs

- SAM3D successful full run:
    - `/ssd1/rxl/zhankaiming/AIWS/outputs/sam3d-aiws52-clean-mesh-stl-20260410-193527`
- Cadrille IMG successful full run:
    - `/ssd1/rxl/zhankaiming/AIWS/outputs/cadrille-img-only-20260411-143505-shmfix`

### 2.3 Current dataset structure and condition

The formal experiments now use `aiws5.2-usable/` as the single dataset root.

Its semantics are:

- top level: `V1 / V2 / NEW`
- under each subset: workpiece folders
- inside each workpiece folder, the typical contents are:
    - `images/`
    - `annotations/`
    - `depth_png/` or `depth_exr/`
- supporting top-level folders:
    - `metadata/`
    - `misc/`
        - `multi_instance/`
        - `multi_label/`
        - `unannotated_images/`

In this structure:

- `V1`: no depth
- `V2`: depth stored as PNG
- `NEW`: depth stored as EXR
- `misc/`: special samples such as multi-instance, multi-label, or unannotated cases

Annotation truth should still be checked against `isat_annotations/`.

The current main benchmark uses single-instance samples, so `misc/` is kept separately.

The current dataset condition can be summarized directly as:

- total usable samples in the main experimental view: **1418**
- `V1`: **593** samples, no depth, currently covering `cover_plate / square_tube / h_beam / bellmouth`
- `V2`: **524** samples, all with PNG depth, currently all labeled as `cover_plate` in the main view
- `NEW`: **301** samples, all with EXR depth, currently all labeled as `cover_plate` in the main view
- `channel_steel`: currently has no populated instances in the main experimental view

The currently identified `misc/` cases include 23 `multi_instance` samples, 0 `multi_label` samples, and 1 unannotated image (`NEW-G90-BLACK-24`).

In the current data, `misc/` is dominated by `multi_instance` samples.

### 2.4 Project structure (upstream repos vs. local scripts)

The project is easiest to understand as a combination of upstream repos and local wrapper scripts.

**Upstream repos**

- SAM3D: `/ssd1/rxl/zhankaiming/AIWS/repos/sam-3d-objects`
    - the main upstream/default entry points used here are `demo.py`, `notebook/inference.py`, and `checkpoints/hf/pipeline.yaml`
- Cadrille: `/ssd1/rxl/zhankaiming/AIWS/repos/cadrille`
    - the main upstream/default scripts used here are `test.py`, `evaluate.py`, and `convert_cadquery.py`

**Local scripts created for this project**

- `scripts/build_aiws52_usable_view.py`: builds the cleaned `aiws5.2-usable/` dataset view
- `scripts/generate_aiws52_instance_masks.py`: prepares instance-level masks and intermediate data from annotations
- `scripts/run_sam3d_aiws52_batch.py`: resumable SAM3D batch runner with sharding, multi-GPU support, and runtime metrics
- `scripts/analyze_sam3d_run_metrics.py`: summarizes and analyzes SAM3D run statistics
- `scripts/run_sam3d_to_cadrille_e2e.py`: bridges SAM3D mesh outputs into Cadrille and writes pipeline summaries
- `scripts/run_cadrille_full_modalities_4gpu.py`: launches full-dataset Cadrille `pc/img` shard jobs across GPUs

### 2.5 Which directory to use in future runs

- **For SAM3D / Cadrille runs**: use `aiws5.2-usable/`
- **For annotation truth**: use `isat_annotations/`
- **For excluded cases that may be fixed later**: inspect `misc/`

---

## 3. SAM3D Environment Setup

## 3.1 Prerequisites

- Linux 64-bit
- NVIDIA GPU, with official guidance recommending at least `32 GB VRAM`
- Current validated platform: `4 × RTX A6000 48 GB`
- `mamba` or `conda`

## 3.2 Get the repository

If direct cloning is unstable in the current network environment, use this workflow:

1. Clone the official repo locally
2. Verify integrity locally
3. Sync to `RXL`

Current production baseline commit:

- `81a82373a3a7f4cbb00bd5b32aaf6b4d0f659ddd`

## 3.3 Create the Conda environment

On `RXL`:

```bash
cd /ssd1/rxl/zhankaiming/AIWS/repos/sam-3d-objects

mamba env create -f environments/default.yml
mamba activate sam3d-objects

export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
pip install -e '.[dev]'
pip install -e '.[p3d]'

export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'

./patching/hydra
```

## 3.4 Download or restore checkpoints

The official route is HuggingFace:

```bash
pip install 'huggingface-hub[cli]<1.0'
hf auth login

cd /ssd1/rxl/zhankaiming/AIWS/repos/sam-3d-objects
TAG=hf
hf download \
  --repo-type model \
  --local-dir checkpoints/${TAG}-download \
  --max-workers 1 \
  facebook/sam-3d-objects
mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
rm -rf checkpoints/${TAG}-download
```

## 3.5 Restore local runtime assets

A stable SAM3D runtime requires the following assets to be present:

- `checkpoints/hf/pipeline.yaml`
- SAM3D checkpoints
- MoGe weights
- DINOv2 cache
- DINOv2 checkpoint

Key paths:

- checkpoints: `/ssd1/rxl/zhankaiming/AIWS/repos/sam-3d-objects/checkpoints/hf`
- MoGe: `/ssd1/rxl/zhankaiming/AIWS/models/moge-vitl/model-real.pt`
- DINO cache: `/home/rxl/.cache/torch/hub/facebookresearch_dinov2_main`
- DINO checkpoint: `/home/rxl/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth`

## 3.6 Enable `flash_attn`

For `RTX A6000`, production inference should explicitly set:

```bash
export ATTN_BACKEND=flash_attn
export SPARSE_ATTN_BACKEND=flash_attn
```

Notes:

- The prebuilt `flash_attn` wheel may not be compatible with the server
- The stable production solution here was to **build `flash_attn` from source on the server**

If the package needs to be rebuilt, the recommended `RXL` runbook is:

```bash
mamba activate sam3d-objects
cd /tmp
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.8.3

pip uninstall -y flash-attn flash_attn || true
pip install -U pip setuptools wheel ninja packaging

MAX_JOBS=8 TORCH_CUDA_ARCH_LIST="8.0;8.6" \
  pip install --no-build-isolation .

python -c "import flash_attn; print('flash_attn ok:', flash_attn.__version__)"
```

Additional notes:

- If `RXL` cannot clone from GitHub reliably, clone the source tree locally first, sync it to the server, and then run `pip install --no-build-isolation .`
- Even after `flash_attn` imports correctly, SAM3D on `RTX A6000` still needs explicit `ATTN_BACKEND=flash_attn` and `SPARSE_ATTN_BACKEND=flash_attn`, because the upstream auto-selection logic does not cover A6000 by default

---

## 4. SAM3D Inference Workflow

## 4.1 Single-image / single-mask quick test

Inside the `sam-3d-objects` repo, the official Python interface can be used directly:

```python
import sys
sys.path.append("notebook")
from inference import Inference, load_image

config_path = "checkpoints/hf/pipeline.yaml"
inference = Inference(config_path, compile=False)

image = load_image("your_image.png")
mask = ...  # binary mask required
output = inference(image, mask, seed=42)
output["glb"].export("mesh.glb")
output["glb"].export("mesh.stl")
```

Use this for:

- single-sample validation
- environment smoke tests
- trying new data before large-scale runs

## 4.2 Dataset batch inference for one shard

Current production script:

- `/ssd1/rxl/zhankaiming/AIWS/scripts/run_sam3d_aiws52_batch.py`

Single-shard example:

```bash
ATTN_BACKEND=flash_attn \
SPARSE_ATTN_BACKEND=flash_attn \
CUDA_VISIBLE_DEVICES=0 \
/home/rxl/anaconda3/envs/sam3d-objects/bin/python -u \
/ssd1/rxl/zhankaiming/AIWS/scripts/run_sam3d_aiws52_batch.py \
  --dataset-root /ssd1/rxl/zhankaiming/AIWS/data/aiws5.2-usable \
  --dataset-layout subset \
  --repo-root /ssd1/rxl/zhankaiming/AIWS/repos/sam-3d-objects \
  --output-root /ssd1/rxl/zhankaiming/AIWS/outputs/sam3d-yourrun/shard-0 \
  --resume \
  --num-shards 4 \
  --shard-index 0
```

## 4.3 Dataset batch inference on 4 GPUs

Recommended `tmux`-friendly launch:

```bash
OUT=/ssd1/rxl/zhankaiming/AIWS/outputs/sam3d-$(date +%Y%m%d-%H%M%S)
PY=/home/rxl/anaconda3/envs/sam3d-objects/bin/python
SCRIPT=/ssd1/rxl/zhankaiming/AIWS/scripts/run_sam3d_aiws52_batch.py
DATA=/ssd1/rxl/zhankaiming/AIWS/data/aiws5.2-usable
REPO=/ssd1/rxl/zhankaiming/AIWS/repos/sam-3d-objects

for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i \
  ATTN_BACKEND=flash_attn \
  SPARSE_ATTN_BACKEND=flash_attn \
  "$PY" -u "$SCRIPT" \
    --dataset-root "$DATA" \
    --dataset-layout subset \
    --repo-root "$REPO" \
    --output-root "$OUT/shard-$i" \
    --resume \
    --num-shards 4 \
    --shard-index $i \
    > "$OUT/shard-$i.log" 2>&1 &
done
wait
```

## 4.4 Monitoring and resume

Monitor shard summaries:

```bash
for s in /ssd1/rxl/zhankaiming/AIWS/outputs/sam3d-yourrun/shard-*; do
  echo "--- ${s} ---"
  jq '{completed_ok, failed, skipped, processed, total_tasks_in_shard, eta_sec}' "${s}/summary.json"
done
```

Resume behavior:

- keep the same parameters
- re-run with `--resume`
- only restart interrupted shards

---

## 5. Continuing from SAM3D to Cadrille

## 5.1 Single end-to-end bridge script

Script:

- `/ssd1/rxl/zhankaiming/AIWS/scripts/run_sam3d_to_cadrille_e2e.py`

This script can:

1. reuse existing SAM3D outputs via `--skip-sam3d`
2. normalize STL into a Cadrille-compatible unit cube
3. run Cadrille
4. export `tmp_py / tmp_mesh / tmp_brep`
5. use `evaluate` to produce `selected_py / selected_mesh / selected_brep`

## 5.2 Recommended future configuration for Cadrille-PC

```bash
OUT=/ssd1/rxl/zhankaiming/AIWS/outputs/cadrille-pc-only-$(date +%Y%m%d-%H%M%S)
/home/rxl/anaconda3/envs/sam3d-objects/bin/python \
/ssd1/rxl/zhankaiming/AIWS/scripts/run_cadrille_full_modalities_4gpu.py \
  --output-root "$OUT" \
  --sam3d-output-root /ssd1/rxl/zhankaiming/AIWS/outputs/sam3d-aiws52-clean-mesh-stl-20260410-193527 \
  --modalities pc \
  --split-prefix sam3d_bridge_pcfull \
  --gpus 0,1,2,3 \
  --pc-n-samples 5 \
  --cadrille-batch-size 64 \
  --selection-mode evaluate \
  --cadrille-runtime docker \
  --cadrille-docker-image cadrille:latest \
  --export-brep
```

Notes:

- This configuration has already been validated for full-dataset PC-mode inference.
- The critical requirement is **strict one-shard-per-GPU placement**.

## 5.3 Recommended future configuration for Cadrille-IMG

```bash
OUT=/ssd1/rxl/zhankaiming/AIWS/outputs/cadrille-img-only-$(date +%Y%m%d-%H%M%S)
/home/rxl/anaconda3/envs/sam3d-objects/bin/python \
/ssd1/rxl/zhankaiming/AIWS/scripts/run_cadrille_full_modalities_4gpu.py \
  --output-root "$OUT" \
  --sam3d-output-root /ssd1/rxl/zhankaiming/AIWS/outputs/sam3d-aiws52-clean-mesh-stl-20260410-193527 \
  --modalities img \
  --split-prefix sam3d_bridge_imgfix \
  --gpus 0,1,2,3 \
  --img-n-samples 1 \
  --cadrille-batch-size 32 \
  --selection-mode evaluate \
  --cadrille-runtime docker \
  --cadrille-docker-image cadrille:latest \
  --cadrille-docker-extra-args '--ipc=host --shm-size=16g' \
  --export-brep
```

Notes:

- Do **not** use the default large-batch IMG configuration for large-scale runs.
- The currently validated stable recipe is:
    - `batch_size=32`
    - `--ipc=host`
    - `--shm-size=16g`
    - reduced dataloader worker pressure

## 5.4 Recommended 100-sample smoke test for new settings

Before a new full-dataset run, validate with a small subset first:

```bash
/home/rxl/anaconda3/envs/sam3d-objects/bin/python \
/ssd1/rxl/zhankaiming/AIWS/scripts/run_sam3d_to_cadrille_e2e.py \
  --skip-sam3d \
  --sam3d-output-root /ssd1/rxl/zhankaiming/AIWS/outputs/sam3d-aiws52-clean-mesh-stl-20260410-193527 \
  --cadrille-output-root /ssd1/rxl/zhankaiming/AIWS/outputs/cadrille-smoke-next \
  --cadrille-split-name sam3d_bridge_smoke_next \
  --cadrille-runtime docker \
  --cadrille-docker-image cadrille:latest \
  --cadrille-mode img \
  --cadrille-input-source mesh \
  --cadrille-n-samples 1 \
  --cadrille-batch-size 32 \
  --cadrille-docker-gpus device=0 \
  --cadrille-docker-extra-args '--ipc=host --shm-size=16g' \
  --selection-mode evaluate \
  --sample-offset 0 \
  --max-samples 100 \
  --export-brep
```

---

## 6. Recommended Operational Strategy

## 6.1 Do not run all stages as one tightly coupled block by default

The safer order is:

1. **Run SAM3D first** and export STL
2. **Run Cadrille-PC separately**
3. **Run Cadrille-IMG separately**

Why:

- each stage has a different bottleneck
- PC mode is more VRAM and GPU-placement sensitive
- IMG mode is more DataLoader / shm sensitive
- failure recovery is much easier when stages are decoupled

## 6.2 Minimum metrics that should always be preserved

SAM3D already logs:

- `duration_sec`
- `peak_memory_allocated_mb`
- `peak_memory_reserved_mb`
- `instances_per_hour`

Cadrille should be upgraded to log the same type of GPU-memory metrics, at least around:

- `test.py`
- `evaluate.py`
- or the outer launcher

Recommended additions:

- `torch.cuda.max_memory_allocated()`
- `torch.cuda.max_memory_reserved()`

That will make future reports much stronger, especially for per-modality memory comparisons.

---

## 7. One-Page Practical Recommendation

For future runs, the recommended procedure is:

1. **Keep using** `/home/rxl/anaconda3/envs/sam3d-objects`
2. **Keep using** the current 4-shard SAM3D batch script
3. **Run Cadrille-PC separately with `batch=64`**
4. **Run Cadrille-IMG separately with `batch=32 + --ipc=host + --shm-size=16g`**
5. **Always do a 100-sample smoke test before launching a new full run**

---

## 8. Conclusion

At this point:

- **SAM3D environment setup is no longer the blocker**. It is already reproducible, resumable, and stable for batch-scale use.
- **The main future engineering focus is not whether inference can run, but how to schedule Cadrille more robustly across modalities.**

In practice, the key operational questions for future work are:

- Is GPU pinning strict enough?
- Is IMG-mode shared memory large enough?
- Was a small smoke test completed before the next large run?
