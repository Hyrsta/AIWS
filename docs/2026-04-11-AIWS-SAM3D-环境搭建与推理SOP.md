# 2026-04-11 AIWS × SAM3D 环境搭建与后续推理 SOP

**作者**: 詹铠铭  
**日期**: 2026-04-11  
**用途**: 面向后续复现实验、继续跑全量推理、或者给新数据做推理时直接复用。

---

## 1. 目标与适用范围

本 SOP 解决三个问题：

1. 如何在 `RXL` 上搭好 **SAM3D** 运行环境。
2. 如何在未来继续做 **SAM3D 全量推理**。
3. 如何把 **SAM3D 输出接到 Cadrille**，继续做 CAD 生成。

---

## 2. 当前正式路径

### 2.1 服务器与路径

- 服务器别名：`RXL`
- 项目根目录：`/ssd1/rxl/zhankaiming/AIWS`
- SAM3D 仓库：`/ssd1/rxl/zhankaiming/AIWS/repos/sam-3d-objects`
- Cadrille 仓库：`/ssd1/rxl/zhankaiming/AIWS/repos/cadrille`
- 数据集目录：`/ssd1/rxl/zhankaiming/AIWS/data/aiws5.2-usable`
- SAM3D Python 环境：`/home/rxl/anaconda3/envs/sam3d-objects`

### 2.2 当前成功输出

- SAM3D 成功全量输出：
  - `/ssd1/rxl/zhankaiming/AIWS/outputs/sam3d-aiws52-clean-mesh-stl-20260410-193527`
- Cadrille IMG 成功全量输出：
  - `/ssd1/rxl/zhankaiming/AIWS/outputs/cadrille-img-only-20260411-143505-shmfix`

### 2.3 数据集目录是怎么整理出来的

原始数据更接近平铺式资源池：

- `aiws5.2-dataset/images/`：RGB 图像混放
- `aiws5.2-dataset/depth/`：深度文件混放
- `isat_annotations/`：标注真值 JSON
- `train.json` / `val.json`：只记录 split 归属

为了方便稳定地跑批量实验，后续统一整理成 `aiws5.2-usable/` 结构。

这个结构的含义是：

- 顶层按 `V1 / V2 / NEW` 分组
- 每个 subset 下再按工件类别分组
- 每个工件目录下通常包含：
  - `images/`
  - `annotations/`
  - `depth_png/` 或 `depth_exr/`
- 顶层辅助目录：
  - `metadata/`
  - `misc/`

其中：

- `V1`：无深度
- `V2`：深度为 PNG
- `NEW`：深度为 EXR
- `misc/`：存放多实例、多类别、无标注等不进入主实验主干的样本

当前没有把 `misc/` 并入正式主实验，核心原因是这部分样本里有较多 **multi-instance 标签**。而当前主流程默认按单实例样本组织和批处理，所以这部分数据后续需要单独修复多实例解析或实例拆分策略后，再考虑纳入。

### 2.4 后续正式运行时应该用哪个目录

- **跑 SAM3D / Cadrille**：使用 `aiws5.2-usable/`
- **检查标注真值**：以 `isat_annotations/` 为准
- **查看被排除且后续可能修复的特殊样本**：看 `misc/`

---

## 3. SAM3D 环境搭建

## 3.1 前置条件

- Linux 64-bit
- NVIDIA GPU，官方文档建议至少 `32 GB VRAM`
- 当前实测平台：`4 × RTX A6000 48 GB`
- 推荐使用 `mamba` / `conda`

## 3.2 获取仓库

如果中国网络环境下直接拉取不稳定，建议采用：

1. 本地机器拉官方仓库
2. 本地确认完整性
3. 同步到 `RXL`

当前正式基线 commit：

- `81a82373a3a7f4cbb00bd5b32aaf6b4d0f659ddd`

## 3.3 创建 Conda 环境

在 `RXL` 上执行：

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

## 3.4 下载或补齐 checkpoints

官方推荐从 HuggingFace 拉取：

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

## 3.5 补齐本地依赖资源

SAM3D 当前稳定运行依赖以下资源齐全：

- `checkpoints/hf/pipeline.yaml`
- MoGe 权重
- DINOv2 cache
- DINOv2 checkpoint

关键路径：

- checkpoints: `/ssd1/rxl/zhankaiming/AIWS/repos/sam-3d-objects/checkpoints/hf`
- MoGe: `/ssd1/rxl/zhankaiming/AIWS/models/moge-vitl/model-real.pt`
- DINO cache: `/home/rxl/.cache/torch/hub/facebookresearch_dinov2_main`
- DINO checkpoint: `/home/rxl/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth`

## 3.6 打开 `flash_attn`

在 `RTX A6000` 上，正式运行时要显式指定：

```bash
export ATTN_BACKEND=flash_attn
export SPARSE_ATTN_BACKEND=flash_attn
```

说明：

- 预编译 `flash_attn` wheel 在服务器上不一定兼容
- 当前稳定方案是**在服务器本地源码编译 `flash_attn`**

---

## 4. SAM3D 推理方法

## 4.1 单图 / 单 mask 快速推理

在 `sam-3d-objects` 仓库里可直接使用官方接口：

```python
import sys
sys.path.append("notebook")
from inference import Inference, load_image

config_path = "checkpoints/hf/pipeline.yaml"
inference = Inference(config_path, compile=False)

image = load_image("your_image.png")
mask = ...  # 需要提供二值 mask
output = inference(image, mask, seed=42)
output["glb"].export("mesh.glb")
output["glb"].export("mesh.stl")
```

适合：

- 单张样本验证
- 调模型是否正常
- 新数据接入前 smoke test

## 4.2 AIWS 数据集批量推理（单个 shard）

当前正式脚本：

- `/ssd1/rxl/zhankaiming/AIWS/scripts/run_sam3d_aiws52_batch.py`

单个 shard 示例：

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

## 4.3 AIWS 数据集批量推理（4 卡并行）

推荐在 `tmux` 中启动：

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

## 4.4 监控与断点续跑

监控汇总：

```bash
for s in /ssd1/rxl/zhankaiming/AIWS/outputs/sam3d-yourrun/shard-*; do
  echo "--- ${s} ---"
  jq '{completed_ok, failed, skipped, processed, total_tasks_in_shard, eta_sec}' "${s}/summary.json"
done
```

断点续跑：

- 保留原参数
- 继续带 `--resume`
- 只重启中断的 shard

---

## 5. SAM3D → Cadrille 的后续推理方法

## 5.1 单次端到端脚本

脚本：

- `/ssd1/rxl/zhankaiming/AIWS/scripts/run_sam3d_to_cadrille_e2e.py`

这个脚本可以：

1. 直接复用已有 SAM3D 输出（`--skip-sam3d`）
2. 把 STL 归一化为 Cadrille 可接受的输入
3. 跑 Cadrille
4. 导出 `tmp_py / tmp_mesh / tmp_brep`
5. 用 `evaluate` 选出 `selected_py / selected_mesh / selected_brep`

## 5.2 未来继续跑 Cadrille-PC（推荐配置）

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

说明：

- 该配置已经验证可以完成全量 PC 推理。
- 关键点是**每个 shard 严格绑定独立 GPU**，避免多进程抢同一张卡。

## 5.3 未来继续跑 Cadrille-IMG（推荐稳定配置）

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

说明：

- **不要**直接使用 IMG 的 `batch_size=64` 默认设置做大规模全量跑。
- 当前稳定经验是：
  - `batch_size=32`
  - `--ipc=host`
  - `--shm-size=16g`
  - 降低 dataloader worker 压力

## 5.4 小规模验证或新数据先做 100 个样本 smoke test

如果不是立刻跑全量，推荐先做一个 100 样本的小规模验证：

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

## 6. 推荐的后续运维策略

## 6.1 不再建议一次性把所有阶段混在一起跑

更稳妥的顺序是：

1. **先完成 SAM3D 全量 STL 导出**
2. **再单独跑 Cadrille-PC**
3. **最后单独跑 Cadrille-IMG**

原因：

- 三个阶段的资源瓶颈不同
- PC 更偏显存与 GPU pinning
- IMG 更偏 DataLoader / shm
- 解耦后，失败更容易恢复

## 6.2 建议保留的最小统计口径

SAM3D 当前已经记录：

- `duration_sec`
- `peak_memory_allocated_mb`
- `peak_memory_reserved_mb`
- `instances_per_hour`

建议后续给 Cadrille 也补上同样的显存记录，至少在：

- `test.py`
- `evaluate.py`
- 或者外层 launcher

记录：

- `torch.cuda.max_memory_allocated()`
- `torch.cuda.max_memory_reserved()`

这样后续汇报就可以直接给出：

- PC 平均/峰值显存
- IMG 平均/峰值显存
- 分 batch size 的资源差异

---

## 7. 一页版执行建议

如果后续要继续推理，建议直接照下面执行：

1. **环境不动**，继续使用 `/home/rxl/anaconda3/envs/sam3d-objects`
2. **SAM3D 继续沿用当前 4 shard 脚本**
3. **Cadrille-PC 用 `batch=64` 单独跑**
4. **Cadrille-IMG 用 `batch=32 + --ipc=host + --shm-size=16g` 单独跑**
5. **每次新配置先做 100 样本 smoke test，再上全量**

---

## 8. 结论

这套 SOP 的意义是：

- **SAM3D 环境已经不是问题**，它已经进入“可复用、可批量、可断点恢复”的阶段。
- **后续重点不在能否推理，而在如何更稳地调度 Cadrille 不同模态。**

也就是说，后续继续做推理时，真正要管的是：

- GPU pinning 是否严格
- IMG 的 shm 是否足够
- 是否先小规模 smoke test 再发全量
