# 2026-04-11 AIWS × SAM3D 环境搭建与后续推理 SOP

**作者**: 詹铠铭  
**日期**: 2026-04-11  
**用途**: 面向后续复现实验，以及继续运行 AIWS 离线 CAD 重建管线时直接复用。

---

## 1. 目标与适用范围

本 SOP 解决三个问题：

1. 如何在 `RXL` 上搭好 **SAM3D** 运行环境。
2. 如何在未来继续对 **AIWS 离线 RGB 图像** 做 SAM3D 推理。
3. 如何把 **AIWS 离线 RGB 图像经由 SAM3D 接到 Cadrille**，继续做下游 CAD 生成。

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

### 2.3 当前数据结构与状态

当前正式实验统一使用 `aiws5.2-usable/`。

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
        - `multi_instance/`
        - `multi_label/`
        - `unannotated_images/`

其中：

- `V1`：无深度
- `V2`：深度为 PNG
- `NEW`：深度为 EXR
- `misc/`：存放多实例、多类别、无标注等不进入主实验主干的样本

标注真值仍以 `isat_annotations/` 为准。

当前主实验主干采用的是单实例样本，因此 `misc/` 单独存放。

当前数据状态可以直接概括为：

- 主实验可用样本总数：**1418**
- `V1`：**593** 个样本，无深度，当前覆盖 `cover_plate / square_tube / h_beam / bellmouth`
- `V2`：**524** 个样本，全部带 PNG 深度，当前主实验样本均为 `cover_plate`
- `NEW`：**301** 个样本，全部带 EXR 深度，当前主实验样本均为 `cover_plate`
- `channel_steel`：当前主实验视图中暂无实例

当前 `misc/` 中已识别的特殊样本包括：23 个 `multi_instance`，0 个 `multi_label`，以及 1 个无标注图像样本（`NEW-G90-BLACK-24`）。

从当前统计看，`misc/` 以 `multi_instance` 样本为主。

### 2.4 AIWS 项目结构

AIWS 当前包含两个相互衔接的部分：

- **在线视觉管线**：用于现场焊接阶段，负责识别真实工件、估计位姿，并与 CAD 模型对齐，以支撑后续定位与焊接路径规划。
    - 模型栈：`YOLOv11-seg + GenPose++ + FoundationPose`
- **离线 CAD 重建管线**：用于部署前构建 CAD 模型库。
    - 模型栈：`SAM3D + Cadrille`
    - `SAM3D`：离线 RGB 图像 → 网格重建
    - `Cadrille`：从重建得到的网格出发，PC 模式从网格采样点云，IMG 模式从网格渲染 4 视图 RGB 图像，随后再做 CAD 重建

**当前仓库结构**

- GitHub：`https://github.com/Hyrsta/AIWS`
- `repos/sam-3d-objects`：SAM3D 官方上游仓库，以 submodule 方式跟踪
- `repos/cadrille`：Cadrille 官方上游仓库，以 submodule 方式跟踪
- `scripts/`：AIWS 自己的编排、wrapper、数据准备与分析脚本
- `docs/`：报告、SOP 与汇报材料
- `gui/`：本地运行与结果查看界面

**离线管线直接使用的上游入口**

- SAM3D（`repos/sam-3d-objects`）：`demo.py`、`notebook/inference.py`、`checkpoints/hf/pipeline.yaml`
- Cadrille（`repos/cadrille`）：`test.py`、`evaluate.py`、`convert_cadquery.py`

**AIWS 的 wrapper 与编排脚本**

- `scripts/cadrille_test_wrapper.py`：薄封装，只负责 processor/checkpoint 覆盖、sample 数控制、batch size 控制，以及 GPU 显存日志
- `scripts/cadrille_evaluate.py`：AIWS e2e 流程里使用的评估 wrapper
- `scripts/cadrille_convert_cadquery.py`：AIWS e2e 流程里使用的 CAD 转换 wrapper
- `scripts/build_aiws52_usable_view.py`：整理并构建 `aiws5.2-usable/` 数据视图
- `scripts/generate_aiws52_instance_masks.py`：根据标注生成实例级 mask/中间数据
- `scripts/sam3d_aiws52_batch.py`：SAM3D 全量批处理入口，支持 resume、shard、多卡与运行指标记录
- `scripts/sam3d_run_metrics_analysis.py`：汇总并分析 SAM3D 运行统计
- `scripts/sam3d_to_cadrille_e2e.py`：把 SAM3D 网格输出整理为 Cadrille 各模式所需的 mesh-derived 输入，并汇总下游结果
- `scripts/cadrille_full_modalities_4gpu.py`：按 GPU 切分 shard，批量启动 Cadrille `pc/img` 全量运行

### 2.5 后续正式运行时应该用哪个目录

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

下面这些命令是干净的从零建环境路径。

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

但昨天真正验证通过的成功恢复路径并不是完整重建环境，而是**保留已有的** `/home/rxl/anaconda3/envs/sam3d-objects`，然后在这个环境里做 import smoke test 验证。

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

但昨天真正验证通过的是**restore 路径**，不是服务器直连 HuggingFace：因为 `RXL` 无法访问 `huggingface.co`，所以最终是把本地旧备份中的 checkpoint 恢复到 `checkpoints/hf`，并补上本地 `pipeline.yaml`。

## 3.5 补齐本地依赖资源

这一步就是昨天成功跑通时**直接需要且已验证**的部分。

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

昨天 `python demo.py` 成功跑通之前，恢复的就是这些本地资源。

## 3.6 打开 `flash_attn`

在 `RTX A6000` 上，正式运行时要显式指定：

```bash
export ATTN_BACKEND=flash_attn
export SPARSE_ATTN_BACKEND=flash_attn
```

说明：

- 预编译 `flash_attn` wheel 在服务器上不一定兼容
- 当前稳定方案是**在服务器本地源码编译 `flash_attn`**

如果需要重新构建，建议按下面的方式在 `RXL` 上执行：

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

补充说明：

- 如果 `RXL` 直接访问 GitHub 不稳定，可在本地机器先拉源码，再同步到服务器后执行 `pip install --no-build-isolation .`
- `flash_attn` 能 import 成功后，SAM3D 在 `RTX A6000` 上仍需显式设置 `ATTN_BACKEND=flash_attn` 与 `SPARSE_ATTN_BACKEND=flash_attn`，因为仓库默认自动切换逻辑并不会覆盖 A6000

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

- `/ssd1/rxl/zhankaiming/AIWS/scripts/sam3d_aiws52_batch.py`

单个 shard 示例：

```bash
ATTN_BACKEND=flash_attn \
SPARSE_ATTN_BACKEND=flash_attn \
CUDA_VISIBLE_DEVICES=0 \
/home/rxl/anaconda3/envs/sam3d-objects/bin/python -u \
/ssd1/rxl/zhankaiming/AIWS/scripts/sam3d_aiws52_batch.py \
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
SCRIPT=/ssd1/rxl/zhankaiming/AIWS/scripts/sam3d_aiws52_batch.py
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

## 5. Cadrille 环境搭建

## 5.1 仓库与运行入口

`RXL` 上当前使用的 Cadrille 仓库：

- `/ssd1/rxl/zhankaiming/AIWS/repos/cadrille`

当前基线使用的是 `AIWS/repos/cadrille` 这份官方上游仓库，AIWS 特有逻辑则通过 wrapper 脚本承载，而不是直接改写上游代码。

AIWS 侧当前使用的编排脚本：

- `/ssd1/rxl/zhankaiming/AIWS/scripts/sam3d_to_cadrille_e2e.py`
- `/ssd1/rxl/zhankaiming/AIWS/scripts/cadrille_full_modalities_4gpu.py`

当前工作流直接调用的 Cadrille 上游/默认入口主要是：

- `test.py`
- `evaluate.py`
- `convert_cadquery.py`

## 5.2 Cadrille 依赖资源

结合之前的建环境历史，Cadrille 可用环境并不只是把仓库放到服务器上，还需要把模型资源一并按本地路径准备好。

当前建议在 `RXL` 上采用这样的仓库树：

```text
/ssd1/rxl/zhankaiming/AIWS/repos/cadrille/
├── ckpt/
│   ├── Qwen2-VL-2B-Instruct/
│   ├── cadrille_rl/
│   └── cadrille_sft/
├── data/
│   └── ...
├── test.py
├── evaluate.py
└── convert_cadquery.py
```

之前实际用过的关键资源包括：

- `Qwen2-VL-2B-Instruct`
- Cadrille RL 权重：`ckpt/cadrille_rl`
- Cadrille SFT 权重：`ckpt/cadrille_sft`
- 可选评估数据：`data/` 下的测试集资源

对当前 AIWS 工作流来说，关键点是：**在正式推理前，把 Cadrille 依赖资源全部提前落到本地仓库树里**。

## 5.3 历史建环境路径，按当前工作区改写

之前的 Cadrille 环境是通过“在另一台网络更稳定的机器准备好，再拷到服务器”的方式搭起来的。这段历史仍然有参考价值，但现在应该改写成面向当前 AIWS 工作区的表述。

当前建议改写为：

1. 在网络稳定的机器上先准备 Docker 镜像：

```bash
docker build -t cadrille:latest .
```

2. 如果服务器侧不适合直接 build / pull，则导出并传到服务器：

```bash
docker save -o cadrille_linux_amd64.tar cadrille:latest
scp cadrille_linux_amd64.tar rxl@<host>:/tmp/
```

3. 在网络稳定的机器上下载所需模型资源，并按当前仓库结构整理到：

- `ckpt/Qwen2-VL-2B-Instruct`
- `ckpt/cadrille_rl`
- `ckpt/cadrille_sft`
- 如有需要，再放入 `data/` 下的数据资源

4. 再把这些资源复制到当前工作区：

- `/ssd1/rxl/zhankaiming/AIWS/repos/cadrille/ckpt/`
- `/ssd1/rxl/zhankaiming/AIWS/repos/cadrille/data/`

这和之前那份历史记录的操作思路是一致的，只是当前激活路径已经变成“官方上游仓库 + AIWS wrapper 脚本”，而不是旧的改动版仓库。

## 5.4 RXL 上当前验证通过的运行方式

当前稳定跑通的生产路径里，**Cadrille 是与 SAM3D 并列的独立模块**，不是一个顺手接上的附属后处理步骤。

在 `RXL` 上，当前验证通过的稳定方式是：

- 使用 `--cadrille-runtime docker` 跑 Cadrille
- Docker 镜像使用 `cadrille:latest`
- 保持**一个 shard 严格绑定一张 GPU**
- IMG 模式额外加 `--ipc=host --shm-size=16g`
- 运行时优先使用本地已准备好的模型资源，而不是临时在线下载

如果 Docker 镜像是 tar 包传过去的，可在 `RXL` 上执行：

```bash
docker load -i /tmp/cadrille_linux_amd64.tar
```

## 5.5 AIWS 内部的数据输入输出约定

在 AIWS 离线管线里，Cadrille 接收的是 SAM3D 重建出来的网格，但在文档结构上应被视为有自己输入准备和输出结果的并列模块。

- Cadrille 输入：重建网格
- PC 模式：从网格采样点云
- IMG 模式：从网格渲染 4 视图 RGB 图像
- 中间输出：`tmp_py / tmp_mesh / tmp_brep`
- 评估筛选后的输出：`selected_py / selected_mesh / selected_brep`

---

## 6. Cadrille 推理方法

## 6.1 单次已准备 split 的快速验证

如果只想直接验证 Cadrille 本身是否能运行，可以进入 Cadrille 运行环境后，对一个已经准备好的 split 直接运行 `test.py`。

当前直接入口主要是：

- `test.py`
- `convert_cadquery.py`
- `evaluate.py`

这种方式适合：

- 单个 split 的快速验证
- 检查 Cadrille 运行环境是否正常
- 在大规模批量推理前先验证 PC 或 IMG 模式

## 6.2 AIWS 数据集批量推理（单个模态）

当前 AIWS 正式脚本：

- `/ssd1/rxl/zhankaiming/AIWS/scripts/cadrille_full_modalities_4gpu.py`

PC 示例：

```bash
OUT=/ssd1/rxl/zhankaiming/AIWS/outputs/cadrille-pc-only-$(date +%Y%m%d-%H%M%S)
/home/rxl/anaconda3/envs/sam3d-objects/bin/python \
/ssd1/rxl/zhankaiming/AIWS/scripts/cadrille_full_modalities_4gpu.py \
  --output-root "$OUT" \
  --sam3d-output-root /ssd1/rxl/zhankaiming/AIWS/outputs/sam3d-aiws52-clean-mesh-stl-20260410-193527 \
  --modalities pc \
  --split-prefix sam3d_bridge_pcfull \
  --gpus 0 \
  --pc-n-samples 5 \
  --cadrille-batch-size 64 \
  --selection-mode evaluate \
  --cadrille-runtime docker \
  --cadrille-docker-image cadrille:latest \
  --export-brep
```

IMG 示例：

```bash
OUT=/ssd1/rxl/zhankaiming/AIWS/outputs/cadrille-img-only-$(date +%Y%m%d-%H%M%S)
/home/rxl/anaconda3/envs/sam3d-objects/bin/python \
/ssd1/rxl/zhankaiming/AIWS/scripts/cadrille_full_modalities_4gpu.py \
  --output-root "$OUT" \
  --sam3d-output-root /ssd1/rxl/zhankaiming/AIWS/outputs/sam3d-aiws52-clean-mesh-stl-20260410-193527 \
  --modalities img \
  --split-prefix sam3d_bridge_imgfix \
  --gpus 0 \
  --img-n-samples 1 \
  --cadrille-batch-size 32 \
  --selection-mode evaluate \
  --cadrille-runtime docker \
  --cadrille-docker-image cadrille:latest \
  --cadrille-docker-extra-args '--ipc=host --shm-size=16g' \
  --export-brep
```

## 6.3 AIWS 数据集批量推理（4 卡并行）

全量批量推理时，建议保持一个 shard 对应一张 GPU。

PC 示例：

```bash
OUT=/ssd1/rxl/zhankaiming/AIWS/outputs/cadrille-pc-only-$(date +%Y%m%d-%H%M%S)
/home/rxl/anaconda3/envs/sam3d-objects/bin/python \
/ssd1/rxl/zhankaiming/AIWS/scripts/cadrille_full_modalities_4gpu.py \
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

IMG 示例：

```bash
OUT=/ssd1/rxl/zhankaiming/AIWS/outputs/cadrille-img-only-$(date +%Y%m%d-%H%M%S)
/home/rxl/anaconda3/envs/sam3d-objects/bin/python \
/ssd1/rxl/zhankaiming/AIWS/scripts/cadrille_full_modalities_4gpu.py \
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

## 6.4 监控与重跑

Cadrille 这部分最重要的运行检查点是：

- 一个 shard 是否真的只对应一张 GPU
- IMG 模式是否保留了 `--ipc=host --shm-size=16g`
- 失败的 shard 是否按原参数重跑
- PC 和 IMG 是否继续分开处理

实际操作上：

- 保持原来的 output-root 结构
- 只重跑失败的 shard 或失败的模态
- 除非明确需要，否则不要把 PC 和 IMG 的恢复混在同一次重启里

---

## 7. SAM3D-Cadrille 端到端桥接与编排

## 7.1 单次端到端脚本

脚本：

- `/ssd1/rxl/zhankaiming/AIWS/scripts/sam3d_to_cadrille_e2e.py`

这个脚本可以：

1. 直接复用已有 SAM3D 输出（`--skip-sam3d`）
2. 把 STL 归一化为 Cadrille 可接受的输入
3. 跑 Cadrille
4. 导出 `tmp_py / tmp_mesh / tmp_brep`
5. 用 `evaluate` 选出 `selected_py / selected_mesh / selected_brep`

---

## 8. 推荐的后续运维策略

## 8.1 不再建议一次性把所有阶段混在一起跑

更稳妥的顺序是：

1. **先完成 SAM3D 全量 STL 导出**
2. **再单独跑 Cadrille-PC**
3. **最后单独跑 Cadrille-IMG**

原因：

- 三个阶段的资源瓶颈不同
- PC 更偏显存与 GPU pinning
- IMG 更偏 DataLoader / shm
- 解耦后，失败更容易恢复

## 8.2 建议保留的最小统计口径

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

## 9. 一页版执行建议

如果后续要继续推理，建议直接照下面执行：

1. **环境不动**，继续使用 `/home/rxl/anaconda3/envs/sam3d-objects`
2. **SAM3D 继续沿用当前 4 shard 脚本**
3. **Cadrille-PC 用 `batch=64` 单独跑**
4. **Cadrille-IMG 用 `batch=32 + --ipc=host + --shm-size=16g` 单独跑**
5. **每次新配置先做 100 样本 smoke test，再上全量**

---

## 10. 结论

这套 SOP 的意义是：

- **SAM3D 环境已经不是问题**，它已经进入“可复用、可批量、可断点恢复”的阶段。
- **后续重点不在能否推理，而在如何更稳地调度 Cadrille 不同模态。**

也就是说，后续继续做推理时，真正要管的是：

- GPU pinning 是否严格
- IMG 的 shm 是否足够
- 是否先小规模 smoke test 再发全量
