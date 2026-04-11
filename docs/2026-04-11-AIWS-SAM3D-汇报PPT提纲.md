# 2026-04-11 AIWS × SAM3D × Cadrille 汇报 PPT 提纲

**作者**: 詹铠铭  
**用途**: 口头汇报或组会展示  
**建议页数**: 6 到 8 页  
**建议风格**: 一页一个结论，少放代码，多放数字和结构图

---

## Slide 1. 工作目标与本次产出

### 标题建议
AIWS 焊接数据上的单图 3D 重建与 CAD 生成实验进展

### 这一页要讲什么
- 本次工作的目标是把 AIWS5.2 数据集接入 **SAM3D → Cadrille** 的完整链路。
- 输出目标包括：
  1. SAM3D 全量 3D 重建
  2. STL 结果可继续进入 Cadrille
  3. 统计运行时间、显存和硬件约束
  4. 固化后续可复现 SOP

### 建议只留 3 个 bullet
- 数据集清洗与结构化完成
- SAM3D 全量跑通，1418 / 1418
- Cadrille 下游链路验证完成

### 口头补充
“这一轮不是只做单次跑通，而是要把整个实验路径沉淀成今后可以复用的正式基线。”

---

## Slide 2. 当前数据结构与正式实验口径

### 标题建议
当前数据结构与正式实验口径

### 可以直接放这张结构图

```text
aiws5.2-usable/
├── V1/                               无深度
├── V2/                               带 PNG 深度
├── NEW/                              带 EXR 深度
├── metadata/                         样本清单与统计
└── misc/                             多实例、多类别、无标注等特殊样本

每个 subset 下再按 5 类工件分组，工件目录中通常包含：
- images/
- annotations/
- depth_png/ 或 depth_exr/
```

### 这一页要讲的核心点
- 后续实验统一基于 `aiws5.2-usable` 结构
- `isat_annotations/` 是标注真值来源
- `misc/` 单独存放不进入主实验主干的样本
- 主实验主干使用的是其中更干净的单实例部分

### 可放数字
- 正式样本总数：**1418**
- `V1`: 593
- `V2`: 524
- `NEW`: 301
- 多实例样本：23
- 无标注样本：1
- 从当前统计看，`misc/` 主要是 `multi_instance` 样本

### 这一页还要交代的当前数据状态
- `V1` 无深度，是当前工件类型最完整的子集
- `V2` 全部带 PNG 深度，当前主实验样本均为 `cover_plate`
- `NEW` 全部带 EXR 深度，当前主实验样本均为 `cover_plate`
- `channel_steel` 当前在主实验视图中没有实例

### 口头补充
“这里直接展示当前正式实验使用的数据结构，主实验主干使用的是其中的单实例部分。”

---

## Slide 3. 项目结构

### 标题建议
项目结构：AIWS 在线/离线划分与离线 CAD 管线

### 建议内容
- AIWS 仓库：`https://github.com/Hyrsta/AIWS`
- 在线管线, 用于现场焊接阶段：
  - 作用：针对真实工件完成识别、位姿估计与基于 CAD 的对齐，支撑后续定位与焊接路径规划
  - 模型栈：`YOLOv11-seg + GenPose++ + FoundationPose`
  - `YOLOv11-seg`：识别与分割
  - `GenPose++`：粗尺寸 / 粗位姿估计
  - `FoundationPose`：基于 CAD 模型的精对齐
- 离线管线, 用于部署前阶段：
  - 作用：构建 CAD 模型库
  - 模型栈：`SAM3D + Cadrille`
  - `SAM3D`：离线 RGB 图像 → 网格重建
  - `Cadrille`：从重建得到的网格出发, PC 模式从网格采样点云，IMG 模式从网格渲染 4 视图 RGB 图像，随后再做 CAD 重建
- 官方仓库，SAM3D：`/ssd1/rxl/zhankaiming/AIWS/repos/sam-3d-objects`
  - 当前直接使用的默认入口：`demo.py`、`notebook/inference.py`、`checkpoints/hf/pipeline.yaml`
- 官方仓库，Cadrille：`/ssd1/rxl/zhankaiming/AIWS/repos/cadrille`
  - 当前直接使用的默认脚本：`test.py`、`evaluate.py`、`convert_cadquery.py`
- 围绕官方 Cadrille 运行的 AIWS wrapper 脚本：
  - `scripts/cadrille_test_wrapper.py`
  - `scripts/cadrille_evaluate.py`
  - `scripts/cadrille_convert_cadquery.py`
- AIWS 离线部分新增脚本：
  - `scripts/build_aiws52_usable_view.py`
  - `scripts/generate_aiws52_instance_masks.py`
  - `scripts/sam3d_aiws52_batch.py`
  - `scripts/sam3d_run_metrics_analysis.py`
  - `scripts/sam3d_to_cadrille_e2e.py`
  - `scripts/cadrille_full_modalities_4gpu.py`

### 要强调的点
- 先讲清楚在线和离线各自是干什么的，再讲每一部分用了什么模型
- 这次汇报讲的是离线 CAD 管线，但它的产出会服务在线视觉管线中的后续对齐与定位

### 口头补充
“后续要复现实验，不能只记住上游模型仓库，还要明确这是 AIWS 的 offline pipeline。”

---

## Slide 4. 运行环境与硬件条件

### 标题建议
实验硬件与运行环境基线

### 建议内容
- 服务器：`RXL`
- GPU：`4 × NVIDIA RTX A6000`
- 单卡显存：`48 GB`
- 系统内存：`503 GiB`
- CPU：`2 × Intel Xeon Gold 6326`
- SAM3D 环境：`/home/rxl/anaconda3/envs/sam3d-objects`
- SAM3D repo：`/ssd1/rxl/zhankaiming/AIWS/repos/sam-3d-objects`

### 要强调的点
- `flash_attn` 已进入正式运行路径
- checkpoints、MoGe、DINO cache 已补齐
- 环境已从“能跑”进入“可复现”状态

### 口头补充
“目前 SAM3D 环境不是主要风险点，关键问题已经从环境搭建转向运行调度与资源控制。”

---

## Slide 5. SAM3D 全量实验结果

### 标题建议
SAM3D 全量 3D 重建结果

### 建议表格

| 指标 | 数值 |
|---|---:|
| 样本数 | 1418 |
| 成功数 | 1418 |
| 失败数 | 0 |
| 完成率 | 100% |
| 总墙钟时间 | 约 1.60 小时 |
| 平均单样本时间 | 14.799 s |
| P95 单样本时间 | 18.977 s |
| 显存 reserved 最大值 | 28076 MB |

### 要讲的结论
- 全量正式样本全部完成
- 在 4×A6000 上具有稳定性
- 峰值 reserved memory 约 28.1 GB，低于 48 GB 上限
- 说明当前 SAM3D 配置可持续复用

### 可加一句分析
- 长尾主要来自 `V1 / bellmouth`

### 口头补充
“从结果上看，SAM3D 这一段已经不是‘能不能跑’的问题，而是‘后续要不要进一步优化长尾类别’的问题。”

---

## Slide 6. Cadrille 下游实验与资源瓶颈

### 标题建议
SAM3D 输出进入 Cadrille 的验证结果

### 可以拆成两栏

#### 左边，PC 模态
- 1418 / 1418 可选中最佳候选
- 成功率 100%
- 墙钟时间约 80.1 分钟
- 说明 STL 输出可以稳定进入 CAD 生成链路

#### 右边，IMG 模态
- 默认大 batch 配置失败
- 失败原因不是 GPU 算力本身，而是 **共享内存 shm 不足**
- 调整后：
    - `batch_size=32`
    - `--ipc=host`
    - `--shm-size=16g`
- 成功重试后：1213 个 STL、1211 个 STEP

### 这一页要讲的核心结论
- PC 模态的主要风险是 **GPU pinning 不当导致 OOM**
- IMG 模态的主要风险是 **DataLoader / shm**
- 两个模态的资源瓶颈不同，所以后续不建议混跑

### 口头补充
“这说明后续运维不能把 PC 和 IMG 当成同一种任务，它们的稳定配置不同。”

---

## Slide 7. 后续如何继续跑实验

### 标题建议
后续正式推理 SOP

### 建议流程图

```text
AIWS 数据集
   ↓
aiws5.2-usable
   ↓
SAM3D 全量推理
   ↓ 输出 STL
Cadrille-PC 单独运行
   ↓
Cadrille-IMG 单独运行
```

### 要讲的建议
- 不要直接把 `pc,img` 混在一个 launcher 里一起跑
- 推荐顺序：
  1. SAM3D
  2. Cadrille-PC
  3. Cadrille-IMG
- 新设置先做 `100` 样本 smoke test，再发全量

### 一句话策略
- **SAM3D 与 Cadrille 解耦**
- **PC / IMG 分开调度**
- **先小规模验证，再全量**

---

## Slide 8. 本次工作结论与下一步

### 标题建议
总结与下一步计划

### 可以放 3 个结论
1. 已建立 AIWS5.2 上的正式实验数据口径
2. 已建立 SAM3D 全量稳定运行基线
3. 已验证 SAM3D → Cadrille 的下游可行性

### 下一步建议
- 给 Cadrille 增加显存 instrumentation
- 分析 IMG 未成功选中的样本
- 对 `bellmouth` 长尾样本做专项分析

### 收尾一句
**当前最重要的进展不是某一个模型单独跑通，而是完整实验链路已经具有复现和扩展能力。**

---

## 附：如果你只讲 5 页

可以压缩成：

1. 目标与贡献
2. 数据集整理与正式口径
3. SAM3D 全量结果
4. Cadrille 下游结果与瓶颈
5. 后续推理 SOP 与下一步

---

## 附：汇报时建议强调的三句话

1. **正式实验数据集已经统一整理为 `aiws5.2-usable` 结构。**
2. **SAM3D 已在 1418 个正式样本上实现 100% 全量重建。**
3. **后续难点不再是能不能跑，而是如何按模态更稳地调度和统计资源。**
