[English](README.md) | 中文

# LLM 对齐：SFT → Expert Iteration → GRPO

在 **Qwen 2.5 Math 1.5B** 上实现 LLM 对齐技术，用于数学推理。从 4.1% 的基线（1024 条验证子集）出发，通过 GRPO 训练的迭代改进，在同一子集上达到 **40.1%**，全量 5000 条验证集上达到 **46.05%**。

基于 [Stanford CS336 Spring 2025 Assignment 5](https://github.com/stanford-cs336/spring2025-assignment5-alignment)。

## 实验结果

| 阶段 | 方法 | 准确率 |
|------|------|--------|
| 基线 | 预训练模型（无训练） | 4.1% |
| SFT | 监督微调（7500 条） | 18.9% |
| Expert Iteration | 5 步 EI（G=8, D_b=1024, E=2） | 21.9% |
| **GRPO** | **GRPO-clip + KL + masked_normalize** | **46.05%** |

SFT、Expert Iteration、GRPO 三个阶段各自独立地从预训练基础模型出发训练，不是串行流水线。模型太小（1.5B），先 SFT 再 GRPO 效果反而不好——直接在基础模型上 GRPO 效果更优。

详细实验过程（遇到的问题与解决方案）见 [实验记录.md](实验记录.md)。

## 环境配置

### 前置要求

- Python 3.11 或 3.12
- [uv](https://github.com/astral-sh/uv) 包管理器
- 支持 CUDA 的 NVIDIA GPU

### 安装依赖

```bash
uv sync --no-install-package flash-attn && uv sync
```

### 下载模型

```bash
mkdir -p models
huggingface-cli download Qwen/Qwen2.5-Math-1.5B --local-dir models/Qwen2.5-Math-1.5B
```

### 设置环境变量

```bash
export ASSIGNMENT5_MODEL_ID=$(pwd)/models/Qwen2.5-Math-1.5B
```

## 使用方法

所有训练脚本在 `scripts/` 目录下，关键超参数可通过环境变量覆盖。

### 评估

在 MATH 验证集上评估任意模型或 checkpoint：

```bash
bash scripts/eval.sh

# 用全量验证集（5000 条）评估训练好的 checkpoint
ASSIGNMENT5_MODEL_ID=outputs/grpo/run10/final_checkpoint \
MAX_EVAL_SAMPLES=0 \
bash scripts/eval.sh
```

### SFT

```bash
# 全量训练（7500 条）
bash scripts/sft.sh

# 子集训练
N=512 RUN_NAME=run_sft_n512 bash scripts/sft.sh
```

### Expert Iteration

```bash
bash scripts/ei.sh

# 自定义配置
ROLLOUTS=8 DB_SIZE=1024 SFT_EPOCHS=2 bash scripts/ei.sh
```

### GRPO（最优结果）

```bash
# 默认配置即为最优 run（run10）：
# grpo_clip, lr=1.5e-5, epochs=2, kl_coef=3e-4,
# masked_normalize, use_std=True, group_filter=True
bash scripts/grpo.sh
```

关键可覆盖参数：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LR` | 1.5e-5 | 学习率 |
| `KL_COEF` | 3e-4 | KL 惩罚系数 |
| `EPOCHS` | 2 | 每个 rollout batch 的 off-policy 训练轮数 |
| `LOSS_TYPE` | grpo_clip | 损失类型（`grpo_clip`、`reinforce_with_baseline`、`no_baseline`） |
| `N_STEPS` | 256 | GRPO 总训练步数 |
| `GROUP_FILTER` | true | 跳过全对/全错的 group |
| `USE_STD` | true | 用组内标准差归一化 advantage |

## 关键发现

1. **masked_normalize >> masked_mean**：loss 除以 `max_gen_len` 而非实际 response 长度，带来了 +10.4pp 的提升。较长的正确推理链贡献更多梯度，隐式鼓励模型生成完整的思维链。

2. **Entropy collapse 是静默的**：没有 KL 惩罚的最优 run（run7）训练时 peak accuracy 达 39.8%，但 final checkpoint 在全量评估时准确率为 **0%**——模型已崩溃到只输出重复的 emoji。1024 子集的 periodic eval 完全没有发现这一点。

3. **Off-policy GRPO 必须有 KL 惩罚**：即使很小的 `kl_coef=3e-4` 就能防止灾难性漂移。过大（1e-3、3e-3）则约束过强，限制学习。

4. **Off-policy 不加 log_ratio clamp = 数值爆炸**：`exp(无界的 log_ratio)` 导致 grad_norm 达到 10¹³。一行 `.clamp(-20, 20)` 即可修复。

## 项目结构

```
├── cs336_alignment/       # 核心实现
│   ├── sft.py             # SFT 训练循环
│   ├── grpo_train_loop.py # GRPO 训练循环
│   ├── expert_iteration_experiment.py
│   ├── utils.py           # 损失函数、奖励计算
│   ├── math_baseline.py   # vLLM 评估
│   └── drgrpo_grader.py   # 数学答案评分
├── scripts/               # 训练与评估脚本
│   ├── sft.sh
│   ├── ei.sh
│   ├── grpo.sh
│   └── eval.sh
├── data/math/             # MATH 数据集（train 7500 / val 5000）
└── 实验记录.md             # 详细实验记录
```
