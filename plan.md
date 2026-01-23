# Stanford CS336 Assignment 5：Alignment 训练计划

## 阶段 0：环境准备与基础知识回顾

### 0.1 环境搭建

```bash
# 克隆仓库
git clone https://github.com/stanford-cs336/assignment5-alignment

# 安装依赖
uv sync --no-install-package flash-attn
uv sync

# 验证安装（所有测试应该失败：NotImplementedError）
uv run pytest
```

### 0.2 前置知识检查清单

- PyTorch 基础（Tensor 操作、autograd、优化器）
- Transformer 架构理解
- 语言模型基础（next-token prediction、cross-entropy loss）
- HuggingFace Transformers 基本使用
- 强化学习基础概念（policy、reward、trajectory）

### 0.3 阅读材料

| 资源                     | 链接                     | 重要性 |
|--------------------------|--------------------------|--------|
| Spinning Up in Deep RL   | OpenAI 教程              | 必读   |
| RLHF Book                | rlhfbook.com             | 推荐   |
| DeepSeek R1 论文         | arXiv:2501.12948         | 推荐   |
| GRPO 原论文 (DeepSeekMath) | arXiv:2402.03300       | 推荐   |

---

## 阶段 1：理解评估与数据（第1-2天）

### 1.1 探索 MATH 数据集

```python
import json

with open('/data/a5-alignment/MATH/validation.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if i < 5:
            example = json.loads(line)
            print(example.keys())
            print(example['problem'][:200])
            print(example['solution'][:200])
            print('---')
```

### 1.2 理解 Prompt 格式

阅读 `cs336_alignment/prompts/r1_zero.prompt`：

```
A conversation between User and Assistant...
User: {question}
Assistant: <think>
```

**关键理解**：

- 模型需要在 `<think>` 标签内生成推理过程
- 最终答案在 `<answer>` 标签内
- 生成在 `</answer>` 时停止

### 1.3 实现 Zero-shot 评估脚本（Problem: math_baseline）

任务分解：

- [ ] 加载 MATH validation 数据
- [ ] 使用 r1_zero prompt 格式化问题
- [ ] 使用 vLLM 生成回答
- [ ] 使用 r1_zero_reward_fn 评估
- [ ] 保存结果到磁盘

代码框架：

```python
from vllm import LLM, SamplingParams

def evaluate_vllm(vllm_model, reward_fn, prompts, ground_truths, sampling_params):
    """评估模型在 MATH 上的表现"""
    outputs = vllm_model.generate(prompts, sampling_params)
    
    results = []
    for output, gt in zip(outputs, ground_truths):
        response = output.outputs[0].text
        reward_info = reward_fn(response, gt)
        results.append({
            'response': response,
            'ground_truth': gt,
            **reward_info
        })
    
    # 计算统计信息
    accuracy = sum(r['reward'] for r in results) / len(results)
    return results, accuracy
```

**检查点**：

- 能成功加载 Qwen 2.5 Math 1.5B
- 能生成 5K 个回答
- 能解析 format_reward 和 answer_reward
- 记录 baseline 准确率（预期：较低）

---

## 阶段 2：SFT 核心实现（第3-5天）

### 2.1 实现辅助函数（按顺序实现 + 测试）

| 序号 | 函数名                          | 分值 | 测试命令                              |
|------|----------------------------------|------|---------------------------------------|
| 2.1.1| `tokenize_prompt_and_output`     | 2    | `uv run pytest -k test_tokenize_prompt_and_output` |
| 2.1.2| `compute_entropy`                | 1    | `uv run pytest -k test_compute_entropy`            |
| 2.1.3| `get_response_log_probs`         | 2    | `uv run pytest -k test_get_response_log_probs`     |
| 2.1.4| `masked_normalize`               | 1    | `uv run pytest -k test_masked_normalize`           |
| 2.1.5| `sft_microbatch_train_step`      | 3    | `uv run pytest -k test_sft_microbatch_train_step`  |

### 2.2 实现完整 SFT 训练循环

**推荐超参数**：

- learning_rate = 2e-5
- batch_size = 32 (effective)
- micro_batch_size = 2
- gradient_accumulation_steps = 16
- max_seq_length = 1024
- num_epochs = 1
- gradient_clip = 1.0

### 2.3 运行 SFT 实验（Problem: sft_experiment）

| 数据量        | 预期验证准确率 |
|---------------|----------------|
| 128 examples  | ~5%            |
| 256 examples  | ~8%            |
| 512 examples  | ~10%           |
| 1024 examples | ~12%           |
| Full dataset  | >15%           |

**检查点**：

- 训练 loss 稳定下降
- 验证准确率随数据量增加而提升
- Full dataset 达到 >15% 准确率
- 保存最佳模型 checkpoint

---

## 阶段 3：Expert Iteration（第6天）

### 3.1 理解算法

1. 从当前 policy 采样多个回答
2. 用 reward function 过滤出正确的回答
3. 在正确回答上做 SFT
4. 重复

### 3.3 运行实验（Problem: expert_iteration_experiment）

实验变量：

- G (rollouts per question): {4, 8, 16}
- epochs_per_step: {1, 2}
- batch_size: {512, 1024, 2048}
- n_ei_steps: 5

**检查点**：

- 验证准确率 >15%
- 观察熵随训练变化
- 对比 SFT 性能

---

## 阶段 4：Policy Gradient 理论（第7天）

核心概念：

1. 语言模型作为 Policy
2. REINFORCE Policy Gradient
3. Baseline 减少方差
4. Off-policy Importance Sampling
5. GRPO 特有：Group-normalized Advantage
6. GRPO-Clip Objective

---

## 阶段 5：GRPO 实现（第8-10天）

### 5.1 实现 GRPO 组件（按顺序 + 测试）

| 序号 | 函数名                              | 分值 | 测试命令                                      |
|------|--------------------------------------|------|-----------------------------------------------|
| 5.1.1| `compute_group_normalized_rewards`   | 2    | `uv run pytest -k test_compute_group_normalized_rewards` |
| 5.1.2| `compute_naive_policy_gradient_loss` | 1    | `uv run pytest -k test_compute_naive_policy_gradient_loss` |
| 5.1.3| `compute_grpo_clip_loss`             | 2    | `uv run pytest -k test_compute_grpo_clip_loss` |
| 5.1.4| `compute_policy_gradient_loss`       | 1    | `uv run pytest -k test_compute_policy_gradient_loss` |
| 5.1.5| `masked_mean`                        | 1    | `uv run pytest -k test_masked_mean`           |
| 5.1.6| `grpo_microbatch_train_step`         | 3    | `uv run pytest -k test_grpo_microbatch_train_step` |

### 5.3 验证实现（Problem: grpo_train_loop）

**Sanity Checks**：

- 训练 reward 上升
- 验证 reward 上升
- 生成的回答质量提升
- 没有 NaN 或梯度爆炸

---

## 阶段 6：GRPO 实验（第11-14天）

6.1 Learning Rate 调优（Problem: grpo_learning_rate）  
目标：验证准确率 >25%

6.2 Baseline 消融（no_baseline vs reinforce_with_baseline）

6.3 Length Normalization 消融（masked_mean vs masked_normalize）

6.4 Std Normalization 消融（normalize_by_std=True/False）

6.5 Off-policy 实验（Problem: grpo_off_policy_sweep）

| epochs_per_rollout | train_batch_size | 是否使用 clip     |
|--------------------|------------------|-------------------|
| 1                  | 256              | No (on-policy)    |
| 2                  | 128              | Yes               |
| 4                  | 64               | Yes               |

6.6 Prompt 消融（r1_zero vs question_only）

---

## 阶段 7：排行榜优化（第15-16天）

**优化方向**：

1. 超参数优化（lr, batch size, group size）
2. 系统优化（torch.compile、混合精度、更高效 GPU 利用）
3. 算法优化（数据过滤、课程学习、自定义 reward）

**约束**：

- 4小时训练时间（2×H100）
- 必须使用 Qwen 2.5 Math 1.5B Base
- 验证使用 r1_zero prompt + r1_zero_reward_fn
- 在完整 5K 验证集上评估

---

## 阶段 8：可选补充作业（额外 3-5 天）

- 8.1 Zero-shot 基线（MMLU / GSM8K / AlpacaEval / SimpleSafetyTests）
- 8.2 Instruction Fine-Tuning
- 8.3 DPO 实现（HH 数据集）

---

## 总时间估计

| 阶段 | 内容             | 预计时间 | GPU 时间       |
|------|------------------|----------|----------------|
| 0    | 环境与基础       | 0.5 天   | -              |
| 1    | 评估与数据       | 1.5 天   | 1 H100 hr      |
| 2    | SFT 实现         | 3 天     | 2 H100 hrs     |
| 3    | Expert Iteration | 1 天     | 6 H100 hrs     |
| 4    | 理论学习         | 1 天     | -              |
| 5    | GRPO 实现        | 3 天     | 2 H100 hrs     |
| 6    | GRPO 实验     | 4 天     | ~30 H100 hrs   |
| 7    | 排行榜           | 2 天     | 16 H100 hrs    |
| **总计** |           | **16 天** | **~57 H100 hrs** |

