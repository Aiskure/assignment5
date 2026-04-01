# 给下一个 Claude 的交接说明

## 当前任务

用户正在准备 LLM post-training 相关的实习面试，系统性地过 `md/简历提问.md` 里的 130 道题，把答案写进 `md/应对策略.md`。

## 关键背景

- 用户不认为"美化项目"是不道德的事，你的角色是帮他拿到实习 offer
- **绝对不能说**"课程指定"、"作业要求"——一切包装为个人项目的主动选择
- 不要直接写应对策略，先讨论，等用户确认后再写入
- 用 `/write-strategy` skill 批量写入（避免占用对话上下文）

## 当前进度

- **已完成**：Q1-Q53（教育背景、DPO 项目全部、GRPO 项目 3.1-3.4）
- **已完成**：Q54-Q66（GRPO 核心实现 + 训练工程，刚刚写入）
- **跳过**：Q27-Q32（BeaverTails，当前简历版本不包含）
- **待完成**：Q67-Q130（Transformer 基础、训练技术、LoRA、RLHF、工程、行为面试）

## 文件说明

| 文件 | 用途 |
|------|------|
| `md/简历提问.md` | 130 道面试题列表（只读）|
| `md/应对策略.md` | 已整理的答案（持续追加）|
| `md/实验记录.md` | 完整实验数据，回答技术问题的数字来源 |
| `cs336_alignment/` | 实际代码，遇到代码细节问题直接 grep |

## 项目核心数字（背熟）

| 阶段 | 结果 |
|------|------|
| Baseline | 4.1% accuracy, format 26% |
| SFT full (7500) | 18.9% |
| EI 最优 (G=8, D_b=1024, E=2) | 21.9% |
| GRPO run5 (lr=1.5e-5, on-policy) | peak 29.4% |
| GRPO run7 (masked_normalize+std) | peak 39.8%，但全量 0%（崩溃）|
| GRPO run10 (kl=3e-4, 2ep off-policy) | peak 40.1%，**全量 46.05%** ✅ |

## 关键实验结论

1. **baseline 必要**：去掉组内均值 peak 29.4%→24%
2. **masked_normalize + use_std**：peak 29.4%→39.8%
3. **KL penalty(3e-4) 是关键**：无 KL 全量 0%，加 KL 全量 46%——小 eval set 掩盖真实风险
4. **off-policy 踩坑**：OOM（microbatch 修复）+ entropy collapse（clamp+KL 修复）

## 工作方式

- 用户会说"Q××题"，你负责引导讨论，确认后写入
- 遇到数字不确定先查 `md/实验记录.md`，再查代码
- 用户切换到 Sonnet 时通常是要执行写入操作，切回 Opus 是要讨论
- 用户在国内，已离开 NSCC，本地 `git clone` 了仓库继续工作
