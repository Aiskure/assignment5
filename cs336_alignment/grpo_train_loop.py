import argparse
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Literal, Any, Callable

import torch
from tqdm import tqdm

try:
    from vllm import SamplingParams
except ModuleNotFoundError:
    SamplingParams = Any  # type: ignore[assignment]


from cs336_alignment import utils

from cs336_alignment.utils import (
    compute_group_normalized_rewards,
    compute_policy_gradient_loss,
    grpo_microbatch_train_step,
)

# ---- NSCC PBS 兼容：UUID 格式的 CUDA_VISIBLE_DEVICES → 数字 index ----
from cs336_alignment.math_baseline import _normalize_cuda_visible_devices_for_vllm
_normalize_cuda_visible_devices_for_vllm()

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.data_utils import extract_question_and_gt

#system prompt
PROMPT_TEMPLATE ="""A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""

# GRPO 名词对照表：
# - n_grpo_steps: 外层训练轮数（1 轮 = rollout + 用该 rollout 做更新）
# - rollout_batch_size: 每轮生成的 response 总数
# - group_size: 每个 prompt 采样多少条 response
# - n_prompts_per_rollout_batch = rollout_batch_size // group_size
# - train_batch_size: 每轮用于训练的样本数（通常等于 rollout_batch_size）
# - gradient_accumulation_steps: 累积多少个 microbatch 再执行一次 optimizer.step()
# - micro_train_batch_size = train_batch_size // gradient_accumulation_steps
# - n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size
#
# 循环层级：
# for step in range(n_grpo_steps):           # 外层 GRPO step
#     ...
#     for mb_idx in range(n_microbatches_per_rollout_batch):  # 内层 microbatch
#         ... backward ...
#         if (mb_idx + 1) % gradient_accumulation_steps == 0:
#             optimizer.step()
#             optimizer.zero_grad()

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_and_derive_config(
    n_grpo_steps: int,
    rollout_batch_size: int,
    group_size: int,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    train_batch_size: int | None, #我们决定要用多少来训练
    epochs_per_rollout_batch: int,
) -> dict[str, int]:
    # on-policy 默认: train_batch_size = rollout_batch_size
    if train_batch_size is None:
        train_batch_size = rollout_batch_size

    # 1) 基础正数检查
    if n_grpo_steps <= 0:
        raise ValueError("n_grpo_steps must be > 0")
    if rollout_batch_size <= 0:
        raise ValueError("rollout_batch_size must be > 0")
    if group_size <= 0:
        raise ValueError("group_size must be > 0")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be > 0")
    if train_batch_size <= 0:
        raise ValueError("train_batch_size must be > 0")
    if epochs_per_rollout_batch <= 0:
        raise ValueError("epochs_per_rollout_batch must be > 0")

    # 2) 结构约束
    if rollout_batch_size % group_size != 0:
        raise ValueError("rollout_batch_size must be divisible by group_size")
    if train_batch_size % gradient_accumulation_steps != 0:
        raise ValueError("train_batch_size must be divisible by gradient_accumulation_steps")
    if train_batch_size < group_size:
        raise ValueError("train_batch_size must be >= group_size")
    if loss_type not in {"no_baseline", "reinforce_with_baseline", "grpo_clip"}:
        raise ValueError(f"invalid loss_type: {loss_type}")

    # 3) 推导中间量
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    if rollout_batch_size % micro_train_batch_size != 0:
        raise ValueError("rollout_batch_size must be divisible by micro_train_batch_size")
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size

    return {
        "train_batch_size": train_batch_size,
        "micro_train_batch_size": micro_train_batch_size,
        "n_prompts_per_rollout_batch": n_prompts_per_rollout_batch,
        "n_microbatches_per_rollout_batch": n_microbatches_per_rollout_batch,
    }


def _sample_prompt_batch(
    train_dataset,
    n_prompts_per_rollout_batch: int,   #几道题
    group_size: int,                    #每道题采样回答的个数
    rollout_batch_size: int,            #总采样的回答
) -> tuple[list[str], list[str], list[str], list[str]]:
    # 1) sample prompt-level examples
    if len(train_dataset) < n_prompts_per_rollout_batch:
        raise ValueError("train_dataset is smaller than n_prompts_per_rollout_batch")

    sample_indices = random.sample(range(len(train_dataset)), n_prompts_per_rollout_batch)

    prompts: list[str] = []
    ground_truths: list[str] = []
    for idx in sample_indices:
        ex = train_dataset[idx]
        question, gt = extract_question_and_gt(ex)
        prompts.append(PROMPT_TEMPLATE.format(question=question))
        ground_truths.append(gt)

    # 2) expand to rollout-level，同一道题要采样多个答案
    expanded_prompts = [p for p in prompts for _ in range(group_size)]
    repeated_ground_truths = [g for g in ground_truths for _ in range(group_size)]

    # 3) sanity check
    if len(expanded_prompts) != rollout_batch_size:
        raise ValueError("expanded_prompts length mismatch")
    if len(repeated_ground_truths) != rollout_batch_size:
        raise ValueError("repeated_ground_truths length mismatch")

    return prompts, ground_truths, expanded_prompts, repeated_ground_truths


#Phase C:采样response

#1)规范化模型输出格式
def _canonicalize_response(raw_text: str) -> str:
    text = raw_text.strip()#去掉字符串首尾的空白
    if not text.startswith("<think>"):#是否以<think>开头
        text = "<think>" + text
    text = re.sub(r"</think>\s*<answer>", "</think> <answer>", text)
    return text

#把训练的模型权重更新到vllm里
def _sync_policy_to_vllm(policy: torch.nn.Module, vllm_model) -> None:
    state_dict = policy.state_dict()
    llm_model = vllm_model.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def _generate_rollouts(
    policy: torch.nn.Module,
    vllm_model,
    expanded_prompts: list[str],
    sampling_params,
    expected_batch_size: int,
) -> list[str]:
    _sync_policy_to_vllm(policy, vllm_model)#同步权重
    outputs = vllm_model.generate(expanded_prompts, sampling_params)#生成回复
    rollout_responses = [_canonicalize_response(o.outputs[0].text) for o in outputs]#规范化输出

    if len(rollout_responses) !=expected_batch_size:
        raise ValueError("rollout_responses length mismatch")
    return rollout_responses


def _build_sampling_params(
    temperature: float,
    top_p: float,
    min_tokens: int,
    max_tokens: int,
):
    # 训练 rollout 和 validation rollout 共用同一种 stop 规则：
    # 生成到第二个答案标签 `</answer>` 就结束。
    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        stop=["</answer>"],
    )
    params.include_stop_str_in_output = True
    return params


def _evaluate_on_validation(
    policy: torch.nn.Module,
    vllm_model,
    val_dataset,
    sampling_params,
    reward_fn: Callable,
    max_eval_samples: int,
) -> tuple[dict[str, float], list[dict]]:
    # Validation 只做"当前策略有多好"的评估，不参与梯度更新。
    if val_dataset is None or len(val_dataset) == 0:
        return {}, []

    # 为了让不同 step 的验证结果可比较，这里固定取验证集前 N 条，
    # 而不是每次随机抽样。
    n_eval_examples = len(val_dataset) if max_eval_samples <= 0 else min(len(val_dataset), max_eval_samples)
    eval_examples = [val_dataset[idx] for idx in range(n_eval_examples)]

    prompts: list[str] = []
    ground_truths: list[str] = []
    for ex in eval_examples:
        question, gt = extract_question_and_gt(ex)
        prompts.append(PROMPT_TEMPLATE.format(question=question))
        ground_truths.append(gt)

    # 这里复用训练时的 vLLM 实例，只是切换成 validation prompt 做生成。
    rollout_responses = _generate_rollouts(
        policy=policy,
        vllm_model=vllm_model,
        expanded_prompts=prompts,
        sampling_params=sampling_params,
        expected_batch_size=n_eval_examples,
    )

    reward_total = 0.0
    format_total = 0.0
    answer_total = 0.0
    type1 = 0
    type2 = 0
    type3 = 0
    examples: list[dict] = []

    # reward_fn 返回的字典里通常同时包含总 reward、格式 reward、答案 reward。
    # 这里把它们分别聚合，方便后续画 validation 曲线或做错误分析。
    for i, (response, gt) in enumerate(zip(rollout_responses, ground_truths)):
        result = reward_fn(response, gt)
        reward_value = float(result.get("reward", 0.0))
        format_value = float(result.get("format_reward", 0.0))
        answer_value = float(result.get("answer_reward", reward_value))

        reward_total += reward_value
        format_total += format_value
        answer_total += answer_value

        if format_value == 1.0 and answer_value == 1.0:
            type1 += 1
        elif format_value == 1.0 and answer_value == 0.0:
            type2 += 1
        else:
            type3 += 1

        # 保存前几条样例，用于 deliverable 中展示 rollout 质量变化
        if i < 5:
            examples.append({
                "prompt": prompts[i],
                "response": response,
                "ground_truth": gt,
                "reward": reward_value,
                "format_reward": format_value,
            })

    metrics = {
        "val/reward": reward_total / n_eval_examples,
        "val/format_reward": format_total / n_eval_examples,
        "val/answer_reward": answer_total / n_eval_examples,
        "val/type1": float(type1),
        "val/type2": float(type2),
        "val/type3": float(type3),
        "val/n_examples": float(n_eval_examples),
    }
    return metrics, examples





def grpo_train_loop(
    policy: torch.nn.Module,
    tokenizer,
    train_dataset,
    val_dataset,
    optimizer: torch.optim.Optimizer,
    n_grpo_steps: int,
    rollout_batch_size: int,
    group_size: int,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    model_id: str,
    loss_aggregation: Literal["masked_mean", "masked_normalize"] = "masked_normalize",
    train_batch_size: int | None = None,
    epochs_per_rollout_batch: int = 1,
    seed: int | None = None,
    vllm_device: str = "cuda:1",
    gpu_memory_utilization: float = 0.85,
    rollout_temperature: float = 1.0,
    rollout_top_p: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1024,
    advantage_eps: float = 1e-6,
    use_std_normalization: bool = True,
    reward_fn: Callable = r1_zero_reward_fn,
    eval_every_steps: int = 10,
    max_eval_samples: int = 1024,
    eval_temperature: float = 1.0,
    eval_top_p: float = 1.0,
    eval_max_tokens: int = 1024,
    output_dir: str | None = None,
) -> tuple[list[dict[str, float]], list[dict]]:
    if seed is not None:
        set_global_seed(seed)
    if eval_every_steps < 0:
        raise ValueError("eval_every_steps must be >= 0")

    # 阶段 A：参数与静态检查 + 推导中间量
    config = _validate_and_derive_config(
        n_grpo_steps=n_grpo_steps,
        rollout_batch_size=rollout_batch_size,
        group_size=group_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        loss_type=loss_type,
        train_batch_size=train_batch_size,
        epochs_per_rollout_batch=epochs_per_rollout_batch,
    )
    micro_train_batch_size = config["micro_train_batch_size"]
    n_microbatches_per_rollout_batch = config["n_microbatches_per_rollout_batch"]
    n_prompts_per_rollout_batch = config["n_prompts_per_rollout_batch"]

    # 阶段 C：rollout 生成初始化（只做一次）
    vllm_model = utils.init_vllm(
        model_id=model_id,
        device=vllm_device,
        seed=seed if seed is not None else 42,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    sampling_params = _build_sampling_params(
        temperature=rollout_temperature,
        top_p=rollout_top_p,
        min_tokens=sampling_min_tokens,
        max_tokens=sampling_max_tokens,
    )
    eval_sampling_params = _build_sampling_params(
        temperature=eval_temperature,
        top_p=eval_top_p,
        min_tokens=sampling_min_tokens,
        max_tokens=eval_max_tokens,
    )

    # metrics_history 是后续画图的基础数据结构：
    # step 0 放 baseline，后面每个训练 step 放 train metrics，
    # 有 validation 时再把 val metrics 合并进来。
    metrics_history: list[dict[str, float]] = []
    # rollout_examples_history 记录不同 step 的样例输出，用于观察模型生成质量的变化。
    rollout_examples_history: list[dict] = []

    if eval_every_steps > 0 and val_dataset is not None and len(val_dataset) > 0:
        # 训练开始前先跑一次 step 0 基线，后面才能看出 validation reward 是否真的上升。
        base_eval_metrics, base_examples = _evaluate_on_validation(
            policy=policy,
            vllm_model=vllm_model,
            val_dataset=val_dataset,
            sampling_params=eval_sampling_params,
            reward_fn=reward_fn,
            max_eval_samples=max_eval_samples,
        )
        base_metrics = {"grpo/step": 0.0, **base_eval_metrics}
        metrics_history.append(base_metrics)
        rollout_examples_history.append({
            "step": 0,
            "source": "validation",
            "examples": base_examples,
        })
        print(
            f"[GRPO step 0] val_reward={base_eval_metrics['val/reward']:.4f} "
            f"format={base_eval_metrics['val/format_reward']:.4f}"
        )

    # 阶段 B + C：先构造 rollout 输入，再生成 rollout responses
    for step in tqdm(range(n_grpo_steps), desc="GRPO steps"):
        prompts, ground_truths, expanded_prompts, repeated_ground_truths = _sample_prompt_batch(
            train_dataset=train_dataset,
            n_prompts_per_rollout_batch=n_prompts_per_rollout_batch,
            group_size=group_size,
            rollout_batch_size=rollout_batch_size,
        )

        rollout_responses = _generate_rollouts(
            policy=policy,
            vllm_model=vllm_model,
            expanded_prompts=expanded_prompts,
            sampling_params=sampling_params,
            expected_batch_size=rollout_batch_size,
        )

        # 阶段 D：奖励与优势计算（内联）
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization,
        )

        if advantages.shape[0] != rollout_batch_size:
            raise ValueError("advantages length mismatch")
        if raw_rewards.shape[0] != rollout_batch_size:
            raise ValueError("raw_rewards length mismatch")
        if not torch.isfinite(advantages).all():
            raise ValueError("advantages contains NaN/Inf")
        if not torch.isfinite(raw_rewards).all():
            raise ValueError("raw_rewards contains NaN/Inf")

        # 这些累加器用于把一个 GRPO step 内多个 epoch / microbatch 的训练统计汇总成一步。
        step_loss_total = 0.0
        step_loss_count = 0
        step_entropy_total = 0.0
        step_entropy_count = 0
        step_clip_fraction_total = 0.0
        step_clip_fraction_count = 0
        last_grad_norm = 0.0

        #阶段 E:把rollout转换为训练张量
        train_device = next(policy.parameters()).device#从policy在的那个设备上

        #去掉冲洗的<think>
        output_strs = [
            (r[len("<think>"):].lstrip() if r.startswith("<think>") else r)
            for r in rollout_responses
        ]
        #得到input_ids,label,response_mask
        batch = utils.tokenize_prompt_and_output(expanded_prompts,output_strs,tokenizer)
        input_ids = batch["input_ids"].to(train_device)
        labels = batch["labels"].to(train_device)
        response_mask = batch["response_mask"].to(train_device)
        raw_rewards = raw_rewards.to(train_device)
        advantages = advantages.to(train_device)

        # 阶段 F: old_log_probs
        # grpo_clip 模式下，每个 rollout batch 只缓存一次 old log-probs（不参与梯度）。
        # 用 microbatch 循环分批计算，避免全批次 forward 导致 OOM。
        if loss_type == "grpo_clip":
            old_log_probs_list = []
            with torch.no_grad():
                for mb_idx in range(n_microbatches_per_rollout_batch):
                    s = mb_idx * micro_train_batch_size
                    e = s + micro_train_batch_size
                    mb_out = utils.get_response_log_probs(
                        model=policy,
                        input_ids=input_ids[s:e],
                        labels=labels[s:e],
                        return_token_entropy=False,
                    )
                    old_log_probs_list.append(mb_out["log_probs"].detach())
            old_log_probs = torch.cat(old_log_probs_list, dim=0)
        else:
            old_log_probs = None

        # 阶段 G:同一批 rollout 可以重复训练多个 epoch
        # 关键：forward pass 在 microbatch 循环内部执行，每个 microbatch 有独立的计算图，
        # 避免 "Trying to backward through the graph a second time" 错误。
        for epoch_idx in range(epochs_per_rollout_batch):
            optimizer.zero_grad(set_to_none=True)

            for mb_idx in tqdm(
                range(n_microbatches_per_rollout_batch),
                desc=f"  step {step+1} epoch {epoch_idx+1} microbatch",
                leave=False,
            ):
                s = mb_idx * micro_train_batch_size
                e = s + micro_train_batch_size

                # 每个 microbatch 独立做 forward，产生独立计算图
                mb_out = utils.get_response_log_probs(
                    model=policy,
                    input_ids=input_ids[s:e],
                    labels=labels[s:e],
                    return_token_entropy=True,
                )
                mb_policy_log_probs = mb_out["log_probs"]
                mb_token_entropy = mb_out["token_entropy"]

                mb_response_mask = response_mask[s:e]
                mb_raw_rewards = raw_rewards[s:e]
                mb_advantages = advantages[s:e]
                mb_old_log_probs = old_log_probs[s:e] if old_log_probs is not None else None

                loss , loss_metadata = grpo_microbatch_train_step(
                    mb_policy_log_probs,
                    mb_response_mask,
                    gradient_accumulation_steps,
                    loss_type,
                    mb_raw_rewards,
                    mb_advantages,
                    mb_old_log_probs,
                    cliprange=0.2,
                    loss_aggregation=loss_aggregation,
                )
                step_loss_total += float(loss.item())
                step_loss_count += 1
                if mb_token_entropy is not None:
                    # 只统计 response token 的熵，prompt 部分不纳入训练监控。
                    response_entropy = utils.masked_mean(mb_token_entropy, mb_response_mask, dim=None)
                    step_entropy_total += float(response_entropy.detach().item())
                    step_entropy_count += 1
                if "clipped_mask" in loss_metadata:
                    # 只在 response token 上统计 clip fraction，避免 prompt/padding 稀释
                    clipped_mask = loss_metadata["clipped_mask"]
                    masked_clip = (clipped_mask.float() * mb_response_mask.float()).sum()
                    n_response_tokens = mb_response_mask.float().sum().clamp(min=1.0)
                    step_clip_fraction_total += float((masked_clip / n_response_tokens).item())
                    step_clip_fraction_count += 1

                if (mb_idx + 1) % gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    last_grad_norm = float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            # 尾部保护：若最后一组 microbatches 不满 gradient_accumulation_steps，也要执行一次 step
            if n_microbatches_per_rollout_batch % gradient_accumulation_steps != 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                last_grad_norm = float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # 把当前 step 的训练侧统计压成一条记录，validation 指标若存在再追加进去。
        step_metrics = {
            "grpo/step": float(step + 1),
            "train/loss": (step_loss_total / step_loss_count) if step_loss_count else 0.0,
            "train/reward_mean": float(reward_metadata["reward_mean"]),
            "train/reward_std": float(reward_metadata["reward_std"]),
            "train/response_entropy": (step_entropy_total / step_entropy_count) if step_entropy_count else 0.0,
            "train/grad_norm": last_grad_norm,
        }
        if step_clip_fraction_count:
            step_metrics["train/clip_fraction"] = step_clip_fraction_total / step_clip_fraction_count

        # Validation 不是每个 step 都跑：按 eval_every_steps 周期执行，
        # 但最后一个 step 无论如何都补一次，保证训练结束时有最终验证结果。
        should_run_validation = (
            eval_every_steps > 0
            and val_dataset is not None
            and len(val_dataset) > 0
            and ((step + 1) % eval_every_steps == 0 or step == n_grpo_steps - 1)
        )
        if should_run_validation:
            val_metrics, val_examples = _evaluate_on_validation(
                policy=policy,
                vllm_model=vllm_model,
                val_dataset=val_dataset,
                sampling_params=eval_sampling_params,
                reward_fn=reward_fn,
                max_eval_samples=max_eval_samples,
            )
            step_metrics.update(val_metrics)
            # 保存 validation rollout 样例
            rollout_examples_history.append({
                "step": step + 1,
                "source": "validation",
                "examples": val_examples,
            })
            # 同时保存当前 step 的几条训练 rollout 样例
            n_train_ex = min(3, n_prompts_per_rollout_batch)
            train_examples = []
            for i in range(n_train_ex):
                train_examples.append({
                    "prompt": prompts[i],
                    "response": rollout_responses[i * group_size],
                    "ground_truth": ground_truths[i],
                    "reward": str(float(raw_rewards[i * group_size].item())),
                })
            rollout_examples_history.append({
                "step": step + 1,
                "source": "train",
                "examples": train_examples,
            })
            print(
                f"[GRPO step {step + 1}] loss={step_metrics['train/loss']:.4f} "
                f"reward={step_metrics['train/reward_mean']:.4f} "
                f"val_reward={val_metrics['val/reward']:.4f} "
                f"val_format={val_metrics['val/format_reward']:.4f}"
            )
        else:
            print(
                f"[GRPO step {step + 1}] loss={step_metrics['train/loss']:.4f} "
                f"reward={step_metrics['train/reward_mean']:.4f}"
            )

        metrics_history.append(step_metrics)

        # 增量落盘：每个 step 都覆写，防止长时间训练中断后丢数据
        if output_dir is not None:
            _save_dir = Path(output_dir)
            _save_dir.mkdir(parents=True, exist_ok=True)
            with open(_save_dir / "metrics_history.json", "w", encoding="utf-8") as f:
                json.dump(metrics_history, f, indent=2, ensure_ascii=False)
            with open(_save_dir / "rollout_examples.json", "w", encoding="utf-8") as f:
                json.dump(rollout_examples_history, f, indent=2, ensure_ascii=False)

    return metrics_history, rollout_examples_history


# ============================================================
# CLI 入口
# ============================================================

def _resolve_model_path() -> str:
    env_model_id = os.environ.get("ASSIGNMENT5_MODEL_ID") or os.environ.get("MODEL_ID")
    if env_model_id:
        return env_model_id
    candidates = [
        "/scratch/users/nus/e1553316/assignment5/models/Qwen2.5-Math-1.5B",
        "/root/autodl-tmp/models/Qwen2.5-Math-1.5B",
        "models/Qwen2.5-Math-1.5B",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return "models/Qwen2.5-Math-1.5B"


def _load_jsonl_dataset(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO train loop on MATH dataset")
    parser.add_argument("--model-path", type=str, default=_resolve_model_path())
    parser.add_argument("--train-path", type=str, default="data/math/train.jsonl")
    parser.add_argument("--validation-path", type=str, default="data/math/validation.jsonl")
    parser.add_argument("--output-dir", type=str, default="grpo_runs")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-grpo-steps", type=int, default=256)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=128)
    parser.add_argument("--loss-type", type=str, default="grpo_clip",
                        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"])
    parser.add_argument("--loss-aggregation", type=str, default="masked_normalize",
                        choices=["masked_mean", "masked_normalize"])
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--epochs-per-rollout-batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)

    parser.add_argument("--vllm-device", type=str, default="cuda:1")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-top-p", type=float, default=1.0)
    parser.add_argument("--sampling-min-tokens", type=int, default=4)
    parser.add_argument("--sampling-max-tokens", type=int, default=1024)

    parser.add_argument("--advantage-eps", type=float, default=1e-6)
    parser.add_argument("--use-std-normalization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-every-steps", type=int, default=10)
    parser.add_argument("--max-eval-samples", type=int, default=1024)
    parser.add_argument("--eval-temperature", type=float, default=1.0)
    parser.add_argument("--eval-top-p", type=float, default=1.0)
    parser.add_argument("--eval-max-tokens", type=int, default=1024)

    # W&B（不传 --wandb-project 则不启用）
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default="grpo")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-offline", action="store_true", default=True,
                        help="使用 offline 模式（默认），跑完后用 wandb sync 上传")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.getLogger("vllm").setLevel(logging.WARNING)
    os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 W&B（可选）
    _wandb = None
    if args.wandb_project:
        try:
            import wandb
            if args.wandb_offline:
                os.environ["WANDB_MODE"] = "offline"
            _wandb = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
                dir=str(output_dir),
            )
            print(f"[INFO] W&B initialized: project={args.wandb_project} offline={args.wandb_offline}")
        except ImportError:
            print("[WARN] wandb not installed, skipping W&B logging")

    # 保存运行配置
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # 加载数据
    train_dataset = _load_jsonl_dataset(args.train_path)
    val_dataset = _load_jsonl_dataset(args.validation_path)
    print(f"[INFO] train: {len(train_dataset)} examples from {args.train_path}")
    print(f"[INFO] val: {len(val_dataset)} examples from {args.validation_path}")

    # 模型 + tokenizer + optimizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"[INFO] loading model from {args.model_path}")
    model_path = Path(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    policy = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, local_files_only=True).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    # 训练
    metrics_history, rollout_examples_history = grpo_train_loop(
        policy=policy,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        n_grpo_steps=args.n_grpo_steps,
        rollout_batch_size=args.rollout_batch_size,
        group_size=args.group_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        loss_type=args.loss_type,
        model_id=args.model_path,
        loss_aggregation=args.loss_aggregation,
        train_batch_size=args.train_batch_size,
        epochs_per_rollout_batch=args.epochs_per_rollout_batch,
        seed=args.seed,
        vllm_device=args.vllm_device,
        gpu_memory_utilization=args.gpu_memory_utilization,
        rollout_temperature=args.rollout_temperature,
        rollout_top_p=args.rollout_top_p,
        sampling_min_tokens=args.sampling_min_tokens,
        sampling_max_tokens=args.sampling_max_tokens,
        advantage_eps=args.advantage_eps,
        use_std_normalization=args.use_std_normalization,
        eval_every_steps=args.eval_every_steps,
        max_eval_samples=args.max_eval_samples,
        eval_temperature=args.eval_temperature,
        eval_top_p=args.eval_top_p,
        eval_max_tokens=args.eval_max_tokens,
        output_dir=str(output_dir),
    )

    # 落盘
    with open(output_dir / "metrics_history.json", "w", encoding="utf-8") as f:
        json.dump(metrics_history, f, indent=2, ensure_ascii=False)
    with open(output_dir / "rollout_examples.json", "w", encoding="utf-8") as f:
        json.dump(rollout_examples_history, f, indent=2, ensure_ascii=False)

    # W&B: 上传全量 metrics（每个 step 一条 log）
    if _wandb is not None:
        for m in metrics_history:
            _wandb.log(m, step=int(m.get("grpo/step", 0)))
        _wandb.finish()
        if args.wandb_offline:
            print(f"[INFO] W&B offline run saved. To upload: wandb sync {output_dir}/wandb/")

    # 保存最终 checkpoint
    ckpt_dir = output_dir / "final_checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    if metrics_history:
        last = metrics_history[-1]
        print(f"[DONE] final val/reward={last.get('val/reward', 'N/A')}")
    print(f"[DONE] outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
