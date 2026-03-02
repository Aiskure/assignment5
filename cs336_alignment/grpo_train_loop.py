import torch
from tqdm import tqdm
from typing import Literal, Any, Callable
import random

import re

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

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.data_utils import extract_question_and_gt

#结构性prompt
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
    train_batch_size: int | None,
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
    n_prompts_per_rollout_batch: int,
    group_size: int,
    rollout_batch_size: int,
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

    # 2) expand to rollout-level
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
):
    if seed is not None:
        set_global_seed(seed)

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
    sampling_params = SamplingParams(
        temperature=rollout_temperature,
        top_p=rollout_top_p,
        min_tokens=sampling_min_tokens,
        max_tokens=sampling_max_tokens,
        stop=["</answer>"],
    )
    sampling_params.include_stop_str_in_output = True

    # 阶段 B + C：先构造 rollout 输入，再生成 rollout responses
    for step in range(n_grpo_steps):
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

        #阶段 E:把rollout转换为训练张量
        train_device = next(policy.parameters()).device#从policy在的那个设备上

        output_strs = [
            (r[len("<think>"):].lstrip() if r.startswith("<think>") else r)
            for r in rollout_responses
        ]
        batch = utils.tokenize_prompt_and_output(expanded_prompts,output_strs,tokenizer)
        model_out = utils.get_response_log_probs(
            model=policy,
            input_ids=batch["input_ids"].to(train_device),
            labels=batch["labels"].to(train_device),
            return_token_entropy=True,
        )

        policy_log_probs = model_out["log_probs"]

        # 阶段 F: old_log_probs
        # grpo_clip 模式下，每个 rollout batch 只缓存一次 old log-probs（不参与梯度）。
        if loss_type == "grpo_clip":
            old_log_probs = policy_log_probs.detach()
            if old_log_probs.shape != policy_log_probs.shape:
                raise ValueError("old_log_probs shape mismatch with policy_log_probs")
            if old_log_probs.requires_grad:
                raise ValueError("old_log_probs must be detached (requires_grad=False)")
        else:
            old_log_probs = None

        response_mask = batch["response_mask"].to(train_device)
        token_entropy = model_out["token_entropy"]

        # 阶段 G:优化步骤

        optimizer.zero_grad(set_to_none=True)

        for mb_idx in range(n_microbatches_per_rollout_batch):
            s = mb_idx * micro_train_batch_size
            e = s + micro_train_batch_size

            mb_policy_log_probs = policy_log_probs[s:e]
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
            )

            if (mb_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        
        # 尾部保护：若最后一组 microbatches 不满 gradient_accumulation_steps，也要执行一次 step
        if n_microbatches_per_rollout_batch % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)       
