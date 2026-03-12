from __future__ import annotations

from typing import Any, TYPE_CHECKING

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel

import torch
from torch import Tensor 
import math

from unittest.mock import patch

if TYPE_CHECKING:
    try:
        from vllm import LLM
    except ModuleNotFoundError:
        LLM = Any  # type: ignore[assignment]



def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """
    Tokenize prompt/output strings, then build shifted labels and response mask.

    处理流程:
    1) 分别 tokenize prompt 和 output
    2) 对每条样本拼接成 full_ids = prompt_ids + output_ids
    3) 先把 full_ids pad 到 batch 最大长度（这个顺序与测试快照一致）
    4) 构造:
       - input_ids = full_ids[:-1]
       - labels = full_ids[1:]
       - response_mask: 仅标记 labels 中属于 output 的位置

    Args:
        prompt_strs: 每条样本的 prompt 文本列表。
        output_strs: 每条样本的 output 文本列表。
        tokenizer: HuggingFace tokenizer（需有 pad_token_id 或 eos_token_id）。

    Returns:
        dict[str, Tensor]:
            - "input_ids": (batch_size, max_len - 1), dtype=torch.long
            - "labels": (batch_size, max_len - 1), dtype=torch.long
            - "response_mask": (batch_size, max_len - 1), dtype=torch.bool
    """
    if len(prompt_strs) != len(output_strs):
        raise ValueError("prompt_strs and output_strs must have the same length.")

    # 分别编码（不加 padding，后面手动对齐到 batch 内最大长度）
    prompt_tokenized = tokenizer(prompt_strs)
    output_tokenized = tokenizer(output_strs)

    prompt_ids_batch = prompt_tokenized["input_ids"]
    output_ids_batch = output_tokenized["input_ids"]

    # pad token 优先使用 pad_token_id；若不存在则回退 eos_token_id
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id.")

    # 先拼接每条样本的 prompt + output，并记录长度用于构造 mask
    full_ids_batch = []
    prompt_lens = []
    output_lens = []
    for prompt_ids, output_ids in zip(prompt_ids_batch, output_ids_batch):
        full_ids = prompt_ids + output_ids
        full_ids_batch.append(full_ids)
        prompt_lens.append(len(prompt_ids))
        output_lens.append(len(output_ids))

    max_full_len = max(len(ids) for ids in full_ids_batch)
    target_len = max_full_len - 1

    input_ids_list = []
    labels_list = []
    response_mask_list = []

    for full_ids, prompt_len, output_len in zip(full_ids_batch, prompt_lens, output_lens):
        # 先 pad full_ids，再做 shift（与快照行为保持一致）
        pad_len = max_full_len - len(full_ids)
        if pad_len > 0:
            full_ids = full_ids + [pad_token_id] * pad_len

        input_ids = full_ids[:-1]
        labels = full_ids[1:]

        # labels 中 output 的起点对应 prompt_len - 1
        # output 实际长度是 output_len，所以只标记这段范围为 1
        response_mask = [0] * target_len
        response_start = max(prompt_len - 1, 0)
        response_end = min(response_start + output_len, target_len)
        for i in range(response_start, response_end):
            response_mask[i] = 1

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        response_mask_list.append(response_mask)

    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.long),
        "response_mask": torch.tensor(response_mask_list, dtype=torch.bool),
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    input:
        logits: (batch_size, seq_len, vocab_size)
    return:
        torch.Tensor Shape (batch_size, sequence_length)
    """
    with torch.no_grad():
        log_prob = torch.nn.functional.log_softmax(logits,dim=-1)
        prob = torch.exp(log_prob)
    return -(torch.sum(prob * log_prob,dim=-1)) 


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
        Args:
        model: PreTrainedModel HuggingFace model used for scoring (placed on the correct device
        and in inference mode if gradients should not be computed).

        inputs:
        input_ids: torch.Tensor shape (batch_size, sequence_length), concatenated prompt +
        response tokens as produced by your tokenization method.
        labels: torch.Tensor shape (batch_size, sequence_length), labels as produced by your
        tokenization method.
        return_token_entropy: bool If True, also return per-token entropy by calling
        compute_entropy.

        Returns:
        dict[str, torch.Tensor].

        "log_probs" shape (batch_size, sequence_length), conditional log-probabilities
        log pθ(xt | x<t).
        "token_entropy" optional, shape (batch_size, sequence_length), per-token entropy
        for each position (present only if return_token_entropy=True).
    """
    logits =  model(input_ids).logits
    log_probs = torch.nn.functional.log_softmax(logits,dim=-1)
    log_probs = log_probs.gather(dim=-1,index=labels.unsqueeze(-1)).squeeze(-1)

    entropy = compute_entropy(logits)

    if return_token_entropy:
        return {
            "log_probs": log_probs,# batch_size, seq_len
            "token_entropy": compute_entropy(logits)
        }
    else:
        return {
            "log_probs": log_probs
        }

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    # 1. 安全检查：形状对齐
    if tensor.shape != mask.shape:
        raise ValueError(f"Shape mismatch: tensor {tensor.shape} != mask {mask.shape}")
    
    # 2. 类型对齐：将 mask 转为与 tensor 相同的数据类型（比如 bfloat16），防止类型冲突
    if mask.dtype != tensor.dtype:
        mask = mask.to(tensor.dtype)
        
    # 3. 核心计算
    masked_tensor = tensor * mask
    
    # torch.sum 在 dim=None 时默认会对所有维度求和，所以不需要额外写 if dim is None
    return masked_tensor.sum(dim=dim) / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # 对整个 microbatch 的 response token log-probs 求和，
    # 再除以 batch_size * normalize_constant。
    batch_size = policy_log_probs.shape[0]
    loss = -masked_normalize(
        tensor=policy_log_probs,
        mask=response_mask,
        normalize_constant=float(batch_size) * float(normalize_constant),
        dim=None,
    )
    loss = loss / gradient_accumulation_steps
    loss.backward()
    return loss, {}



def init_vllm(
    model_id: str,
    device: str,
    seed: int,
    gpu_memory_utilization: float = 0.85,
) -> LLM:
    """
    启动推理进程，使用 vLLM 在单独的 GPU 上加载模型
    """
    try:
        from vllm import LLM
        from vllm.model_executor import set_random_seed as vllm_set_random_seed
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "vllm is required for init_vllm(). Install vllm before calling this function."
        ) from exc

    vllm_set_random_seed(seed)  # 设置随机种子确保可重复性
    
    # Monkeypatch from TRL: 打补丁解决兼容性问题
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    # 作用：告诉分布式系统 world_size=1（只有1个GPU），避免多卡通信问题
    
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    # 作用：跳过 vLLM 的内存检查，因为在训练场景下这个检查不适用
    
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,           # 模型名称，如 "Qwen/Qwen2.5-1.5B"
            device=device,            # 指定 GPU，如 "cuda:1"
            dtype=torch.bfloat16,      # 使用 bfloat16 节省显存
            enable_prefix_caching=True, # 启用前缀缓存加速生成
            gpu_memory_utilization=gpu_memory_utilization, # GPU内存使用率，默认0.85
        )
 


#===============================进入grpo===================================
def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
    filter_zero_std_groups: bool = False,
):
    """
    为每个 rollout 响应计算原始奖励，并在组内进行归一化。

    每个问题会生成 `group_size` 个回答（一个 group）。
    本函数会：
        1. 使用 reward_fn 为每个回答计算原始奖励；
        2. 按 group 划分（同一问题的回答属于同一组）；
        3. 在组内减去组均值（baseline）；
        4. 若 normalize_by_std=True，则再除以组标准差（加上 advantage_eps 防止除零）；
        5. 返回归一化后的优势值、原始奖励以及一些统计信息。

    参数：
        reward_fn (Callable[[str, str], dict[str, float]]):
            奖励函数。输入为 (模型回答, 对应 ground truth)，
            返回一个字典，至少包含键 "reward"。

        rollout_responses (List[str]):
            模型生成的回答列表。
            长度必须为：
                rollout_batch_size = n_prompts_per_rollout_batch * group_size。
            并且排列顺序为：每连续 `group_size` 个元素对应同一个问题。

        repeated_ground_truths (List[str]):
            与 rollout_responses 等长的 ground truth 列表。
            每个问题的 ground truth 会重复 `group_size` 次，
            与对应的回答逐一匹配。

        group_size (int):
            每个问题对应的回答数量（即组大小）。

        advantage_eps (float):
            一个很小的常数，用于在标准化时避免除以 0：
                std + advantage_eps。

        normalize_by_std (bool):
            若为 True：
                执行 (reward - group_mean) / (group_std + advantage_eps)
            若为 False：
                仅执行 (reward - group_mean)

    返回：
        advantages (torch.Tensor):
            形状为 (rollout_batch_size,) 的张量，
            表示组归一化后的奖励（优势值）。

        raw_rewards (torch.Tensor):
            形状为 (rollout_batch_size,) 的张量，
            表示未归一化的原始奖励。

        metadata (dict[str, float]):
            可选统计信息，例如：
                - 奖励均值
                - 奖励标准差
                - 最小/最大值
                - 组统计信息等
    """
    #safty_cheak
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError("rollout_responses and repeated_ground_truths must have the same length.")
    if group_size <= 0:
        raise ValueError("group_size must be > 0.")
    if len(rollout_responses) % group_size != 0:
        raise ValueError("rollout batch size must be divisible by group_size.")

    raw_reward_list: list[float] = []
    for response, gt in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(response, gt)
        raw_reward_list.append(float(reward_dict["reward"]))

    raw_rewards = torch.tensor(raw_reward_list, dtype=torch.float32)

    # 按组重排，组内做 baseline/归一化。
    num_groups = raw_rewards.numel() // group_size
    grouped = raw_rewards.view(num_groups, group_size)
    group_mean = grouped.mean(dim=1, keepdim=True)
    group_std = grouped.std(dim=1, keepdim=True, unbiased=True)

    if normalize_by_std:
        normalized_rewards = (grouped - group_mean) / (group_std + advantage_eps)
    else:
        normalized_rewards = grouped - group_mean

    # 过滤全0或全1的 group（std ≈ 0，无梯度信号）
    if filter_zero_std_groups:
        zero_std_mask = (group_std.squeeze(1) < advantage_eps)  # (num_groups,)
        zero_std_expanded = zero_std_mask.unsqueeze(1).expand_as(normalized_rewards)
        normalized_rewards = normalized_rewards.masked_fill(zero_std_expanded, 0.0)
        n_filtered = int(zero_std_mask.sum().item())
    else:
        n_filtered = 0

    normalized_rewards = normalized_rewards.reshape(-1)
    metadata = {
        "reward_mean": float(raw_rewards.mean().item()),
        "reward_std": float(raw_rewards.std(unbiased=False).item()),
        "reward_min": float(raw_rewards.min().item()),
        "reward_max": float(raw_rewards.max().item()),
        "num_groups": float(num_groups),
        "group_size": float(group_size),
        "filtered_groups": float(n_filtered),
    }
    return normalized_rewards, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    计算每个 token 的朴素策略梯度损失（naive policy gradient loss）。

    根据公式：
        loss_{i,t} = - A_i * log_prob_{i,t}

    其中 A_i 是每条 rollout 的标量 reward 或 advantage，
    log_prob_{i,t} 是该 rollout 在第 t 个 token 上的对数概率。

    参数：
        raw_rewards_or_advantages (torch.Tensor):
            形状 (batch_size, 1) 或 (batch_size,)。
            每条 rollout 对应一个标量 reward 或已经计算好的 advantage。

        policy_log_probs (torch.Tensor):
            形状 (batch_size, sequence_length)。
            每条 rollout 中每个 token 的 log-prob。

    返回：
        torch.Tensor:
            形状 (batch_size, sequence_length)。
            每个 token 的策略梯度损失（尚未在 batch 或序列维度上聚合）。
    """
    # 接受 (B,) 或 (B,1)，统一到 (B,1)
    if policy_log_probs.ndim != 2:
        raise ValueError("policy_log_probs must have shape (B, T).")
    if raw_rewards_or_advantages.ndim == 1:
        raw_rewards_or_advantages = raw_rewards_or_advantages.unsqueeze(1)
    elif raw_rewards_or_advantages.ndim == 2 and raw_rewards_or_advantages.shape[1] == 1:
        pass
    else:
        raise ValueError(
            "raw_rewards_or_advantages must have shape (B,) or (B,1)."
        )

    if raw_rewards_or_advantages.shape[0] != policy_log_probs.shape[0]:
        raise ValueError("batch size mismatch.")

    if not torch.isfinite(raw_rewards_or_advantages).all():
        raise ValueError("raw_rewards_or_advantages contains NaN/Inf.")
    if not torch.isfinite(policy_log_probs).all():
        raise ValueError("policy_log_probs contains NaN/Inf.")

    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    计算每个 token 的 GRPO-Clip 损失。

    根据公式：

        r_t = exp(policy_log_probs - old_log_probs)

        loss_{i,t} = - min(
            r_t * A_i,
            clip(r_t, 1 - cliprange, 1 + cliprange) * A_i
        )

    其中 A_i 是每条 rollout 的标量优势值（advantage）。

    参数：
        advantages (torch.Tensor):
            形状 (batch_size, 1) 或 (batch_size,)。
            每条 rollout 的优势值 A。

        policy_log_probs (torch.Tensor):
            形状 (batch_size, sequence_length)。
            当前策略在每个 token 上的 log-prob。

        old_log_probs (torch.Tensor):
            形状 (batch_size, sequence_length)。
            旧策略在每个 token 上的 log-prob。

        cliprange (float):
            裁剪范围 ε，例如 0.2。
            ratio 会被限制在 [1 - ε, 1 + ε] 之间。

    返回：
        loss (torch.Tensor):
            形状 (batch_size, sequence_length)。
            每个 token 的裁剪策略梯度损失。

        metadata (dict[str, torch.Tensor]):
            可选统计信息，例如：
                - ratio
                - clipped_ratio
                - 是否发生裁剪的布尔 mask
                - clip 比例
    """
        # ====== sanity checks ======
    if policy_log_probs.ndim != 2:
        raise ValueError("policy_log_probs must have shape (B, T).")

    if old_log_probs.ndim != 2:
        raise ValueError("old_log_probs must have shape (B, T).")

    if policy_log_probs.shape != old_log_probs.shape:
        raise ValueError("policy_log_probs and old_log_probs must have same shape.")

    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(1)
    elif advantages.ndim == 2 and advantages.shape[1] == 1:
        pass
    else:
        raise ValueError("advantages must have shape (B,) or (B,1).")

    if advantages.shape[0] != policy_log_probs.shape[0]:
        raise ValueError("batch size mismatch.")

    if cliprange <= 0:
        raise ValueError("cliprange must be positive.")

    if not torch.isfinite(policy_log_probs).all():
        raise ValueError("policy_log_probs contains NaN/Inf.")

    if not torch.isfinite(old_log_probs).all():
        raise ValueError("old_log_probs contains NaN/Inf.")

    if not torch.isfinite(advantages).all():
        raise ValueError("advantages contains NaN/Inf.")
    log_ratio = (policy_log_probs - old_log_probs).clamp(-20, 20)
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio,1-cliprange,1+cliprange)

    loss = -torch.minimum(advantages * ratio,advantages * clipped_ratio)

    # ====== metadata ======
    clipped_mask = (ratio != clipped_ratio)
    clip_fraction = clipped_mask.float().mean()

    metadata = {
        "ratio": ratio.detach(),
        "clipped_ratio": clipped_ratio.detach(),
        "clip_fraction": clip_fraction.detach(),
        "clipped_mask": clipped_mask.detach(),
    }

    return loss, metadata

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    #sanity check
    valid_types = {"no_baseline", "reinforce_with_baseline", "grpo_clip"}
    if loss_type not in valid_types:
        raise ValueError(f"Invalid loss_type: {loss_type}")

    if loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError("raw_rewards required for no_baseline.")
        return compute_naive_policy_gradient_loss(raw_rewards,policy_log_probs),{}

    elif loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError("advantages required for reinforce_with_baseline.")
        return compute_naive_policy_gradient_loss(advantages,policy_log_probs),{}
    elif loss_type == "grpo_clip":
        if advantages is None:
            raise ValueError("advantages required for grpo_clip.")
        if old_log_probs is None:
            raise ValueError("old_log_probs required for grpo_clip.")
        if cliprange is None:
            raise ValueError("cliprange required for grpo_clip.")
        loss,metadata = compute_grpo_clip_loss(advantages,policy_log_probs,old_log_probs,cliprange)
       
        return loss,metadata

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """
    我们只希望计算response部分的loss，所以只对这部分进行计算平均
    """
    if tensor.shape != mask.shape:
        raise ValueError("tensor and mask must have the same shape.")

    # Convert mask to float for multiplication
    if mask.dtype != torch.float32 and mask.dtype != torch.float64:
        mask = mask.float()

    masked_tensor = tensor * mask

    if dim is None:
        total = masked_tensor.sum()
        count = mask.sum()
        return total / count

    total = masked_tensor.sum(dim=dim)
    count = mask.sum(dim=dim)

    return total / count

    
def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_aggregation: Literal["masked_mean", "masked_normalize"] = "masked_normalize",
    ref_log_probs: torch.Tensor | None = None,
    kl_coef: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    #sanity check
    if policy_log_probs.ndim !=2:
        raise ValueError("policy_log_probs must have shape (B, T).")
    if response_mask.shape != policy_log_probs.shape:
        raise ValueError("response_mask must have the same shape as policy_log_probs.")
    if not isinstance(gradient_accumulation_steps, int) or gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be a positive integer.")
    if not torch.isfinite(policy_log_probs).all().item():
        raise ValueError("policy_log_probs contains NaN/Inf.")

    # 统一 mask 类型
    if response_mask.dtype != torch.bool:
        response_mask = response_mask.bool()

    # 避免 masked_mean 出现除零 NaN（训练时更好定位问题）
    response_token_counts = response_mask.sum(dim=1)
    if (response_token_counts == 0).any().item():
        raise ValueError("each sample must have at least one response token.")
    

    per_token_loss,loss_metadata = compute_policy_gradient_loss(policy_log_probs,loss_type,raw_rewards,advantages,old_log_probs,cliprange)

    if per_token_loss.shape != policy_log_probs.shape:
        raise ValueError("per_token_loss shape mismatch with policy_log_probs.")

    # 3) aggregate + backward
    if loss_aggregation == "masked_normalize":
        max_gen_len = float(response_mask.shape[1])
        masked_loss = masked_normalize(per_token_loss, response_mask, normalize_constant=max_gen_len, dim=1)  # (B,)
    else:
        masked_loss = masked_mean(per_token_loss, response_mask, dim=1)  # (B,)
    if not torch.isfinite(masked_loss).all().item():
        raise ValueError("masked_loss contains NaN/Inf.")  
    
    loss = masked_loss.mean()

    # KL-in-loss: low_var_kl (k3 estimator) = exp(log_ref - log_θ) - (log_ref - log_θ) - 1
    if ref_log_probs is not None and kl_coef > 0.0:
        log_diff = (ref_log_probs - policy_log_probs).clamp(-20, 20)
        low_var_kl = torch.exp(log_diff) - log_diff - 1  # (B, T), always >= 0
        kl_loss = masked_mean(low_var_kl, response_mask, dim=None)
        loss = loss + kl_coef * kl_loss
        loss_metadata["kl_loss"] = kl_loss.detach()

    loss = loss / gradient_accumulation_steps
    if not torch.isfinite(loss).item():
        raise ValueError("final loss is NaN/Inf.")

    loss.backward()

    metadata = dict(loss_metadata)
    metadata["mean_response_tokens"] = response_token_counts.float().mean().detach()
    metadata["mean_masked_loss"] = masked_loss.mean().detach()
    return loss.detach(), metadata


