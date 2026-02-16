from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel

import torch
from torch import Tensor 
import math

from unittest.mock import patch
from vllm import LLM
from vllm.model_executor import set_random_seed as vllm_set_random_seed



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
    return (tensor * mask).sum(dim=dim)/normalize_constant

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

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    启动推理进程，使用 vLLM 在单独的 GPU 上加载模型
    """
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
 