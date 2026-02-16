try:
    from drgrpo_grader import r1_zero_reward_fn
except:
    from .drgrpo_grader import r1_zero_reward_fn
import json
import logging
import os
import gc
import re
import subprocess
import torch
from typing import List


def _normalize_cuda_visible_devices_for_vllm() -> None:
    """Map PBS/NVIDIA UUID-based CUDA_VISIBLE_DEVICES to index IDs for vLLM."""
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible_devices or "GPU-" not in visible_devices:
        return

    raw_tokens = [token.strip() for token in visible_devices.split(",") if token.strip()]
    if not raw_tokens:
        return

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Failed to map CUDA_VISIBLE_DEVICES UUIDs to indices (%s).", exc
        )
        return

    uuid_to_index = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            uuid_to_index[parts[1]] = parts[0]

    mapped_tokens = []
    for token in raw_tokens:
        if token.isdigit():
            mapped_tokens.append(token)
            continue
        mapped = uuid_to_index.get(token)
        if mapped is None:
            logging.getLogger(__name__).warning(
                "Cannot map CUDA device token '%s'; keep original CUDA_VISIBLE_DEVICES.",
                token,
            )
            return
        mapped_tokens.append(mapped)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(mapped_tokens)


_normalize_cuda_visible_devices_for_vllm()
from vllm import LLM, SamplingParams

# 设置日志级别，减少 VLLM 的输出
logging.getLogger("vllm").setLevel(logging.WARNING)
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: callable,
    prompts: List[str],
    eval_sampling_params
) -> None:
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    res = [output.outputs[0].text for output in outputs]
    return res
    
def evaluate(model_path,llm=None,rl=False,reward_fn=None,prompt=None):
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
    )
    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True
    if llm is None:
        llm = LLM(model=model_path, gpu_memory_utilization=0.8,dtype=torch.float16)
    if reward_fn is None:
        reward_fn = r1_zero_reward_fn
    if prompt is None:
        prompt = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""
    gsm8k = []
    with open("data/gsm8k/test.jsonl") as f:
        lines = f.readlines()
        for line in lines:
            gsm8k.append(json.loads(line))
    prompts = []
    answer = []
    for dict in gsm8k:
        prompts.append(prompt.format(question=dict['question']))
        answer.append(dict['answer'][dict['answer'].find("####") + 5:])
    print(len(prompts))
    outputs = evaluate_vllm(llm, reward_fn, prompts, sampling_params)
    acc = 0
    format_reward = 0
    type1_num = 0
    type2_num = 0
    type3_num = 0
    for i in range(len(outputs)):
        full_response = outputs[i]
        if not full_response.lstrip().startswith("<think>"):
            full_response = "<think>" + full_response

        # r1_zero_reward_fn strictly checks for "</think> <answer>".
        full_response = re.sub(r"</think>\s*<answer>", "</think> <answer>", full_response)

        gsm8k[i]['outputs'] = full_response
        result = reward_fn(full_response, answer[i])
        gsm8k[i]['result'] = result
        if result['format_reward'] == 1.0 and result['answer_reward'] == 1.0:
            type = 1
            type1_num += 1
        elif result['format_reward'] == 1.0 and result['answer_reward'] == 0.0:
            type = 2
            type2_num += 1
        else:
            type = 3
            type3_num += 1
        gsm8k[i]['type'] = type
        acc += result['reward']
        if rl == True:
            format_reward += result['format_reward']
    accuracy = acc / len(outputs)
    format_reward = format_reward / len(outputs)
    with open(f"{model_path}/test_log.json",'w') as f:
        json.dump(gsm8k,f,indent=4)
    if rl == True:
        return accuracy, format_reward
    return accuracy, type1_num, type2_num, type3_num
if __name__ == "__main__":
    model_path = "/home/users/nus/e1553316/scratch/assignment5/models/Qwen2.5-Math-1.5B"
    evaluate(model_path)
