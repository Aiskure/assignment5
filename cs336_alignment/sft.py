from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import logging
import os
import gc
import wandb
from tqdm import tqdm
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment import utils
from cs336_alignment.math_baseline import evaluate_vllm, evaluate

model_path = "models/Qwen2.5-Math-1.5B"
train_path = "data/gsm8k/train.jsonl"  # 训练集
test_path = "data/gsm8k/test.jsonl"    # 测试集（用作验证）

# 设置日志级别，减少 VLLM 的输出
logging.getLogger("vllm").setLevel(logging.WARNING)
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
def load_policy_into_vllm_instance(policy, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

#加载模型

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16
).to(device)
model = model.to(device)
use_vllm_eval = torch.cuda.device_count() >= 2
llm = None
if use_vllm_eval:
    llm = utils.init_vllm(model_id=model_path, device="cuda:1", seed=42, gpu_memory_utilization=0.8)
else:
    print("[WARN] <2 visible GPUs; skipping vLLM eval.")
tokenizer = AutoTokenizer.from_pretrained(model_path)

#优化器
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-5,)


#============================================data prepare=========================================

reward = r1_zero_reward_fn
r1_zero_prompt = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""

gsm8k =[]
with open(train_path) as f:
    lines = f.readlines()
    for line in lines:
        gsm8k.append(json.loads(line))

prompts = []
answers = []

for dict in gsm8k:
    prompts.append(r1_zero_prompt.format(question=dict['question']))
    answers.append(" " + dict['answer'].replace("#### "," </think> <answer> ") + " </answer>")

# ==========================================train step==========================================


epochs = 3
micro_batch_size = 1
local_batch_size = 32
gradient_accumulation_steps = local_batch_size//micro_batch_size

log_directory = "cs336_alignment/sft_eval"
opt_step = 0

# wandb 初始化（可通过环境变量覆盖）
wandb_run = wandb.init(
    project=os.getenv("WANDB_PROJECT", "cs336-a5-sft"),
    entity=os.getenv("WANDB_ENTITY"),
    name=os.getenv("WANDB_RUN_NAME"),
    config={
        "model_path": model_path,
        "train_path": train_path,
        "epochs": epochs,
        "micro_batch_size": micro_batch_size,
        "local_batch_size": local_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lr": 1e-5,
        "eval_every_local_steps": 1000,
        "use_vllm_eval": use_vllm_eval,
    },
)

local_step = 0
for i in range(epochs):
    pbar = tqdm(range(len(prompts) // micro_batch_size), desc=f"Epoch {i+1}/{epochs}")

    for j in pbar:
        prompt_strs = prompts[j * micro_batch_size:(j+1) * micro_batch_size ]
        answer_strs = answers[j * micro_batch_size:(j+1) * micro_batch_size ]
        #拼接和logists
        train_batch = utils.tokenize_prompt_and_output(prompt_strs,answer_strs,tokenizer)
        result_batch = utils.get_response_log_probs(model,train_batch["input_ids"].to(device),train_batch["labels"].to(device),return_token_entropy=False)
        policy_log_probs = result_batch["log_probs"]

        #计算loss
        loss,_= utils.sft_microbatch_train_step(policy_log_probs,train_batch["response_mask"].to(device),gradient_accumulation_steps,normalize_constant = 1.0)
        
        if (local_step + 1) % gradient_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            opt_step += 1
            wandb.log(
                {
                    "train/grad_norm": float(grad_norm),
                    "train/opt_step": opt_step,
                },
                step=local_step,
            )
        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Step': local_step})
        if local_step % 10 == 0:
            wandb.log({"train/loss": float(loss.item())}, step=local_step)

        #vllm推理
        if use_vllm_eval and local_step % 1000 == 0:

            save_directory = f'{log_directory}/{local_step}'
            os.makedirs(save_directory, exist_ok=True)
            # model.save_pretrained(save_directory=save_directory)
            # tokenizer.save_pretrained(save_directory=save_directory)
            load_policy_into_vllm_instance(model, llm)
            accuracy, type1_num, type2_num, type3_num = evaluate(save_directory,llm)
            wandb.log(
                {
                    "eval/accuracy": float(accuracy),
                    "eval/type1": int(type1_num),
                    "eval/type2": int(type2_num),
                    "eval/type3": int(type3_num),
                },
                step=local_step,
            )
            # 额外的显存清理，确保彻底释放
            gc.collect()
            torch.cuda.empty_cache()        

        local_step+=1

wandb.finish()





