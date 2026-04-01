from vllm.model_executor import set_random_seed as vllm_set_random_seed


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
    
def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    将训练好的策略模型权重加载到 vLLM 实例中
    从 TRL 库复制而来
    """
    # 1. 获取 policy 模型的权重字典
    state_dict = policy.state_dict()  # 包含所有参数名称和数值
    
    # 2. 获取 vLLM 内部的模型对象
    # llm.llm_engine: vLLM 的引擎
    # .model_executor: 模型执行器
    # .driver_worker: 驱动工作进程
    # .model_runner: 模型运行器
    # .model: 实际的 PyTorch 模型
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    
    # 3. 将 policy 的权重加载到 vLLM 模型中
    # state_dict.items() 返回 (参数名, 参数值) 的迭代器
    llm_model.load_weights(state_dict.items())


#wandb

# Setup wandb metrics
wandb.define_metric("train_step")  # 训练步骤的x轴
wandb.define_metric("eval_step")    # 评估步骤的x轴

# 所有以 train/ 开头的指标都与 train_step 关联
wandb.define_metric("train/*", step_metric="train_step")

# 所有以 eval/ 开头的指标都与 eval_step 关联
wandb.define_metric("eval/*", step_metric="eval_step")