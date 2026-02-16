def log_generations(
    prompts: list[str],
    generateds: list[str],
    ground_truths: list[str],
    rewards: list[dict],  # 包含 format_reward, answer_reward, total_reward
    entropies: list[float],
    epoch: int
):
    # 计算统计信息
    response_lengths = [len(g.split()) for g in generateds]
    correct_mask = [r.get('answer_reward', 0) > 0.5 for r in rewards]  # 假设阈值
    
    # 记录表格（wandb.Table）
    table_data = []
    for p, g, gt, r, e in zip(prompts, generateds, ground_truths, rewards, entropies):
        table_data.append([
            p, g, gt, 
            r.get('format_reward', 0),
            r.get('answer_reward', 0), 
            r.get('total_reward', 0),
            e
        ])
    
    wandb.log({
        "generations": wandb.Table(
            columns=["prompt", "generated", "ground_truth", 
                     "format_reward", "answer_reward", "total_reward", "entropy"],
            data=table_data
        ),
        "avg_response_length": np.mean(response_lengths),
        "avg_length_correct": np.mean([l for l, c in zip(response_lengths, correct_mask) if c]),
        "avg_length_incorrect": np.mean([l for l, c in zip(response_lengths, correct_mask) if not c]),
    }, step=epoch)