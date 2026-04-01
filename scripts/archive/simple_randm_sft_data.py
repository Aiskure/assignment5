import json
import random

def sample_and_display(jsonl_path, num_samples=10):
    """从JSONL文件中随机抽取指定数量的样本并显示"""
    
    # 读取所有行
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"总样本数: {len(lines)}")
    print(f"随机抽取 {num_samples} 个样本:\n")
    print("=" * 80)
    
    # 随机抽取
    sampled_lines = random.sample(lines, min(num_samples, len(lines)))
    
    for i, line in enumerate(sampled_lines, 1):
        data = json.loads(line.strip())
        prompt = data.get('prompt', '')
        response = data.get('response', '')
        
        print(f"\n样本 {i}:")
        print("-" * 80)
        print(f"【Prompt】:\n{prompt[:500]}{'...' if len(prompt) > 500 else ''}\n")
        print(f"【Response】:\n{response[:500]}{'...' if len(response) > 500 else ''}\n")
        print("=" * 80)

if __name__ == "__main__":
    # 设置随机种子以便复现（可选）
    random.seed(42)
    
    # 数据文件路径
    data_path = "/home/users/nus/e1553316/scratch/assignment5/data/safety_augmented_ultrachat_200k_single_turn/test.jsonl"
    
    # 随机抽取10个样本
    sample_and_display(data_path, num_samples=10)
