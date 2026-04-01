import json

#提取数据，添加来源标签
def load_hh_data(file_path: str) -> list[dict]:
    data = []
    with open(file_path, 'r') as f:
        data = [json.loads(line.strip()) for line in f]
    return data

def load_multiple_files(file_paths: list[str]) -> list[dict]:
    all_data = []
    for file_path in file_paths:
        data = load_hh_data(file_path)
        for item in data:
            item["source"] = file_path.split("/")[-1].replace(".jsonl", "")  # 添加来源标签
        all_data.extend(data)
    return all_data


#过滤多轮对话
def is_single_turn(text: str) -> bool:
    return text.count("Human:") == 1

def filter_to_single_turn(data: list[dict]) -> list[dict]:
    return [item for item in data if is_single_turn(item["chosen"]) and is_single_turn(item["rejected"])]


#提取指令和回复
def parse_conversation(data: list[dict]) -> list[dict]:
    extracted_data = []
    for item in data:
        text = item["chosen"]
        human_part, assistant_part = text.split("\n\nAssistant:",maxsplit=1)
        instruction = human_part.replace("\n\nHuman:", "").strip()
        chosen_response = assistant_part.strip()
        rejected_response = item["rejected"].split("\n\nAssistant:",maxsplit=1)[1].strip()
        extracted_data.append({
            "instruction": instruction,
            "chosen_response": chosen_response,
            "rejected_response": rejected_response,
            "source": item["source"]})
    return extracted_data


if __name__ == "__main__":
    files = [
    "data/hh/harmless-base.jsonl",
    "data/hh/helpful-base.jsonl",
    "data/hh/helpful-online.jsonl",
    "data/hh/helpful-rejection-sampled.jsonl",
    ]
    data = load_multiple_files(files)
    data = filter_to_single_turn(data)
    parsed = parse_conversation(data)
    
    import random
    random.seed(42)

    helpful = [x for x in parsed if x["source"].startswith("helpful")]
    harmless = [x for x in parsed if x["source"].startswith("harmless")]

    print(f"Helpful: {len(helpful)}, Harmless: {len(harmless)}")

    for label, subset in [("HELPFUL", helpful), ("HARMLESS", harmless)]:
        samples = random.sample(subset, 3)
        for i, s in enumerate(samples):
            print(f"\n=== {label} Example {i+1} ({s['source']}) ===")
            print(f"Instruction: {s['instruction'][:200]}")
            print(f"Chosen: {s['chosen_response'][:200]}")
            print(f"Rejected: {s['rejected_response'][:200]}")