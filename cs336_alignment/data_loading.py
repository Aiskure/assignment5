import json
import random
import torch
from torch.utils.data import Dataset,DataLoader


class PackedSFTDataset(Dataset):
    def __init__(self, tokenizer, dataset_path, seq_length, shuffle=True):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.shuffle = shuffle

        with open(dataset_path, 'r') as f:
            data = [json.loads(line) for line in f]
        documents = []

        for ex in data:
            text = (
                "Below is an instruction that describes a task. Write a response that appropriately "
                "completes the request.\n\n"
                "### Instruction:\n"
                f"{ex['prompt']}\n\n"
                "### Response:\n"
                f"{ex['response']}"
            )
            documents.append(text)
        if self.shuffle:
            random.shuffle(documents)

        self.all_tokens = []
        for doc in documents:
            self.all_tokens.append(self.tokenizer.bos_token_id)
            doc_tokens = self.tokenizer.encode(doc, add_special_tokens=False)
            self.all_tokens.extend(doc_tokens)
            self.all_tokens.append(self.tokenizer.eos_token_id)
        self.num_examples = (len(self.all_tokens) - 1) // self.seq_length

    def __len__(self):
        return self.num_examples

    def __getitem__(self, i):
        start = i * self.seq_length
        input_ids = torch.tensor(self.all_tokens[start : start + self.seq_length])
        labels = torch.tensor(self.all_tokens[start + 1 : start + self.seq_length + 1])
        return  {"input_ids":   input_ids, 
                 "labels":      labels}



def iterate_batches(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)