import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch

class CritiqueDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, device):
        self.data = []
        self.tokenizer = tokenizer
        self.device = device
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = f"{item['question']}{self.tokenizer.sep_token}{item['initial_response']}{self.tokenizer.sep_token}{item['feedback']}"
        label = torch.tensor(item["score"], dtype=torch.float).to(self.device)

        return {
            "text": input_text,
            "label": label
        }
def collate_fn(batch, tokenizer):
    texts = [item['text'] for item in batch]
    labels = torch.tensor([item["label"].item() for item in batch], dtype=torch.float)
    
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'labels': labels
    }

def get_dataloader(jsonl_path, tokenizer, device,  batch_size=16, shuffle=True):
    dataset = CritiqueDataset(jsonl_path, tokenizer, device)
    collator = lambda batch: collate_fn(batch, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)
    return dataloader