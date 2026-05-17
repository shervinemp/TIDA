import os
from datasets import load_dataset
from torch.utils.data import Dataset
import torch

class TIDADataset(Dataset):
    def __init__(self, tokenizer, data_source, max_length, split="train", text_key="text"):
        self.text_key = text_key
        if data_source == "wikitext":
            self.data = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        elif os.path.isfile(data_source):
            ext = os.path.splitext(data_source)[1].lower()
            if ext in (".json", ".jsonl"):
                self.data = load_dataset("json", data_files=data_source, split=split)
            elif ext == ".csv":
                self.data = load_dataset("csv", data_files=data_source, split=split)
            else:
                self.data = load_dataset("text", data_files=data_source, split=split)
        else:
            self.data = load_dataset(data_source, split=split)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.text_key in item:
            text = item[self.text_key]
        else:
            text = item[list(item.keys())[0]]
        if not text:
            text = " "

        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encodings['input_ids'].squeeze(0)
        labels = input_ids.clone()

        labels[encodings['attention_mask'].squeeze(0) == 0] = -100

        return {
            "input_ids": input_ids,
            "labels": labels
        }

def get_collate_fn(pad_token_id):
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        return input_ids, labels
    return collate_fn
