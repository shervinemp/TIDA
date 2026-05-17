import logging
import os
from datasets import load_dataset
from torch.utils.data import Dataset
import torch

class TIDADataset(Dataset):
    def __init__(self, tokenizer, data_source, max_length, split="train", text_key="text"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id or 0

        if data_source == "wikitext":
            raw = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        elif os.path.isfile(data_source):
            ext = os.path.splitext(data_source)[1].lower()
            if ext in (".json", ".jsonl"):
                raw = load_dataset("json", data_files=data_source, split=split)
            elif ext == ".csv":
                raw = load_dataset("csv", data_files=data_source, split=split)
            else:
                raw = load_dataset("text", data_files=data_source, split=split)
        else:
            raw = load_dataset(data_source, split=split)

        # Filter: skip empty lines and wiki markup headers
        texts = [
            item[text_key] for item in raw
            if item.get(text_key) and not item[text_key].startswith(" =")
        ]
        if not texts:
            texts = [" "]

        sep = tokenizer.eos_token or ""
        full = sep.join(texts)

        logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
        token_ids = tokenizer(full, truncation=False, return_tensors="pt")["input_ids"][0]

        chunks = torch.split(token_ids, max_length)
        self.examples = []
        for c in chunks:
            if c.numel() < max_length:
                pad = torch.full((max_length - c.numel(),), self.pad_token_id, dtype=c.dtype)
                c = torch.cat([c, pad])
            self.examples.append(c)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids = self.examples[idx].clone()
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100
        return {"input_ids": input_ids, "labels": labels}


def get_collate_fn(pad_token_id):
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        return input_ids, labels
    return collate_fn
