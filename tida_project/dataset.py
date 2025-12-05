from datasets import load_dataset
from torch.utils.data import Dataset
import torch

class TIDADataset(Dataset):
    def __init__(self, tokenizer, data_path, max_length):
        # Using a small subset for verification purposes if needed, but standard logic follows
        # We handle "wikitext" specifically or generic file
        if data_path == "wikitext":
             self.data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        else:
             self.data = load_dataset(data_path, split="train")

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text'] # Adjust key based on dataset
        if not text:
             # Handle empty strings in wikitext
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

        # Mask pad tokens in labels
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
