from datasets import load_dataset
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence

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
            padding=False, # CHANGED: Do not pad here
            return_tensors="pt"
        )

        input_ids = encodings['input_ids'].squeeze(0)
        labels = input_ids.clone()

        # No padding here, so no need to mask pad tokens yet.
        # But we must ensure no partial padding exists from tokenizer (it shouldn't with padding=False)

        return {
            "input_ids": input_ids,
            "labels": labels
        }

def get_collate_fn(pad_token_id):
    def collate_fn(batch):
        # Extract inputs and labels
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]

        # Dynamic padding
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

        return input_ids_padded, labels_padded
    return collate_fn
