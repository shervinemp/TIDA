import argparse
from config import TIDAConfig, load_config, MODEL_PRESETS
from modeling_tida import TIDAModel
from dataset import TIDADataset, get_collate_fn
from trainer import TIDATrainer
from inference import generate_tida
from transformers import AutoTokenizer
import torch
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_PRESETS), default="tiny", help="Model preset")
    parser.add_argument("--verify", action="store_true", help="Alias for --model verify")
    args = parser.parse_args()

    preset = "verify" if args.verify else args.model
    config = load_config(preset)

    if preset == "verify":
        torch.autograd.set_detect_anomaly(True)
        print("Running in verification mode with tiny model...")

    data_path = "wikitext"

    print(f"Loading model: {config.base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Data
    print("Loading dataset...")
    train_dataset = TIDADataset(tokenizer, data_path, config.max_seq_len, split="train")

    if data_path == "wikitext":
        val_dataset = TIDADataset(tokenizer, data_path, config.max_seq_len, split="validation")
    else:
        val_dataset = None

    if preset == "verify":
        train_dataset.examples = train_dataset.examples[:10]
        if val_dataset is not None:
            val_dataset.examples = val_dataset.examples[:5]

    collate_fn = get_collate_fn(tokenizer.pad_token_id)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        shuffle=True
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            collate_fn=collate_fn,
            shuffle=False
        )

    # 3. Model
    print("Initializing model...")
    model = TIDAModel(config)

    # 4. Train
    print("Starting training...")
    trainer = TIDATrainer(model, train_loader, config, tokenizer, val_loader=val_loader)
    trainer.train()
    print("Training finished.")

    # 5. Inference Demonstration
    print("\n--- Inference Demonstration ---")
    if preset == "verify":
        prompt = "Once upon a time"
    else:
        prompt = "The future of AI is"

    generated_text = generate_tida(model, tokenizer, prompt, max_new_tokens=10 if preset == "verify" else 30)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")

    if preset == "verify":
        print("\n--- Testing Model Loading ---")
        last_epoch = config.num_epochs - 1
        checkpoint_dir = f"./checkpoints/epoch_{last_epoch}"
        if os.path.exists(checkpoint_dir):
            try:
                loaded_model = TIDAModel.from_pretrained(config, checkpoint_dir)
                print("Model loaded successfully from checkpoint.")
            except Exception as e:
                print(f"Failed to load model: {e}")

if __name__ == "__main__":
    main()
