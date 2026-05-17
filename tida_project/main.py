from config import TIDAConfig
from modeling_tida import TIDAModel
from dataset import TIDADataset, get_collate_fn
from trainer import TIDATrainer
from inference import generate_tida
from transformers import AutoTokenizer
import torch
import sys
import os

def main():
    config = TIDAConfig()

    is_verify = len(sys.argv) > 1 and sys.argv[1] == "--verify"

    # Enable anomaly detection (slows training, used for debugging)
    if is_verify:
        torch.autograd.set_detect_anomaly(True)

    # 1. Setup
    if is_verify:
        print("Running in verification mode with tiny model...")
        config.base_model_name = "HuggingFaceTB/SmolLM-135M"
        config.batch_size = 1
        config.max_seq_len = 32
        config.num_epochs = 1
        config.gradient_accumulation_steps = 1

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

    # Subset for verification
    if is_verify:
        train_dataset.data = train_dataset.data.select(range(10))
        if val_dataset is not None:
            val_dataset.data = val_dataset.data.select(range(5))

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
    if is_verify:
        prompt = "Once upon a time"
    else:
        prompt = "The future of AI is"

    generated_text = generate_tida(model, tokenizer, prompt, max_new_tokens=10 if is_verify else 30)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")

    # 6. Test Loading
    if is_verify:
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
