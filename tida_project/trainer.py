import torch
from accelerate import Accelerator
from tqdm import tqdm
import random
import os

class TIDATrainer:
    def __init__(self, model, train_loader, config, tokenizer):
        self.model = model
        self.config = config
        self.loader = train_loader
        self.tokenizer = tokenizer
        self.accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

        # Prepare everything
        self.model, self.optimizer, self.loader = self.accelerator.prepare(
            self.model, self.optimizer, self.loader
        )

        # Placeholder for validation loader if we had one
        self.val_loader = None

    def train(self):
        self.model.train()

        for epoch in range(self.config.num_epochs):
            if self.accelerator.is_local_main_process:
                print(f"Starting Epoch {epoch}")

            for step, (batch_inputs, batch_labels) in enumerate(self.loader):
                with self.accelerator.accumulate(self.model):
                    # Randomize K for this step (Curriculum / Anti-Lazy)
                    k_step = random.randint(self.config.k_min, self.config.k_max)

                    # Forward
                    loss = self.model(
                        input_ids=batch_inputs,
                        labels=batch_labels,
                        k_steps=k_step
                    )

                    # Backward
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if step % 10 == 0 and self.accelerator.is_local_main_process:
                        print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} | K: {k_step}")

            # Validation Step (Simple implementation using same loader subset or similar)
            # Since we don't have a separate Val Set in this setup, we skip or use train subset
            # self.validate(epoch)

            # Save checkpoint
            if self.accelerator.is_local_main_process:
                self.save_checkpoint(epoch)

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        steps = 0
        # Dummy validation on a few batches from train loader for demonstration
        with torch.no_grad():
             for i, (batch_inputs, batch_labels) in enumerate(self.loader):
                 if i > 5: break
                 k_step = self.config.k_min # Fixed K for validation
                 loss = self.model(input_ids=batch_inputs, labels=batch_labels, k_steps=k_step)
                 total_loss += loss.item()
                 steps += 1

        avg_loss = total_loss / steps if steps > 0 else 0
        if self.accelerator.is_local_main_process:
            print(f"Validation Epoch {epoch} | Loss: {avg_loss:.4f}")
        self.model.train()

    def save_checkpoint(self, epoch):
        unwrapped = self.accelerator.unwrap_model(self.model)
        output_dir = f"./checkpoints/epoch_{epoch}"
        os.makedirs(output_dir, exist_ok=True)

        # Save LoRA weights and Custom Heads separately
        unwrapped.base_model.save_pretrained(f"{output_dir}/lora")
        torch.save(unwrapped.time_head.state_dict(), f"{output_dir}/time_head.pt")
        torch.save(unwrapped.blank_embedding, f"{output_dir}/blank_emb.pt")
        print(f"Checkpoint saved to {output_dir}")
