import json
import torch
from accelerate import Accelerator
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import random
import os

class TIDATrainer:
    def __init__(self, model, train_loader, config, tokenizer, val_loader=None):
        self.model = model
        self.config = config
        self.loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

        total_steps = len(self.loader) * config.num_epochs
        warmup = config.warmup_steps
        if warmup > 0 and warmup < total_steps:
            warmup_sched = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup)
            cosine_sched = CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup)
            self.scheduler = SequentialLR(self.optimizer, [warmup_sched, cosine_sched], milestones=[warmup])
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)

        # Prepare everything
        if self.val_loader is not None:
            self.model, self.optimizer, self.loader, self.scheduler, self.val_loader = self.accelerator.prepare(
                self.model, self.optimizer, self.loader, self.scheduler, self.val_loader
            )
        else:
            self.model, self.optimizer, self.loader, self.scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.loader, self.scheduler
            )

    def curriculum_k(self, step, total_steps):
        frac = step / max(total_steps, 1)
        if frac < 0.10:
            return 0
        elif frac < 0.30:
            return self.config.k_min
        else:
            return random.randint(self.config.k_min, self.config.k_max)

    def curriculum_lambda(self, step, total_steps):
        """Ramp lambda_budget from 0 → target over Phase 2 (10-30%),
           then hold at target for Phase 3."""
        frac = step / max(total_steps, 1)
        if frac < 0.10:
            return 0.0
        elif frac < 0.30:
            ramp = (frac - 0.10) / 0.20
            return ramp * self.config.lambda_budget
        else:
            return self.config.lambda_budget

    def train(self):
        self.model.train()
        total_steps = len(self.loader) * self.config.num_epochs

        for epoch in range(self.config.num_epochs):
            if self.accelerator.is_local_main_process:
                print(f"Starting Epoch {epoch}")

            for step, (batch_inputs, batch_labels) in enumerate(self.loader):
                global_step = epoch * len(self.loader) + step
                with self.accelerator.accumulate(self.model):
                    k_step = self.curriculum_k(global_step, total_steps)
                    current_lambda = self.curriculum_lambda(global_step, total_steps)

                    loss = self.model(
                        input_ids=batch_inputs,
                        labels=batch_labels,
                        k_steps=k_step,
                        lambda_budget=current_lambda,
                    )

                    # Backward
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.accelerator.sync_gradients:
                        self.scheduler.step()

                    if step % 10 == 0 and self.accelerator.is_local_main_process:
                        print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} | K: {k_step}")

            if self.val_loader is not None:
                self.validate(epoch)

            # Save checkpoint
            if self.accelerator.is_local_main_process:
                self.save_checkpoint(epoch)

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        steps = 0
        with torch.no_grad():
            for batch_inputs, batch_labels in self.val_loader:
                k_step = self.config.k_min
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

        # Save config metadata for correct reloading
        meta = {
            "fractional_positions": self.config.fractional_positions,
            "k_max": self.config.k_max,
            "base_model_name": self.config.base_model_name,
        }
        with open(os.path.join(output_dir, "config_meta.json"), "w") as f:
            json.dump(meta, f)

        print(f"Checkpoint saved to {output_dir}")
