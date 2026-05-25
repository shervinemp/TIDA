import json
import torch
from accelerate import Accelerator
from peft import set_peft_model_state_dict
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
import random
import os

class TIDATrainer:
    def __init__(self, model, train_loader, config, tokenizer, val_loader=None, run_name=None):
        self.model = model
        self.config = config
        self.loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.run_name = run_name
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

    def _checkpoint_dir(self, epoch):
        if self.run_name:
            return f"./checkpoints/{self.run_name}/epoch_{epoch}"
        return f"./checkpoints/epoch_{epoch}"

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

    def train(self, resume_epoch=0):
        self.model.train()
        total_steps = len(self.loader) * self.config.num_epochs

        for epoch in range(resume_epoch, self.config.num_epochs):
            if self.accelerator.is_local_main_process:
                print(f"Starting Epoch {epoch}")

            pbar = tqdm(self.loader, desc=f"Epoch {epoch}", disable=not self.accelerator.is_local_main_process)
            for step, (batch_inputs, batch_labels) in enumerate(pbar):
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

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.accelerator.sync_gradients:
                        self.scheduler.step()

                pbar.set_postfix(loss=f"{loss.item():.3f}", K=k_step)

            if self.val_loader is not None:
                self.validate(epoch)

            # Save checkpoint
            if self.accelerator.is_local_main_process:
                self.save_checkpoint(epoch)

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        steps = 0
        pbar = tqdm(self.val_loader, desc=f"Val {epoch}", disable=not self.accelerator.is_local_main_process)
        with torch.no_grad():
            for batch_inputs, batch_labels in pbar:
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
        output_dir = self._checkpoint_dir(epoch)
        os.makedirs(output_dir, exist_ok=True)

        # Save LoRA weights and Custom Heads separately
        unwrapped.base_model.save_pretrained(f"{output_dir}/lora")
        torch.save(unwrapped.time_head.state_dict(), f"{output_dir}/time_head.pt")
        torch.save(unwrapped.blank_embedding, f"{output_dir}/blank_emb.pt")

        # Save optimizer and scheduler state
        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(self.scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

        # Save config metadata for correct reloading
        meta = {
            "fractional_positions": self.config.fractional_positions,
            "k_max": self.config.k_max,
            "base_model_name": self.config.base_model_name,
        }
        with open(os.path.join(output_dir, "config_meta.json"), "w") as f:
            json.dump(meta, f)

        # Save training state for resume
        training_state = {"epoch": epoch}
        with open(os.path.join(output_dir, "training_state.json"), "w") as f:
            json.dump(training_state, f)

        print(f"Checkpoint saved to {output_dir}")

    def load_checkpoint(self, checkpoint_dir):
        """Load model weights, optimizer, and scheduler from a checkpoint. Returns the next epoch to train."""
        device = self.accelerator.device
        unwrapped = self.accelerator.unwrap_model(self.model)

        # Load LoRA weights (handle both .safetensors and .bin)
        lora_path = os.path.join(checkpoint_dir, "lora")
        if os.path.exists(lora_path):
            adapter_file = os.path.join(lora_path, "adapter_model.safetensors")
            if not os.path.exists(adapter_file):
                adapter_file = os.path.join(lora_path, "adapter_model.bin")
            if os.path.exists(adapter_file):
                if adapter_file.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    adapter_weights = load_file(adapter_file)
                else:
                    adapter_weights = torch.load(adapter_file, map_location="cpu")
                set_peft_model_state_dict(unwrapped.base_model, adapter_weights, adapter_name="default")
                print(f"LoRA weights loaded from {adapter_file}")

        # Load time head
        time_head_path = os.path.join(checkpoint_dir, "time_head.pt")
        if os.path.exists(time_head_path):
            unwrapped.time_head.load_state_dict(torch.load(time_head_path, map_location=device))
            print(f"Time head loaded from {time_head_path}")

        # Load blank embedding
        blank_emb_path = os.path.join(checkpoint_dir, "blank_emb.pt")
        if os.path.exists(blank_emb_path):
            loaded = torch.load(blank_emb_path, map_location="cpu")
            unwrapped.blank_embedding.data.copy_(loaded)
            print(f"Blank embedding loaded from {blank_emb_path}")

        # Load optimizer (move state tensors to the correct device)
        optim_path = os.path.join(checkpoint_dir, "optimizer.pt")
        if os.path.exists(optim_path):
            optim_state = torch.load(optim_path, map_location="cpu")
            for state in optim_state["state"].values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            self.optimizer.load_state_dict(optim_state)
            print(f"Optimizer loaded from {optim_path}")

        # Load scheduler
        sched_path = os.path.join(checkpoint_dir, "scheduler.pt")
        if os.path.exists(sched_path):
            self.scheduler.load_state_dict(torch.load(sched_path, map_location="cpu"))
            print(f"Scheduler loaded from {sched_path}")

        # Determine next epoch
        state_path = os.path.join(checkpoint_dir, "training_state.json")
        epoch = -1
        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
            epoch = state.get("epoch", -1)
            if epoch >= 0:
                print(f"Resuming from epoch {epoch + 1}")

        return epoch + 1
