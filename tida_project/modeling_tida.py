import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import AutoModelForCausalLM
import types
import os

class TIDATimeHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, hidden_states):
        return F.hardsigmoid(self.mlp(hidden_states))

def patched_rope_forward(self, x, position_ids, seq_len=None, **kwargs):
    if not hasattr(self, "inv_freq"):
        return x, x

    inv_freq = self.inv_freq

    if inv_freq.device != x.device:
        inv_freq = inv_freq.to(x.device)

    inv_freq_expanded = inv_freq.view(1, 1, -1)

    if position_ids is None:
         return x, x

    position_ids_expanded = position_ids.unsqueeze(-1).float()

    with torch.autocast(device_type=x.device.type, enabled=False):
        freqs = (position_ids_expanded * inv_freq_expanded.float())
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def apply_rope_patch(model):
    patched = False
    for name, module in model.named_modules():
        if "RotaryEmbedding" in module.__class__.__name__:
             module.forward = types.MethodType(patched_rope_forward, module)
             patched = True

    if not patched:
        print("Warning: No RotaryEmbedding module found to patch.")

class TIDAModel(nn.Module):
    def __init__(self, config: 'TIDAConfig', base_model_instance=None):
        super().__init__()
        self.config = config

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

        if base_model_instance:
             self.base_model = base_model_instance
        else:
            # Force eager attention to avoid issues with custom masks and FA2 on CPU/Verification
            self.base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name,
                dtype=self.dtype,
                attn_implementation="eager"
            )

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.target_modules
            )
            self.base_model = get_peft_model(self.base_model, peft_config)

            # Enable gradient checkpointing for VRAM efficiency
            if hasattr(self.base_model, "gradient_checkpointing_enable"):
                 self.base_model.gradient_checkpointing_enable()

        if hasattr(self.base_model, "get_base_model"):
             underlying = self.base_model.get_base_model()
             base_config = underlying.config
        elif hasattr(self.base_model, "config"):
             underlying = self.base_model
             base_config = underlying.config
        else:
             underlying = self.base_model.base_model
             base_config = underlying.config

        self.hidden_size = base_config.hidden_size

        # Locate the transformer backbone (without LM head) so we can access last_hidden_state
        self.backbone = getattr(underlying, 'model', None) or getattr(underlying, 'transformer', None)
        if self.backbone is None:
            print("Warning: Could not find transformer backbone — falling back to output_hidden_states=True")

        # Zero-init for stable start; model learns thought embeddings
        self.blank_embedding = nn.Parameter(torch.zeros(1, config.k_max, self.hidden_size, dtype=self.dtype, device=self.device))
        self.time_head = TIDATimeHead(self.hidden_size).to(dtype=self.dtype, device=self.device)

        if config.fractional_positions:
            apply_rope_patch(self.base_model)

    @classmethod
    def from_pretrained(cls, config, checkpoint_dir):
        print(f"Loading TIDA model from {checkpoint_dir}")

        # Restore config metadata saved during training
        meta_path = os.path.join(checkpoint_dir, "config_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            for k, v in meta.items():
                if hasattr(config, k):
                    setattr(config, k, v)
                    print(f"  Restored config.{k} = {v}")

        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
            attn_implementation="eager"
        )

        lora_path = os.path.join(checkpoint_dir, "lora")
        if os.path.exists(lora_path):
            base_model = PeftModel.from_pretrained(base_model, lora_path)
        else:
            print("Warning: LoRA weights not found, using initialized weights.")

        instance = cls(config, base_model_instance=base_model)

        # Re-enable gradient checkpointing (not applied in __init__ when base_model_instance is given)
        if hasattr(instance.base_model, "gradient_checkpointing_enable"):
            instance.base_model.gradient_checkpointing_enable()

        time_head_path = os.path.join(checkpoint_dir, "time_head.pt")
        if os.path.exists(time_head_path):
            instance.time_head.load_state_dict(torch.load(time_head_path, map_location=instance.device, weights_only=True))

        blank_emb_path = os.path.join(checkpoint_dir, "blank_emb.pt")
        if os.path.exists(blank_emb_path):
             loaded_emb = torch.load(blank_emb_path, map_location=instance.device, weights_only=True)
             saved_k = loaded_emb.shape[-2]
             model_k = instance.blank_embedding.shape[-2]
             if saved_k == model_k:
                 instance.blank_embedding.data = loaded_emb.data
             elif saved_k == 1 and model_k > 1:
                 instance.blank_embedding.data[:, 0:1, :] = loaded_emb.data
             else:
                 n = min(saved_k, model_k)
                 instance.blank_embedding.data[:, :n, :] = loaded_emb.data[:, :n, :]
                 print(f"  Resized blank_embedding: saved {saved_k} → model {model_k}")

        return instance

    def forward(self, input_ids, attention_mask=None, labels=None, k_steps=None, lambda_budget=None):
        batch_size, seq_len = input_ids.shape
        K = k_steps if k_steps is not None else self.config.k_min

        # --- A. Embeddings & Expansion ---
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        blank_embeds = self.blank_embedding[:, :K, :].unsqueeze(1).expand(batch_size, seq_len, K, -1)

        combined_embeds = torch.cat([inputs_embeds.unsqueeze(2), blank_embeds], dim=2)
        combined_embeds = combined_embeds.view(batch_size, seq_len * (K + 1), -1)

        # --- B. Position Calculation ---
        macro_positions = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(1)
        if self.config.fractional_positions:
            micro_positions = torch.linspace(0, 0.99, K+1, device=inputs_embeds.device, dtype=torch.float32)
            macro_f = macro_positions.float()
            position_ids = (macro_f + micro_positions).view(1, -1).expand(batch_size, -1)
        else:
            micro_positions = torch.arange(K+1, device=inputs_embeds.device)
            position_ids = (macro_positions * (K + 1) + micro_positions).view(1, -1).expand(batch_size, -1)

        # --- Create Custom Attention Mask (Staircase) ---
        total_len = seq_len * (K + 1)

        indices = torch.arange(total_len, device=inputs_embeds.device)
        rows = indices.unsqueeze(1)
        cols = indices.unsqueeze(0)

        bucket_rows = rows // (K + 1)
        bucket_cols = cols // (K + 1)
        micro_cols = cols % (K + 1)

        # Condition 1: Causal
        mask_cond = (cols <= rows)

        # Condition 2: Visibility (Staircase)
        is_think_col = (micro_cols > 0)
        same_bucket = (bucket_rows == bucket_cols)

        allowed = mask_cond & (~is_think_col | same_bucket)

        # Create additive mask: 0.0 for allowed, -10000.0 for masked
        min_val = -10000.0
        custom_mask = torch.full((batch_size, 1, total_len, total_len), min_val, device=inputs_embeds.device, dtype=self.dtype)

        mask_expanded = allowed.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        custom_mask = torch.where(mask_expanded, torch.tensor(0.0, dtype=self.dtype, device=self.device), custom_mask)

        # --- C. Transformer Pass ---
        if self.backbone is not None:
            backbone_out = self.backbone(
                inputs_embeds=combined_embeds,
                position_ids=position_ids,
                attention_mask=custom_mask,
                use_cache=False,
            )
            last_hidden = backbone_out.last_hidden_state
            logits = self.base_model.get_output_embeddings()(last_hidden)
        else:
            outputs = self.base_model(
                inputs_embeds=combined_embeds,
                position_ids=position_ids,
                attention_mask=custom_mask,
                use_cache=False,
                output_hidden_states=True,
            )
            logits = outputs.logits
            last_hidden = outputs.hidden_states[-1]

        p_vals = self.time_head(last_hidden)

        # --- D. Integral Logic & Loss ---
        if labels is not None:
            total_loss, integrated_logits = self.compute_loss(logits, p_vals, labels, K, seq_len, lambda_budget=lambda_budget)

            # KL regularization: keep TIDA output close to the model's own K=0 output
            if self.config.lambda_kl > 0 and K > 0:
                with torch.no_grad():
                    ref = self.forward(input_ids, labels=None, k_steps=0)
                log_p = F.log_softmax(integrated_logits, dim=-1)
                p_ref = F.softmax(ref, dim=-1)
                kl = F.kl_div(log_p, p_ref, reduction='batchmean')
                total_loss = total_loss + self.config.lambda_kl * kl

            return total_loss

        return logits

    def compute_loss(self, logits, p_vals, labels, K, seq_len, lambda_budget=None):
        batch_size = logits.shape[0]

        logits = logits.view(batch_size, seq_len, K+1, -1)
        p_vals = p_vals.view(batch_size, seq_len, K+1)

        t_accum = torch.zeros(batch_size, seq_len, K+1, device=logits.device, dtype=logits.dtype)
        b_rem = torch.ones(batch_size, seq_len, device=logits.device, dtype=logits.dtype)
        current_t = torch.zeros(batch_size, seq_len, device=logits.device, dtype=logits.dtype)

        for k in range(K):
            p_k = p_vals[:, :, k]
            delta = b_rem * p_k
            current_t = current_t + delta
            b_rem = b_rem - delta
            t_accum[:, :, k+1] = current_t

        weights = torch.exp(t_accum).unsqueeze(-1)
        integrated_logits = torch.sum(logits * weights, dim=2)

        shift_logits = integrated_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = torch.nan_to_num(shift_logits, nan=0.0, posinf=1e4, neginf=-1e4)

        loss_task = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        if K > 0:
            final_t = t_accum[:, :, -1]
            loss_budget = torch.mean((1.0 - final_t) ** 2)
            lb = lambda_budget if lambda_budget is not None else self.config.lambda_budget
            total_loss = loss_task + lb * loss_budget
        else:
            total_loss = loss_task

        return total_loss, integrated_logits
