import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import AutoModelForCausalLM, AutoConfig
import types
import os

class TIDATimeHead(nn.Module):
    """Predicts consumption fraction p_k."""
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, hidden_states):
        # HardSigmoid: max(0, min(1, (x+3)/6))
        return F.hardsigmoid(self.mlp(hidden_states))

def patched_rope_forward(self, x, position_ids, seq_len=None, **kwargs):
    """
    Patched forward for RoPE to handle float position_ids.
    Calculates cos/sin on the fly.
    """
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
    """
    Monkey-patch the base model's RoPE.
    """
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
                torch_dtype=self.dtype,
                attn_implementation="eager"
            )

            # Enable gradient checkpointing for VRAM efficiency as per spec
            if hasattr(self.base_model, "gradient_checkpointing_enable"):
                 self.base_model.gradient_checkpointing_enable()

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.target_modules
            )
            self.base_model = get_peft_model(self.base_model, peft_config)

        if hasattr(self.base_model, "get_base_model"):
             base_config = self.base_model.get_base_model().config
        elif hasattr(self.base_model, "config"):
             base_config = self.base_model.config
        else:
             base_config = self.base_model.base_model.config

        self.hidden_size = base_config.hidden_size

        # Use smaller initialization for stability
        self.blank_embedding = nn.Parameter(torch.randn(1, 1, self.hidden_size, dtype=self.dtype, device=self.device) * 0.02)
        self.time_head = TIDATimeHead(self.hidden_size).to(dtype=self.dtype, device=self.device)

        apply_rope_patch(self.base_model)

    @classmethod
    def from_pretrained(cls, config, checkpoint_dir):
        print(f"Loading TIDA model from {checkpoint_dir}")

        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
            attn_implementation="eager"
        )

        lora_path = os.path.join(checkpoint_dir, "lora")
        if os.path.exists(lora_path):
            base_model = PeftModel.from_pretrained(base_model, lora_path)
        else:
            print("Warning: LoRA weights not found, using initialized weights.")

        instance = cls(config, base_model_instance=base_model)

        time_head_path = os.path.join(checkpoint_dir, "time_head.pt")
        if os.path.exists(time_head_path):
            instance.time_head.load_state_dict(torch.load(time_head_path, map_location=instance.device))

        blank_emb_path = os.path.join(checkpoint_dir, "blank_emb.pt")
        if os.path.exists(blank_emb_path):
             loaded_emb = torch.load(blank_emb_path, map_location=instance.device)
             instance.blank_embedding.data = loaded_emb.data

        return instance

    def forward(self, input_ids, attention_mask=None, labels=None, k_steps=None):
        """
        Parallel Training Forward Pass.
        """
        batch_size, seq_len = input_ids.shape
        K = k_steps if k_steps else self.config.k_min

        # --- A. Embeddings & Expansion ---
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        blank_embeds = self.blank_embedding.expand(batch_size, seq_len, K, -1)

        combined_embeds = torch.cat([inputs_embeds.unsqueeze(2), blank_embeds], dim=2)
        combined_embeds = combined_embeds.view(batch_size, seq_len * (K + 1), -1)

        # --- B. Physics Calculation ---
        micro_positions = torch.linspace(0, 0.99, K+1, device=inputs_embeds.device, dtype=self.dtype)
        macro_positions = torch.arange(seq_len, device=inputs_embeds.device, dtype=self.dtype).unsqueeze(1)
        position_ids = (macro_positions + micro_positions).view(1, -1).expand(batch_size, -1)

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
        outputs = self.base_model(
            inputs_embeds=combined_embeds,
            position_ids=position_ids,
            attention_mask=custom_mask,
            output_hidden_states=True
        )

        logits = outputs.logits
        last_hidden = outputs.hidden_states[-1]
        p_vals = self.time_head(last_hidden)

        # --- D. Integral Logic & Loss ---
        if labels is not None:
            return self.compute_loss(logits, p_vals, labels, K, seq_len)

        return logits

    def compute_loss(self, logits, p_vals, labels, K, seq_len):
        batch_size = logits.shape[0]

        logits = logits.view(batch_size, seq_len, K+1, -1)
        p_vals = p_vals.view(batch_size, seq_len, K+1)

        # t_accum stores [t_0, t_1, ... t_K]
        # Boundary Condition: t_0 = 0.0
        t_accum = torch.zeros(batch_size, seq_len, K+1, device=logits.device, dtype=logits.dtype)
        b_rem = torch.ones(batch_size, seq_len, device=logits.device, dtype=logits.dtype)
        current_t = torch.zeros(batch_size, seq_len, device=logits.device, dtype=logits.dtype)

        # Calculate t_{k+1} based on p_k. t_0 is already 0.0.
        # Loop K times to fill indices 1..K
        for k in range(K):
            p_k = p_vals[:, :, k]
            delta = b_rem * p_k
            current_t = current_t + delta
            b_rem = b_rem - delta

            # Update next step
            t_accum[:, :, k+1] = current_t

        weights = torch.exp(t_accum).unsqueeze(-1)

        integrated_logits = torch.sum(logits * weights, dim=2)

        shift_logits = integrated_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        if torch.isnan(shift_logits).any():
             print("NaN detected in shift_logits - likely mask or scale issue. Returning zero loss.")
             # Return a dummy loss with grad
             return torch.tensor(0.0, device=logits.device, requires_grad=True)

        loss_task = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        final_t = t_accum[:, :, -1]
        loss_budget = torch.mean((1.0 - final_t) ** 2)

        total_loss = loss_task + self.config.lambda_budget * loss_budget
        return total_loss
