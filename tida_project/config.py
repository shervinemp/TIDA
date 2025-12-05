from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TIDAConfig:
    # Base Model
    base_model_name: str = "mistralai/Mistral-7B-v0.1"

    # LoRA Parameters
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    # TIDA Specifics
    k_min: int = 2          # Min micro-steps during training
    k_max: int = 6          # Max micro-steps during training
    lambda_budget: float = 1.0  # Loss weight for budget convergence

    # Training
    batch_size: int = 4
    learning_rate: float = 2e-4
    # Reduced to 256 to ensure L * (K+1) fits in standard context (e.g. 2048)
    # 256 * (6+1) = 1792 < 2048
    max_seq_len: int = 256
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
