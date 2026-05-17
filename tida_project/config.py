from dataclasses import dataclass, field
from typing import Optional

MODEL_PRESETS = {
    "verify": {
        "base_model_name": "HuggingFaceTB/SmolLM-135M",
        "batch_size": 1,
        "max_seq_len": 32,
        "num_epochs": 1,
        "gradient_accumulation_steps": 1,
        "warmup_steps": 0,
    },
    "tiny": {
        "base_model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    },
    "small": {
        "base_model_name": "microsoft/Phi-3-mini-4k-instruct",
    },
    "medium": {
        "base_model_name": "mistralai/Mistral-7B-v0.1",
    },
}

def load_config(preset: Optional[str] = None, **overrides) -> "TIDAConfig":
    config = TIDAConfig()
    if preset in MODEL_PRESETS:
        for k, v in MODEL_PRESETS[preset].items():
            setattr(config, k, v)
    for k, v in overrides.items():
        if hasattr(config, k):
            setattr(config, k, v)
    return config

@dataclass
class TIDAConfig:
    # Base Model
    base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # LoRA Parameters
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    # TIDA Specifics
    k_min: int = 2
    k_max: int = 6
    lambda_budget: float = 1.0
    lambda_kl: float = 0.0
    fractional_positions: bool = True

    # Training
    batch_size: int = 2
    learning_rate: float = 2e-4
    max_seq_len: int = 256
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_steps: int = 500
