# TIDA — Time-Integrated Diffusion Attention

Adaptive computation for language models. TIDA inserts a variable number of "thought" tokens between each real token, letting the model spend more compute on difficult tokens and less on easy ones. Training uses parallel computation of all thought tokens per macro-step with a staircase attention mask. Inference runs sequentially but can early-exit when p > 0.99.

## Project Structure

```
tida_project/
  config.py        — TIDAConfig dataclass (model, LoRA, training hyperparams)
  modeling_tida.py — TIDAModel, TIDATimeHead, RoPE patch, loss computation
  dataset.py       — TIDADataset (HF datasets + local files), collate fn
  trainer.py       — TIDATrainer (Accelerate-based train/val loop, checkpointing)
  inference.py     — generate_tida — adaptive autoregressive generation
  main.py          — Entry point: train, infer, test checkpoint loading
```

## Setup

```bash
pip install torch transformers accelerate peft datasets tqdm
```

## Usage

**Full training:**
```bash
python tida_project/main.py
```

**Quick verification (SmolLM-135M, 10 samples, 1 epoch):**
```bash
python tida_project/main.py --verify
```

**Inference with a trained checkpoint:**
```python
from modeling_tida import TIDAModel
from config import TIDAConfig
from transformers import AutoTokenizer

config = TIDAConfig()
model = TIDAModel.from_pretrained(config, "./checkpoints/epoch_2")
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)

from inference import generate_tida
print(generate_tida(model, tokenizer, "The future of AI is", max_new_tokens=30))
```

## How It Works

For each macro-step (one real token), the model runs **K micro-steps** (configurable `k_min`–`k_max`). Each thought slot k has its own **learned embedding** (`blank_embedding[:, k, :]`), giving each step a distinct identity:

| Step | Token | Description |
|------|-------|-------------|
| k=0 | Real token embedding | Position offset 0 |
| k=1..K | Learned per-step blank embedding | Position offset 0.99/K per step |

A **time head** predicts a "consumption fraction" p_k ∈ [0,1]. The budget b_k starts at 1.0 and decrements: t_{k+1} = t_k + b_k · p_k. The weighted logits `exp(t_k) · logits_k` are integrated via summation, and CE loss is applied on the result. A budget loss `(1 - t_K)²` encourages full budget consumption.

**Training** is fully parallel — all K+1 tokens per macro-step are computed in a single transformer pass with a staircase causal mask (thoughts can only attend within their own bucket, not to future real tokens). The KV cache is explicitly disabled (`use_cache=False`) since it's unused during training.

**Inference** runs the micro-loop sequentially with KV cache, and early-exits when p_k > 0.99. The KV cache is truncated after each macro-step to remove thought tokens.

## Configuration

Key settings in `TIDAConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_model_name` | `mistralai/Mistral-7B-v0.1` | Base LM |
| `lora_r` / `lora_alpha` / `lora_dropout` | 64 / 128 / 0.05 | LoRA params |
| `k_min` / `k_max` | 2 / 6 | Thought step range |
| `lambda_budget` | 1.0 | Budget loss weight |
| `max_seq_len` | 256 | Must respect `L·(K+1) < model max` |
| `batch_size` / `gradient_accumulation_steps` | 4 / 4 | Effective batch = 16 |

## Checkpoints

Saved to `./checkpoints/epoch_{N}/`:
- `lora/` — LoRA adapter weights
- `time_head.pt` — TIDATimeHead state dict
- `blank_emb.pt` — Learned blank embedding tensor

## Key Design Decisions

- **Eager attention** — required for custom staircase mask; FA2 not compatible.
- **Float32 position IDs** — fractional positions for thought tokens need higher precision than bf16.
- **Gradient checkpointing** after LoRA wrap — reduces VRAM.
- **No early exit during training** — inference-only for reproducibility.
- **Cosine LR schedule** with gradient clipping (max_norm=1.0) for stability.
