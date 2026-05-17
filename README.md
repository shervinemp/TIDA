# TIDA — Time-Integrated Diffusion Attention

Adaptive computation for language models. TIDA inserts a variable number of "thought" tokens between each real token, letting the model spend more compute on difficult tokens and less on easy ones. Training uses parallel computation of all thought tokens per macro-step with a staircase attention mask. Inference runs sequentially and can early-exit when the time head predicts near-complete budget consumption.

## Project Structure

```
tida_project/
  config.py        — TIDAConfig dataclass + MODEL_PRESETS + load_config()
  modeling_tida.py — TIDAModel, TIDATimeHead, RoPE patch, loss computation
  dataset.py       — TIDADataset (chunked LM corpus), collate fn
  trainer.py       — TIDATrainer (curriculum, budget ramp, checkpointing)
  inference.py     — generate_tida — adaptive generation with KV pruning
  experiments.py   — Experiment runner (12 experiments × 3 seeds)
  main.py          — Entry point for single runs
results/
  experiments.json — Aggregate results across seeds
  samples/         — Generation samples with adaptivity metrics
colab_run.ipynb   — Colab notebook for GPU runs
```

## Setup

```bash
pip install torch transformers accelerate peft datasets tqdm
```

## Usage

**Single training run:**
```bash
# Default: TinyLlama-1.1B, 3 epochs
python tida_project/main.py

# Quick verification with SmolLM-135M
python tida_project/main.py --verify

# Specify model
python tida_project/main.py --model tiny       # TinyLlama (default)
python tida_project/main.py --model small      # Phi-3-mini
python tida_project/main.py --model medium     # Mistral-7B
```

**Experiment suite:**
```bash
cd tida_project

# Preview
python experiments.py --dry-run

# Full suite (12 experiments × 3 seeds)
python experiments.py

# TinyLlama only
python experiments.py --name _tiny

# Single experiment
python experiments.py --name baseline_k0_tiny
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

For each macro-step (one real token), the model runs **K micro-steps**. Each thought slot k has its own learned embedding:

| Step | Token | Description |
|------|-------|-------------|
| k=0 | Real token embedding | Base input |
| k=1..K | Learned per-step embedding | Each slot has its own `blank_embedding[:, k, :]` |

A **time head** (2-layer MLP + hardsigmoid) predicts a consumption fraction p_k ∈ [0,1]. The budget b_k starts at 1.0 and decrements: t_{k+1} = t_k + b_k · p_k. The weighted logits `exp(t_k) · logits_k` are integrated via summation, and CE loss is applied on the result. A budget loss `(1 - t_K)²` encourages full budget consumption.

**Training** uses a three-phase curriculum:

| Phase | % Steps | K | Budget weight |
|-------|---------|---|---------------|
| 1 | 0–10% | 0 | 0 |
| 2 | 10–30% | k_min | 0 → target (linear ramp) |
| 3 | 30–100% | random k_min..k_max | Full target |

All K+1 tokens per macro-step are computed in parallel with a staircase causal mask (thoughts can only attend within their own bucket). The KV cache is disabled during training.

**Inference** runs the micro-loop sequentially with KV cache, early-exiting when p_k > 0.99. The KV cache is truncated after each macro-step to remove thought tokens.

## Configuration

Key settings in `TIDAConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_model_name` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Base LM |
| `lora_r` / `lora_alpha` / `lora_dropout` | 64 / 128 / 0.05 | LoRA params |
| `k_min` / `k_max` | 2 / 6 | Thought step range |
| `lambda_budget` | 1.0 | Budget loss weight |
| `lambda_kl` | 0.0 | KL regularization (disabled by default) |
| `fractional_positions` | True | Fractional vs integer position encoding |
| `max_seq_len` | 256 | Must respect `L·(K+1) < model context` |
| `batch_size` / `gradient_accumulation_steps` | 2 / 4 | Effective batch = 8 |
| `warmup_steps` | 500 | Linear LR warmup |
| `num_epochs` | 3 | Full training epochs |

## Model Presets

| Preset | Model | Params | Use |
|--------|-------|--------|-----|
| `verify` | SmolLM-135M | 135M | CI / smoke test |
| `tiny` | TinyLlama-1.1B | 1.1B | Default — experiments & eval |
| `small` | Phi-3-mini | 3.8B | Stronger baseline |
| `medium` | Mistral-7B | 7B | Full-scale comparison |

## Checkpoints

Saved to `./checkpoints/epoch_{N}/`:
- `lora/` — LoRA adapter weights
- `time_head.pt` — TIDATimeHead state dict
- `blank_emb.pt` — Learned blank embedding tensor
- `config_meta.json` — Training config (fractional_positions, k_max, etc.)

Loading automatically restores the config metadata so checkpoints are self-describing.

## Experiment Suite

12 experiments × 3 seeds = 36 runs:

| Experiment | Models | What it tests |
|------------|--------|---------------|
| `baseline_k0` | tiny, small, medium | Standard LoRA fine-tune (no thoughts) |
| `tida_k6` | tiny, small, medium | Fixed 6 thought tokens |
| `tida_k2` | tiny | Fixed 2 thought tokens |
| `tida_k2-6` | tiny | Variable curriculum 2–6 |
| `tida_integer_pos` | tiny | Integer positions vs fractional |
| `tida_nobudget` | tiny | No budget loss (λ=0) |
| `tida_lambda_0.1` | tiny | Weak budget pressure |
| `tida_rank32` | tiny | Lower LoRA rank (r=32) for limited data |

Each measures: validation/test perplexity, Lambada accuracy, adaptivity (steps/token, budget used), training time, and FLOPs.

## Key Design Decisions

- **Eager attention** — required for custom staircase mask; Flash Attention 2 not compatible.
- **Float32 position IDs** — fractional positions need higher precision than bf16; integer positions skip the RoPE patch.
- **Gradient checkpointing** applied after LoRA wrap — reduces VRAM.
- **Three-phase curriculum** — warmup at K=0, then fixed K=k_min, then random.
- **Budget ramp** — lambda_budget linearly increases over phase 2 to avoid cold-start over-regularization.
- **Chunked dataset** — concatenate corpus with EOS, chunk into fixed blocks. Every position predicts a real token.
