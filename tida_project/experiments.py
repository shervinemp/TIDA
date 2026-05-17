"""
TIDA Experiment Suite

Covers: baselines, multiple datasets, inference analysis, compute cost,
statistical rigor, ablations, and qualitative analysis.

Usage:
    python experiments.py --dry-run              # Preview
    python experiments.py                        # Run all experiments
    python experiments.py --name baseline_k0     # Run one experiment
    python experiments.py --name tida_k6 --seeds 5   # Override seeds
"""

import argparse
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

from config import load_config, MODEL_PRESETS

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / "experiments.json"
INFERENCE_DIR = RESULTS_DIR / "samples"
INFERENCE_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["tiny", "small", "medium"]

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENT_SUITE = [
    # === Baselines (all models) ===
    {
        "name": "baseline_k0_{m}",
        "description": "No thought tokens — standard LM fine-tune",
        "preset": "{m}",
        "overrides": {"k_min": 0, "k_max": 0},
    },

    # === Fixed K (all models) ===
    {
        "name": "tida_k6_{m}",
        "description": "Fixed 6 thought tokens",
        "preset": "{m}",
        "overrides": {"k_min": 6, "k_max": 6},
    },

    # === Fixed K=2, curriculum, ablations (Tiny only) ===
    {
        "name": "tida_k2_tiny",
        "description": "Fixed 2 thought tokens",
        "preset": "tiny",
        "overrides": {"k_min": 2, "k_max": 2},
    },
    {
        "name": "tida_k2-6_tiny",
        "description": "Variable 2-6 thought tokens (curriculum)",
        "preset": "tiny",
        "overrides": {"k_min": 2, "k_max": 6},
    },
    {
        "name": "tida_integer_pos_k6_tiny",
        "description": "Integer positions, not fractional",
        "preset": "tiny",
        "overrides": {"k_min": 6, "k_max": 6, "fractional_positions": False},
    },
    {
        "name": "tida_nobudget_k2-6_tiny",
        "description": "No budget loss (lambda=0)",
        "preset": "tiny",
        "overrides": {"k_min": 2, "k_max": 6, "lambda_budget": 0.0},
    },
    {
        "name": "tida_lambda_0.1_tiny",
        "description": "Weak budget pressure",
        "preset": "tiny",
        "overrides": {"k_min": 2, "k_max": 6, "lambda_budget": 0.1},
    },
    {
        "name": "tida_lambda_2.0_tiny",
        "description": "Strong budget pressure",
        "preset": "tiny",
        "overrides": {"k_min": 2, "k_max": 6, "lambda_budget": 2.0},
    },
    {
        "name": "tida_rank32_k2-6_tiny",
        "description": "Lower LoRA rank (r=32) for better generalization with limited data",
        "preset": "tiny",
        "overrides": {"k_min": 2, "k_max": 6, "lora_r": 32, "lora_alpha": 64},
    },
]


def expand_suite(suite, models):
    expanded = []
    for exp in suite:
        if "{m}" in exp["name"]:
            for m in models:
                row = {**exp, "name": exp["name"].format(m=m), "preset": exp["preset"].format(m=m)}
                expanded.append(row)
        else:
            expanded.append(dict(exp))
    return expanded


# ---------------------------------------------------------------------------
# FLOP estimation
# ---------------------------------------------------------------------------

def estimate_flops(config) -> dict:
    L = config.max_seq_len
    K = config.k_max
    total_tokens = L * (K + 1)
    attn_flops = 2 * (total_tokens ** 2)
    mlp_flops = 2 * total_tokens * 4096
    return {
        "total_tokens_per_fwd": total_tokens,
        "attn_flops_approx": attn_flops,
        "mlp_flops_approx": mlp_flops,
        "total_flops_approx": attn_flops + mlp_flops,
    }


# ---------------------------------------------------------------------------
# Inference analysis
# ---------------------------------------------------------------------------

def analyze_adaptivity(model, tokenizer, config, device, prompts, max_micro_k=None):
    import torch

    if max_micro_k is None:
        max_micro_k = config.k_max

    model.eval()
    results = []

    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        if input_ids.shape[1] > 1:
            context_ids = input_ids[:, :-1]
            outputs = model.base_model(input_ids=context_ids, use_cache=True)
            past_kv = outputs.past_key_values
            generated = input_ids
        else:
            past_kv = None
            generated = input_ids

        steps = []
        budgets = []
        times = []

        for _ in range(30):
            current_input = generated[:, -1:]
            t = 0.0
            budget = 1.0

            bucket_start_kv = past_kv
            current_bucket_kv = bucket_start_kv

            for k in range(max_micro_k):
                seq_len_so_far = generated.shape[1]
                if config.fractional_positions:
                    pos_ids = torch.tensor([[seq_len_so_far - 1 + t]], device=device, dtype=torch.float32)
                else:
                    pos_ids = torch.tensor([[seq_len_so_far - 1 + k]], device=device, dtype=torch.long)
                embeds = (
                    model.base_model.get_input_embeddings()(current_input)
                    if k == 0
                    else model.blank_embedding[:, k - 1 : k, :]
                )
                out = model.base_model(
                    inputs_embeds=embeds,
                    position_ids=pos_ids,
                    past_key_values=current_bucket_kv,
                    use_cache=True,
                    output_hidden_states=True,
                )
                p = model.time_head(out.hidden_states[-1])
                delta = budget * p
                t = t + delta.item()
                budget = budget - delta.item()
                current_bucket_kv = out.past_key_values
                if p > 0.99:
                    steps.append(k + 1)
                    break
            else:
                steps.append(max_micro_k)

            budgets.append(1.0 - budget)
            times.append(t)

            logits = model.base_model.get_output_embeddings()(out.hidden_states[-1])
            next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=1)

            past_kv = current_bucket_kv
            if past_kv is not None:
                target_len = generated.shape[1] - 1
                if hasattr(past_kv, "key_cache"):
                    for idx in range(len(past_kv.key_cache)):
                        past_kv.key_cache[idx] = past_kv.key_cache[idx][..., :target_len, :]
                        past_kv.value_cache[idx] = past_kv.value_cache[idx][..., :target_len, :]

        results.append({
            "prompt": prompt,
            "avg_steps": sum(steps) / len(steps),
            "avg_budget_used": sum(budgets) / len(budgets),
            "avg_t_K": sum(times) / len(times),
            "generated": tokenizer.decode(generated[0]),
        })

    model.train()
    return results


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------

def seed_worker(worker_id):
    import random
    random.seed(worker_id)


def run_experiment(exp: dict, seed: int) -> dict:
    import torch
    import random
    import numpy as np
    from transformers import AutoTokenizer
    from modeling_tida import TIDAModel
    from dataset import TIDADataset, get_collate_fn
    from trainer import TIDATrainer

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    name = exp["name"]
    preset = exp.get("preset", "tiny")
    print(f"\n{'='*60}")
    print(f"Experiment: {name}  |  seed: {seed}")
    print(f"  {exp.get('description', '')}")
    print(f"{'='*60}")

    config = load_config(preset, **exp["overrides"])
    print(f"  model: {config.base_model_name}  k=[{config.k_min}, {config.k_max}]")
    print(f"  max_seq_len={config.max_seq_len}  lambda={config.lambda_budget}")

    flop_info = estimate_flops(config)
    print(f"  ~{flop_info['total_tokens_per_fwd']} tokens/fwd  ~{flop_info['total_flops_approx']/1e6:.0f}M FLOPs")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Train dataset ---
    print("Loading training data...")
    train_data = TIDADataset(tokenizer, "wikitext", config.max_seq_len, split="train")

    # Shared-embedding ablation: override model init
    shared_emb = exp.get("shared_embedding", False)

    collate_fn = get_collate_fn(tokenizer.pad_token_id)
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        generator=g,
        worker_init_fn=seed_worker,
    )

    print("Initializing model...")
    if shared_emb:
        model = TIDAModel(config)
        with torch.no_grad():
            model.blank_embedding.data = model.blank_embedding.data[:, 0:1, :].expand(
                -1, config.k_max, -1
            ).clone()
    else:
        model = TIDAModel(config)

    # --- Validation datasets ---
    val_wikitext = TIDADataset(tokenizer, "wikitext", config.max_seq_len, split="validation")
    val_loader = torch.utils.data.DataLoader(
        val_wikitext, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=False
    )

    print("Starting training...")
    start = time.time()

    # Wrap model blank embeddings before training if using shared_emb
    trainer = TIDATrainer(model, train_loader, config, tokenizer, val_loader=val_loader)
    trainer.train()
    elapsed = time.time() - start

    device = next(model.parameters()).device

    # --- Perplexity evaluations ---
    metrics = {"train_time_s": round(elapsed, 1), "seed": seed}
    metrics.update(flop_info)

    def eval_ppl(loader, name):
        import torch
        model.eval()
        total = 0.0
        steps = 0
        with torch.no_grad():
            for batch_inputs, batch_labels in loader:
                loss = model(input_ids=batch_inputs, labels=batch_labels, k_steps=config.k_min)
                total += loss.item()
                steps += 1
        avg = total / max(steps, 1)
        ppl = float(torch.exp(torch.tensor(avg)).item())
        metrics[f"val_ppl_{name}"] = round(ppl, 4)
        metrics[f"val_loss_{name}"] = round(avg, 4)
        print(f"  {name}: loss={avg:.4f}  ppl={ppl:.2f}")
        model.train()

    eval_ppl(val_loader, "wikitext2")

    # Held-out test set (Wikitext-2 test split)
    try:
        test_data = TIDADataset(tokenizer, "wikitext", config.max_seq_len, split="test")
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=False
        )
        eval_ppl(test_loader, "wikitext2_test")
    except Exception as e:
        print(f"  Wikitext-2 test not available: {e}")

    # Lambada: last-token accuracy (cloze task)
    try:
        from datasets import load_dataset

        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            lambada_ds = load_dataset("lambada", split="test")
            for item in lambada_ds:
                text = item["text"]
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=config.max_seq_len)
                input_ids = enc["input_ids"].to(device)
                if input_ids.shape[1] < 2:
                    continue
                logits = model(input_ids=input_ids, k_steps=0)
                pred = logits[0, -2, :].argmax().item()
                target = input_ids[0, -1].item()
                total += 1
                if pred == target:
                    correct += 1

        acc = correct / max(total, 1)
        metrics["lambada_acc"] = round(acc, 4)
        metrics["lambada_samples"] = total
        print(f"  lambada: acc={acc:.4f}  ({correct}/{total})")
        model.train()
    except Exception as e:
        print(f"  Lambada evaluation failed: {e}")

    # --- Inference adaptivity ---
    prompts = [
        "The future of AI is",
        "In the beginning",
        "The key to understanding quantum mechanics is",
        "Once upon a time",
    ]
    adaptivity = analyze_adaptivity(model, tokenizer, config, device, prompts, max_micro_k=config.k_max)
    metrics["adaptivity"] = {
        "avg_steps": round(sum(a["avg_steps"] for a in adaptivity) / len(adaptivity), 2),
        "avg_budget_used": round(sum(a["avg_budget_used"] for a in adaptivity) / len(adaptivity), 3),
        "avg_t_K": round(sum(a["avg_t_K"] for a in adaptivity) / len(adaptivity), 3),
    }

    # Save generation samples
    sample_path = INFERENCE_DIR / f"{name}_seed{seed}.txt"
    with open(sample_path, "w") as f:
        for a in adaptivity:
            f.write(f"Prompt: {a['prompt']}\n")
            f.write(f"Generated: {a['generated']}\n")
            f.write(f"  avg_steps={a['avg_steps']:.1f}  budget_used={a['avg_budget_used']:.2f}\n\n")
    print(f"  Samples saved to {sample_path}")

    # Clean up intermediate checkpoints — keep only the final epoch
    for ep in range(config.num_epochs - 1):
        ckpt = f"./checkpoints/epoch_{ep}"
        if os.path.exists(ckpt):
            shutil.rmtree(ckpt, ignore_errors=True)

    metrics["status"] = "completed"
    return metrics


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarize_results(all_results: dict):
    exp_groups = defaultdict(list)
    for key, data in all_results.items():
        base_name = key.rsplit("_seed", 1)[0]
        exp_groups[base_name].append(data)

    summary = {}
    for base, runs in exp_groups.items():
        completed = [r for r in runs if r.get("status") == "completed"]
        if not completed:
            continue

        def agg(key):
            vals = [r[key] for r in completed if key in r]
            if not vals:
                return None
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0.0
            return round(mean, 4), round(std, 4)

        summary[base] = {
            "n_seeds": len(completed),
            "val_ppl_wikitext2": agg("val_ppl_wikitext2"),
            "test_ppl_wikitext2": agg("val_ppl_wikitext2_test"),
            "lambada_acc": agg("lambada_acc"),
            "train_time_s": agg("train_time_s"),
            "adaptivity_avg_steps": agg("adaptivity_avg_steps"),
            "adaptivity_avg_budget_used": agg("adaptivity_avg_budget_used"),
        }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TIDA experiment suite")
    parser.add_argument("--name", help="Run only experiments containing this name")
    parser.add_argument("--list", action="store_true", help="List all available experiment names")
    parser.add_argument("--dry-run", action="store_true", help="Preview experiments")
    parser.add_argument("--force", action="store_true", help="Re-run even if results exist")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds (default: 3)")
    args = parser.parse_args()

    experiments = expand_suite(EXPERIMENT_SUITE, MODELS)

    if args.name:
        experiments = [e for e in experiments if args.name in e["name"]]
        if not experiments:
            print(f"No experiments matching '{args.name}'")
            return

    existing = load_results()

    if args.list:
        print("\nAvailable experiments:")
        for exp in experiments:
            tag = " [done]" if exp["name"] in existing else ""
            print(f"  {exp['name']}{tag}")
        return

    if args.dry_run:
        print(f"\nExperiments ({len(experiments)} total):\n")
        for exp in experiments:
            tag = ""
            if exp["name"] in existing:
                tag = " [done]"
            ov = exp["overrides"]
            shared = " [shared]" if exp.get("shared_embedding") else ""
            print(f"  {exp['name']:35s}  {exp['preset']:8s}  k=[{ov.get('k_min', '?'):>2},{ov.get('k_max', '?'):>2}]"
                  f"  w={ov.get('lambda_budget', 1.0):<4}{shared}{tag}")
        return

    any_failed = False
    for exp in experiments:
        name = exp["name"]
        for seed in range(args.seeds):
            seed_key = f"{name}_seed{seed}"

            if seed_key in existing and existing[seed_key].get("status") == "completed" and not args.force:
                print(f"Skipping {seed_key} (use --force to re-run)")
                continue

            try:
                metrics = run_experiment(exp, seed)
                existing[seed_key] = {**exp, **metrics}
                save_results(existing)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Experiment {seed_key} FAILED: {e}")
                existing[seed_key] = {**exp, "status": "failed", "error": str(e), "seed": seed}
                save_results(existing)
                any_failed = True

    # Print summary table
    summary = summarize_results(existing)
    if summary:
        print(f"\n{'='*80}")
        print(f"Summary — mean ± std across {args.seeds} seeds")
        print(f"{'='*80}")
        header = f"{'Experiment':35s} {'Test PPL':>10s} {'Lmb Acc':>8s} {'Steps':>7s} {'Budget':>8s} {'Time':>8s}"
        print(header)
        print("-" * len(header))
        for base, s in sorted(summary.items()):
            tp = f"{s['test_ppl_wikitext2'][0]:.2f}±{s['test_ppl_wikitext2'][1]:.2f}" if s['test_ppl_wikitext2'] else "N/A"
            la = f"{s['lambada_acc'][0]:.3f}±{s['lambada_acc'][1]:.3f}" if s['lambada_acc'] else "N/A"
            st = f"{s['adaptivity_avg_steps'][0]:.1f}" if s['adaptivity_avg_steps'] else "N/A"
            bu = f"{s['adaptivity_avg_budget_used'][0]:.2f}" if s['adaptivity_avg_budget_used'] else "N/A"
            tm = f"{s['train_time_s'][0]:.0f}s" if s['train_time_s'] else "N/A"
            print(f"{base:35s} {tp:>10s} {la:>8s} {st:>7s} {bu:>8s} {tm:>8s}")

    if any_failed:
        print("\nSome experiments failed. Check results/experiments.json for details.")


def load_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_results(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
