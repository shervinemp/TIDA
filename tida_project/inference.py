import torch
import torch.nn.functional as F

@torch.no_grad()
def generate_tida(model, tokenizer, prompt, max_new_tokens=50, max_micro_k=8):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.base_model.device)

    # Batch size check
    if input_ids.shape[0] > 1:
        print("Warning: generate_tida currently supports adaptive inference for batch_size=1 only. Output might be incorrect for larger batches.")

    # 1. Pre-process prompt to get initial KV cache
    print(f"Generating for prompt: '{prompt}'")

    if input_ids.shape[1] > 1:
        context_ids = input_ids[:, :-1]
        outputs = model.base_model(
            input_ids=context_ids,
            use_cache=True
        )
        past_key_values = outputs.past_key_values
        generated = input_ids
    else:
        past_key_values = None
        generated = input_ids

    # The current input token for the bucket logic is the last token of 'generated'

    for i in range(max_new_tokens):
        # 1. Macro Step Initialization
        current_input = generated[:, -1:]

        # Initialize Bucket State
        t = 0.0
        budget = 1.0
        logit_sum = 0.0

        # Store kv from start of bucket
        bucket_start_kv = past_key_values
        current_bucket_kv = bucket_start_kv

        # Micro Loop
        for k in range(max_micro_k):
            # Calculate P_eff
            seq_len_so_far = generated.shape[1]
            current_pos_ids = torch.tensor([[seq_len_so_far - 1 + t]], device=input_ids.device, dtype=model.dtype)

            # Forward
            if k == 0:
                 embeds = model.base_model.get_input_embeddings()(current_input)
            else:
                 embeds = model.blank_embedding

            outputs = model.base_model(
                inputs_embeds=embeds,
                position_ids=current_pos_ids,
                past_key_values=current_bucket_kv,
                use_cache=True,
                output_hidden_states=True
            )

            # Extract outputs
            last_hidden = outputs.hidden_states[-1]
            next_kv = outputs.past_key_values

            # Physics
            p = F.hardsigmoid(model.time_head(last_hidden))
            delta = budget * p
            t += delta.item()
            budget -= delta.item()

            # Integration
            if hasattr(model.base_model, "lm_head"):
                 lm_head = model.base_model.lm_head
            elif hasattr(model.base_model, "base_model") and hasattr(model.base_model.base_model, "lm_head"):
                 lm_head = model.base_model.base_model.lm_head
            else:
                 lm_head = model.base_model.model.lm_head

            logits = lm_head(last_hidden)

            weight = torch.exp(torch.tensor(t))
            logit_sum += logits * weight

            # Update loop variables
            current_bucket_kv = next_kv

            # Early Exit
            if p > 0.99:
                break

        # Finalize Bucket
        next_token_logits = logit_sum[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        generated = torch.cat([generated, next_token], dim=1)

        # Pruning: Truncate the KV cache to remove thought tokens
        # current_bucket_kv includes [History + Input + Thought_1 + ... + Thought_K]
        # We want to keep [History + Input]
        # Length of generated is now L + 1 (History + Input + NewToken).
        # But 'NewToken' is not in the cache yet.
        # The cache contains processing up to 'Input' (step k=0) and Thoughts.
        # So we want length of 'generated' BEFORE cat? i.e. History + Input.
        # generated.shape[1] includes the new token now.
        # So we want generated.shape[1] - 1.

        past_key_values = current_bucket_kv
        if past_key_values is not None:
             target_len = generated.shape[1] - 1

             # Handle DynamicCache (standard in recent transformers)
             if hasattr(past_key_values, "key_cache"):
                 for idx in range(len(past_key_values.key_cache)):
                     # Truncate time dimension (usually dim 2)
                     past_key_values.key_cache[idx] = past_key_values.key_cache[idx][..., :target_len, :]
                     past_key_values.value_cache[idx] = past_key_values.value_cache[idx][..., :target_len, :]

             # Handle Tuple Cache (older models)
             elif isinstance(past_key_values, tuple):
                 new_kv = []
                 for layer_kv in past_key_values:
                     k_state, v_state = layer_kv
                     k_state = k_state[..., :target_len, :]
                     v_state = v_state[..., :target_len, :]
                     new_kv.append((k_state, v_state))
                 past_key_values = tuple(new_kv)

    return tokenizer.decode(generated[0])
