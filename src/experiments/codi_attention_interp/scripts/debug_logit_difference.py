#!/usr/bin/env python3
"""
Debug script to investigate logit difference computation.

Examines:
1. How answers tokenize
2. Top predicted tokens and their logits
3. Whether logit difference computation is correct
"""

import sys
import torch
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'codi'))
activation_patching_core = project_root / 'src' / 'experiments' / 'activation_patching' / 'core'
sys.path.insert(0, str(activation_patching_core))

from cache_activations_llama import ActivationCacherLLaMA


def debug_answer_prediction(model_path: str):
    """Debug answer prediction and logit difference."""

    print("="*80)
    print("DEBUGGING LOGIT DIFFERENCE")
    print("="*80)

    # Load model
    print(f"\nLoading model from {model_path}...")
    cacher = ActivationCacherLLaMA(model_path)
    model = cacher.model
    tokenizer = cacher.tokenizer
    device = cacher.device

    # Test problem
    problem = "John has 3 bags with 7 apples each. How many apples does he have in total?"
    correct_answer = 21

    print(f"\nProblem: {problem}")
    print(f"Correct answer: {correct_answer}")

    # ========================================
    # 1. EXAMINE TOKENIZATION
    # ========================================
    print("\n" + "="*80)
    print("1. TOKENIZATION ANALYSIS")
    print("="*80)

    # How does the answer tokenize?
    answer_str = str(correct_answer)
    answer_tokens = tokenizer.encode(answer_str, add_special_tokens=False)
    answer_text = [tokenizer.decode([t]) for t in answer_tokens]

    print(f"\nAnswer string: '{answer_str}'")
    print(f"Token IDs: {answer_tokens}")
    print(f"Token text: {answer_text}")
    print(f"Number of tokens: {len(answer_tokens)}")

    # Try with space prefix (common in generation)
    answer_str_space = " " + str(correct_answer)
    answer_tokens_space = tokenizer.encode(answer_str_space, add_special_tokens=False)
    answer_text_space = [tokenizer.decode([t]) for t in answer_tokens_space]

    print(f"\nWith space prefix: ' {correct_answer}'")
    print(f"Token IDs: {answer_tokens_space}")
    print(f"Token text: {answer_text_space}")

    # ========================================
    # 2. GENERATE FULL ANSWER
    # ========================================
    print("\n" + "="*80)
    print("2. FULL ANSWER GENERATION")
    print("="*80)

    model.eval()
    with torch.no_grad():
        inputs = tokenizer(problem, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        input_embd = model.get_embd(model.codi, model.model_name)(input_ids).to(device)

        outputs = model.codi(
            inputs_embeds=input_embd,
            use_cache=True,
            output_hidden_states=True
        )
        past_key_values = outputs.past_key_values

        bot_emb = model.get_embd(model.codi, model.model_name)(
            torch.tensor([model.bot_id], dtype=torch.long, device=device)
        ).unsqueeze(0)

        latent_embd = bot_emb

        # Process latent thoughts
        for _ in range(cacher.num_latent):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if model.use_prj:
                latent_embd = model.prj(latent_embd)

        # EOT token - this is where we get logits for the first answer token
        eot_emb = model.get_embd(model.codi, model.model_name)(
            torch.tensor([model.eot_id], dtype=torch.long, device=device)
        ).unsqueeze(0)

        final_outputs = model.codi(
            inputs_embeds=eot_emb,
            use_cache=True,
            past_key_values=past_key_values
        )

        # Get logits
        final_logits = final_outputs.logits[:, -1, :model.codi.config.vocab_size-1]

        print(f"\nLogits shape: {final_logits.shape}")
        print(f"Vocab size: {model.codi.config.vocab_size-1}")

        # ========================================
        # 3. ANALYZE TOP PREDICTIONS
        # ========================================
        print("\n" + "="*80)
        print("3. TOP PREDICTED TOKENS")
        print("="*80)

        # Get top 20 tokens
        top_k = 20
        top_logits, top_indices = torch.topk(final_logits[0], top_k)

        print(f"\nTop {top_k} predictions:")
        print(f"{'Rank':<6} {'Token ID':<10} {'Logit':<12} {'Token Text'}")
        print("-" * 60)

        for rank, (logit, idx) in enumerate(zip(top_logits, top_indices), 1):
            token_text = tokenizer.decode([idx.item()])
            print(f"{rank:<6} {idx.item():<10} {logit.item():<12.4f} '{token_text}'")

        # ========================================
        # 4. ANALYZE CORRECT ANSWER TOKENS
        # ========================================
        print("\n" + "="*80)
        print("4. CORRECT ANSWER TOKEN ANALYSIS")
        print("="*80)

        print(f"\nAnalyzing logits for correct answer: {correct_answer}")

        # Check all possible tokenizations
        for prefix, tokens, desc in [
            ("", answer_tokens, "no prefix"),
            (" ", answer_tokens_space, "space prefix")
        ]:
            print(f"\n{desc.upper()}: '{prefix}{correct_answer}'")
            print(f"Tokens: {tokens}")

            for i, token_id in enumerate(tokens):
                logit_value = final_logits[0, token_id].item()
                rank = (final_logits[0] > logit_value).sum().item() + 1
                token_text = tokenizer.decode([token_id])
                print(f"  Token {i}: '{token_text}' (ID {token_id})")
                print(f"    Logit: {logit_value:.4f}")
                print(f"    Rank: {rank}/{final_logits.shape[1]}")

        # ========================================
        # 5. LOGIT DIFFERENCE COMPUTATION
        # ========================================
        print("\n" + "="*80)
        print("5. LOGIT DIFFERENCE ANALYSIS")
        print("="*80)

        # First token of answer (no space)
        correct_token_id = answer_tokens[0]
        correct_logit = final_logits[0, correct_token_id].item()

        # Max incorrect (excluding correct)
        logits_copy = final_logits.clone()
        logits_copy[0, correct_token_id] = float('-inf')
        max_incorrect_logit = torch.max(logits_copy).item()
        max_incorrect_idx = torch.argmax(logits_copy).item()
        max_incorrect_text = tokenizer.decode([max_incorrect_idx])

        logit_diff = correct_logit - max_incorrect_logit

        print(f"\nCorrect token: '{tokenizer.decode([correct_token_id])}' (ID {correct_token_id})")
        print(f"  Logit: {correct_logit:.4f}")

        print(f"\nMax incorrect token: '{max_incorrect_text}' (ID {max_incorrect_idx})")
        print(f"  Logit: {max_incorrect_logit:.4f}")

        print(f"\nLogit difference: {logit_diff:.4f}")
        print(f"  Interpretation: {'Model prefers CORRECT' if logit_diff > 0 else 'Model prefers INCORRECT'}")

        # ========================================
        # 6. GENERATE ACTUAL OUTPUT
        # ========================================
        print("\n" + "="*80)
        print("6. ACTUAL GENERATION")
        print("="*80)

        # Continue generation to see what model actually produces
        output_emb = eot_emb
        pred_tokens = []

        for step in range(20):  # Generate 20 tokens max
            out = model.codi(
                inputs_embeds=output_emb,
                use_cache=True,
                past_key_values=past_key_values
            )

            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :model.codi.config.vocab_size-1]

            # Greedy decoding
            next_token_id = torch.argmax(logits, dim=-1)

            if next_token_id.item() == tokenizer.eos_token_id:
                break

            pred_tokens.append(next_token_id.item())
            output_emb = model.get_embd(model.codi, model.model_name)(
                next_token_id
            ).unsqueeze(1)

        generated_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        print(f"\nGenerated text: '{generated_text}'")

        # Show first few tokens
        print(f"\nFirst 10 tokens of generation:")
        for i, token_id in enumerate(pred_tokens[:10]):
            token_text = tokenizer.decode([token_id])
            print(f"  {i}: '{token_text}' (ID {token_id})")


if __name__ == "__main__":
    model_path = str(Path.home() / 'codi_ckpt' / 'llama_gsm8k')
    debug_answer_prediction(model_path)
