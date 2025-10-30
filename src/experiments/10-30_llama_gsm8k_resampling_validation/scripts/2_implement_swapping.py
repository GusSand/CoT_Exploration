#!/usr/bin/env python3
"""
Story 1.3: Implement CT Token Swapping Function

Implement and validate CT token swapping for resampling experiment.

Architecture:
- Load cached CT hidden states
- Implement swapping function using KV cache manipulation
- Run validation tests (no-swap, high-impact, low-impact positions)

Time estimate: 3.5 hours (3h coding + 30m testing)

Usage:
    python 2_implement_swapping.py --cache_file ../data/ct_hidden_states_cache_pilot.pkl
"""

# CRITICAL: Set PYTHONHASHSEED before imports
import os
os.environ['PYTHONHASHSEED'] = '42'

import torch
import pickle
import argparse
from pathlib import Path
from typing import Dict
import sys

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import set_seed, load_model, extract_answer


def generate_with_swapped_ct(
    model,
    tokenizer,
    problem_A_cache: dict,
    problem_B_cache: dict,
    swap_position: int,  # 0-5
    device: str = "cuda",
    max_new_tokens: int = 100
) -> str:
    """
    Generate answer for problem A with CT position swapped from problem B.

    Architecture (from architecture spec):
    - Generate CT tokens 0 through 5
    - At swap_position: INJECT hidden state from problem B
    - This "contaminates" the reasoning chain
    - Measure if final answer changes

    Args:
        model: CODI model
        tokenizer: Model tokenizer
        problem_A_cache: Cached data for problem A
        problem_B_cache: Cached data for problem B
        swap_position: Which CT position to swap (0-5)
        device: 'cuda' or 'cpu'
        max_new_tokens: Max answer tokens

    Returns:
        Generated answer text (str)
    """

    with torch.no_grad():
        # Step 1: Question forward pass (problem A)
        inputs = tokenizer(problem_A_cache['question'], return_tensors='pt').to(device)
        input_ids = inputs['input_ids']

        question_outputs = model.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True
        )
        past_key_values = question_outputs.past_key_values

        # Step 2: BOT token
        bot_emb = model.get_embd(model.codi, model.model_name)(
            torch.tensor([model.bot_id], dtype=torch.long, device=device)
        ).unsqueeze(0)

        latent_embd = bot_emb

        # Step 3: Generate CT tokens with swap
        for step in range(6):
            # BUGFIX: Check if we should swap THIS step BEFORE forward pass
            if step == swap_position:
                # SWAP: Use problem B's hidden state
                # Cache stores [6, 2048], need [1, 1, 2048] for inputs_embeds
                latent_embd = problem_B_cache['ct_hidden_states'][step].unsqueeze(0).unsqueeze(0).to(device)
                # Apply projection (model architecture requirement)
                if model.use_prj:
                    latent_embd = model.prj(latent_embd)

            # Forward pass with chosen embedding (either swapped or from previous step)
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values

            # Prepare embedding for NEXT iteration (only if not swapping next time)
            if step + 1 < 6:  # Not the last iteration
                # NORMAL: Use generated hidden state for next iteration
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                # Apply projection
                if model.use_prj:
                    latent_embd = model.prj(latent_embd)

        # Step 4: EOT + answer generation
        eot_emb = model.get_embd(model.codi, model.model_name)(
            torch.tensor([model.eot_id], dtype=torch.long, device=device)
        ).unsqueeze(0)

        output_emb = eot_emb
        pred_tokens = []

        for _ in range(max_new_tokens):
            out = model.codi(
                inputs_embeds=output_emb,
                use_cache=True,
                past_key_values=past_key_values
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :model.codi.config.vocab_size-1]
            next_token_id = torch.argmax(logits, dim=-1)

            if next_token_id.item() == tokenizer.eos_token_id:
                break

            pred_tokens.append(next_token_id.item())
            output_emb = model.get_embd(model.codi, model.model_name)(
                next_token_id
            ).unsqueeze(1)

        generated_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()

        return generated_text


def validate_swapping_function(model, tokenizer, cache, device='cuda'):
    """
    Test suite to validate CT swapping implementation.

    Tests:
    1. No-swap baseline (swap from same problem)
    2. High-impact position swap (CT5 - 26% ablation impact)
    3. Low-impact position swap (CT4 - 3-4% impact)
    4. Position variance (different positions produce different outputs)
    5. Self-swap (swap position from same problem A→A)

    Args:
        model: CODI model
        tokenizer: Model tokenizer
        cache: Loaded cache dict
        device: Device

    Returns:
        Dict with test results
    """
    results = {}

    print(f"\n{'='*60}")
    print("VALIDATION TESTS")
    print(f"{'='*60}\n")

    # Get test problems
    problem_A = cache[0]
    problem_B = cache[1]
    problem_C = cache[2]

    # Test 1: Self-swap (swap from same problem)
    print("Test 1: Self-swap baseline (A→A at CT0)")
    answer_self_swap = generate_with_swapped_ct(
        model, tokenizer, problem_A, problem_A, swap_position=0, device=device
    )
    answer_baseline = problem_A['baseline_prediction']

    # Extract numeric answers
    numeric_self = extract_answer(answer_self_swap)
    numeric_baseline = extract_answer(answer_baseline)

    match = (numeric_self == numeric_baseline)
    results['test_1_self_swap'] = {
        'passed': match,
        'baseline_answer': numeric_baseline,
        'self_swap_answer': numeric_self,
        'note': 'Swapping from same problem should reproduce baseline'
    }

    print(f"  Baseline: {numeric_baseline}")
    print(f"  Self-swap: {numeric_self}")
    print(f"  Match: {match} {'✓' if match else '✗ WARNING'}\n")

    # Test 2: High-impact position swap (CT5)
    print("Test 2: High-impact position swap (A→B at CT5)")
    answer_ct5_swap = generate_with_swapped_ct(
        model, tokenizer, problem_A, problem_B, swap_position=5, device=device
    )

    numeric_ct5 = extract_answer(answer_ct5_swap)
    changed = (numeric_ct5 != numeric_baseline)

    results['test_2_ct5_swap'] = {
        'passed': changed,
        'baseline_answer': numeric_baseline,
        'swapped_answer': numeric_ct5,
        'changed': changed,
        'note': 'CT5 swap should likely change answer (26% ablation impact)'
    }

    print(f"  Baseline: {numeric_baseline}")
    print(f"  CT5 swap: {numeric_ct5}")
    print(f"  Changed: {changed} {'✓' if changed else '(ok - may not always change)'}\n")

    # Test 3: Low-impact position swap (CT4)
    print("Test 3: Low-impact position swap (A→B at CT4)")
    answer_ct4_swap = generate_with_swapped_ct(
        model, tokenizer, problem_A, problem_B, swap_position=4, device=device
    )

    numeric_ct4 = extract_answer(answer_ct4_swap)
    ct4_changed = (numeric_ct4 != numeric_baseline)

    results['test_3_ct4_swap'] = {
        'baseline_answer': numeric_baseline,
        'swapped_answer': numeric_ct4,
        'changed': ct4_changed,
        'note': 'CT4 swap may or may not change (3-4% impact expected)'
    }

    print(f"  Baseline: {numeric_baseline}")
    print(f"  CT4 swap: {numeric_ct4}")
    print(f"  Changed: {ct4_changed} (ok either way)\n")

    # Test 4: Position variance (different positions produce different contamination)
    print("Test 4: Position variance (A→B at CT0 vs CT2)")
    answer_ct0 = generate_with_swapped_ct(
        model, tokenizer, problem_A, problem_B, swap_position=0, device=device
    )
    answer_ct2 = generate_with_swapped_ct(
        model, tokenizer, problem_A, problem_B, swap_position=2, device=device
    )

    numeric_ct0 = extract_answer(answer_ct0)
    numeric_ct2 = extract_answer(answer_ct2)

    # They don't need to be different (both could contaminate to same answer)
    # but if they are, that shows position-specific contamination
    ct0_vs_ct2_different = (numeric_ct0 != numeric_ct2)

    results['test_4_position_variance'] = {
        'ct0_answer': numeric_ct0,
        'ct2_answer': numeric_ct2,
        'different': ct0_vs_ct2_different,
        'note': 'Different swap positions may produce different contamination'
    }

    print(f"  CT0 swap: {numeric_ct0}")
    print(f"  CT2 swap: {numeric_ct2}")
    print(f"  Different: {ct0_vs_ct2_different}\n")

    # Test 5: Multiple swaps produce consistent results
    print("Test 5: Reproducibility (run CT5 swap twice)")
    answer_ct5_run1 = generate_with_swapped_ct(
        model, tokenizer, problem_A, problem_B, swap_position=5, device=device
    )
    answer_ct5_run2 = generate_with_swapped_ct(
        model, tokenizer, problem_A, problem_B, swap_position=5, device=device
    )

    numeric_ct5_run1 = extract_answer(answer_ct5_run1)
    numeric_ct5_run2 = extract_answer(answer_ct5_run2)

    reproducible = (numeric_ct5_run1 == numeric_ct5_run2)

    results['test_5_reproducibility'] = {
        'passed': reproducible,
        'run1_answer': numeric_ct5_run1,
        'run2_answer': numeric_ct5_run2,
        'note': 'Same swap should produce identical results (deterministic)'
    }

    print(f"  Run 1: {numeric_ct5_run1}")
    print(f"  Run 2: {numeric_ct5_run2}")
    print(f"  Reproducible: {reproducible} {'✓' if reproducible else '✗ ERROR'}\n")

    # Summary
    print(f"{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")

    critical_tests = ['test_1_self_swap', 'test_5_reproducibility']
    critical_passed = all(results[t].get('passed', True) for t in critical_tests)

    if critical_passed:
        print("✓ All critical tests passed")
        print("  - Self-swap matches baseline")
        print("  - Swapping is reproducible")
    else:
        print("✗ CRITICAL TESTS FAILED")
        for t in critical_tests:
            if not results[t].get('passed', True):
                print(f"  - {t}: {results[t]['note']}")

    print(f"\nSwapping function validation: {'PASSED' if critical_passed else 'FAILED'}")
    print(f"{'='*60}\n")

    return results, critical_passed


def main():
    parser = argparse.ArgumentParser(description="Implement and validate CT swapping function")
    parser.add_argument('--cache_file', type=str,
                        default='../data/ct_hidden_states_cache_pilot.pkl',
                        help='Path to cached CT hidden states')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set all seeds
    set_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"Story 1.3: CT Token Swapping Implementation")
    print(f"{'='*60}")
    print(f"Cache file: {args.cache_file}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}\n")

    # Load cache
    cache_path = Path(__file__).parent / args.cache_file
    print(f"Loading cache from {cache_path}...")

    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    print(f"✓ Loaded cache with {len(cache)} problems")

    # Load model
    model, tokenizer = load_model(device=args.device)

    # Run validation tests
    results, passed = validate_swapping_function(model, tokenizer, cache, device=args.device)

    # Save results
    results_dir = Path(__file__).parent / '../results'
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / 'swapping_validation_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"Validation results saved to {results_path}\n")

    if not passed:
        print("⚠️  WARNING: Critical tests failed. Review implementation before proceeding.")
        exit(1)
    else:
        print("✓ Swapping function ready for resampling experiment\n")


if __name__ == "__main__":
    main()
