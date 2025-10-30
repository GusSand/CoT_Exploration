"""
Smoking Gun Test - Qualitative CT Token Analysis

Tests the interpretation that CT tokens contain problem-specific information
by decoding them and examining contamination effects.

Expected Results (if interpretation is correct):
- CT4 should decode to problem-specific numbers/entities
- CT5 should decode to generic computation tokens
- Swapping CT4 should contaminate with problem B's numbers
- Swapping CT5 should not contaminate (or contaminate less)
"""

import sys
import os
import torch
import pickle
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_model, extract_answer, set_seed

# Import swapping function
import importlib.util
spec = importlib.util.spec_from_file_location("swapping", "2_implement_swapping.py")
swapping_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(swapping_module)
generate_with_swapped_ct = swapping_module.generate_with_swapped_ct


def decode_ct_token(model, tokenizer, ct_hidden_state, top_k=10):
    """
    Decode a CT token's hidden state to see what tokens it's "thinking about".

    Args:
        model: CODI model
        tokenizer: Tokenizer
        ct_hidden_state: Hidden state tensor [2048]
        top_k: Number of top tokens to return

    Returns:
        Dict with top tokens and their probabilities
    """
    # Get the LM head (output projection) from the underlying language model
    lm_head = model.codi.get_output_embeddings()  # More robust than accessing lm_head directly

    # Project hidden state to vocabulary logits
    with torch.no_grad():
        # ct_hidden_state is [2048], need [1, 2048]
        hidden = ct_hidden_state.unsqueeze(0).to(next(model.parameters()).device)
        logits = lm_head(hidden)  # [1, vocab_size]

        # Get probabilities
        probs = torch.softmax(logits, dim=-1)

        # Get top-k tokens
        top_probs, top_indices = torch.topk(probs[0], k=top_k)

        # Decode tokens
        tokens = [tokenizer.decode([idx.item()]) for idx in top_indices]
        probs_list = top_probs.cpu().tolist()

    return {
        'tokens': tokens,
        'probabilities': probs_list,
        'top_token': tokens[0],
        'top_prob': probs_list[0]
    }


def analyze_problem_pair(model, tokenizer, problem_A, problem_B):
    """
    Comprehensive analysis of a problem pair.

    Args:
        problem_A: First problem dict from cache
        problem_B: Second problem dict from cache
    """
    print("="*80)
    print("SMOKING GUN TEST - QUALITATIVE ANALYSIS")
    print("="*80)
    print()

    # Show problems
    print("="*80)
    print("PROBLEM A")
    print("="*80)
    print(f"Question: {problem_A['question']}")
    print(f"Gold answer: {problem_A['gold_numeric']}")
    baseline_A = extract_answer(problem_A['baseline_prediction'])
    print(f"Baseline prediction: {baseline_A}")
    print(f"Baseline correct: {'✓' if baseline_A == problem_A['gold_numeric'] else '✗'}")
    print()

    print("="*80)
    print("PROBLEM B")
    print("="*80)
    print(f"Question: {problem_B['question']}")
    print(f"Gold answer: {problem_B['gold_numeric']}")
    baseline_B = extract_answer(problem_B['baseline_prediction'])
    print(f"Baseline prediction: {baseline_B}")
    print(f"Baseline correct: {'✓' if baseline_B == problem_B['gold_numeric'] else '✗'}")
    print()

    # Decode all CT positions for both problems
    print("="*80)
    print("DECODING CT TOKENS")
    print("="*80)
    print()

    for position in range(6):
        print(f"\n{'='*40}")
        print(f"CT{position}")
        print(f"{'='*40}")

        # Decode from problem A
        ct_A = problem_A['ct_hidden_states'][position]
        decoded_A = decode_ct_token(model, tokenizer, ct_A, top_k=10)

        # Decode from problem B
        ct_B = problem_B['ct_hidden_states'][position]
        decoded_B = decode_ct_token(model, tokenizer, ct_B, top_k=10)

        print(f"\nProblem A (gold={problem_A['gold_numeric']}):")
        print(f"  Top 10 tokens: {decoded_A['tokens']}")
        print(f"  Top 3 probs: {[f'{p:.3f}' for p in decoded_A['probabilities'][:3]]}")

        print(f"\nProblem B (gold={problem_B['gold_numeric']}):")
        print(f"  Top 10 tokens: {decoded_B['tokens']}")
        print(f"  Top 3 probs: {[f'{p:.3f}' for p in decoded_B['probabilities'][:3]]}")

        # Check for overlap
        overlap = set(decoded_A['tokens'][:5]) & set(decoded_B['tokens'][:5])
        print(f"\n  Overlap in top 5: {overlap if overlap else 'None'}")

        # Check for problem-specific numbers
        gold_A_str = str(problem_A['gold_numeric'])
        gold_B_str = str(problem_B['gold_numeric'])

        has_A_answer = any(gold_A_str in token for token in decoded_A['tokens'][:5])
        has_B_answer = any(gold_B_str in token for token in decoded_B['tokens'][:5])

        if has_A_answer:
            print(f"  ⚡ Problem A: Contains gold answer '{gold_A_str}' in top 5!")
        if has_B_answer:
            print(f"  ⚡ Problem B: Contains gold answer '{gold_B_str}' in top 5!")

    # Now test swapping
    print("\n" + "="*80)
    print("SWAPPING TEST - ALL POSITIONS")
    print("="*80)
    print()

    for position in range(6):
        print(f"\n--- Swapping CT{position} ---")

        # Generate with swap
        set_seed(42)
        swapped_output = generate_with_swapped_ct(
            model, tokenizer, problem_A, problem_B, swap_position=position
        )
        swapped_answer = extract_answer(swapped_output)

        print(f"Baseline A: {baseline_A}")
        print(f"Baseline B: {baseline_B}")
        print(f"Swapped: {swapped_answer}")

        # Analyze contamination
        if swapped_answer == baseline_A:
            print(f"Result: ✓ No contamination (answer unchanged)")
        elif swapped_answer == baseline_B:
            print(f"Result: ⚠️ FULL CONTAMINATION (now produces B's answer!)")
        elif swapped_answer == problem_B['gold_numeric']:
            print(f"Result: ⚡ EXTREME CONTAMINATION (now produces B's GOLD answer!)")
        else:
            print(f"Result: ⚠️ Partial contamination (changed to {swapped_answer})")

        # Check if swapped output contains B's gold answer
        if str(problem_B['gold_numeric']) in swapped_output:
            print(f"⚡ Output contains B's gold answer ({problem_B['gold_numeric']})")

    # Summary
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print()

    print("Expected if CT4/CT5 contain problem-specific information:")
    print("  - CT4, CT5 should decode to numbers/entities from the problem")
    print("  - Swapping CT4/CT5 should contaminate the answer")
    print()

    print("Expected if CT0-CT3 are more generic:")
    print("  - CT0-CT3 might decode to generic computation tokens")
    print("  - Swapping CT0-CT3 might have less contamination effect")
    print()


def main():
    print("\n" + "="*80)
    print("SMOKING GUN TEST - CT TOKEN DECODING")
    print("="*80)
    print()

    # Load cache
    cache_file = '../data/ct_hidden_states_cache_pilot.pkl'
    print(f"Loading cache from: {cache_file}")
    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)
    print(f"✓ Loaded {len(cache)} problems")
    print()

    # Load model
    print("Loading model...")
    model, tokenizer = load_model()
    print("✓ Model loaded")
    print()

    # Test on multiple problem pairs
    test_pairs = [
        (0, 1),   # First pair
        (4, 5),   # Pair that showed high contamination in diagnostics
        (3, 4),   # Pair that showed no contamination in diagnostics
    ]

    for i, (idx_A, idx_B) in enumerate(test_pairs):
        print(f"\n{'='*80}")
        print(f"TEST PAIR {i+1}: Problems {idx_A} and {idx_B}")
        print(f"{'='*80}\n")

        analyze_problem_pair(model, tokenizer, cache[idx_A], cache[idx_B])

        print("\n" + "="*80)
        print(f"END OF TEST PAIR {i+1}")
        print("="*80 + "\n\n")


if __name__ == '__main__':
    main()
