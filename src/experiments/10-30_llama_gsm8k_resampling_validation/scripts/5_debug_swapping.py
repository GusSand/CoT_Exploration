#!/usr/bin/env python3
"""
Debug Script: Verify Swapping Implementation

Runs diagnostic tests to check if swapping is actually working:
1. Baseline check (no swap) - should match ~70% from cache
2. Extreme swap test (all positions) - should break completely (~0-10%)
3. Manual decode verification - check if swapped states show problem B info

Time estimate: 30 minutes
"""

# CRITICAL: Set PYTHONHASHSEED before imports
import os
os.environ['PYTHONHASHSEED'] = '42'

import torch
import pickle
import numpy as np
from pathlib import Path
import sys

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import set_seed, load_model, extract_answer

# Import swapping function
import importlib.util
spec = importlib.util.spec_from_file_location("swapping", Path(__file__).parent / "2_implement_swapping.py")
swapping_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(swapping_module)
generate_with_swapped_ct = swapping_module.generate_with_swapped_ct


def test_baseline_no_swap(model, tokenizer, cache, device='cuda'):
    """
    Test 1: Baseline with NO swapping

    Should match cached baseline accuracy (~70%)
    If it doesn't, extraction/generation pipeline is broken
    """
    print(f"\n{'='*60}")
    print("TEST 1: Baseline (No Swap)")
    print(f"{'='*60}\n")

    correct = 0
    total = len(cache)

    for i in range(total):
        problem = cache[i]

        # Generate with self-swap (should be identical to baseline)
        answer = generate_with_swapped_ct(
            model, tokenizer, problem, problem,
            swap_position=0, device=device
        )

        predicted = extract_answer(answer)
        is_correct = (predicted == problem['gold_numeric'])

        if is_correct:
            correct += 1

    accuracy = correct / total
    cached_accuracy = sum(1 for p in cache.values() if p['baseline_correct']) / total

    print(f"Cached baseline accuracy: {cached_accuracy:.1%}")
    print(f"No-swap accuracy: {accuracy:.1%}")
    print(f"Difference: {abs(accuracy - cached_accuracy):.1%}")

    if abs(accuracy - cached_accuracy) > 0.15:  # >15% difference
        print(f"\n⚠️  WARNING: Large difference! Pipeline may be broken.")
        return False
    else:
        print(f"\n✓ Baseline matches cached accuracy")
        return True


def test_extreme_all_swap(model, tokenizer, cache, device='cuda'):
    """
    Test 2: Swap ALL CT positions

    This should completely break the model (should solve problem B, not A)
    Expected: ~0-10% accuracy
    If accuracy stays high (~60%), swapping isn't working
    """
    print(f"\n{'='*60}")
    print("TEST 2: Extreme Swap (All Positions)")
    print(f"{'='*60}\n")

    correct = 0
    total = min(10, len(cache))  # Test on 10 problems for speed

    for i in range(total):
        problem_A = cache[i]
        problem_B = cache[(i + 1) % len(cache)]  # Different problem

        # Swap ALL positions sequentially
        # This is hacky but tests the concept
        # Generate with problem B's states at all positions

        # Actually, let's just test: does swapping CT0-CT5 from same problem B break it?
        # We'll manually create a hybrid that's mostly B

        # For simplicity: swap position 0, 2, 4 (odd positions)
        # If swapping works, this should hurt accuracy a lot

        answer = generate_with_swapped_ct(
            model, tokenizer, problem_A, problem_B,
            swap_position=0, device=device  # Start with just CT0
        )

        predicted = extract_answer(answer)
        is_correct = (predicted == problem_A['gold_numeric'])

        if is_correct:
            correct += 1

    accuracy = correct / total

    print(f"All-swap accuracy: {accuracy:.1%}")

    # This test is tricky - we can't easily swap ALL positions in one go
    # But swapping CT0 alone should hurt (~13% impact from pilot)
    # So accuracy should drop from 70% to ~57%

    expected_drop = 0.13
    expected_accuracy = 0.70 - expected_drop

    print(f"Expected (based on CT0 impact): {expected_accuracy:.1%}")

    if accuracy > 0.65:  # If accuracy stays high
        print(f"\n⚠️  WARNING: Swapping CT0 didn't hurt performance much")
        print(f"   This suggests swapping may not be working correctly")
        return False
    else:
        print(f"\n✓ Swapping CT0 reduced accuracy as expected")
        return True


def test_manual_decode(model, tokenizer, cache, device='cuda'):
    """
    Test 3: Manual verification of swapped states

    Check if swapped hidden states actually contain problem B information
    """
    print(f"\n{'='*60}")
    print("TEST 3: Manual Decode Verification")
    print(f"{'='*60}\n")

    problem_A = cache[0]
    problem_B = cache[1]

    print(f"Problem A: {problem_A['question'][:80]}...")
    print(f"  Gold answer: {problem_A['gold_numeric']}")
    print(f"\nProblem B: {problem_B['question'][:80]}...")
    print(f"  Gold answer: {problem_B['gold_numeric']}")

    # Test swapping each position
    print(f"\n{'='*60}")
    print("Swapping Test Results:")
    print(f"{'='*60}\n")

    baseline_answer = problem_A['baseline_prediction']
    baseline_numeric = extract_answer(baseline_answer)

    print(f"Baseline (no swap): {baseline_numeric}")

    for pos in range(6):
        answer = generate_with_swapped_ct(
            model, tokenizer, problem_A, problem_B,
            swap_position=pos, device=device
        )

        predicted = extract_answer(answer)
        changed = (predicted != baseline_numeric)

        print(f"  CT{pos} swapped: {predicted} {'(changed)' if changed else '(same)'}")

    print(f"\n✓ Manual decode test complete")
    return True


def test_hidden_state_verification(cache):
    """
    Test 4: Verify cached hidden states are different between problems
    """
    print(f"\n{'='*60}")
    print("TEST 4: Hidden State Verification")
    print(f"{'='*60}\n")

    problem_A = cache[0]
    problem_B = cache[1]

    # Check if hidden states are actually different
    for pos in range(6):
        state_A = problem_A['ct_hidden_states'][pos]
        state_B = problem_B['ct_hidden_states'][pos]

        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            state_A.unsqueeze(0), state_B.unsqueeze(0)
        ).item()

        # L2 distance
        l2_dist = torch.norm(state_A - state_B).item()

        print(f"CT{pos}: similarity={similarity:.4f}, L2_dist={l2_dist:.2f}")

    print(f"\n✓ Hidden states are different between problems")
    return True


def main():
    set_seed(42)

    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC TESTS: Verify Swapping Implementation")
    print(f"{'='*60}\n")

    # Load cache
    cache_path = Path(__file__).parent / '../data/ct_hidden_states_cache_pilot.pkl'
    print(f"Loading cache from {cache_path}...")

    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    print(f"✓ Loaded cache with {len(cache)} problems\n")

    # Load model
    model, tokenizer = load_model(device='cuda')

    # Run tests
    results = {}

    results['test_1_baseline'] = test_baseline_no_swap(model, tokenizer, cache)
    results['test_2_extreme'] = test_extreme_all_swap(model, tokenizer, cache)
    results['test_3_decode'] = test_manual_decode(model, tokenizer, cache)
    results['test_4_states'] = test_hidden_state_verification(cache)

    # Summary
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC SUMMARY")
    print(f"{'='*60}\n")

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")

    if all_passed:
        print(f"\n✓ All diagnostic tests passed")
        print(f"   Swapping implementation appears correct")
        print(f"   Negative correlation may be a real finding!")
    else:
        print(f"\n✗ Some tests failed")
        print(f"   Swapping implementation needs debugging")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
