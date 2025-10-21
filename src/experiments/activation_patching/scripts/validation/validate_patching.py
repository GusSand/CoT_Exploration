"""
Quick validation script to inspect patching behavior.
Tests a few examples with detailed logging to understand negative recovery rates.

Usage:
    python validate_patching.py --model_path ~/codi_ckpt/gpt2_gsm8k
"""

import json
import torch
import argparse
from cache_activations import ActivationCacher
from patch_and_eval import ActivationPatcher
import re


def extract_answer_number(text: str) -> int:
    """Extract numerical answer from text."""
    patterns = [
        r'####\s*(-?\d+)',
        r'(?:answer|total)(?:\s+is)?\s*[:=]?\s*(-?\d+)',
        r'\$?\s*(-?\d+)\s*$',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))

    numbers = re.findall(r'-?\d+', text)
    if numbers:
        return int(numbers[-1])
    return None


def validate_single_pair(pair, cacher, patcher, layer_name='middle'):
    """Validate a single problem pair with detailed logging."""

    pair_id = pair['pair_id']
    clean_q = pair['clean']['question']
    corrupted_q = pair['corrupted']['question']
    clean_ans = pair['clean']['answer']
    corrupted_ans = pair['corrupted']['answer']

    print(f"\n{'='*80}")
    print(f"PAIR {pair_id}")
    print(f"{'='*80}")
    print(f"Clean Q: {clean_q[:100]}...")
    print(f"Corrupted Q: {corrupted_q[:100]}...")
    print(f"Expected Clean Answer: {clean_ans}")
    print(f"Expected Corrupted Answer: {corrupted_ans}")
    print(f"Number Change: {pair['corrupted']['changed_number']}")

    # 1. Run clean (no patch)
    print(f"\n{'='*40}")
    print("1. CLEAN (No Patch)")
    print(f"{'='*40}")
    clean_generated = patcher.run_without_patch(clean_q, max_new_tokens=200)
    clean_predicted = extract_answer_number(clean_generated)
    clean_correct = (clean_predicted == clean_ans)
    print(f"Generated: {clean_generated[:150]}...")
    print(f"Predicted: {clean_predicted}, Expected: {clean_ans}, Correct: {clean_correct}")

    # 2. Run corrupted (no patch)
    print(f"\n{'='*40}")
    print("2. CORRUPTED (No Patch)")
    print(f"{'='*40}")
    corrupted_generated = patcher.run_without_patch(corrupted_q, max_new_tokens=200)
    corrupted_predicted = extract_answer_number(corrupted_generated)
    corrupted_correct = (corrupted_predicted == corrupted_ans)
    print(f"Generated: {corrupted_generated[:150]}...")
    print(f"Predicted: {corrupted_predicted}, Expected: {corrupted_ans}, Correct: {corrupted_correct}")

    # 3. Cache clean activation
    print(f"\n{'='*40}")
    print("3. CACHE CLEAN ACTIVATION")
    print(f"{'='*40}")
    clean_activations = cacher.cache_problem_activations(clean_q, pair_id)
    clean_act = clean_activations[layer_name]
    print(f"Cached {layer_name} activation shape: {clean_act.shape}")
    print(f"Activation stats: mean={clean_act.mean():.4f}, std={clean_act.std():.4f}, min={clean_act.min():.4f}, max={clean_act.max():.4f}")

    # 4. Run corrupted WITH patch
    print(f"\n{'='*40}")
    print(f"4. CORRUPTED + PATCHED ({layer_name})")
    print(f"{'='*40}")
    patched_generated = patcher.run_with_patch(
        corrupted_q,
        clean_act,
        layer_name,
        max_new_tokens=200
    )
    patched_predicted = extract_answer_number(patched_generated)

    # Check against both answers to understand what's happening
    patched_matches_clean = (patched_predicted == clean_ans)
    patched_matches_corrupted = (patched_predicted == corrupted_ans)

    print(f"Generated: {patched_generated[:150]}...")
    print(f"Predicted: {patched_predicted}")
    print(f"Matches Clean Answer ({clean_ans}): {patched_matches_clean}")
    print(f"Matches Corrupted Answer ({corrupted_ans}): {patched_matches_corrupted}")
    print(f"Matches Neither: {not patched_matches_clean and not patched_matches_corrupted}")

    # Summary
    print(f"\n{'='*40}")
    print("SUMMARY")
    print(f"{'='*40}")
    print(f"Clean (baseline):     {'✓' if clean_correct else '✗'} (predicted {clean_predicted})")
    print(f"Corrupted (baseline): {'✓' if corrupted_correct else '✗'} (predicted {corrupted_predicted})")
    print(f"Patched:              Predicted {patched_predicted}")
    print(f"  - Recovers to clean?    {'✓' if patched_matches_clean else '✗'}")
    print(f"  - Still corrupted?      {'✓' if patched_matches_corrupted else '✗'}")
    print(f"  - Completely wrong?     {'✓' if not patched_matches_clean and not patched_matches_corrupted else '✗'}")

    return {
        'pair_id': pair_id,
        'clean_correct': clean_correct,
        'corrupted_correct': corrupted_correct,
        'patched_predicted': patched_predicted,
        'patched_matches_clean': patched_matches_clean,
        'patched_matches_corrupted': patched_matches_corrupted,
        'clean_ans': clean_ans,
        'corrupted_ans': corrupted_ans,
        'clean_predicted': clean_predicted,
        'corrupted_predicted': corrupted_predicted,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--problem_pairs', type=str, default='problem_pairs.json')
    parser.add_argument('--num_examples', type=int, default=5, help='Number of examples to validate')
    parser.add_argument('--layer', type=str, default='middle', choices=['early', 'middle', 'late'])
    args = parser.parse_args()

    print(f"Loading problem pairs from {args.problem_pairs}...")
    with open(args.problem_pairs, 'r') as f:
        pairs = json.load(f)

    print(f"Testing {args.num_examples} examples from {len(pairs)} total pairs")
    print(f"Layer to test: {args.layer}")

    # Initialize
    print(f"\nLoading CODI model from {args.model_path}...")
    cacher = ActivationCacher(args.model_path)
    patcher = ActivationPatcher(cacher)

    # Test first N pairs
    results = []
    for i, pair in enumerate(pairs[:args.num_examples]):
        result = validate_single_pair(pair, cacher, patcher, args.layer)
        results.append(result)

    # Overall summary
    print(f"\n\n{'='*80}")
    print("OVERALL VALIDATION SUMMARY")
    print(f"{'='*80}")

    clean_correct = sum(r['clean_correct'] for r in results)
    corrupted_correct = sum(r['corrupted_correct'] for r in results)
    patched_recover_to_clean = sum(r['patched_matches_clean'] for r in results)
    patched_stay_corrupted = sum(r['patched_matches_corrupted'] for r in results)
    patched_completely_wrong = sum(not r['patched_matches_clean'] and not r['patched_matches_corrupted'] for r in results)

    total = len(results)

    print(f"Clean Baseline:          {clean_correct}/{total} ({100*clean_correct/total:.1f}%)")
    print(f"Corrupted Baseline:      {corrupted_correct}/{total} ({100*corrupted_correct/total:.1f}%)")
    print(f"\nPatching Outcomes:")
    print(f"  Recovered to clean:    {patched_recover_to_clean}/{total} ({100*patched_recover_to_clean/total:.1f}%)")
    print(f"  Stayed corrupted:      {patched_stay_corrupted}/{total} ({100*patched_stay_corrupted/total:.1f}%)")
    print(f"  Completely wrong:      {patched_completely_wrong}/{total} ({100*patched_completely_wrong/total:.1f}%)")

    if patched_completely_wrong > patched_recover_to_clean:
        print(f"\n⚠️  WARNING: Patching makes things WORSE in {patched_completely_wrong}/{total} cases!")
        print("This suggests the patching is disrupting reasoning rather than correcting it.")
    elif patched_recover_to_clean > corrupted_correct:
        print(f"\n✓ POSITIVE: Patching recovers {patched_recover_to_clean - corrupted_correct} additional correct answers!")
    else:
        print(f"\n⚠️  Patching shows minimal positive effect")

    # Show detailed table
    print(f"\n\nDetailed Results:")
    print(f"{'Pair':<6} {'Clean':<8} {'Corrupt':<8} {'Patched':<10} {'Outcome':<20}")
    print(f"{'-'*60}")
    for r in results:
        clean_mark = '✓' if r['clean_correct'] else '✗'
        corrupt_mark = '✓' if r['corrupted_correct'] else '✗'

        if r['patched_matches_clean']:
            outcome = "→ Clean (Good!)"
        elif r['patched_matches_corrupted']:
            outcome = "→ Corrupted (No effect)"
        else:
            outcome = "→ Wrong (Worse!)"

        print(f"{r['pair_id']:<6} {clean_mark}{r['clean_predicted']:<7} {corrupt_mark}{r['corrupted_predicted']:<7} {r['patched_predicted']:<10} {outcome:<20}")


if __name__ == "__main__":
    main()
