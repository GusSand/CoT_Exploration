#!/usr/bin/env python3
"""
Baseline Validation Script for Problem Pairs

Validates all problem pairs on a given model (LLaMA or GPT-2) by running
inference and checking correctness. Saves detailed results for filtering.

Usage:
    python validate_baseline.py \
        --model_path ~/codi_ckpt/llama_gsm8k/ \
        --problem_pairs problem_pairs_with_answers_300.json \
        --output validation_results_llama_300.json \
        --model_name llama
"""

import json
import sys
import re
import argparse
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, '.')


def extract_answer_number(text: str):
    """Extract the numerical answer from generated text."""
    patterns = [
        r'####\s*(-?\d+(?:\.\d+)?)',
        r'(?:answer|total|result)(?:\s+is)?\s*[:=]?\s*\$?\s*(-?\d+(?:\.\d+)?)',
        r'\$?\s*(-?\d+(?:\.\d+)?)\s*$',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                num = float(match.group(1))
                return int(num) if num.is_integer() else num
            except ValueError:
                continue

    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if numbers:
        try:
            num = float(numbers[-1])
            return int(num) if num.is_integer() else num
        except ValueError:
            pass

    return None


def answers_match(predicted, expected):
    """Check if predicted answer matches expected."""
    if predicted is None or expected is None:
        return False

    try:
        pred_float = float(predicted)
        exp_float = float(expected)
        return abs(pred_float - exp_float) < 0.01
    except (ValueError, TypeError):
        return False


def validate_pair(patcher, pair, max_new_tokens=200):
    """Validate a single problem pair.

    Returns:
        dict with validation results
    """
    # Test clean
    clean_output = patcher.run_without_patch(
        problem_text=pair['clean']['question'],
        max_new_tokens=max_new_tokens
    )
    clean_pred = extract_answer_number(clean_output)
    clean_is_correct = answers_match(clean_pred, pair['clean']['answer'])

    # Test corrupted
    corrupted_output = patcher.run_without_patch(
        problem_text=pair['corrupted']['question'],
        max_new_tokens=max_new_tokens
    )
    corrupted_pred = extract_answer_number(corrupted_output)
    corrupted_is_correct = answers_match(corrupted_pred, pair['corrupted']['answer'])

    return {
        'pair_id': pair['pair_id'],
        'clean': {
            'output': clean_output,
            'predicted': clean_pred,
            'expected': pair['clean']['answer'],
            'correct': clean_is_correct,
            'output_length': len(clean_output)
        },
        'corrupted': {
            'output': corrupted_output,
            'predicted': corrupted_pred,
            'expected': pair['corrupted']['answer'],
            'correct': corrupted_is_correct,
            'output_length': len(corrupted_output)
        },
        'both_correct': clean_is_correct and corrupted_is_correct,
        'clean_only': clean_is_correct and not corrupted_is_correct,
        'corrupted_only': not clean_is_correct and corrupted_is_correct,
        'both_wrong': not clean_is_correct and not corrupted_is_correct
    }


def main():
    parser = argparse.ArgumentParser(description="Validate problem pairs on model baseline")
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--problem_pairs', type=str, required=True, help='Path to problem pairs JSON')
    parser.add_argument('--output', type=str, required=True, help='Output validation results JSON')
    parser.add_argument('--model_name', type=str, required=True, choices=['llama', 'gpt2'],
                        help='Model type (llama or gpt2)')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Max tokens to generate')

    args = parser.parse_args()

    # Import appropriate modules based on model
    if args.model_name == 'llama':
        from cache_activations_llama import ActivationCacherLLaMA
        from patch_and_eval_llama import ActivationPatcher

        print(f"Loading LLaMA model from {args.model_path}...")
        cacher = ActivationCacherLLaMA(args.model_path)
        patcher = ActivationPatcher(cacher)
    else:  # gpt2
        from cache_activations import ActivationCacher
        from patch_and_eval import ActivationPatcher

        print(f"Loading GPT-2 model from {args.model_path}...")
        cacher = ActivationCacher(args.model_path)
        patcher = ActivationPatcher(cacher)

    # Load pairs
    print(f"Loading problem pairs from {args.problem_pairs}...")
    with open(args.problem_pairs, 'r') as f:
        pairs = json.load(f)

    print(f"\nValidating {len(pairs)} pairs on {args.model_name.upper()}...")
    print("This will take approximately 45-60 minutes...\n")

    # Validate all pairs
    results = []
    stats = {
        'clean_correct': 0,
        'corrupted_correct': 0,
        'both_correct': 0,
        'clean_only': 0,
        'corrupted_only': 0,
        'both_wrong': 0
    }

    for pair in tqdm(pairs, desc="Validating pairs"):
        try:
            result = validate_pair(patcher, pair, args.max_new_tokens)
            results.append(result)

            # Update stats
            if result['clean']['correct']:
                stats['clean_correct'] += 1
            if result['corrupted']['correct']:
                stats['corrupted_correct'] += 1
            if result['both_correct']:
                stats['both_correct'] += 1
            if result['clean_only']:
                stats['clean_only'] += 1
            if result['corrupted_only']:
                stats['corrupted_only'] += 1
            if result['both_wrong']:
                stats['both_wrong'] += 1

        except Exception as e:
            print(f"\n⚠️  Error on pair {pair['pair_id']}: {e}")
            # Add error result
            results.append({
                'pair_id': pair['pair_id'],
                'error': str(e),
                'both_correct': False,
                'clean_only': False,
                'corrupted_only': False,
                'both_wrong': True
            })
            stats['both_wrong'] += 1

    # Save detailed results
    output_data = {
        'model_name': args.model_name,
        'model_path': args.model_path,
        'problem_pairs_file': args.problem_pairs,
        'num_pairs': len(pairs),
        'statistics': stats,
        'results': results
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print(f"{args.model_name.upper()} BASELINE VALIDATION COMPLETE")
    print("=" * 80)
    print(f"Total pairs tested:               {len(pairs)}")
    print(f"Clean correct:                    {stats['clean_correct']}/{len(pairs)} ({100*stats['clean_correct']/len(pairs):.1f}%)")
    print(f"Corrupted correct:                {stats['corrupted_correct']}/{len(pairs)} ({100*stats['corrupted_correct']/len(pairs):.1f}%)")
    print(f"Both correct:                     {stats['both_correct']}/{len(pairs)} ({100*stats['both_correct']/len(pairs):.1f}%)")
    print(f"Clean✓ + Corrupted✗:              {stats['clean_only']}/{len(pairs)} ({100*stats['clean_only']/len(pairs):.1f}%)")
    print(f"Clean✗ + Corrupted✓:              {stats['corrupted_only']}/{len(pairs)} ({100*stats['corrupted_only']/len(pairs):.1f}%)")
    print(f"Both wrong:                       {stats['both_wrong']}/{len(pairs)} ({100*stats['both_wrong']/len(pairs):.1f}%)")
    print("=" * 80)
    print(f"\n✓ Results saved to {args.output}")
    print(f"\nNext: Run validation on other model, then filter pairs with auto_filter_pairs.py")


if __name__ == "__main__":
    main()
