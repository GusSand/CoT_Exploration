#!/usr/bin/env python3
"""
CoT Necessity Test

Tests whether models actually NEED latent chain-of-thought tokens to solve problems.

Methodology:
1. Baseline: Model solves with latent tokens (already validated)
2. Ablated: Model solves with ALL latent tokens replaced by zeros
3. CoT-dependent: Baseline correct AND ablated incorrect

This ensures fair cross-model comparison - both models are using latent
reasoning, not just direct computation.

Usage:
    python test_cot_necessity.py \
        --model_path ~/codi_ckpt/llama_gsm8k/ \
        --problem_pairs data/problem_pairs_matched.json \
        --output results/cot_necessity_llama.json \
        --model_name llama
"""

import json
import sys
import re
import argparse
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path for imports
script_dir = Path(__file__).parent.parent.parent
project_root = script_dir.parent.parent.parent  # /home/paperspace/dev/CoT_Exploration/
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir / 'core'))
sys.path.insert(0, str(project_root / 'codi'))  # Add codi to path for src.model imports


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


def ablate_all_latent_tokens(patcher, problem_text, max_new_tokens=200):
    """Ablate ALL latent tokens by replacing with zeros.

    Uses the existing run_with_patch but patches all latent positions with zeros.
    """
    import torch

    # First, cache activations from clean problem to get shape
    clean_cache = patcher.cacher.cache_activations(
        problem_text=problem_text,
        max_new_tokens=max_new_tokens
    )

    # Get the shape of latent activations (6 tokens x hidden_dim)
    # Using 'middle' layer as representative
    sample_activation = clean_cache['middle'][0]  # First position
    hidden_dim = sample_activation.shape[-1]

    # Create zero tensor for all 6 latent tokens
    zero_activation = torch.zeros(6, hidden_dim, device=sample_activation.device, dtype=sample_activation.dtype)

    # Patch at middle layer (could be any layer, zeros will kill latent reasoning)
    ablated_output = patcher.run_with_patch(
        problem_text=problem_text,
        patch_activation=zero_activation,
        patch_layer_name='middle',
        max_new_tokens=max_new_tokens
    )

    return ablated_output


def test_cot_necessity(patcher, pair, max_new_tokens=200):
    """Test if model needs CoT for a problem pair.

    Returns:
        dict with necessity test results
    """
    results = {
        'pair_id': pair['pair_id'],
        'clean': {},
        'corrupted': {}
    }

    for problem_type in ['clean', 'corrupted']:
        question = pair[problem_type]['question']
        expected = pair[problem_type]['answer']

        # Baseline (with latent tokens) - from validation we know this works
        baseline_correct = pair['matched_validation'][patcher.model_name][f'{problem_type}_correct']
        baseline_pred = pair['matched_validation'][patcher.model_name][f'{problem_type}_predicted']

        # Ablated (ALL latent tokens replaced with zeros)
        ablated_output = ablate_all_latent_tokens(
            patcher=patcher,
            problem_text=question,
            max_new_tokens=max_new_tokens
        )
        ablated_pred = extract_answer_number(ablated_output)
        ablated_correct = answers_match(ablated_pred, expected)

        # CoT-dependent if baseline correct but ablated fails
        needs_cot = baseline_correct and not ablated_correct

        results[problem_type] = {
            'expected': expected,
            'baseline_predicted': baseline_pred,
            'baseline_correct': baseline_correct,
            'ablated_output': ablated_output,
            'ablated_predicted': ablated_pred,
            'ablated_correct': ablated_correct,
            'needs_cot': needs_cot,
            'output_length': len(ablated_output)
        }

    # Overall: model needs CoT if EITHER problem is CoT-dependent
    results['needs_cot_clean'] = results['clean']['needs_cot']
    results['needs_cot_corrupted'] = results['corrupted']['needs_cot']
    results['needs_cot_either'] = results['clean']['needs_cot'] or results['corrupted']['needs_cot']
    results['needs_cot_both'] = results['clean']['needs_cot'] and results['corrupted']['needs_cot']

    return results


def main():
    parser = argparse.ArgumentParser(description="Test CoT necessity")
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--problem_pairs', type=str, required=True, help='Path to matched pairs JSON')
    parser.add_argument('--output', type=str, required=True, help='Output necessity test results JSON')
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
        patcher.model_name = 'llama'
    else:  # gpt2
        from cache_activations import ActivationCacher
        from patch_and_eval import ActivationPatcher

        print(f"Loading GPT-2 model from {args.model_path}...")
        cacher = ActivationCacher(args.model_path)
        patcher = ActivationPatcher(cacher)
        patcher.model_name = 'gpt2'

    # Load pairs
    print(f"Loading matched pairs from {args.problem_pairs}...")
    with open(args.problem_pairs, 'r') as f:
        pairs = json.load(f)

    print(f"\nTesting CoT necessity on {len(pairs)} pairs for {args.model_name.upper()}...")
    print("This will take approximately 5-10 minutes...\n")

    # Test all pairs
    results = []
    stats = {
        'total_pairs': len(pairs),
        'needs_cot_clean': 0,
        'needs_cot_corrupted': 0,
        'needs_cot_either': 0,
        'needs_cot_both': 0,
        'ablated_still_correct_clean': 0,
        'ablated_still_correct_corrupted': 0
    }

    for pair in tqdm(pairs, desc="Testing CoT necessity"):
        try:
            result = test_cot_necessity(patcher, pair, args.max_new_tokens)
            results.append(result)

            # Update stats
            if result['needs_cot_clean']:
                stats['needs_cot_clean'] += 1
            if result['needs_cot_corrupted']:
                stats['needs_cot_corrupted'] += 1
            if result['needs_cot_either']:
                stats['needs_cot_either'] += 1
            if result['needs_cot_both']:
                stats['needs_cot_both'] += 1
            if result['clean']['ablated_correct']:
                stats['ablated_still_correct_clean'] += 1
            if result['corrupted']['ablated_correct']:
                stats['ablated_still_correct_corrupted'] += 1

        except Exception as e:
            print(f"\n⚠️  Error on pair {pair['pair_id']}: {e}")
            results.append({
                'pair_id': pair['pair_id'],
                'error': str(e),
                'needs_cot_either': False,
                'needs_cot_both': False
            })

    # Save results
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
    print(f"{args.model_name.upper()} COT NECESSITY TEST COMPLETE")
    print("=" * 80)
    print(f"Total pairs tested:                    {len(pairs)}")
    print(f"Needs CoT for CLEAN:                   {stats['needs_cot_clean']}/{len(pairs)} ({100*stats['needs_cot_clean']/len(pairs):.1f}%)")
    print(f"Needs CoT for CORRUPTED:               {stats['needs_cot_corrupted']}/{len(pairs)} ({100*stats['needs_cot_corrupted']/len(pairs):.1f}%)")
    print(f"Needs CoT for EITHER:                  {stats['needs_cot_either']}/{len(pairs)} ({100*stats['needs_cot_either']/len(pairs):.1f}%)")
    print(f"Needs CoT for BOTH:                    {stats['needs_cot_both']}/{len(pairs)} ({100*stats['needs_cot_both']/len(pairs):.1f}%)")
    print()
    print(f"Ablated still correct (CLEAN):         {stats['ablated_still_correct_clean']}/{len(pairs)} ({100*stats['ablated_still_correct_clean']/len(pairs):.1f}%)")
    print(f"Ablated still correct (CORRUPTED):     {stats['ablated_still_correct_corrupted']}/{len(pairs)} ({100*stats['ablated_still_correct_corrupted']/len(pairs):.1f}%)")
    print("=" * 80)
    print(f"\n✓ Results saved to {args.output}")
    print(f"\nNext: Run on other model, then filter to CoT-dependent pairs")


if __name__ == "__main__":
    main()
