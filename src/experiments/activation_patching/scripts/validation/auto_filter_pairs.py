#!/usr/bin/env python3
"""
Auto-Filter Problem Pairs Based on Validation Results

Filters problem pairs based on model baseline performance to select
the highest quality pairs for experiments.

Usage:
    python auto_filter_pairs.py \
        --pairs problem_pairs_with_answers_300.json \
        --llama_results validation_results_llama_300.json \
        --gpt2_results validation_results_gpt2_300.json \
        --output data/problem_pairs_200.json \
        --target_count 200
"""

import json
import argparse
from typing import Dict, List
from pathlib import Path


def load_validation_results(filepath: str) -> Dict:
    """Load validation results from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_pairs(filepath: str) -> List[Dict]:
    """Load problem pairs from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def merge_validation_data(pairs: List[Dict], llama_results: Dict, gpt2_results: Dict) -> List[Dict]:
    """Merge validation results into pair data.

    Returns:
        List of pairs with added validation metadata
    """
    # Index results by pair_id
    llama_by_id = {r['pair_id']: r for r in llama_results['results']}
    gpt2_by_id = {r['pair_id']: r for r in gpt2_results['results']}

    enriched_pairs = []
    for pair in pairs:
        pair_id = pair['pair_id']

        # Get validation results
        llama_val = llama_by_id.get(pair_id, {})
        gpt2_val = gpt2_by_id.get(pair_id, {})

        # Add validation metadata
        pair['validation'] = {
            'llama': {
                'clean_correct': llama_val.get('clean', {}).get('correct', False),
                'corrupted_correct': llama_val.get('corrupted', {}).get('correct', False),
                'both_correct': llama_val.get('both_correct', False),
                'clean_only': llama_val.get('clean_only', False),
                'output_length': {
                    'clean': llama_val.get('clean', {}).get('output_length', 0),
                    'corrupted': llama_val.get('corrupted', {}).get('output_length', 0)
                }
            },
            'gpt2': {
                'clean_correct': gpt2_val.get('clean', {}).get('correct', False),
                'corrupted_correct': gpt2_val.get('corrupted', {}).get('correct', False),
                'both_correct': gpt2_val.get('both_correct', False),
                'clean_only': gpt2_val.get('clean_only', False),
                'output_length': {
                    'clean': gpt2_val.get('clean', {}).get('output_length', 0),
                    'corrupted': gpt2_val.get('corrupted', {}).get('output_length', 0)
                }
            }
        }

        enriched_pairs.append(pair)

    return enriched_pairs


def calculate_quality_score(pair: Dict) -> tuple:
    """Calculate quality score and tier for a pair.

    Returns:
        (tier, score) where lower tier = higher priority
    """
    val = pair['validation']
    llama = val['llama']
    gpt2 = val['gpt2']

    # Check for gibberish (too long outputs)
    MAX_LENGTH = 500
    llama_gibberish = (llama['output_length']['clean'] > MAX_LENGTH or
                       llama['output_length']['corrupted'] > MAX_LENGTH)
    gpt2_gibberish = (gpt2['output_length']['clean'] > MAX_LENGTH or
                     gpt2['output_length']['corrupted'] > MAX_LENGTH)

    if llama_gibberish and gpt2_gibberish:
        return (99, 0)  # Reject - both models gibberish

    # Quality tiers (lower = better)
    if llama['both_correct'] and gpt2['both_correct']:
        tier = 1  # Both models both-correct (best!)
    elif llama['both_correct'] or gpt2['both_correct']:
        tier = 2  # One model both-correct
    elif llama['clean_only'] and gpt2['clean_only']:
        tier = 3  # Both models target case (clean✓ + corrupted✗)
    elif llama['clean_only'] or gpt2['clean_only']:
        tier = 4  # One model target case
    elif llama['clean_correct'] or gpt2['clean_correct']:
        tier = 5  # At least one clean correct
    else:
        tier = 99  # Reject - neither model gets clean correct

    # Score within tier (higher = better)
    score = 0

    # Prefer pairs where both models get clean correct
    if llama['clean_correct'] and gpt2['clean_correct']:
        score += 100
    elif llama['clean_correct'] or gpt2['clean_correct']:
        score += 50

    # Prefer high confidence auto-calculations
    if pair.get('calculation_confidence') == 'high':
        score += 10
    elif pair.get('calculation_confidence') == 'medium':
        score += 5

    # Penalize gibberish
    if llama_gibberish or gpt2_gibberish:
        score -= 50

    return (tier, score)


def filter_and_rank_pairs(pairs: List[Dict], target_count: int) -> List[Dict]:
    """Filter and rank pairs by quality, returning top N.

    Args:
        pairs: List of enriched pairs with validation data
        target_count: Number of pairs to return

    Returns:
        Top N pairs sorted by quality
    """
    # Calculate quality for each pair
    pairs_with_quality = []
    for pair in pairs:
        tier, score = calculate_quality_score(pair)
        pair['quality_tier'] = tier
        pair['quality_score'] = score

        if tier < 99:  # Not rejected
            pairs_with_quality.append(pair)

    # Sort by tier (ascending), then score (descending)
    pairs_with_quality.sort(key=lambda p: (p['quality_tier'], -p['quality_score']))

    # Return top N
    return pairs_with_quality[:target_count]


def print_statistics(pairs: List[Dict], filtered_pairs: List[Dict]):
    """Print filtering statistics."""
    val_counts = {
        'llama_clean': 0,
        'llama_corrupted': 0,
        'llama_both': 0,
        'llama_target': 0,
        'gpt2_clean': 0,
        'gpt2_corrupted': 0,
        'gpt2_both': 0,
        'gpt2_target': 0
    }

    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    for pair in filtered_pairs:
        val = pair['validation']

        if val['llama']['clean_correct']:
            val_counts['llama_clean'] += 1
        if val['llama']['corrupted_correct']:
            val_counts['llama_corrupted'] += 1
        if val['llama']['both_correct']:
            val_counts['llama_both'] += 1
        if val['llama']['clean_only']:
            val_counts['llama_target'] += 1

        if val['gpt2']['clean_correct']:
            val_counts['gpt2_clean'] += 1
        if val['gpt2']['corrupted_correct']:
            val_counts['gpt2_corrupted'] += 1
        if val['gpt2']['both_correct']:
            val_counts['gpt2_both'] += 1
        if val['gpt2']['clean_only']:
            val_counts['gpt2_target'] += 1

        tier = pair['quality_tier']
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    print("\n" + "=" * 80)
    print("AUTO-FILTERING COMPLETE")
    print("=" * 80)
    print(f"Total candidates:                 {len(pairs)}")
    print(f"Filtered pairs:                   {len(filtered_pairs)}")
    print()
    print("LLaMA Performance on Filtered Pairs:")
    print(f"  Clean correct:                  {val_counts['llama_clean']}/{len(filtered_pairs)} ({100*val_counts['llama_clean']/len(filtered_pairs):.1f}%)")
    print(f"  Corrupted correct:              {val_counts['llama_corrupted']}/{len(filtered_pairs)} ({100*val_counts['llama_corrupted']/len(filtered_pairs):.1f}%)")
    print(f"  Both correct:                   {val_counts['llama_both']}/{len(filtered_pairs)} ({100*val_counts['llama_both']/len(filtered_pairs):.1f}%)")
    print(f"  Target case (clean✓+corrupt✗):  {val_counts['llama_target']}/{len(filtered_pairs)} ({100*val_counts['llama_target']/len(filtered_pairs):.1f}%)")
    print()
    print("GPT-2 Performance on Filtered Pairs:")
    print(f"  Clean correct:                  {val_counts['gpt2_clean']}/{len(filtered_pairs)} ({100*val_counts['gpt2_clean']/len(filtered_pairs):.1f}%)")
    print(f"  Corrupted correct:              {val_counts['gpt2_corrupted']}/{len(filtered_pairs)} ({100*val_counts['gpt2_corrupted']/len(filtered_pairs):.1f}%)")
    print(f"  Both correct:                   {val_counts['gpt2_both']}/{len(filtered_pairs)} ({100*val_counts['gpt2_both']/len(filtered_pairs):.1f}%)")
    print(f"  Target case (clean✓+corrupt✗):  {val_counts['gpt2_target']}/{len(filtered_pairs)} ({100*val_counts['gpt2_target']/len(filtered_pairs):.1f}%)")
    print()
    print("Quality Tier Distribution:")
    print(f"  Tier 1 (both models both-correct):       {tier_counts.get(1, 0)}")
    print(f"  Tier 2 (one model both-correct):         {tier_counts.get(2, 0)}")
    print(f"  Tier 3 (both models target case):        {tier_counts.get(3, 0)}")
    print(f"  Tier 4 (one model target case):          {tier_counts.get(4, 0)}")
    print(f"  Tier 5 (at least one clean correct):     {tier_counts.get(5, 0)}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Auto-filter problem pairs by validation results")
    parser.add_argument('--pairs', type=str, required=True, help='Problem pairs JSON file')
    parser.add_argument('--llama_results', type=str, required=True, help='LLaMA validation results JSON')
    parser.add_argument('--gpt2_results', type=str, required=True, help='GPT-2 validation results JSON')
    parser.add_argument('--output', type=str, required=True, help='Output filtered pairs JSON')
    parser.add_argument('--target_count', type=int, default=200, help='Target number of pairs')

    args = parser.parse_args()

    print("Loading data...")
    pairs = load_pairs(args.pairs)
    llama_results = load_validation_results(args.llama_results)
    gpt2_results = load_validation_results(args.gpt2_results)

    print("Merging validation results...")
    enriched_pairs = merge_validation_data(pairs, llama_results, gpt2_results)

    print(f"Filtering and ranking pairs (target: {args.target_count})...")
    filtered_pairs = filter_and_rank_pairs(enriched_pairs, args.target_count)

    # Save filtered pairs
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(filtered_pairs, f, indent=2)

    print_statistics(pairs, filtered_pairs)
    print(f"\n✓ Saved {len(filtered_pairs)} filtered pairs to {args.output}")
    print(f"\nNext: Run experiments with {args.output}")


if __name__ == "__main__":
    main()
