#!/usr/bin/env python3
"""
Filter to CoT-Dependent Pairs

Filters matched pairs to only include those where BOTH models demonstrably need
latent chain-of-thought tokens to solve the problems correctly.

Since GPT-2 needs CoT for ALL 101 pairs, we filter to the pairs where LLaMA also needs CoT.

Usage:
    python filter_cot_dependent_pairs.py
"""

import json
from pathlib import Path

def filter_cot_dependent():
    """Filter to pairs where both models need CoT."""

    # Load original matched pairs
    pairs_file = 'data/problem_pairs_matched.json'
    with open(pairs_file, 'r') as f:
        matched_pairs = json.load(f)

    # Load CoT necessity results
    with open('results/cot_necessity_llama_simple.json', 'r') as f:
        llama_results = json.load(f)

    with open('results/cot_necessity_gpt2_simple.json', 'r') as f:
        gpt2_results = json.load(f)

    # Index by pair_id
    llama_by_id = {r['pair_id']: r for r in llama_results['results']}
    gpt2_by_id = {r['pair_id']: r for r in gpt2_results['results']}

    # Filter to CoT-dependent pairs
    # Since GPT-2 needs CoT for ALL pairs, we filter where LLaMA also needs it
    cot_dependent_pairs = []

    for pair in matched_pairs:
        pair_id = pair['pair_id']

        # Get necessity results
        llama_needs = llama_by_id[pair_id]['needs_cot_either']
        gpt2_needs = gpt2_by_id[pair_id]['needs_cot_either']

        # Both models need CoT for at least one problem
        if llama_needs and gpt2_needs:
            # Add necessity metadata
            pair['cot_necessity'] = {
                'llama': {
                    'needs_cot_clean': llama_by_id[pair_id]['clean']['needs_cot'],
                    'needs_cot_corrupted': llama_by_id[pair_id]['corrupted']['needs_cot'],
                    'needs_cot_either': llama_needs,
                    'needs_cot_both': llama_by_id[pair_id]['needs_cot_both']
                },
                'gpt2': {
                    'needs_cot_clean': gpt2_by_id[pair_id]['clean']['needs_cot'],
                    'needs_cot_corrupted': gpt2_by_id[pair_id]['corrupted']['needs_cot'],
                    'needs_cot_either': gpt2_needs,
                    'needs_cot_both': gpt2_by_id[pair_id]['needs_cot_both']
                }
            }
            cot_dependent_pairs.append(pair)

    # Save filtered pairs
    output_file = 'data/problem_pairs_cot_dependent.json'
    with open(output_file, 'w') as f:
        json.dump(cot_dependent_pairs, f, indent=2)

    # Print statistics
    print("=" * 80)
    print("COT-DEPENDENT PAIR FILTERING")
    print("=" * 80)
    print(f"Original matched pairs:               {len(matched_pairs)}")
    print(f"LLaMA needs CoT (either):             {llama_results['statistics']['needs_cot_either']}")
    print(f"GPT-2 needs CoT (either):             {gpt2_results['statistics']['needs_cot_either']}")
    print(f"BOTH models need CoT:                 {len(cot_dependent_pairs)}")
    print()
    print(f"Excluded (LLaMA doesn't need CoT):    {len(matched_pairs) - len(cot_dependent_pairs)}")
    print("=" * 80)
    print(f"\n✓ Filtered to {len(cot_dependent_pairs)} CoT-dependent pairs")
    print(f"✓ Saved to {output_file}")

    # Breakdown by necessity type
    both_both = sum(1 for p in cot_dependent_pairs
                    if p['cot_necessity']['llama']['needs_cot_both'])
    llama_either = len(cot_dependent_pairs) - both_both

    print(f"\nBreakdown:")
    print(f"  LLaMA needs CoT for BOTH problems:   {both_both}")
    print(f"  LLaMA needs CoT for EITHER:          {llama_either}")
    print(f"  GPT-2 always needs CoT for BOTH:     {len(cot_dependent_pairs)}")


if __name__ == "__main__":
    filter_cot_dependent()
