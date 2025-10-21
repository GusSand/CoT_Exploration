#!/usr/bin/env python3
"""
Analyze Difficulty of CoT-Dependent Pairs

Analyzes the difficulty distribution and provides stratification recommendations.
"""

import json
import re

def count_reasoning_steps(solution: str) -> int:
    """Count reasoning steps in the solution."""
    # Count << >> delimited calculations
    steps = len(re.findall(r'<<[^>]+>>', solution))
    return max(steps, 1)  # At least 1 step

def analyze_difficulty():
    """Analyze difficulty of CoT-dependent pairs."""

    with open('data/problem_pairs_cot_dependent.json', 'r') as f:
        pairs = json.load(f)

    print("=" * 80)
    print(f"DIFFICULTY ANALYSIS: {len(pairs)} CoT-Dependent Pairs")
    print("=" * 80)

    # Analyze difficulty
    difficulties = []
    for pair in pairs:
        steps = count_reasoning_steps(pair['clean']['full_answer'])
        pair['difficulty'] = {
            'reasoning_steps': steps,
            'answer_magnitude': pair['clean']['answer']
        }
        difficulties.append(steps)

    # Statistics
    from collections import Counter
    step_counts = Counter(difficulties)

    print("\nReasoning Steps Distribution:")
    for steps in sorted(step_counts.keys()):
        count = step_counts[steps]
        pct = 100 * count / len(pairs)
        print(f"  {steps} steps: {count:2d} pairs ({pct:5.1f}%)")

    # Difficulty stratification
    print("\nRecommended Stratification:")

    easy = [p for p in pairs if p['difficulty']['reasoning_steps'] <= 2]
    medium = [p for p in pairs if p['difficulty']['reasoning_steps'] == 3]
    hard = [p for p in pairs if p['difficulty']['reasoning_steps'] >= 4]

    print(f"  Easy (≤2 steps):    {len(easy):2d} pairs ({100*len(easy)/len(pairs):5.1f}%)")
    print(f"  Medium (3 steps):   {len(medium):2d} pairs ({100*len(medium)/len(pairs):5.1f}%)")
    print(f"  Hard (≥4 steps):    {len(hard):2d} pairs ({100*len(hard)/len(pairs):5.1f}%)")

    # Save stratified data
    stratified = {
        'easy': [p['pair_id'] for p in easy],
        'medium': [p['pair_id'] for p in medium],
        'hard': [p['pair_id'] for p in hard],
        'metadata': {
            'total_pairs': len(pairs),
            'easy_count': len(easy),
            'medium_count': len(medium),
            'hard_count': len(hard)
        }
    }

    with open('results/cot_dependent_stratification.json', 'w') as f:
        json.dump(stratified, f, indent=2)

    print(f"\n✓ Stratification saved to results/cot_dependent_stratification.json")
    print("=" * 80)

    # Summary statistics
    print(f"\nSummary:")
    print(f"  Total CoT-dependent pairs: {len(pairs)}")
    print(f"  Mean reasoning steps: {sum(difficulties)/len(difficulties):.1f}")
    print(f"  Range: {min(difficulties)}-{max(difficulties)} steps")


if __name__ == "__main__":
    analyze_difficulty()
