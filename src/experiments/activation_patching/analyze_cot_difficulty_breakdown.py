#!/usr/bin/env python3
"""
Analyze Difficulty Breakdown: CoT-Needed vs CoT-Not-Needed

Compares the difficulty distribution of problems where LLaMA needs CoT
versus problems where it can skip CoT and use direct computation.

This analysis addresses the question: Is LLaMA's CoT-skipping strategic?
Does it only skip CoT on easier problems?

Usage:
    python analyze_cot_difficulty_breakdown.py
"""

import json
import re
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple


def count_reasoning_steps(solution: str) -> int:
    """Count reasoning steps in the solution (based on << >> delimited calculations)."""
    steps = len(re.findall(r'<<[^>]+>>', solution))
    return max(steps, 1)  # At least 1 step


def load_matched_pairs() -> Tuple[List[Dict], List[int]]:
    """Load matched pairs from validation results."""
    
    # Load validation results
    validation_file = 'validation_results_llama_gpt4_532.json'
    with open(validation_file, 'r') as f:
        validation = json.load(f)
    
    # Load problem pairs
    pairs_file = 'problem_pairs_gpt4_answers.json'
    with open(pairs_file, 'r') as f:
        all_pairs = json.load(f)
    
    # Index all pairs by pair_id
    pairs_by_id = {p['pair_id']: p for p in all_pairs}
    
    # Filter to matched (both-correct) pairs
    matched_pairs = []
    matched_ids = []
    
    for result in validation['results']:
        if result.get('both_correct', False):
            pair_id = result['pair_id']
            if pair_id in pairs_by_id:
                pair = pairs_by_id[pair_id].copy()
                # Add validation metadata
                pair['validation'] = {
                    'clean_correct': result['clean']['correct'],
                    'corrupted_correct': result['corrupted']['correct']
                }
                matched_pairs.append(pair)
                matched_ids.append(pair_id)
    
    print(f"Found {len(matched_pairs)} matched pairs (both-correct baseline)")
    return matched_pairs, matched_ids


def simulate_cot_necessity(matched_pairs: List[Dict]) -> Dict[int, Dict]:
    """
    Since the CoT necessity results files don't exist, we simulate the analysis
    using the documented results from the research journal.
    
    According to the journal:
    - LLaMA needs CoT for 44/101 pairs (43.6%)
    - LLaMA doesn't need CoT for 57/101 pairs (56.4%)
    
    We'll use a heuristic: harder problems (more steps) are more likely to need CoT.
    This is for demonstration purposes.
    """
    print("\n⚠️  NOTE: CoT necessity results files not found.")
    print("Using heuristic simulation based on problem difficulty...")
    print("For accurate results, run manual_cot_necessity_test.py first.\n")
    
    # Calculate difficulty for each pair
    pairs_with_difficulty = []
    for pair in matched_pairs:
        steps = count_reasoning_steps(pair['clean']['full_answer'])
        pairs_with_difficulty.append({
            'pair_id': pair['pair_id'],
            'steps': steps,
            'pair': pair
        })
    
    # Sort by difficulty
    pairs_with_difficulty.sort(key=lambda x: x['steps'])
    
    # Heuristic: Harder problems (top 44) need CoT
    # Easier problems (bottom 57) don't need CoT
    # This is based on the hypothesis that CoT is needed for harder problems
    
    cot_results = {}
    total_pairs = len(pairs_with_difficulty)
    needs_cot_count = 44  # From journal
    
    # Assign CoT necessity (harder problems need CoT)
    for i, item in enumerate(pairs_with_difficulty):
        # Top 44 hardest problems need CoT
        needs_cot = i >= (total_pairs - needs_cot_count)
        cot_results[item['pair_id']] = {
            'needs_cot_either': needs_cot,
            'simulated': True
        }
    
    return cot_results


def try_load_actual_cot_results() -> Dict[int, Dict]:
    """Try to load actual CoT necessity results if they exist."""
    
    # Try multiple possible locations
    possible_files = [
        'results/cot_necessity_llama_simple.json',
        '../results/cot_necessity_llama_simple.json',
        '../../results/cot_necessity_llama_simple.json',
    ]
    
    for filepath in possible_files:
        if Path(filepath).exists():
            print(f"✓ Found CoT necessity results: {filepath}")
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Convert to dict indexed by pair_id
            results = {}
            for result in data['results']:
                results[result['pair_id']] = {
                    'needs_cot_either': result['needs_cot_either'],
                    'needs_cot_both': result['needs_cot_both'],
                    'needs_cot_clean': result['clean']['needs_cot'],
                    'needs_cot_corrupted': result['corrupted']['needs_cot'],
                    'simulated': False
                }
            return results
    
    return None


def analyze_difficulty_breakdown():
    """Main analysis function."""
    
    print("=" * 80)
    print("LLAMA COT NECESSITY: DIFFICULTY BREAKDOWN ANALYSIS")
    print("=" * 80)
    
    # Load matched pairs
    matched_pairs, matched_ids = load_matched_pairs()
    
    # Try to load actual CoT results, fall back to simulation
    cot_results = try_load_actual_cot_results()
    using_simulation = False
    
    if cot_results is None:
        print("\nActual CoT necessity results not found. Using heuristic simulation.")
        cot_results = simulate_cot_necessity(matched_pairs)
        using_simulation = True
    
    # Separate pairs into CoT-needed and CoT-not-needed
    needs_cot_pairs = []
    no_cot_pairs = []
    
    for pair in matched_pairs:
        pair_id = pair['pair_id']
        if pair_id not in cot_results:
            continue
            
        steps = count_reasoning_steps(pair['clean']['full_answer'])
        difficulty_info = {
            'pair_id': pair_id,
            'steps': steps,
            'question': pair['clean']['question'][:80] + '...',
            'answer': pair['clean']['answer']
        }
        
        if cot_results[pair_id]['needs_cot_either']:
            needs_cot_pairs.append(difficulty_info)
        else:
            no_cot_pairs.append(difficulty_info)
    
    # Analyze distributions
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print(f"\nTotal matched pairs analyzed: {len(needs_cot_pairs) + len(no_cot_pairs)}")
    print(f"  Needs CoT: {len(needs_cot_pairs)} pairs ({100*len(needs_cot_pairs)/(len(needs_cot_pairs)+len(no_cot_pairs)):.1f}%)")
    print(f"  Doesn't need CoT: {len(no_cot_pairs)} pairs ({100*len(no_cot_pairs)/(len(needs_cot_pairs)+len(no_cot_pairs)):.1f}%)")
    
    if using_simulation:
        print("\n⚠️  These results are SIMULATED based on difficulty heuristic.")
        print("For accurate results, run: python manual_cot_necessity_test.py")
    
    # Difficulty distributions
    needs_cot_steps = [p['steps'] for p in needs_cot_pairs]
    no_cot_steps = [p['steps'] for p in no_cot_pairs]
    
    needs_cot_counter = Counter(needs_cot_steps)
    no_cot_counter = Counter(no_cot_steps)
    
    print("\n" + "-" * 80)
    print("DIFFICULTY DISTRIBUTION: NEEDS CoT")
    print("-" * 80)
    
    for steps in sorted(needs_cot_counter.keys()):
        count = needs_cot_counter[steps]
        pct = 100 * count / len(needs_cot_pairs)
        print(f"  {steps} steps: {count:2d} pairs ({pct:5.1f}%)")
    
    print(f"\n  Mean: {sum(needs_cot_steps)/len(needs_cot_steps):.2f} steps")
    print(f"  Range: {min(needs_cot_steps)}-{max(needs_cot_steps)} steps")
    
    # Stratification for CoT-needed
    easy_cot = [p for p in needs_cot_pairs if p['steps'] <= 2]
    medium_cot = [p for p in needs_cot_pairs if p['steps'] == 3]
    hard_cot = [p for p in needs_cot_pairs if p['steps'] >= 4]
    
    print(f"\n  Stratification:")
    print(f"    Easy (≤2 steps):  {len(easy_cot):2d} pairs ({100*len(easy_cot)/len(needs_cot_pairs):5.1f}%)")
    print(f"    Medium (3 steps): {len(medium_cot):2d} pairs ({100*len(medium_cot)/len(needs_cot_pairs):5.1f}%)")
    print(f"    Hard (≥4 steps):  {len(hard_cot):2d} pairs ({100*len(hard_cot)/len(needs_cot_pairs):5.1f}%)")
    
    print("\n" + "-" * 80)
    print("DIFFICULTY DISTRIBUTION: DOESN'T NEED CoT")
    print("-" * 80)
    
    for steps in sorted(no_cot_counter.keys()):
        count = no_cot_counter[steps]
        pct = 100 * count / len(no_cot_pairs)
        print(f"  {steps} steps: {count:2d} pairs ({pct:5.1f}%)")
    
    print(f"\n  Mean: {sum(no_cot_steps)/len(no_cot_steps):.2f} steps")
    print(f"  Range: {min(no_cot_steps)}-{max(no_cot_steps)} steps")
    
    # Stratification for no-CoT
    easy_no_cot = [p for p in no_cot_pairs if p['steps'] <= 2]
    medium_no_cot = [p for p in no_cot_pairs if p['steps'] == 3]
    hard_no_cot = [p for p in no_cot_pairs if p['steps'] >= 4]
    
    print(f"\n  Stratification:")
    print(f"    Easy (≤2 steps):  {len(easy_no_cot):2d} pairs ({100*len(easy_no_cot)/len(no_cot_pairs):5.1f}%)")
    print(f"    Medium (3 steps): {len(medium_no_cot):2d} pairs ({100*len(medium_no_cot)/len(no_cot_pairs):5.1f}%)")
    print(f"    Hard (≥4 steps):  {len(hard_no_cot):2d} pairs ({100*len(hard_no_cot)/len(no_cot_pairs):5.1f}%)")
    
    # Comparative analysis
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80)
    
    mean_needs_cot = sum(needs_cot_steps) / len(needs_cot_steps)
    mean_no_cot = sum(no_cot_steps) / len(no_cot_steps)
    mean_diff = mean_needs_cot - mean_no_cot
    
    print(f"\nMean difficulty (reasoning steps):")
    print(f"  Needs CoT:        {mean_needs_cot:.2f} steps")
    print(f"  Doesn't need CoT: {mean_no_cot:.2f} steps")
    print(f"  Difference:       {mean_diff:+.2f} steps")
    
    if mean_diff > 0.5:
        print("\n✓ LLaMA's CoT-skipping appears STRATEGIC - skips on easier problems")
    elif mean_diff < -0.5:
        print("\n✗ UNEXPECTED - LLaMA skips CoT on HARDER problems")
    else:
        print("\n~ LLaMA's CoT-skipping is NOT strongly correlated with difficulty")
    
    # Percentage in each category
    print(f"\nPercentage distribution by difficulty:")
    print(f"{'Category':<12} {'Needs CoT':>12} {'No CoT':>12} {'Difference':>12}")
    print("-" * 50)
    
    pct_easy_cot = 100 * len(easy_cot) / len(needs_cot_pairs)
    pct_easy_no_cot = 100 * len(easy_no_cot) / len(no_cot_pairs)
    print(f"{'Easy':<12} {pct_easy_cot:>11.1f}% {pct_easy_no_cot:>11.1f}% {pct_easy_cot-pct_easy_no_cot:>+11.1f}%")
    
    pct_med_cot = 100 * len(medium_cot) / len(needs_cot_pairs)
    pct_med_no_cot = 100 * len(medium_no_cot) / len(no_cot_pairs)
    print(f"{'Medium':<12} {pct_med_cot:>11.1f}% {pct_med_no_cot:>11.1f}% {pct_med_cot-pct_med_no_cot:>+11.1f}%")
    
    pct_hard_cot = 100 * len(hard_cot) / len(needs_cot_pairs)
    pct_hard_no_cot = 100 * len(hard_no_cot) / len(no_cot_pairs)
    print(f"{'Hard':<12} {pct_hard_cot:>11.1f}% {pct_hard_no_cot:>11.1f}% {pct_hard_cot-pct_hard_no_cot:>+11.1f}%")
    
    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    print(f"\n1. Mean difficulty gap: {abs(mean_diff):.2f} steps")
    if abs(mean_diff) > 0.5:
        if mean_diff > 0:
            print("   → LLaMA needs CoT for harder problems on average")
        else:
            print("   → LLaMA needs CoT for easier problems (unexpected!)")
    else:
        print("   → Similar difficulty for both categories")
    
    # Check if there are hard problems without CoT
    if len(hard_no_cot) > 0:
        print(f"\n2. ⚠️  {len(hard_no_cot)} HARD problems (≥4 steps) solved WITHOUT CoT")
        print("   → LLaMA can use direct computation even on complex problems")
    else:
        print(f"\n2. ✓ ALL hard problems require CoT")
        print("   → LLaMA consistently uses CoT for complex reasoning")
    
    # Check if there are easy problems that need CoT
    if len(easy_cot) > 0:
        print(f"\n3. {len(easy_cot)} EASY problems (≤2 steps) require CoT")
        pct_easy_need_cot = 100 * len(easy_cot) / (len(easy_cot) + len(easy_no_cot))
        print(f"   → {pct_easy_need_cot:.1f}% of easy problems need CoT")
    
    # Save results
    output_data = {
        'analysis_type': 'cot_difficulty_breakdown',
        'model': 'llama',
        'using_simulation': using_simulation,
        'total_pairs': len(needs_cot_pairs) + len(no_cot_pairs),
        'needs_cot': {
            'count': len(needs_cot_pairs),
            'percentage': 100 * len(needs_cot_pairs) / (len(needs_cot_pairs) + len(no_cot_pairs)),
            'mean_steps': mean_needs_cot,
            'range': [min(needs_cot_steps), max(needs_cot_steps)],
            'distribution': dict(needs_cot_counter),
            'stratification': {
                'easy': len(easy_cot),
                'medium': len(medium_cot),
                'hard': len(hard_cot)
            },
            'pair_ids': [p['pair_id'] for p in needs_cot_pairs]
        },
        'no_cot': {
            'count': len(no_cot_pairs),
            'percentage': 100 * len(no_cot_pairs) / (len(needs_cot_pairs) + len(no_cot_pairs)),
            'mean_steps': mean_no_cot,
            'range': [min(no_cot_steps), max(no_cot_steps)],
            'distribution': dict(no_cot_counter),
            'stratification': {
                'easy': len(easy_no_cot),
                'medium': len(medium_no_cot),
                'hard': len(hard_no_cot)
            },
            'pair_ids': [p['pair_id'] for p in no_cot_pairs]
        },
        'comparison': {
            'mean_difference': mean_diff,
            'strategic_cot_use': mean_diff > 0.5
        }
    }
    
    output_file = 'results_cot_difficulty_breakdown.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"✓ Analysis complete. Results saved to {output_file}")
    print("=" * 80)
    
    if using_simulation:
        print("\n⚠️  IMPORTANT: These results are based on SIMULATED CoT necessity.")
        print("To get actual results:")
        print("  1. Run: python manual_cot_necessity_test.py")
        print("  2. Wait for results file: results/cot_necessity_llama_simple.json")
        print("  3. Re-run this script for accurate analysis")


if __name__ == "__main__":
    analyze_difficulty_breakdown()


