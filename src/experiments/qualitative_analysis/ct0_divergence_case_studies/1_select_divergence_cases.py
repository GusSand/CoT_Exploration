#!/usr/bin/env python3
"""
Select Case Studies: CT Hidden State Divergence

Goal: Select diverse, interesting problems that show cascading divergence
      when CT0 attention is blocked.

Selection Criteria:
1. High divergence: CT1-CT4 show significant divergence (< 0.7 similarity)
2. Cascading pattern: Divergence increases step-by-step
3. Problem impact: Include both degradation and no_change cases
4. Diverse divergence profiles:
   - Early divergence (CT1 low similarity)
   - Late divergence (CT1 ok, later steps diverge)
   - Monotonic decrease vs recovery patterns

Output:
- 10-15 selected case studies
- Each with: problem ID, divergence profile, baseline/blocked predictions
- Saved to results/selected_divergence_cases.json
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# Load divergence data
DIVERGENCE_FILE = Path('/home/paperspace/dev/CoT_Exploration/src/experiments/codi_attention_flow/results/ct_hidden_state_divergence.json')
METADATA_FILE = Path('/home/paperspace/dev/CoT_Exploration/src/experiments/codi_attention_flow/ablation/results/attention_data/llama_metadata_final.json')


def load_data():
    """Load divergence analysis and metadata."""
    with open(DIVERGENCE_FILE) as f:
        divergence_data = json.load(f)

    with open(METADATA_FILE) as f:
        metadata = json.load(f)

    return divergence_data, metadata


def compute_divergence_metrics(problem_result: Dict) -> Dict:
    """Compute aggregate divergence metrics for a problem."""
    divergence = problem_result.get('divergence', {})

    # Extract similarity at each step
    similarities = {}
    for step_str, metrics in divergence.items():
        step = int(step_str)
        similarities[step] = metrics['cosine_similarity_mean']

    # Compute metrics
    ct1_similarity = similarities.get(1, 1.0)
    ct4_similarity = similarities.get(4, 1.0)

    # Divergence magnitude
    total_divergence = 1.0 - np.mean([similarities.get(i, 1.0) for i in range(1, 6)])

    # Divergence rate (slope from CT1 to CT5)
    steps = sorted([s for s in similarities.keys() if s > 0])
    if len(steps) > 1:
        sims = [similarities[s] for s in steps]
        divergence_slope = (sims[-1] - sims[0]) / len(steps)
    else:
        divergence_slope = 0.0

    # Pattern type
    if ct1_similarity < 0.6:
        pattern = "early_divergence"
    elif ct4_similarity < 0.4:
        pattern = "late_divergence"
    elif divergence_slope < -0.1:
        pattern = "cascading"
    else:
        pattern = "stable"

    return {
        'ct1_similarity': ct1_similarity,
        'ct4_similarity': ct4_similarity,
        'total_divergence': total_divergence,
        'divergence_slope': divergence_slope,
        'pattern': pattern,
        'similarities_by_step': similarities
    }


def select_case_studies(divergence_data: Dict, metadata: List[Dict], n_cases: int = 12) -> List[Dict]:
    """
    Select diverse case studies showing interesting divergence patterns.

    Selection strategy:
    1. High divergence examples (top N by total divergence)
    2. Diverse patterns (early, late, cascading, recovering)
    3. Both degradation and no_change impact types
    4. Range of problem complexities
    """

    # Get problem-level results
    problem_results = divergence_data.get('problem_level_results', [])

    # Compute metrics for each problem
    enriched_problems = []
    for problem in problem_results:
        problem_id = problem['problem_id']

        # Find metadata
        meta = next((m for m in metadata if m['problem_id'] == problem_id), None)
        if not meta:
            continue

        # Compute divergence metrics
        metrics = compute_divergence_metrics(problem)

        enriched_problems.append({
            **problem,
            'question': meta.get('baseline', {}).get('question', 'N/A'),
            'baseline_answer': meta.get('baseline', {}).get('pred_answer', 'N/A'),
            'ct0_blocked_answer': meta.get('ct0_blocked', {}).get('pred_answer', 'N/A'),
            'gold_answer': meta.get('baseline', {}).get('gold_answer', 'N/A'),
            **metrics
        })

    # Sort by total divergence
    enriched_problems.sort(key=lambda x: x['total_divergence'], reverse=True)

    # Selection buckets
    selected = []

    # 1. Top 3 highest divergence
    print("Selecting top 3 highest divergence cases...")
    for problem in enriched_problems[:3]:
        selected.append({
            'selection_reason': f"Highest divergence ({problem['total_divergence']:.3f})",
            **problem
        })

    # 2. Early divergence pattern (CT1 < 0.6)
    print("Selecting early divergence cases...")
    early_div = [p for p in enriched_problems if p['ct1_similarity'] < 0.6 and p['problem_id'] not in [s['problem_id'] for s in selected]]
    if early_div:
        selected.append({
            'selection_reason': f"Early divergence (CT1={early_div[0]['ct1_similarity']:.3f})",
            **early_div[0]
        })

    # 3. Late divergence pattern (CT1 ok, CT4 very low)
    print("Selecting late divergence cases...")
    late_div = [p for p in enriched_problems if p['ct1_similarity'] > 0.7 and p['ct4_similarity'] < 0.4 and p['problem_id'] not in [s['problem_id'] for s in selected]]
    if late_div:
        selected.append({
            'selection_reason': f"Late divergence (CT1={late_div[0]['ct1_similarity']:.3f}, CT4={late_div[0]['ct4_similarity']:.3f})",
            **late_div[0]
        })

    # 4. Steep cascading (large negative slope)
    print("Selecting steep cascading cases...")
    cascading = [p for p in enriched_problems if p['divergence_slope'] < -0.15 and p['problem_id'] not in [s['problem_id'] for s in selected]]
    if cascading:
        selected.append({
            'selection_reason': f"Steep cascade (slope={cascading[0]['divergence_slope']:.3f})",
            **cascading[0]
        })

    # 5. Degradation impact (baseline correct, blocked wrong)
    print("Selecting degradation cases...")
    degradation = [p for p in enriched_problems
                   if p['impact'] == 'degradation'
                   and p['baseline_correct'] and not p['ct0_blocked_correct']
                   and p['problem_id'] not in [s['problem_id'] for s in selected]]
    for i, problem in enumerate(degradation[:3]):
        selected.append({
            'selection_reason': f"Degradation case #{i+1} (baseline ✓, blocked ✗)",
            **problem
        })

    # 6. No change impact (both correct despite divergence)
    print("Selecting no_change cases...")
    no_change = [p for p in enriched_problems
                 if p['impact'] == 'no_change'
                 and p['baseline_correct'] and p['ct0_blocked_correct']
                 and p['total_divergence'] > 0.3  # High divergence but still correct
                 and p['problem_id'] not in [s['problem_id'] for s in selected]]
    for i, problem in enumerate(no_change[:2]):
        selected.append({
            'selection_reason': f"High divergence but still correct (robustness)",
            **problem
        })

    # 7. Fill remaining with diverse patterns
    print("Filling remaining slots with diverse cases...")
    remaining = [p for p in enriched_problems if p['problem_id'] not in [s['problem_id'] for s in selected]]

    # Ensure we have variety in divergence profiles
    while len(selected) < n_cases and remaining:
        # Pick next problem with most different profile from selected
        best_candidate = None
        best_diversity_score = -1

        for candidate in remaining[:20]:  # Check top 20
            # Diversity score: how different is this from selected cases?
            diversity = 0
            for existing in selected:
                # Compare similarity profiles
                ct1_diff = abs(candidate['ct1_similarity'] - existing['ct1_similarity'])
                ct4_diff = abs(candidate['ct4_similarity'] - existing['ct4_similarity'])
                slope_diff = abs(candidate['divergence_slope'] - existing['divergence_slope'])

                diversity += ct1_diff + ct4_diff + slope_diff

            diversity_score = diversity / len(selected)

            if diversity_score > best_diversity_score:
                best_diversity_score = diversity_score
                best_candidate = candidate

        if best_candidate:
            selected.append({
                'selection_reason': f"Diverse profile (score={best_diversity_score:.3f})",
                **best_candidate
            })
            remaining.remove(best_candidate)
        else:
            break

    return selected[:n_cases]


def main():
    print("="*60)
    print("Selecting CT0 Divergence Case Studies")
    print("="*60)
    print()

    # Load data
    print("Loading data...")
    divergence_data, metadata = load_data()
    print(f"Loaded {len(divergence_data.get('problem_level_results', []))} problems with divergence data")
    print()

    # Select cases
    selected_cases = select_case_studies(divergence_data, metadata, n_cases=12)

    print()
    print("="*60)
    print(f"Selected {len(selected_cases)} Case Studies")
    print("="*60)
    print()

    # Print summary
    for i, case in enumerate(selected_cases, 1):
        print(f"Case {i}: Problem {case['problem_id']}")
        print(f"  Reason: {case['selection_reason']}")
        print(f"  Impact: {case['impact']}")
        print(f"  Baseline: {case['baseline_correct']} | CT0-blocked: {case['ct0_blocked_correct']}")
        print(f"  Divergence: CT1={case['ct1_similarity']:.3f}, CT4={case['ct4_similarity']:.3f}, slope={case['divergence_slope']:.3f}")
        print(f"  Pattern: {case['pattern']}")
        print()

    # Save results
    output_file = RESULTS_DIR / 'selected_divergence_cases.json'
    with open(output_file, 'w') as f:
        json.dump({
            'n_cases': len(selected_cases),
            'selection_criteria': {
                'high_divergence': 3,
                'early_divergence': 1,
                'late_divergence': 1,
                'steep_cascading': 1,
                'degradation_impact': 3,
                'no_change_impact': 2,
                'diverse_remaining': 1
            },
            'cases': selected_cases
        }, f, indent=2)

    print(f"✓ Saved {len(selected_cases)} case studies to: {output_file}")
    print()

    # Print statistics
    print("="*60)
    print("Case Study Statistics")
    print("="*60)
    print()

    patterns = defaultdict(int)
    impacts = defaultdict(int)

    for case in selected_cases:
        patterns[case['pattern']] += 1
        impacts[case['impact']] += 1

    print("Divergence Patterns:")
    for pattern, count in sorted(patterns.items()):
        print(f"  {pattern}: {count}")

    print()
    print("Impact Types:")
    for impact, count in sorted(impacts.items()):
        print(f"  {impact}: {count}")

    print()
    print("Divergence Ranges:")
    ct1_sims = [c['ct1_similarity'] for c in selected_cases]
    ct4_sims = [c['ct4_similarity'] for c in selected_cases]
    print(f"  CT1 similarity: {min(ct1_sims):.3f} - {max(ct1_sims):.3f} (mean: {np.mean(ct1_sims):.3f})")
    print(f"  CT4 similarity: {min(ct4_sims):.3f} - {max(ct4_sims):.3f} (mean: {np.mean(ct4_sims):.3f})")


if __name__ == '__main__':
    main()
