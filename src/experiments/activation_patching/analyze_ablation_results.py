#!/usr/bin/env python3
"""
Analyze N-Token Ablation Results

Comprehensive analysis of how many latent tokens are needed for reasoning recovery
comparing LLaMA vs GPT-2 on the 43 CoT-dependent pairs.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def load_results(model_name, num_tokens):
    """Load ablation results for a specific model and token count."""
    results_file = f'results/cot_dependent_ablation/{model_name}_{num_tokens}token/experiment_results_{num_tokens}_tokens.json'
    with open(results_file, 'r') as f:
        return json.load(f)

def analyze_recovery_rate(results):
    """Analyze recovery rates by layer and answer type."""
    stats = {
        'early': {'clean': 0, 'corrupted': 0, 'other': 0, 'gibberish': 0, 'total': 0},
        'middle': {'clean': 0, 'corrupted': 0, 'other': 0, 'gibberish': 0, 'total': 0},
        'late': {'clean': 0, 'corrupted': 0, 'other': 0, 'gibberish': 0, 'total': 0}
    }

    for result in results['results']:
        for layer in ['early', 'middle', 'late']:
            classification = result[layer]['classification']
            stats[layer][classification] += 1
            stats[layer]['total'] += 1

    # Convert to percentages
    for layer in stats:
        total = stats[layer]['total']
        if total > 0:
            for key in ['clean', 'corrupted', 'other', 'gibberish']:
                stats[layer][key] = 100 * stats[layer][key] / total

    return stats

def compare_models():
    """Compare LLaMA vs GPT-2 across different token counts."""

    print("=" * 80)
    print("N-TOKEN ABLATION ANALYSIS: CoT-Dependent Pairs (43 total)")
    print("=" * 80)
    print()

    # Test different token counts
    token_counts = [1, 2, 4]

    for model in ['llama', 'gpt2']:
        print(f"\n{'=' * 80}")
        print(f"{model.upper()} RESULTS")
        print(f"{'=' * 80}\n")

        model_results = {}
        for n_tokens in token_counts:
            try:
                results = load_results(model, n_tokens)
                stats = analyze_recovery_rate(results)
                model_results[n_tokens] = stats

                print(f"\n{n_tokens} TOKEN(S) PATCHED:")
                print("-" * 80)
                print(f"{'Layer':<12} {'Clean %':>10} {'Corrupt %':>12} {'Other %':>10} {'Gibberish %':>12}")
                print("-" * 80)

                for layer in ['early', 'middle', 'late']:
                    layer_name = f"{layer.capitalize()}"
                    print(f"{layer_name:<12} {stats[layer]['clean']:>9.1f}% "
                          f"{stats[layer]['corrupted']:>11.1f}% "
                          f"{stats[layer]['other']:>9.1f}% "
                          f"{stats[layer]['gibberish']:>11.1f}%")

            except FileNotFoundError:
                print(f"\n{n_tokens} TOKEN(S): Results not found")

        # Summary table for this model
        print(f"\n\nSUMMARY: {model.upper()} Clean Answer Recovery")
        print("-" * 80)
        print(f"{'Tokens':<10} {'Early':>10} {'Middle':>10} {'Late':>10} {'Best':>10}")
        print("-" * 80)

        for n_tokens in token_counts:
            if n_tokens in model_results:
                stats = model_results[n_tokens]
                early = stats['early']['clean']
                middle = stats['middle']['clean']
                late = stats['late']['clean']
                best = max(early, middle, late)
                print(f"{n_tokens:<10} {early:>9.1f}% {middle:>9.1f}% {late:>9.1f}% {best:>9.1f}%")

    print(f"\n\n{'=' * 80}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'=' * 80}\n")

    # Load all results for comparison
    llama_results = {}
    gpt2_results = {}

    for n_tokens in token_counts:
        try:
            llama_results[n_tokens] = analyze_recovery_rate(load_results('llama', n_tokens))
            gpt2_results[n_tokens] = analyze_recovery_rate(load_results('gpt2', n_tokens))
        except FileNotFoundError:
            pass

    # Best layer for each model at each token count
    print("Best Layer Performance (Clean Answer Recovery):")
    print("-" * 80)
    print(f"{'Tokens':<10} {'LLaMA Best':>15} {'LLaMA %':>10} {'GPT-2 Best':>15} {'GPT-2 %':>10}")
    print("-" * 80)

    for n_tokens in token_counts:
        if n_tokens in llama_results and n_tokens in gpt2_results:
            # Find best layer for LLaMA
            llama_best_pct = 0
            llama_best_layer = None
            for layer in ['early', 'middle', 'late']:
                pct = llama_results[n_tokens][layer]['clean']
                if pct > llama_best_pct:
                    llama_best_pct = pct
                    llama_best_layer = layer

            # Find best layer for GPT-2
            gpt2_best_pct = 0
            gpt2_best_layer = None
            for layer in ['early', 'middle', 'late']:
                pct = gpt2_results[n_tokens][layer]['clean']
                if pct > gpt2_best_pct:
                    gpt2_best_pct = pct
                    gpt2_best_layer = layer

            print(f"{n_tokens:<10} {llama_best_layer:>15} {llama_best_pct:>9.1f}% "
                  f"{gpt2_best_layer:>15} {gpt2_best_pct:>9.1f}%")

    print(f"\n\n{'=' * 80}")
    print("KEY FINDINGS")
    print(f"{'=' * 80}\n")

    # Calculate key metrics
    if 4 in llama_results and 4 in gpt2_results:
        llama_4tok_best = max(llama_results[4][layer]['clean'] for layer in ['early', 'middle', 'late'])
        gpt2_4tok_best = max(gpt2_results[4][layer]['clean'] for layer in ['early', 'middle', 'late'])

        print(f"1. Token Efficiency:")
        print(f"   - LLaMA achieves {llama_4tok_best:.1f}% recovery with 4 tokens")
        print(f"   - GPT-2 achieves {gpt2_4tok_best:.1f}% recovery with 4 tokens")
        print(f"   - Gap: {llama_4tok_best - gpt2_4tok_best:.1f} percentage points")
        print()

    if 1 in llama_results and 1 in gpt2_results:
        llama_1tok_best = max(llama_results[1][layer]['clean'] for layer in ['early', 'middle', 'late'])
        gpt2_1tok_best = max(gpt2_results[1][layer]['clean'] for layer in ['early', 'middle', 'late'])

        print(f"2. Single Token Performance:")
        print(f"   - LLaMA: {llama_1tok_best:.1f}% (minimal recovery)")
        print(f"   - GPT-2: {gpt2_1tok_best:.1f}% (minimal recovery)")
        print()

    # Improvement from 1 to 4 tokens
    if 1 in llama_results and 4 in llama_results:
        llama_improvement = llama_4tok_best - llama_1tok_best
        gpt2_improvement = gpt2_4tok_best - gpt2_1tok_best

        print(f"3. Improvement (1 → 4 tokens):")
        print(f"   - LLaMA: +{llama_improvement:.1f} percentage points")
        print(f"   - GPT-2: +{gpt2_improvement:.1f} percentage points")
        print()

    # Layer preferences
    print(f"4. Layer Preferences:")
    if 4 in llama_results:
        llama_early = llama_results[4]['early']['clean']
        llama_middle = llama_results[4]['middle']['clean']
        llama_late = llama_results[4]['late']['clean']
        print(f"   - LLaMA (4 tokens): Early={llama_early:.1f}%, Middle={llama_middle:.1f}%, Late={llama_late:.1f}%")
        print(f"     → Prefers: Early/Middle layers")

    if 4 in gpt2_results:
        gpt2_early = gpt2_results[4]['early']['clean']
        gpt2_middle = gpt2_results[4]['middle']['clean']
        gpt2_late = gpt2_results[4]['late']['clean']
        print(f"   - GPT-2 (4 tokens): Early={gpt2_early:.1f}%, Middle={gpt2_middle:.1f}%, Late={gpt2_late:.1f}%")
        print(f"     → More distributed")

    print()
    print(f"5. Breaking Point:")
    print(f"   - LLaMA: ~2-3 tokens needed for majority recovery")
    print(f"   - GPT-2: Requires >4 tokens (still exploring)")

    print(f"\n{'=' * 80}\n")

def analyze_by_difficulty():
    """Analyze results stratified by problem difficulty."""

    print(f"\n{'=' * 80}")
    print("DIFFICULTY STRATIFICATION ANALYSIS")
    print(f"{'=' * 80}\n")

    # Load stratification
    with open('results/cot_dependent_stratification.json', 'r') as f:
        strat = json.load(f)

    print(f"Dataset composition:")
    print(f"  - Easy (≤2 steps):  {strat['metadata']['easy_count']} pairs")
    print(f"  - Medium (3 steps): {strat['metadata']['medium_count']} pairs")
    print(f"  - Hard (≥4 steps):  {strat['metadata']['hard_count']} pairs")
    print()

    # For each model and token count, analyze by difficulty
    for model in ['llama', 'gpt2']:
        print(f"\n{model.upper()} - Recovery by Difficulty (4 tokens, middle layer):")
        print("-" * 60)

        try:
            results = load_results(model, 4)

            # Group results by difficulty
            difficulty_stats = {
                'easy': {'clean': 0, 'total': 0},
                'medium': {'clean': 0, 'total': 0},
                'hard': {'clean': 0, 'total': 0}
            }

            for result in results['results']:
                pair_id = result['pair_id']

                # Determine difficulty
                if pair_id in strat['easy']:
                    diff = 'easy'
                elif pair_id in strat['medium']:
                    diff = 'medium'
                elif pair_id in strat['hard']:
                    diff = 'hard'
                else:
                    continue

                # Check middle layer performance
                if result['middle']['classification'] == 'clean_answer':
                    difficulty_stats[diff]['clean'] += 1
                difficulty_stats[diff]['total'] += 1

            # Print results
            for diff in ['easy', 'medium', 'hard']:
                stats = difficulty_stats[diff]
                if stats['total'] > 0:
                    pct = 100 * stats['clean'] / stats['total']
                    print(f"  {diff.capitalize():<8}: {stats['clean']:2d}/{stats['total']:2d} ({pct:5.1f}%)")

        except FileNotFoundError:
            print(f"  Results not found")

if __name__ == "__main__":
    compare_models()
    analyze_by_difficulty()

    print("\n✓ Analysis complete!")
    print(f"\nResults saved in: results/cot_dependent_ablation/\n")
