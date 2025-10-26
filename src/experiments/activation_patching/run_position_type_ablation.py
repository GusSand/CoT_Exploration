#!/usr/bin/env python3
"""
Position-type ablation experiment.

Ablate positions based on whether they decode to numbers vs non-numbers.
Tests causal importance of "number-encoding" positions.
"""

import sys
import json
import torch
from pathlib import Path
from tqdm import tqdm
from typing import List

# Add paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'core'))
sys.path.insert(0, str(Path(__file__).parent / 'scripts' / 'experiments'))
sys.path.insert(0, str(project_root / 'codi'))

from cache_activations import ActivationCacher
from run_ablation_N_tokens import NTokenPatcher, extract_answer_number, answers_match


def run_gpt2_position_ablation(max_samples=None):
    """Run GPT-2 position-type ablation experiment."""
    print("=" * 70)
    print("GPT-2 POSITION-TYPE ABLATION EXPERIMENT")
    print("=" * 70)

    # Paths
    model_path = str(Path.home() / 'codi_ckpt' / 'gpt2_gsm8k')
    decoding_file = Path("/home/paperspace/dev/CoT_Exploration/src/experiments/gpt2_token_ablation/results/gpt2_final_layer_decoding.json")
    output_file = Path("/home/paperspace/dev/CoT_Exploration/src/experiments/gpt2_token_ablation/results/gpt2_position_ablation.json")

    # Load model
    print(f"\nLoading GPT-2 model from {model_path}...")
    cacher = ActivationCacher(model_path)
    patcher = NTokenPatcher(cacher, num_tokens=6)

    # Load decoding results
    print(f"Loading decoding results...")
    with open(decoding_file, 'r') as f:
        decoding_data = json.load(f)

    samples = decoding_data['samples']
    if max_samples:
        samples = samples[:max_samples]
        print(f"  Running on first {max_samples} samples (test mode)")
    else:
        print(f"  Running on all {len(samples)} samples")

    results = []
    stats = {
        'n_samples': 0,
        'baseline_correct': 0,
        'ablate_numbers_correct': 0,
        'ablate_non_numbers_correct': 0,
        'n_has_numbers': 0,
        'n_has_non_numbers': 0
    }

    for sample in tqdm(samples, desc="GPT-2 Position Ablation"):
        question = sample['question']
        ground_truth = sample['ground_truth']

        # Identify position types
        number_positions = [
            d['position'] for d in sample['decoded_positions']
            if d['is_number']
        ]
        non_number_positions = [
            d['position'] for d in sample['decoded_positions']
            if not d['is_number']
        ]

        # Baseline (from existing results)
        baseline_correct = sample['is_correct']

        # Condition A: Ablate number positions
        ablate_numbers_correct = None
        if len(number_positions) > 0:
            output_a = ablate_positions(cacher, patcher, question, number_positions)
            pred_a = extract_answer_number(output_a)
            ablate_numbers_correct = answers_match(pred_a, ground_truth)
            stats['n_has_numbers'] += 1

        # Condition B: Ablate non-number positions
        ablate_non_numbers_correct = None
        if len(non_number_positions) > 0:
            output_b = ablate_positions(cacher, patcher, question, non_number_positions)
            pred_b = extract_answer_number(output_b)
            ablate_non_numbers_correct = answers_match(pred_b, ground_truth)
            stats['n_has_non_numbers'] += 1

        # Update stats
        stats['n_samples'] += 1
        if baseline_correct:
            stats['baseline_correct'] += 1
        if ablate_numbers_correct:
            stats['ablate_numbers_correct'] += 1
        if ablate_non_numbers_correct:
            stats['ablate_non_numbers_correct'] += 1

        results.append({
            'id': sample['id'],
            'number_positions': number_positions,
            'non_number_positions': non_number_positions,
            'baseline_correct': baseline_correct,
            'ablate_numbers_correct': ablate_numbers_correct,
            'ablate_non_numbers_correct': ablate_non_numbers_correct
        })

    # Calculate accuracies
    baseline_acc = 100 * stats['baseline_correct'] / stats['n_samples']
    ablate_num_acc = 100 * stats['ablate_numbers_correct'] / stats['n_has_numbers'] if stats['n_has_numbers'] > 0 else 0
    ablate_non_acc = 100 * stats['ablate_non_numbers_correct'] / stats['n_has_non_numbers'] if stats['n_has_non_numbers'] > 0 else 0

    output = {
        'model': 'GPT-2',
        'n_samples': stats['n_samples'],
        'accuracy': {
            'baseline': round(baseline_acc, 1),
            'ablate_number_positions': round(ablate_num_acc, 1),
            'ablate_non_number_positions': round(ablate_non_acc, 1),
            'drop_from_ablating_numbers': round(baseline_acc - ablate_num_acc, 1),
            'drop_from_ablating_non_numbers': round(baseline_acc - ablate_non_acc, 1)
        },
        'counts': stats,
        'samples': results
    }

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    # Print results
    print("\n" + "=" * 70)
    print("GPT-2 RESULTS")
    print("=" * 70)
    print(f"Baseline accuracy: {baseline_acc:.1f}%")
    print(f"Ablate number positions: {ablate_num_acc:.1f}% (drop: {baseline_acc - ablate_num_acc:.1f}%)")
    print(f"Ablate non-number positions: {ablate_non_acc:.1f}% (drop: {baseline_acc - ablate_non_acc:.1f}%)")
    print(f"\nSaved to: {output_file}")

    return output


def ablate_positions(cacher, patcher, question: str, positions_to_ablate: List[int], layer_name: str = 'middle'):
    """Ablate specific positions with zeros."""
    # Cache activations for all 6 positions
    all_activations = patcher.cache_N_token_activations(question, layer_name)

    # Create patched activations: zeros for selected positions, original for others
    patched_activations = []
    for pos in range(6):
        if pos in positions_to_ablate:
            # Ablate: replace with zeros
            patched_activations.append(torch.zeros_like(all_activations[pos]))
        else:
            # Keep original
            patched_activations.append(all_activations[pos])

    # Run with patched activations
    output = patcher.run_with_N_tokens_patched(
        problem_text=question,
        patch_activations=patched_activations,
        layer_name=layer_name,
        max_new_tokens=200
    )

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run on 100 samples only')
    args = parser.parse_args()

    max_samples = 100 if args.test else None
    run_gpt2_position_ablation(max_samples=max_samples)
