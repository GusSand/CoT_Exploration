#!/usr/bin/env python3
"""
Prepare LLaMA Steering Dataset - Full 532 Pairs

Uses ALL 532 pairs (relaxing CoT-dependency filter) to maximize statistical power.
Creates balanced train/test split with 80/20 ratio.
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# Set seed for reproducibility
random.seed(42)

def load_baseline_results():
    """Load existing LLaMA baseline results from validation."""
    baseline_file = Path(__file__).parent / 'validation_results_llama_gpt4_532.json'
    with open(baseline_file) as f:
        data = json.load(f)

    # Handle nested format
    baseline_data = data.get('results', data)
    return baseline_data


def create_balanced_split(correct_pairs, wrong_pairs, train_ratio=0.8):
    """Create balanced train/test split.

    Args:
        correct_pairs: List of correctly answered pairs
        wrong_pairs: List of wrongly answered pairs
        train_ratio: Fraction of data for training

    Returns:
        dict with train_correct, train_wrong, test_correct, test_wrong
    """
    # Shuffle both lists
    random.shuffle(correct_pairs)
    random.shuffle(wrong_pairs)

    # Split correct pairs
    n_correct = len(correct_pairs)
    n_train_correct = int(n_correct * train_ratio)
    train_correct = correct_pairs[:n_train_correct]
    test_correct = correct_pairs[n_train_correct:]

    # Split wrong pairs
    n_wrong = len(wrong_pairs)
    n_train_wrong = int(n_wrong * train_ratio)
    train_wrong = wrong_pairs[:n_train_wrong]
    test_wrong = wrong_pairs[n_train_wrong:]

    return {
        'train_correct': train_correct,
        'train_wrong': train_wrong,
        'test_correct': test_correct,
        'test_wrong': test_wrong
    }


def main():
    print("="*80)
    print("PREPARE LLAMA STEERING DATASET - FULL 532 PAIRS")
    print("="*80)

    # Load baseline results
    print("\nLoading baseline results...")
    baseline_results = load_baseline_results()

    # Load problem pairs
    pairs_file = Path(__file__).parent / 'problem_pairs_gpt4_answers.json'
    with open(pairs_file) as f:
        all_problems = json.load(f)

    problem_lookup = {p['pair_id']: p for p in all_problems}

    # Separate correct vs wrong
    correct_pairs = []
    wrong_pairs = []

    for result in baseline_results:
        pair_id = result['pair_id']

        # Check if baseline correct
        baseline_correct = result.get('clean', {}).get('correct', False)

        # Get expected answer
        problem = problem_lookup[pair_id]
        expected = problem['clean']['answer']

        pair_data = {
            'pair_id': pair_id,
            'expected': expected
        }

        if baseline_correct:
            correct_pairs.append(pair_data)
        else:
            wrong_pairs.append(pair_data)

    print(f"\nTotal pairs: {len(baseline_results)}")
    print(f"  Correct: {len(correct_pairs)} ({100*len(correct_pairs)/len(baseline_results):.1f}%)")
    print(f"  Wrong:   {len(wrong_pairs)} ({100*len(wrong_pairs)/len(baseline_results):.1f}%)")

    # Create balanced split (80/20)
    dataset = create_balanced_split(correct_pairs, wrong_pairs, train_ratio=0.8)

    print(f"\nBalanced Dataset (80/20 split):")
    print(f"  Training:")
    print(f"    Correct: {len(dataset['train_correct'])}")
    print(f"    Wrong:   {len(dataset['train_wrong'])}")
    print(f"    Total:   {len(dataset['train_correct']) + len(dataset['train_wrong'])}")
    print(f"  Test:")
    print(f"    Correct: {len(dataset['test_correct'])}")
    print(f"    Wrong:   {len(dataset['test_wrong'])}")
    print(f"    Total:   {len(dataset['test_correct']) + len(dataset['test_wrong'])}")

    # Save dataset
    output_file = Path(__file__).parent / 'results' / 'steering_dataset_llama_full.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"\n✓ Saved dataset to {output_file}")

    # Print comparison to small dataset
    print("\n" + "="*80)
    print("COMPARISON TO SMALL DATASET")
    print("="*80)

    small_dataset_file = Path(__file__).parent / 'results' / 'steering_dataset_llama.json'
    if small_dataset_file.exists():
        with open(small_dataset_file) as f:
            small_dataset = json.load(f)

        small_train = len(small_dataset['train_correct']) + len(small_dataset['train_wrong'])
        small_test = len(small_dataset['test_correct']) + len(small_dataset['test_wrong'])

        full_train = len(dataset['train_correct']) + len(dataset['train_wrong'])
        full_test = len(dataset['test_correct']) + len(dataset['test_wrong'])

        print(f"\nTraining samples:")
        print(f"  Small (CoT-dependent):  {small_train}")
        print(f"  Full (all pairs):       {full_train}")
        print(f"  Increase:               {full_train/small_train:.1f}×")

        print(f"\nTest samples:")
        print(f"  Small (CoT-dependent):  {small_test}")
        print(f"  Full (all pairs):       {full_test}")
        print(f"  Increase:               {full_test/small_test:.1f}×")

    print("\n" + "="*80)
    print("✅ DATASET PREPARATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
