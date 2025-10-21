#!/usr/bin/env python3
"""
Prepare balanced dataset for GPT-2 activation steering experiment.

This script:
1. Loads GPT-2 validation results on 532 GPT-4 pairs
2. Separates clean problems into CORRECT and WRONG
3. Balances the dataset (n = min(correct, wrong))
4. Creates 80/20 train/test split
5. Saves problem IDs and metadata
"""

import json
import random
from pathlib import Path
from collections import Counter

# Set random seed for reproducibility
random.seed(42)

# Paths
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def load_validation_results():
    """Load GPT-2 validation results."""
    print("Loading GPT-2 validation results...")

    with open(BASE_DIR / "validation_results_gpt2_gpt4_532.json") as f:
        data = json.load(f)

    print(f"Total problems: {data['num_pairs']}")
    print(f"Statistics: {data['statistics']}")

    return data['results']

def load_difficulty_metrics():
    """Load difficulty metrics for problems."""
    print("\nLoading difficulty metrics...")

    with open(RESULTS_DIR / "matched_pairs_difficulty_analysis.json") as f:
        difficulty_data = json.load(f)

    # Create lookup by pair_id
    difficulty_lookup = {d['pair_id']: d for d in difficulty_data}

    print(f"Loaded difficulty metrics for {len(difficulty_lookup)} problems")

    return difficulty_lookup

def categorize_difficulty(steps):
    """Categorize problem difficulty by reasoning steps."""
    if steps <= 2:
        return 'easy'
    elif steps == 3:
        return 'medium'
    else:
        return 'hard'

def prepare_dataset(results, difficulty_lookup):
    """Separate into CORRECT and WRONG, then balance."""
    print("\n" + "="*80)
    print("CATEGORIZING PROBLEMS")
    print("="*80)

    correct_problems = []
    wrong_problems = []

    for result in results:
        pair_id = result['pair_id']
        clean_correct = result['clean']['correct']

        # Get difficulty if available
        difficulty_info = difficulty_lookup.get(pair_id, None)

        problem_info = {
            'pair_id': pair_id,
            'predicted': result['clean']['predicted'],
            'expected': result['clean']['expected'],
            'correct': clean_correct
        }

        # Add difficulty metrics if available
        if difficulty_info:
            problem_info['steps'] = difficulty_info['steps']
            problem_info['total_operations'] = difficulty_info['total_operations']
            problem_info['difficulty'] = categorize_difficulty(difficulty_info['steps'])
        else:
            problem_info['steps'] = None
            problem_info['total_operations'] = None
            problem_info['difficulty'] = 'unknown'

        if clean_correct:
            correct_problems.append(problem_info)
        else:
            wrong_problems.append(problem_info)

    print(f"\nCORRECT problems: {len(correct_problems)}")
    print(f"WRONG problems: {len(wrong_problems)}")

    # Show difficulty distribution for CORRECT
    if any(p['difficulty'] != 'unknown' for p in correct_problems):
        correct_diff_dist = Counter(p['difficulty'] for p in correct_problems if p['difficulty'] != 'unknown')
        print(f"\nCORRECT difficulty distribution: {dict(correct_diff_dist)}")

    # Show difficulty distribution for WRONG
    if any(p['difficulty'] != 'unknown' for p in wrong_problems):
        wrong_diff_dist = Counter(p['difficulty'] for p in wrong_problems if p['difficulty'] != 'unknown')
        print(f"WRONG difficulty distribution: {dict(wrong_diff_dist)}")

    return correct_problems, wrong_problems

def balance_dataset(correct_problems, wrong_problems):
    """Balance dataset by randomly sampling equal numbers from each category."""
    print("\n" + "="*80)
    print("BALANCING DATASET")
    print("="*80)

    n = min(len(correct_problems), len(wrong_problems))
    print(f"\nBalanced n = {n} (using {n} from each category)")

    # Randomly sample n from each
    random.shuffle(correct_problems)
    random.shuffle(wrong_problems)

    balanced_correct = correct_problems[:n]
    balanced_wrong = wrong_problems[:n]

    # Verify difficulty distribution in balanced set
    if any(p['difficulty'] != 'unknown' for p in balanced_correct):
        balanced_correct_diff = Counter(p['difficulty'] for p in balanced_correct if p['difficulty'] != 'unknown')
        print(f"\nBalanced CORRECT difficulty: {dict(balanced_correct_diff)}")

    if any(p['difficulty'] != 'unknown' for p in balanced_wrong):
        balanced_wrong_diff = Counter(p['difficulty'] for p in balanced_wrong if p['difficulty'] != 'unknown')
        print(f"Balanced WRONG difficulty: {dict(balanced_wrong_diff)}")

    return balanced_correct, balanced_wrong, n

def create_train_test_split(correct_problems, wrong_problems, test_ratio=0.2):
    """Create 80/20 train/test split."""
    print("\n" + "="*80)
    print("CREATING TRAIN/TEST SPLIT")
    print("="*80)

    n = len(correct_problems)
    test_size = int(n * test_ratio)
    train_size = n - test_size

    print(f"\nTotal per category: {n}")
    print(f"Train size per category: {train_size} ({100*(1-test_ratio):.0f}%)")
    print(f"Test size per category: {test_size} ({100*test_ratio:.0f}%)")

    # Split CORRECT
    train_correct = correct_problems[:train_size]
    test_correct = correct_problems[train_size:]

    # Split WRONG
    train_wrong = wrong_problems[:train_size]
    test_wrong = wrong_problems[train_size:]

    print(f"\nTrain set: {len(train_correct)} correct + {len(train_wrong)} wrong = {len(train_correct) + len(train_wrong)} total")
    print(f"Test set: {len(test_correct)} correct + {len(test_wrong)} wrong = {len(test_correct) + len(test_wrong)} total")

    return {
        'train_correct': train_correct,
        'train_wrong': train_wrong,
        'test_correct': test_correct,
        'test_wrong': test_wrong
    }

def save_dataset(dataset, n):
    """Save dataset to JSON."""
    output_file = RESULTS_DIR / "steering_dataset_gpt2.json"

    output = {
        'model': 'gpt2',
        'total_pairs_per_category': n,
        'train_size_per_category': len(dataset['train_correct']),
        'test_size_per_category': len(dataset['test_correct']),
        'train_correct': dataset['train_correct'],
        'train_wrong': dataset['train_wrong'],
        'test_correct': dataset['test_correct'],
        'test_wrong': dataset['test_wrong'],
        'summary': {
            'total_problems': n * 2,
            'train_problems': len(dataset['train_correct']) + len(dataset['train_wrong']),
            'test_problems': len(dataset['test_correct']) + len(dataset['test_wrong'])
        }
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*80)
    print(f"Dataset saved to: {output_file}")
    print("="*80)

    return output_file

def main():
    """Main pipeline."""
    print("="*80)
    print("GPT-2 STEERING DATASET PREPARATION")
    print("="*80)

    # Load data
    results = load_validation_results()
    difficulty_lookup = load_difficulty_metrics()

    # Categorize
    correct_problems, wrong_problems = prepare_dataset(results, difficulty_lookup)

    # Balance
    balanced_correct, balanced_wrong, n = balance_dataset(correct_problems, wrong_problems)

    # Train/test split
    dataset = create_train_test_split(balanced_correct, balanced_wrong, test_ratio=0.2)

    # Save
    output_file = save_dataset(dataset, n)

    print("\n✅ Dataset preparation complete!")
    print(f"✅ Using {n} balanced pairs ({n} correct, {n} wrong)")
    print(f"✅ Train: {dataset['train_correct'][0]['pair_id']} problems per category")
    print(f"✅ Test: {len(dataset['test_correct'])} problems per category")

    return output_file

if __name__ == "__main__":
    main()
