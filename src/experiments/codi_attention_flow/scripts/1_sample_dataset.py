#!/usr/bin/env python3
"""
Dataset Sampler - Story 1.1

Sample 100 problems from GSM8K training set for attention flow analysis.

Usage:
    python 1_sample_dataset.py [--seed SEED] [--n_samples N]

Output:
    ../data/attention_dataset_100_train.json
"""
import json
import random
import argparse
from pathlib import Path
from datasets import load_dataset


def extract_answer(answer_str: str) -> str:
    """
    Extract numerical answer from GSM8K answer string.

    GSM8K format: "Step 1\nStep 2\n#### 42"

    Args:
        answer_str: Full answer string with reasoning and final answer

    Returns:
        answer: Numerical answer as string (e.g., "42")
    """
    if '####' in answer_str:
        return answer_str.split('####')[-1].strip()
    return answer_str.strip()


def sample_gsm8k_training(n_samples: int = 100, seed: int = 42) -> list[dict]:
    """
    Sample N problems from GSM8K training set.

    Args:
        n_samples: Number of problems to sample
        seed: Random seed for reproducibility

    Returns:
        dataset: List of dicts with keys:
            - gsm8k_id: "train_{idx}"
            - question: Problem text
            - answer: Numerical answer
            - full_solution: Complete GSM8K solution
    """
    print("=" * 80)
    print("DATASET SAMPLER - Story 1.1")
    print("=" * 80)

    # Load GSM8K training set
    print("\nLoading GSM8K training set from HuggingFace...")
    train_dataset = load_dataset('gsm8k', 'main', split='train')
    print(f"✓ Loaded {len(train_dataset)} training problems")

    # Random sampling with seed
    print(f"\nSampling {n_samples} problems with seed={seed}...")
    random.seed(seed)
    sampled_indices = random.sample(range(len(train_dataset)), n_samples)
    sampled_indices.sort()  # Keep original order for reproducibility

    # Create dataset
    dataset = []
    for idx in sampled_indices:
        problem = train_dataset[idx]

        dataset.append({
            'gsm8k_id': f'train_{idx}',
            'question': problem['question'],
            'answer': extract_answer(problem['answer']),
            'full_solution': problem['answer']
        })

    # Validation checks
    print("\nValidation checks:")

    # Check 1: No duplicates
    ids = [p['gsm8k_id'] for p in dataset]
    assert len(set(ids)) == len(ids), "Duplicate IDs found!"
    print(f"✓ No duplicates: {len(set(ids))} unique IDs")

    # Check 2: All questions non-empty
    assert all(len(p['question']) > 0 for p in dataset), "Empty questions found!"
    print(f"✓ All questions non-empty: min length = {min(len(p['question']) for p in dataset)}")

    # Check 3: All answers present
    assert all(p['answer'] for p in dataset), "Missing answers found!"
    print(f"✓ All answers present: {len([p for p in dataset if p['answer']])} problems")

    # Check 4: All IDs are from training set
    assert all(id.startswith('train_') for id in ids), "Non-training IDs found!"
    print(f"✓ All IDs from training set: {ids[0]} ... {ids[-1]}")

    # Statistics
    print("\nDataset statistics:")
    print(f"  Total problems: {len(dataset)}")
    print(f"  Avg question length: {sum(len(p['question']) for p in dataset) / len(dataset):.1f} chars")
    print(f"  Sample question: {dataset[0]['question'][:80]}...")
    print(f"  Sample answer: {dataset[0]['answer']}")

    return dataset


def save_dataset(dataset: list[dict], output_path: Path) -> None:
    """Save dataset to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"\n✓ Saved dataset to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description='Sample GSM8K training problems')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_samples', type=int, default=100, help='Number of samples')
    args = parser.parse_args()

    # Sample dataset
    dataset = sample_gsm8k_training(n_samples=args.n_samples, seed=args.seed)

    # Save to data directory
    output_path = Path(__file__).parent.parent / 'data' / 'attention_dataset_100_train.json'
    save_dataset(dataset, output_path)

    print("\n" + "=" * 80)
    print("STORY 1.1 COMPLETE ✓")
    print("=" * 80)
    print("\nNext step: Run Story 1.2 to extract attention patterns")
    print("  python 2_extract_attention_6x6.py")


if __name__ == '__main__':
    main()
