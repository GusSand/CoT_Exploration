"""
Operation classification using Matryoshka SAE features.

Tests classification at each hierarchy level (512, 1024, 2048) and compares
to ReLU baseline.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
import json
import re

from matryoshka_sae import MatryoshkaSAE


def extract_operation_from_cot(cot_text: str) -> str:
    """Extract primary operation type from CoT text.

    Args:
        cot_text: Chain-of-thought text

    Returns:
        Operation type: 'multiplication', 'addition', or 'division'
    """
    # Look for operation indicators in the CoT
    mult_indicators = ['*', '×', 'multiply', 'product', 'times']
    add_indicators = ['+', 'add', 'sum', 'total', 'plus']
    div_indicators = ['/', '÷', 'divide', 'quotient', 'per']

    # Count occurrences
    mult_count = sum(1 for ind in mult_indicators if ind in cot_text.lower())
    add_count = sum(1 for ind in add_indicators if ind in cot_text.lower())
    div_count = sum(1 for ind in div_indicators if ind in cot_text.lower())

    # Return most common operation
    counts = {
        'multiplication': mult_count,
        'addition': add_count,
        'division': div_count
    }

    return max(counts, key=counts.get)


def load_gsm8k_with_operations(n_samples: int = 1000):
    """Load GSM8K data and extract operation labels.

    Args:
        n_samples: Number of samples to load

    Returns:
        List of (problem_id, operation_type) tuples
    """
    from datasets import load_dataset

    print(f"Loading {n_samples} GSM8K problems...")
    dataset = load_dataset('gsm8k', 'main', split='train')

    problem_ops = []
    for i, example in enumerate(dataset):
        if i >= n_samples:
            break

        cot = example['answer']  # GSM8K answer contains the CoT
        op_type = extract_operation_from_cot(cot)
        problem_ops.append((f"train_{i}", op_type))

    print(f"  Loaded {len(problem_ops)} problems")

    # Print distribution
    from collections import Counter
    op_dist = Counter([op for _, op in problem_ops])
    print(f"  Operation distribution:")
    for op, count in op_dist.most_common():
        print(f"    {op}: {count} ({count/len(problem_ops)*100:.1f}%)")

    return problem_ops


def extract_matryoshka_features(model, data, device='cuda'):
    """Extract features at all three Matryoshka levels.

    Args:
        model: Trained Matryoshka SAE
        data: Activation data

    Returns:
        Dictionary mapping level → features
    """
    print("\nExtracting Matryoshka features at all levels...")

    model.eval()
    with torch.no_grad():
        # Extract features at all levels
        z1 = model.encode_level(data.to(device), 0)  # 512 features
        z2 = model.encode_level(data.to(device), 1)  # 1024 features
        z3 = model.encode_level(data.to(device), 2)  # 2048 features

    features = {
        'level_1': z1.cpu().numpy(),
        'level_2': z2.cpu().numpy(),
        'level_3': z3.cpu().numpy(),
        'concatenated': np.concatenate([
            z1.cpu().numpy(),
            z2.cpu().numpy(),
            z3.cpu().numpy()
        ], axis=1)  # Total: 512 + 1024 + 2048 = 3584 features
    }

    print(f"  Level 1 (512): {features['level_1'].shape}")
    print(f"  Level 2 (1024): {features['level_2'].shape}")
    print(f"  Level 3 (2048): {features['level_3'].shape}")
    print(f"  Concatenated (3584): {features['concatenated'].shape}")

    return features


def classify_operations(features_dict, labels, test_size=0.2):
    """Train classifiers for each feature level.

    Args:
        features_dict: Dictionary mapping level → features
        labels: Operation labels
        test_size: Test split ratio

    Returns:
        Results dictionary
    """
    results = {}

    for level_name, features in features_dict.items():
        print(f"\n{level_name}:")
        print(f"  Features shape: {features.shape}")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )

        # Train logistic regression
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_test)

        # Compute accuracy
        accuracy = accuracy_score(y_test, y_pred)

        print(f"  Train samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Accuracy: {accuracy:.1%}")

        # Store results
        results[level_name] = {
            'accuracy': accuracy,
            'n_features': features.shape[1],
            'n_train': len(X_train),
            'n_test': len(X_test)
        }

    return results


def compare_to_baseline(matryoshka_results):
    """Compare to ReLU SAE baseline.

    Args:
        matryoshka_results: Results from Matryoshka classification

    Returns:
        Comparison dictionary
    """
    # ReLU baseline from sae_pilot
    relu_baseline = {
        'accuracy': 0.70,  # 70% from classification_results.json
        'n_features': 8192,
        'architecture': 'ReLU SAE'
    }

    print("\n" + "=" * 60)
    print("Classification Results: Matryoshka vs ReLU Baseline")
    print("=" * 60)

    print(f"\n{'Level':<20} {'Features':<12} {'Accuracy':<12} {'vs Baseline':<15}")
    print("-" * 60)

    print(f"{'ReLU Baseline':<20} {relu_baseline['n_features']:<12,} {relu_baseline['accuracy']:<11.1%} {'---':<15}")

    for level_name, results in matryoshka_results.items():
        acc = results['accuracy']
        n_feat = results['n_features']
        diff = (acc - relu_baseline['accuracy']) * 100

        print(f"{level_name:<20} {n_feat:<12,} {acc:<11.1%} {diff:+.1f} pts")

    return {
        'relu_baseline': relu_baseline,
        'matryoshka': matryoshka_results
    }


if __name__ == "__main__":
    # Paths
    base_dir = Path(__file__).parent
    model_path = base_dir / "models/pos3_hierarchical.pt"
    data_path = base_dir / "data/position_3_activations.pt"
    output_path = base_dir / "results/classification_results.json"

    print("=" * 60)
    print("Matryoshka SAE - Operation Classification")
    print("=" * 60)

    # Load model
    print("\nLoading Matryoshka SAE...")
    checkpoint = torch.load(model_path)
    config = checkpoint['config']

    model = MatryoshkaSAE(
        input_dim=config['input_dim'],
        levels=config['levels'],
        l1_coefficient=config['l1_coefficient'],
        level_weights=config['level_weights']
    ).to('cuda')

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  Loaded from epoch {checkpoint['epoch'] + 1}")

    # Load position 3 data
    print("\nLoading Position 3 data...")
    data_full = torch.load(data_path)
    activations = data_full['activations']
    problem_ids = data_full['metadata']['problem_ids']

    print(f"  Total samples: {len(activations):,}")

    # Load GSM8K and extract operation labels
    # For simplicity, we'll use the first N unique problems
    n_unique_problems = len(set(problem_ids))
    print(f"  Unique problems: {n_unique_problems:,}")

    # Extract operation labels
    problem_ops = load_gsm8k_with_operations(n_samples=min(6000, n_unique_problems))
    op_labels_dict = {pid: op for pid, op in problem_ops}

    # Match problem IDs to labels
    labels = []
    valid_indices = []

    for idx, pid in enumerate(problem_ids):
        if pid in op_labels_dict:
            labels.append(op_labels_dict[pid])
            valid_indices.append(idx)

    labels = np.array(labels)
    valid_activations = activations[valid_indices]

    print(f"\n  Matched samples: {len(labels):,}")

    # Extract features at all levels
    features_dict = extract_matryoshka_features(model, valid_activations)

    # Classify operations
    print("\n" + "=" * 60)
    print("Training Classifiers")
    print("=" * 60)

    results = classify_operations(features_dict, labels)

    # Compare to baseline
    comparison = compare_to_baseline(results)

    # Save results
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")
    print("\n✓ Classification complete!")
