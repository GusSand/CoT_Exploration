"""
Fair comparison: Test ReLU SAE on the SAME dataset as Matryoshka.

Critical controls:
1. Same test set (15,290 samples)
2. Same position (Position 3)
3. Same evaluation protocol
4. Same train/test split
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path
import json

from matryoshka_sae import MatryoshkaSAE


class ReLUSAE(nn.Module):
    """ReLU SAE from sae_pilot."""

    def __init__(self, input_dim=2048, n_features=8192):
        super().__init__()
        self.input_dim = input_dim
        self.n_features = n_features
        self.encoder = nn.Linear(input_dim, n_features, bias=True)
        self.decoder = nn.Linear(n_features, input_dim, bias=True)

    def encode(self, x):
        return torch.relu(self.encoder(x))

    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z), z


def load_relu_sae(checkpoint_path, device='cuda'):
    """Load ReLU SAE from checkpoint."""
    print("Loading ReLU SAE...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint['config']
    model = ReLUSAE(
        input_dim=config['input_dim'],
        n_features=config['n_features']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  Input dim: {config['input_dim']}")
    print(f"  Features: {config['n_features']}")

    return model


def load_matryoshka_sae(checkpoint_path, device='cuda'):
    """Load Matryoshka SAE from checkpoint."""
    print("\nLoading Matryoshka SAE...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint['config']
    model = MatryoshkaSAE(
        input_dim=config['input_dim'],
        levels=config['levels'],
        l1_coefficient=config['l1_coefficient'],
        level_weights=config['level_weights']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  Levels: {config['levels']}")

    return model


def extract_relu_features(model, data, device='cuda'):
    """Extract ReLU SAE features."""
    print("\nExtracting ReLU SAE features...")
    model.eval()
    with torch.no_grad():
        _, features = model(data.to(device))

    features_np = features.cpu().numpy()
    print(f"  Shape: {features_np.shape}")

    return features_np


def extract_matryoshka_features(model, data, device='cuda'):
    """Extract Matryoshka features at all levels."""
    print("\nExtracting Matryoshka features...")
    model.eval()
    with torch.no_grad():
        z1 = model.encode_level(data.to(device), 0)
        z2 = model.encode_level(data.to(device), 1)
        z3 = model.encode_level(data.to(device), 2)

    features = {
        'level_1': z1.cpu().numpy(),
        'level_2': z2.cpu().numpy(),
        'level_3': z3.cpu().numpy(),
        'concatenated': np.concatenate([
            z1.cpu().numpy(),
            z2.cpu().numpy(),
            z3.cpu().numpy()
        ], axis=1)
    }

    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")

    return features


def classify_fair_comparison(relu_features, matryoshka_features, labels, test_size=0.2):
    """Run classification with identical train/test split for both models."""
    print("\n" + "=" * 60)
    print("Fair Classification Comparison")
    print("=" * 60)

    # Use SAME random state for identical splits
    random_state = 42

    results = {}

    # ReLU SAE
    print("\nReLU SAE (8192 features):")
    X_train, X_test, y_train, y_test = train_test_split(
        relu_features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    relu_accuracy = accuracy_score(y_test, y_pred)

    print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")
    print(f"  Accuracy: {relu_accuracy:.1%}")

    results['relu'] = {
        'accuracy': relu_accuracy,
        'n_features': relu_features.shape[1],
        'n_train': len(X_train),
        'n_test': len(X_test)
    }

    # Matryoshka levels
    for level_name, features in matryoshka_features.items():
        print(f"\nMatryoshka {level_name}:")

        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=random_state, stratify=labels
        )

        clf = LogisticRegression(max_iter=1000, random_state=random_state)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")
        print(f"  Accuracy: {accuracy:.1%}")

        results[f'matryoshka_{level_name}'] = {
            'accuracy': accuracy,
            'n_features': features.shape[1],
            'n_train': len(X_train),
            'n_test': len(X_test)
        }

    return results


def print_comparison_table(results):
    """Print comparison table."""
    print("\n" + "=" * 60)
    print("FAIR COMPARISON: Same Data, Same Protocol")
    print("=" * 60)

    relu_acc = results['relu']['accuracy']

    print(f"\n{'Model':<30} {'Features':<12} {'Accuracy':<12} {'vs ReLU':<15}")
    print("-" * 70)

    print(f"{'ReLU SAE':<30} {results['relu']['n_features']:<12,} {relu_acc:<11.1%} {'baseline':<15}")

    for key in ['matryoshka_level_1', 'matryoshka_level_2', 'matryoshka_level_3', 'matryoshka_concatenated']:
        if key in results:
            acc = results[key]['accuracy']
            n_feat = results[key]['n_features']
            diff = (acc - relu_acc) * 100

            name = key.replace('matryoshka_', '').replace('_', ' ').title()
            print(f"{name:<30} {n_feat:<12,} {acc:<11.1%} {diff:+.1f} pts")

    print("\n" + "=" * 60)
    print("Key Insights")
    print("=" * 60)

    best_matryoshka = max(
        [(k, v['accuracy']) for k, v in results.items() if 'matryoshka' in k],
        key=lambda x: x[1]
    )

    best_name = best_matryoshka[0].replace('matryoshka_', '').replace('_', ' ').title()
    best_acc = best_matryoshka[1]
    diff = (best_acc - relu_acc) * 100

    print(f"\nBest Matryoshka: {best_name}")
    print(f"  Accuracy: {best_acc:.1%}")
    print(f"  vs ReLU: {diff:+.1f} pts")

    if diff > 0:
        print(f"\n✓ Matryoshka OUTPERFORMS ReLU by {diff:.1f} pts")
    elif diff > -5:
        print(f"\n≈ Matryoshka MATCHES ReLU ({diff:.1f} pts difference)")
    else:
        print(f"\n✗ Matryoshka UNDERPERFORMS ReLU by {abs(diff):.1f} pts")


if __name__ == "__main__":
    base_dir = Path(__file__).parent

    # Paths
    relu_checkpoint = base_dir.parent / "sae_pilot/results/sae_weights.pt"
    matryoshka_checkpoint = base_dir / "models/pos3_hierarchical.pt"
    data_path = base_dir / "data/position_3_activations.pt"
    labels_path = base_dir / "results/classification_results.json"
    output_path = base_dir / "results/fair_comparison.json"

    print("=" * 60)
    print("FAIR COMPARISON: ReLU SAE vs Matryoshka SAE")
    print("=" * 60)
    print("\nControls:")
    print("  ✓ Same dataset (Position 3, 15,290 test samples)")
    print("  ✓ Same train/test split (random_state=42)")
    print("  ✓ Same classifier (LogisticRegression)")
    print("  ✓ Same evaluation protocol")

    # Load models
    relu_model = load_relu_sae(str(relu_checkpoint))
    matryoshka_model = load_matryoshka_sae(str(matryoshka_checkpoint))

    # Load data and labels
    print("\nLoading test data...")
    data_full = torch.load(data_path)
    activations = data_full['activations']

    # Load labels from previous classification
    with open(labels_path) as f:
        prev_results = json.load(f)

    # Reconstruct labels (need to re-extract from GSM8K)
    print("Loading GSM8K operation labels...")
    from datasets import load_dataset

    dataset = load_dataset('gsm8k', 'main', split='train')
    problem_ids = data_full['metadata']['problem_ids']

    # Extract operation labels
    def extract_operation(cot_text):
        mult_count = sum(1 for c in ['*', '×'] if c in cot_text)
        add_count = sum(1 for c in ['+'] if c in cot_text)
        div_count = sum(1 for c in ['/', '÷'] if c in cot_text)

        counts = {'multiplication': mult_count, 'addition': add_count, 'division': div_count}
        return max(counts, key=counts.get)

    labels = []
    valid_indices = []

    for idx, pid in enumerate(problem_ids):
        # Extract problem index from pid (e.g., "train_410" -> 410)
        try:
            problem_idx = int(pid.split('_')[1])
            if problem_idx < len(dataset):
                cot = dataset[problem_idx]['answer']
                op = extract_operation(cot)
                labels.append(op)
                valid_indices.append(idx)
        except:
            continue

    labels = np.array(labels)
    valid_activations = activations[valid_indices]

    print(f"  Matched samples: {len(labels):,}")

    # Extract features
    relu_features = extract_relu_features(relu_model, valid_activations)
    matryoshka_features = extract_matryoshka_features(matryoshka_model, valid_activations)

    # Run fair comparison
    results = classify_fair_comparison(relu_features, matryoshka_features, labels)

    # Print comparison
    print_comparison_table(results)

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")
    print("\n✓ Fair comparison complete!")
