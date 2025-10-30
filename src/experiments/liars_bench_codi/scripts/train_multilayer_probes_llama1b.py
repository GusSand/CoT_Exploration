"""
Train logistic regression probes for all layer-position combinations.

This script trains 48 probes (6 layers × 8 positions) to detect deception
at different points in the model's processing pipeline.

Input:
- multilayer_activations_llama1b_{epoch}_train.json
- multilayer_activations_llama1b_{epoch}_test.json

Output:
- multilayer_probe_results_llama1b_{epoch}.json
  Contains accuracy, precision, recall, F1 for each layer-position combination

Author: Claude Code
Date: 2025-10-30
Experiment: Pre-compression deception signal analysis
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import argparse


def load_activations(data_path):
    """Load extracted activations."""
    print(f"  Loading: {data_path}")
    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"    Samples: {len(data['labels'])}")
    print(f"    Layers: {data['metadata']['layers']}")
    print(f"    Positions: {data['metadata']['positions']}")

    return data


def train_probe(X_train, y_train, X_test, y_test, layer_name, position_name):
    """
    Train a single logistic regression probe.

    Args:
        X_train: Training features (n_samples, hidden_dim)
        y_train: Training labels (n_samples,)
        X_test: Test features (n_samples, hidden_dim)
        y_test: Test labels (n_samples,)
        layer_name: e.g., 'layer_0'
        position_name: e.g., 'ct0'

    Returns:
        dict: Probe results with metrics
    """
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression
    clf = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        random_state=42
    )
    clf.fit(X_train_scaled, y_train)

    # Predict
    y_train_pred = clf.predict(X_train_scaled)
    y_test_pred = clf.predict(X_test_scaled)

    # Compute metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)

    return {
        'layer': layer_name,
        'position': position_name,
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1': float(test_f1),
        'overfitting_gap': float(train_acc - test_acc),
        'confusion_matrix': cm.tolist(),
        'train_samples': len(y_train),
        'test_samples': len(y_test)
    }


def main():
    print("\n" + "=" * 80)
    print("MULTI-LAYER PROBE TRAINING - LLAMA-3.2-1B")
    print("Pre-Compression Deception Signal Analysis")
    print("=" * 80)
    print()

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epoch',
        type=str,
        default='5ep',
        help='Epoch identifier (5ep, 10ep, 15ep)'
    )
    args = parser.parse_args()

    epoch_str = args.epoch

    # Paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data" / "processed"

    train_path = data_dir / f"multilayer_activations_llama1b_{epoch_str}_train.json"
    test_path = data_dir / f"multilayer_activations_llama1b_{epoch_str}_test.json"

    print("=" * 80)
    print("LOADING ACTIVATIONS")
    print("=" * 80)

    train_data = load_activations(train_path)
    test_data = load_activations(test_path)
    print()

    # Extract labels
    y_train = np.array(train_data['labels'])
    y_test = np.array(test_data['labels'])

    print("Label distribution:")
    print(f"  Train: {np.sum(y_train == 0)} honest / {np.sum(y_train == 1)} deceptive")
    print(f"  Test: {np.sum(y_test == 0)} honest / {np.sum(y_test == 1)} deceptive")
    print()

    # Train probes for all layer-position combinations
    layers = train_data['metadata']['layers']
    positions = train_data['metadata']['positions']

    print("=" * 80)
    print(f"TRAINING PROBES: {len(layers)} layers × {len(positions)} positions = {len(layers) * len(positions)} probes")
    print("=" * 80)
    print()

    results = {}

    total_probes = len(layers) * len(positions)
    pbar = tqdm(total=total_probes, desc="Training probes")

    for layer_idx in layers:
        layer_name = f'layer_{layer_idx}'
        results[layer_name] = {}

        for position_name in positions:
            # Extract activations for this layer-position
            X_train = np.array(train_data['activations'][layer_name][position_name])
            X_test = np.array(test_data['activations'][layer_name][position_name])

            # Train probe
            probe_result = train_probe(
                X_train, y_train,
                X_test, y_test,
                layer_name, position_name
            )

            results[layer_name][position_name] = probe_result

            pbar.set_description(
                f"{layer_name}/{position_name}: {probe_result['test_accuracy']:.1%}"
            )
            pbar.update(1)

    pbar.close()
    print()

    # Add metadata
    results['metadata'] = {
        'model': 'LLaMA-3.2-1B',
        'epoch': epoch_str,
        'layers': layers,
        'positions': positions,
        'train_samples': len(y_train),
        'test_samples': len(y_test),
        'total_probes': len(layers) * len(positions)
    }

    # Save results
    results_dir = script_dir.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / f"multilayer_probe_results_llama1b_{epoch_str}.json"

    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    print(f"  Output: {output_path}")
    print()

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary statistics
    print("=" * 80)
    print("PROBE ACCURACY SUMMARY")
    print("=" * 80)
    print()

    # Compute statistics by layer
    print("By Layer:")
    for layer_idx in layers:
        layer_name = f'layer_{layer_idx}'
        layer_accs = [
            results[layer_name][pos]['test_accuracy']
            for pos in positions
        ]
        mean_acc = np.mean(layer_accs)
        std_acc = np.std(layer_accs)
        print(f"  {layer_name:10s}: {mean_acc:.1%} ± {std_acc:.1%}")

    print()

    # Compute statistics by position
    print("By Position:")
    for position_name in positions:
        pos_accs = [
            results[f'layer_{layer_idx}'][position_name]['test_accuracy']
            for layer_idx in layers
        ]
        mean_acc = np.mean(pos_accs)
        std_acc = np.std(pos_accs)
        print(f"  {position_name:15s}: {mean_acc:.1%} ± {std_acc:.1%}")

    print()

    # Find best and worst probes
    all_results = []
    for layer_idx in layers:
        layer_name = f'layer_{layer_idx}'
        for position_name in positions:
            all_results.append({
                'layer': layer_name,
                'position': position_name,
                'accuracy': results[layer_name][position_name]['test_accuracy']
            })

    all_results.sort(key=lambda x: x['accuracy'], reverse=True)

    print("Best Probes (Top 5):")
    for i, r in enumerate(all_results[:5], 1):
        print(f"  {i}. {r['layer']:10s} / {r['position']:15s}: {r['accuracy']:.1%}")

    print()

    print("Worst Probes (Bottom 5):")
    for i, r in enumerate(all_results[-5:], 1):
        print(f"  {i}. {r['layer']:10s} / {r['position']:15s}: {r['accuracy']:.1%}")

    print()
    print("=" * 80)
    print("✅ PROBE TRAINING COMPLETE")
    print("=" * 80)
    print()
    print("Next step: Generate visualizations with scripts/visualize_multilayer_results.py")
    print()


if __name__ == "__main__":
    main()
