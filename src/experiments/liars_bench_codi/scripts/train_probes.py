"""
Story 5.2: Train Deception Detection Probes

Trains logistic regression probes on continuous thought activations to detect deception.
Similar to GSM8K linear probe analysis but for honest vs deceptive classification.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_probe_dataset(data_path: str):
    """Load extracted activations."""
    print(f"Loading dataset from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"  Loaded {len(data['samples'])} samples")
    print(f"  Honest: {data['n_honest']}, Deceptive: {data['n_deceptive']}")
    print(f"  Layers: {data['layers']}")
    print(f"  Tokens: {data['num_tokens']}")

    return data


def extract_features(samples, layer: str, token: int):
    """Extract features for a specific layer and token."""
    X = []
    y = []

    for sample in samples:
        # Structure: thoughts[layer] = [token0, token1, ..., token5]
        # Each token is a 768-dim activation vector
        activation = sample['thoughts'][layer][token]
        X.append(activation)
        y.append(1 if sample['is_honest'] else 0)

    return np.array(X), np.array(y)


def train_single_probe(X, y, layer, token):
    """Train a single probe."""
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegressionCV(
        Cs=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        cv=cv,
        scoring='accuracy',
        random_state=42,
        max_iter=1000,
        n_jobs=-1
    )

    clf.fit(X_scaled, y)

    # Predictions
    y_pred = clf.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)

    # Bootstrap CI
    accuracies = []
    for _ in range(100):
        indices = np.random.choice(len(y), size=len(y), replace=True)
        y_boot = y[indices]
        y_pred_boot = clf.predict(X_scaled[indices])
        accuracies.append(accuracy_score(y_boot, y_pred_boot))

    ci_lower = np.percentile(accuracies, 2.5)
    ci_upper = np.percentile(accuracies, 97.5)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)

    return {
        "layer": layer,
        "token": token,
        "accuracy": accuracy,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "best_C": clf.C_[0] if hasattr(clf, 'C_') else None,
        "cv_scores_mean": clf.scores_[1].mean(),
        "cv_scores_std": clf.scores_[1].std(),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y, y_pred, output_dict=True)
    }


def train_all_probes(model_name: str = "gpt2"):
    """Train probes for all layer-token combinations."""

    print("=" * 80)
    print(f"STORY 5.2: Training Deception Detection Probes for {model_name.upper()}")
    print("=" * 80)

    # Load data
    script_dir = Path(__file__).parent
    data_path = script_dir.parent / "data" / "processed" / f"probe_dataset_{model_name}.json"
    data = load_probe_dataset(str(data_path))

    layers = data['layers']
    num_tokens = data['num_tokens']
    samples = data['samples']

    print(f"\n[1/4] Training {len(layers)} × {num_tokens} = {len(layers) * num_tokens} probes...")

    results = []

    for layer in layers:
        for token in range(num_tokens):
            print(f"\n  Training probe: {layer}, Token {token}...")

            # Extract features
            X, y = extract_features(samples, layer, token)

            print(f"    Features: {X.shape}, Labels: {y.shape}")
            print(f"    Honest: {(y == 1).sum()}, Deceptive: {(y == 0).sum()}")

            # Train probe
            result = train_single_probe(X, y, layer, token)
            results.append(result)

            print(f"    ✅ Accuracy: {result['accuracy']*100:.2f}% "
                  f"[{result['ci_lower']*100:.1f}%, {result['ci_upper']*100:.1f}%]")

    print(f"\n[2/4] Computing summary statistics...")

    # Overall statistics
    accuracies = [r['accuracy'] for r in results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    min_acc = np.min(accuracies)
    max_acc = np.max(accuracies)

    print(f"  Mean accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"  Range: {min_acc*100:.2f}% - {max_acc*100:.2f}%")

    # Best and worst probes
    best_idx = np.argmax(accuracies)
    worst_idx = np.argmin(accuracies)

    best_probe = results[best_idx]
    worst_probe = results[worst_idx]

    print(f"\n  Best probe: {best_probe['layer']}, Token {best_probe['token']} "
          f"({best_probe['accuracy']*100:.2f}%)")
    print(f"  Worst probe: {worst_probe['layer']}, Token {worst_probe['token']} "
          f"({worst_probe['accuracy']*100:.2f}%)")

    print(f"\n[3/4] Creating visualizations...")

    # Create heatmap
    accuracy_matrix = np.zeros((len(layers), num_tokens))
    for i, layer in enumerate(layers):
        for j in range(num_tokens):
            result = [r for r in results if r['layer'] == layer and r['token'] == j][0]
            accuracy_matrix[i, j] = result['accuracy'] * 100

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        accuracy_matrix,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        xticklabels=[f'Token {i}' for i in range(num_tokens)],
        yticklabels=layers,
        vmin=50,
        vmax=100,
        cbar_kws={'label': 'Accuracy (%)'}
    )
    plt.title(f'{model_name.upper()} Deception Detection Probe Accuracy\nHonest vs Deceptive Classification')
    plt.xlabel('Continuous Thought Token')
    plt.ylabel('Layer')
    plt.tight_layout()

    output_dir = script_dir.parent / "results"
    output_dir.mkdir(exist_ok=True)

    heatmap_file = output_dir / f"probe_heatmap_{model_name}.png"
    plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved heatmap: {heatmap_file}")
    plt.close()

    print(f"\n[4/4] Saving results...")

    # Save detailed results
    output = {
        "model": model_name,
        "summary": {
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "min_accuracy": min_acc,
            "max_accuracy": max_acc,
            "best_probe": {
                "layer": best_probe['layer'],
                "token": best_probe['token'],
                "accuracy": best_probe['accuracy']
            },
            "worst_probe": {
                "layer": worst_probe['layer'],
                "token": worst_probe['token'],
                "accuracy": worst_probe['accuracy']
            }
        },
        "probes": results,
        "accuracy_matrix": accuracy_matrix.tolist()
    }

    results_file = output_dir / f"probe_results_{model_name}.json"
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"  ✅ Saved results: {results_file}")

    print("\n" + "=" * 80)
    print("✅ STORY 5.2 COMPLETE: Probe training finished!")
    print("=" * 80)

    print(f"\n📊 Summary:")
    print(f"  Total probes trained: {len(results)}")
    print(f"  Mean accuracy: {mean_acc*100:.2f}%")
    print(f"  Best: {best_probe['layer']}, Token {best_probe['token']} ({best_probe['accuracy']*100:.2f}%)")
    print(f"  Worst: {worst_probe['layer']}, Token {worst_probe['token']} ({worst_probe['accuracy']*100:.2f}%)")

    if mean_acc > 0.7:
        print(f"\n✅ SUCCESS: Probes achieve >{mean_acc*100:.0f}% accuracy!")
        print(f"   Continuous thoughts encode deception information.")
    elif mean_acc > 0.6:
        print(f"\n⚠️  MODERATE: Probes achieve {mean_acc*100:.0f}% accuracy.")
        print(f"   Some deception signal present but weak.")
    else:
        print(f"\n❌ POOR: Probes achieve only {mean_acc*100:.0f}% accuracy.")
        print(f"   Limited deception signal in continuous thoughts.")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train deception detection probes")
    parser.add_argument("--model", type=str, default="gpt2", choices=["gpt2", "llama"],
                       help="Model to train probes for")

    args = parser.parse_args()

    train_all_probes(model_name=args.model)
