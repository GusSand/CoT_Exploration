"""
GPT-2 Experiment 2: Linear Probes (FULL IMPLEMENTATION)

Trains 18 logistic regression probes on GPT-2 activations:
- Layers: 4 (early), 8 (middle), 11 (late)
- Tokens: 0-5 (6 continuous thought tokens)
- Features: 768 dims (GPT-2 hidden size)

Predicts correctness (correct vs incorrect) from continuous thoughts.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def load_balanced_dataset(data_path: str, n_samples: int = 100):
    """Load balanced dataset (50 correct, 50 incorrect)."""
    print(f"Loading shared data from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    samples = data['samples']

    # Split by correctness
    correct = [s for s in samples if s['is_correct']]
    incorrect = [s for s in samples if not s['is_correct']]

    print(f"  Available: {len(correct)} correct, {len(incorrect)} incorrect")

    # Balance
    n_each = n_samples // 2
    import random
    random.seed(42)

    selected_correct = random.sample(correct, min(n_each, len(correct)))
    selected_incorrect = random.sample(incorrect, min(n_each, len(incorrect)))

    balanced = selected_correct + selected_incorrect
    random.shuffle(balanced)

    print(f"  Sampled: {len(selected_correct)} correct + {len(selected_incorrect)} incorrect = {len(balanced)} total")

    return balanced


def extract_features(samples, layer: int, token: int):
    """Extract features for specific layer and token."""
    X = []
    y = []

    layer_key = f'layer_{layer}'

    for sample in samples:
        # Get activation: thoughts[layer_key][token] is a list of 768 floats
        activation = sample['thoughts'][layer_key][token]
        X.append(activation)
        y.append(1 if sample['is_correct'] else 0)

    return np.array(X), np.array(y)


def train_probe(X, y, layer, token):
    """Train a single probe with cross-validation."""

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train with CV
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
    for _ in range(100):  # Reduced from 1000 for speed
        indices = np.random.choice(len(y), size=len(y), replace=True)
        y_boot = y[indices]
        y_pred_boot = clf.predict(X_scaled[indices])
        accuracies.append(accuracy_score(y_boot, y_pred_boot))

    ci_lower = np.percentile(accuracies, 2.5)
    ci_upper = np.percentile(accuracies, 97.5)

    print(f"    Layer {layer}, Token {token}: {accuracy:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")

    return {
        'layer': layer,
        'token': token,
        'accuracy': float(accuracy),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'best_C': float(clf.C_[0])
    }


def create_heatmap(results, output_path):
    """Create accuracy heatmap."""
    layers = sorted(set(r['layer'] for r in results))
    tokens = sorted(set(r['token'] for r in results))

    # Create matrix
    matrix = np.zeros((len(layers), len(tokens)))
    for r in results:
        layer_idx = layers.index(r['layer'])
        token_idx = tokens.index(r['token'])
        matrix[layer_idx, token_idx] = r['accuracy']

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0.5,
        vmax=1.0,
        xticklabels=[f'T{t}' for t in tokens],
        yticklabels=[f'L{l}' for l in layers],
        cbar_kws={'label': 'Accuracy'},
        ax=ax
    )
    ax.set_title('GPT-2 Linear Probe Accuracy', fontweight='bold')
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Layer')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Heatmap saved: {output_path}")


def main():
    """Train all probes."""
    print("="*60)
    print("GPT-2 EXPERIMENT 2: LINEAR PROBES")
    print("="*60)

    project_root = Path(__file__).parent.parent.parent.parent.parent
    data_path = project_root / "src/experiments/gpt2_shared_data/gpt2_predictions_1000.json"
    output_dir = project_root / "src/experiments/gpt2_linear_probes/results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load balanced dataset
    samples = load_balanced_dataset(str(data_path), n_samples=100)

    # Train probes
    layers = [4, 8, 11]  # Early, middle, late (GPT-2 has 12 layers, 0-indexed)
    tokens = list(range(6))

    print(f"\nTraining {len(layers) * len(tokens)} probes...")
    print(f"  Layers: {layers}")
    print(f"  Tokens: {tokens}")

    results = []
    count = 0
    total = len(layers) * len(tokens)

    for layer in layers:
        print(f"\n  Layer {layer}:")
        for token in tokens:
            count += 1
            print(f"  [{count}/{total}]", end=" ")

            # Extract features
            X, y = extract_features(samples, layer, token)

            # Train probe
            result = train_probe(X, y, layer, token)
            results.append(result)

    # Save results
    results_path = output_dir / "probe_results_gpt2.json"
    with open(results_path, 'w') as f:
        json.dump({
            'metadata': {
                'model': 'GPT-2 CODI',
                'n_samples': len(samples),
                'n_probes': len(results),
                'layers': layers,
                'tokens': tokens,
                'feature_dim': 768,
                'date': datetime.now().isoformat()
            },
            'results': results
        }, f, indent=2)

    print(f"\n  Results saved: {results_path}")

    # Create heatmap
    heatmap_path = output_dir / "probe_heatmap_gpt2.png"
    create_heatmap(results, heatmap_path)

    # Summary statistics
    accuracies = [r['accuracy'] for r in results]
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Mean accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"  Range: [{np.min(accuracies):.4f}, {np.max(accuracies):.4f}]")

    best = max(results, key=lambda r: r['accuracy'])
    print(f"  Best: Layer {best['layer']}, Token {best['token']} = {best['accuracy']:.4f}")

    print("="*60)
    print("✅ GPT-2 LINEAR PROBES COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
