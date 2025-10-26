"""
Mean pooling probe training - aggregate across all 6 continuous thought tokens.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def load_and_pool(data_path: str):
    """Load dataset and create pooled features."""
    print(f"Loading dataset from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"  Loaded {len(data['samples'])} samples")

    layers = data['layers']
    samples = data['samples']

    results = {}

    for layer in layers:
        X_list = []
        y_list = []

        seen_questions = set()

        for idx, sample in enumerate(samples):
            # Skip duplicates
            if sample['question'] in seen_questions:
                continue
            seen_questions.add(sample['question'])

            batch_data = sample['thoughts'][layer]
            batch_size = len(batch_data)
            batch_elem_idx = idx % batch_size

            # Get all 6 token activations for this layer
            token_activations = batch_data[batch_elem_idx]  # List of 6 activations

            # Mean pool across tokens
            pooled = np.mean(token_activations, axis=0)  # Shape: (hidden_size,)

            X_list.append(pooled)
            y_list.append(1 if sample['is_honest'] else 0)

        results[layer] = {
            'X': np.array(X_list),
            'y': np.array(y_list)
        }

        print(f"  {layer}: {results[layer]['X'].shape} samples")

    return results, layers


def train_pooled_probes():
    """Train probes on mean-pooled features."""
    print("=" * 80)
    print("Training Probes with Mean Pooling Across Tokens")
    print("=" * 80)

    script_dir = Path(__file__).parent
    data_path = script_dir.parent / "data" / "processed" / "probe_dataset_gpt2.json"

    layer_data, layers = load_and_pool(str(data_path))

    print(f"\n[1/2] Training probes for {len(layers)} layers...")

    results = {}

    for layer in layers:
        X = layer_data[layer]['X']
        y = layer_data[layer]['y']

        print(f"\n  Training {layer}...")
        print(f"    Features: {X.shape}")
        print(f"    Honest: {(y == 1).sum()}, Deceptive: {(y == 0).sum()}")

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
            max_iter=2000,
            n_jobs=-1
        )

        clf.fit(X_scaled, y)
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

        results[layer] = {
            'accuracy': accuracy,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'report': classification_report(y, y_pred, output_dict=True)
        }

        print(f"    ✅ Accuracy: {accuracy*100:.2f}% [{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]")

    print(f"\n[2/2] Summary:")
    mean_acc = np.mean([r['accuracy'] for r in results.values()])
    print(f"  Mean accuracy across layers: {mean_acc*100:.2f}%")

    for layer in layers:
        print(f"  {layer}: {results[layer]['accuracy']*100:.2f}%")

    # Save results
    output_dir = script_dir.parent / "results"
    output_file = output_dir / "probe_results_pooled_gpt2.json"

    with open(output_file, 'w') as f:
        json.dump({
            'method': 'mean_pooling',
            'results': {k: {'accuracy': float(v['accuracy']),
                           'ci_lower': float(v['ci_lower']),
                           'ci_upper': float(v['ci_upper'])}
                       for k, v in results.items()},
            'mean_accuracy': float(mean_acc)
        }, f, indent=2)

    print(f"\n  ✅ Saved: {output_file}")

    print("\n" + "=" * 80)
    print("✅ Complete!")
    print("=" * 80)

    return results


if __name__ == "__main__":
    train_pooled_probes()
