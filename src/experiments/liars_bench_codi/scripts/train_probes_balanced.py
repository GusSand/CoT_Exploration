"""
Balanced probe training - deduplicate within each class separately to maintain balance.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


def load_dataset(data_path: str):
    """Load extracted activations."""
    print(f"Loading dataset from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"  Loaded {len(data['samples'])} samples")
    print(f"  Honest: {data['n_honest']}, Deceptive: {data['n_deceptive']}")
    print(f"  Layers: {data['layers']}")
    print(f"  Tokens: {data['num_tokens']}")

    return data


def extract_features_balanced(samples, layer: str, token: int, target_per_class=500):
    """
    Extract features with balanced sampling.
    Deduplicates within each class separately to maintain balance.
    """
    # Separate by class first
    honest_samples = [s for s in samples if s['is_honest']]
    deceptive_samples = [s for s in samples if not s['is_honest']]

    print(f"\n  Class distribution before deduplication:")
    print(f"    Honest: {len(honest_samples)}")
    print(f"    Deceptive: {len(deceptive_samples)}")

    # Deduplicate within each class
    def deduplicate_class(class_samples, class_name):
        seen_questions = set()
        unique_samples = []

        for idx, sample in enumerate(class_samples):
            if sample['question'] in seen_questions:
                continue
            seen_questions.add(sample['question'])
            unique_samples.append((idx, sample))

        print(f"    {class_name} unique: {len(unique_samples)} / {len(class_samples)}")
        return unique_samples

    honest_unique = deduplicate_class(honest_samples, "Honest")
    deceptive_unique = deduplicate_class(deceptive_samples, "Deceptive")

    # Limit to target_per_class for balance
    if len(honest_unique) > target_per_class:
        honest_unique = honest_unique[:target_per_class]
    if len(deceptive_unique) > target_per_class:
        deceptive_unique = deceptive_unique[:target_per_class]

    # Extract features
    X_honest = []
    X_deceptive = []

    for idx, sample in honest_unique:
        batch_data = sample['thoughts'][layer]
        batch_size = len(batch_data)
        batch_elem_idx = idx % batch_size
        activation = batch_data[batch_elem_idx][token]
        X_honest.append(activation)

    for idx, sample in deceptive_unique:
        batch_data = sample['thoughts'][layer]
        batch_size = len(batch_data)
        batch_elem_idx = idx % batch_size
        activation = batch_data[batch_elem_idx][token]
        X_deceptive.append(activation)

    # Combine
    X = np.array(X_honest + X_deceptive)
    y = np.array([1] * len(X_honest) + [0] * len(X_deceptive))  # 1=honest, 0=deceptive

    print(f"  Final balanced dataset:")
    print(f"    Honest: {len(X_honest)}")
    print(f"    Deceptive: {len(X_deceptive)}")
    print(f"    Total: {len(X)}")
    print(f"    Balance: {len(X_honest)/(len(X))*100:.1f}% honest")

    return X, y


def train_single_probe(X, y, layer, token):
    """Train a single probe with comprehensive metrics."""
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
    y_proba = clf.predict_proba(X_scaled)[:, 1]  # Probability of honest class

    # Comprehensive metrics
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auroc = roc_auc_score(y, y_proba)
    conf_matrix = confusion_matrix(y, y_pred)

    # Bootstrap confidence intervals for accuracy
    accuracies = []
    f1_scores = []
    aurocs = []

    for _ in range(100):
        indices = np.random.choice(len(y), size=len(y), replace=True)
        y_boot = y[indices]
        y_pred_boot = clf.predict(X_scaled[indices])
        y_proba_boot = clf.predict_proba(X_scaled[indices])[:, 1]

        accuracies.append(accuracy_score(y_boot, y_pred_boot))
        f1_scores.append(f1_score(y_boot, y_pred_boot))
        aurocs.append(roc_auc_score(y_boot, y_proba_boot))

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'auroc': auroc,
        'ci_accuracy_lower': np.percentile(accuracies, 2.5),
        'ci_accuracy_upper': np.percentile(accuracies, 97.5),
        'ci_f1_lower': np.percentile(f1_scores, 2.5),
        'ci_f1_upper': np.percentile(f1_scores, 97.5),
        'ci_auroc_lower': np.percentile(aurocs, 2.5),
        'ci_auroc_upper': np.percentile(aurocs, 97.5),
        'confusion_matrix': conf_matrix.tolist(),
        'report': classification_report(y, y_pred, output_dict=True),
        'best_C': clf.C_[0],
        'cv_scores_mean': clf.scores_[1].mean(axis=0).max(),
        'cv_scores_std': clf.scores_[1].mean(axis=0).std()
    }


def train_all_probes():
    """Train probes on balanced dataset with comprehensive metrics."""
    print("=" * 80)
    print("Training Probes with Balanced Dataset")
    print("=" * 80)

    script_dir = Path(__file__).parent
    data_path = script_dir.parent / "data" / "processed" / "probe_dataset_gpt2.json"

    data = load_dataset(str(data_path))

    layers = data['layers']
    num_tokens = data['num_tokens']
    samples = data['samples']

    print(f"\n[1/3] Extracting features for {len(layers)} layers × {num_tokens} tokens...")

    # Extract features for all layer-token combinations
    features = {}
    for layer in layers:
        for token in range(num_tokens):
            print(f"\n  Extracting {layer}, token {token}...")
            X, y = extract_features_balanced(samples, layer, token, target_per_class=500)
            features[(layer, token)] = (X, y)

    print(f"\n[2/3] Training {len(layers) * num_tokens} probes...")

    results = {}
    accuracy_matrix = []

    for layer_idx, layer in enumerate(layers):
        layer_accuracies = []

        for token in range(num_tokens):
            X, y = features[(layer, token)]

            print(f"\n  Training {layer}, token {token}...")
            print(f"    Features: {X.shape}")
            print(f"    Honest: {(y == 1).sum()}, Deceptive: {(y == 0).sum()}")

            probe_results = train_single_probe(X, y, layer, token)

            results[f"{layer}_token_{token}"] = {
                'layer': layer,
                'token': token,
                **probe_results
            }

            layer_accuracies.append(probe_results['accuracy'] * 100)

            print(f"    ✅ Accuracy: {probe_results['accuracy']*100:.2f}% "
                  f"[{probe_results['ci_accuracy_lower']*100:.1f}%, {probe_results['ci_accuracy_upper']*100:.1f}%]")
            print(f"    ✅ F1-Score: {probe_results['f1_score']:.3f} "
                  f"[{probe_results['ci_f1_lower']:.3f}, {probe_results['ci_f1_upper']:.3f}]")
            print(f"    ✅ AUROC: {probe_results['auroc']:.3f} "
                  f"[{probe_results['ci_auroc_lower']:.3f}, {probe_results['ci_auroc_upper']:.3f}]")
            print(f"    Confusion Matrix:")
            print(f"      {probe_results['confusion_matrix']}")

        accuracy_matrix.append(layer_accuracies)

    print(f"\n[3/3] Summary:")

    # Compute summary statistics
    all_accuracies = [r['accuracy'] for r in results.values()]
    all_f1s = [r['f1_score'] for r in results.values()]
    all_aurocs = [r['auroc'] for r in results.values()]

    print(f"  Mean accuracy: {np.mean(all_accuracies)*100:.2f}% (std: {np.std(all_accuracies)*100:.2f}%)")
    print(f"  Mean F1-score: {np.mean(all_f1s):.3f} (std: {np.std(all_f1s):.3f})")
    print(f"  Mean AUROC: {np.mean(all_aurocs):.3f} (std: {np.std(all_aurocs):.3f})")

    print(f"\n  Per layer:")
    for layer in layers:
        layer_results = [r for k, r in results.items() if r['layer'] == layer]
        layer_accs = [r['accuracy'] for r in layer_results]
        layer_f1s = [r['f1_score'] for r in layer_results]
        layer_aurocs = [r['auroc'] for r in layer_results]

        print(f"    {layer}: Acc={np.mean(layer_accs)*100:.2f}%, "
              f"F1={np.mean(layer_f1s):.3f}, AUROC={np.mean(layer_aurocs):.3f}")

    # Save results
    output_dir = script_dir.parent / "results"
    output_file = output_dir / "probe_results_balanced_gpt2.json"

    summary = {
        'mean_accuracy': float(np.mean(all_accuracies)),
        'std_accuracy': float(np.std(all_accuracies)),
        'min_accuracy': float(np.min(all_accuracies)),
        'max_accuracy': float(np.max(all_accuracies)),
        'mean_f1': float(np.mean(all_f1s)),
        'std_f1': float(np.std(all_f1s)),
        'mean_auroc': float(np.mean(all_aurocs)),
        'std_auroc': float(np.std(all_aurocs)),
    }

    with open(output_file, 'w') as f:
        json.dump({
            'model': 'gpt2',
            'balanced': True,
            'target_per_class': 500,
            'summary': summary,
            'probes': list(results.values()),
            'accuracy_matrix': accuracy_matrix
        }, f, indent=2)

    print(f"\n  ✅ Saved: {output_file}")

    # Create visualization
    create_heatmap(accuracy_matrix, layers, num_tokens, output_dir)

    print("\n" + "=" * 80)
    print("✅ Complete!")
    print("=" * 80)

    return results


def create_heatmap(accuracy_matrix, layers, num_tokens, output_dir):
    """Create heatmap visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        accuracy_matrix,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        vmin=50,
        vmax=100,
        xticklabels=[f'Token {i}' for i in range(num_tokens)],
        yticklabels=layers,
        cbar_kws={'label': 'Accuracy (%)'},
        ax=ax
    )

    ax.set_title('GPT-2 Deception Detection Probe Accuracy (Balanced Dataset)\nHonest vs Deceptive Classification')
    ax.set_xlabel('Continuous Thought Token')
    ax.set_ylabel('Layer')

    plt.tight_layout()
    plt.savefig(output_dir / 'probe_heatmap_balanced_gpt2.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {output_dir / 'probe_heatmap_balanced_gpt2.png'}")
    plt.close()


if __name__ == "__main__":
    train_all_probes()
