"""
Train probes on response token activations (Apollo Research baseline methodology).

This replicates Apollo's approach: train linear probes on final layer hidden states
from generated response tokens to detect deception.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score


def load_response_activations(data_path: str):
    """Load response token activations."""
    print(f"Loading response activations from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"  Loaded {len(data['samples'])} samples")
    print(f"  Honest: {data['n_honest']}, Deceptive: {data['n_deceptive']}")
    print(f"  Extraction type: {data['extraction_type']}")
    print(f"  Pooling: {data['pooling']}")

    return data


def train_response_probe():
    """Train probe on response token activations."""
    print("=" * 80)
    print("Training Probe on Response Token Activations")
    print("=" * 80)

    script_dir = Path(__file__).parent
    data_path = script_dir.parent / "data" / "processed" / "response_activations_gpt2.json"

    data = load_response_activations(str(data_path))
    samples = data['samples']

    print(f"\n[1/3] Preparing data...")

    # Extract features and labels
    X = []
    y = []

    for sample in samples:
        X.append(sample['response_activation'])
        y.append(1 if sample['is_honest'] else 0)  # 1=honest, 0=deceptive

    X = np.array(X)
    y = np.array(y)

    print(f"  Features shape: {X.shape}")
    print(f"  Labels: Honest={np.sum(y == 1)}, Deceptive={np.sum(y == 0)}")
    print(f"  Balance: {np.mean(y)*100:.1f}% honest")

    print(f"\n[2/3] Training probe...")

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
    y_proba = clf.predict_proba(X_scaled)[:, 1]

    # Metrics
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auroc = roc_auc_score(y, y_proba)
    conf_matrix = confusion_matrix(y, y_pred)

    print(f"\n  ✅ Accuracy: {accuracy*100:.2f}%")
    print(f"  ✅ F1-Score: {f1:.3f}")
    print(f"  ✅ AUROC: {auroc:.3f}")
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted")
    print(f"                Deceptive  Honest")
    print(f"  Actual:")
    print(f"    Deceptive      {conf_matrix[0][0]:3d}      {conf_matrix[0][1]:3d}")
    print(f"    Honest         {conf_matrix[1][0]:3d}      {conf_matrix[1][1]:3d}")

    # Bootstrap confidence intervals
    print(f"\n  Computing bootstrap confidence intervals...")
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

    ci_acc_lower = np.percentile(accuracies, 2.5)
    ci_acc_upper = np.percentile(accuracies, 97.5)
    ci_f1_lower = np.percentile(f1_scores, 2.5)
    ci_f1_upper = np.percentile(f1_scores, 97.5)
    ci_auroc_lower = np.percentile(aurocs, 2.5)
    ci_auroc_upper = np.percentile(aurocs, 97.5)

    print(f"\n[3/3] Final Results:")
    print(f"  Accuracy: {accuracy*100:.2f}% [{ci_acc_lower*100:.1f}%, {ci_acc_upper*100:.1f}%]")
    print(f"  F1-Score: {f1:.3f} [{ci_f1_lower:.3f}, {ci_f1_upper:.3f}]")
    print(f"  AUROC:    {auroc:.3f} [{ci_auroc_lower:.3f}, {ci_auroc_upper:.3f}]")

    # Comparison to continuous thoughts
    print(f"\n  Comparison to Continuous Thought Probes:")
    print(f"    Response tokens:      {accuracy*100:.2f}% (AUROC: {auroc:.3f})")
    print(f"    Continuous thoughts:  59.73% (AUROC: 0.643)")
    print(f"    Difference:           {(accuracy - 0.5973)*100:+.2f} percentage points")

    # Comparison to Apollo baseline
    apollo_baseline = 0.70
    print(f"\n  Comparison to Apollo Research Baseline:")
    print(f"    Apollo (response):    {apollo_baseline*100:.1f}%")
    print(f"    Ours (response):      {accuracy*100:.2f}%")
    if accuracy >= apollo_baseline:
        print(f"    ✅ MATCHED/EXCEEDED Apollo baseline by {(accuracy - apollo_baseline)*100:+.2f} points")
    else:
        print(f"    ❌ Below Apollo baseline by {(apollo_baseline - accuracy)*100:.2f} points")

    # Save results
    output_dir = script_dir.parent / "results"
    output_file = output_dir / "probe_results_response_gpt2.json"

    results = {
        'model': 'gpt2',
        'probe_type': 'response_tokens',
        'layer': 'final_layer',
        'pooling': 'mean_across_tokens',
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'auroc': float(auroc),
        'ci_accuracy_lower': float(ci_acc_lower),
        'ci_accuracy_upper': float(ci_acc_upper),
        'ci_f1_lower': float(ci_f1_lower),
        'ci_f1_upper': float(ci_f1_upper),
        'ci_auroc_lower': float(ci_auroc_lower),
        'ci_auroc_upper': float(ci_auroc_upper),
        'confusion_matrix': conf_matrix.tolist(),
        'report': classification_report(y, y_pred, output_dict=True),
        'best_C': float(clf.C_[0]),
        'comparison': {
            'continuous_thoughts_accuracy': 0.5973,
            'continuous_thoughts_auroc': 0.643,
            'response_vs_continuous_diff': float(accuracy - 0.5973),
            'apollo_baseline': apollo_baseline,
            'vs_apollo_diff': float(accuracy - apollo_baseline)
        }
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  ✅ Saved: {output_file}")

    print("\n" + "=" * 80)
    print("✅ Complete!")
    print("=" * 80)

    return results


if __name__ == "__main__":
    train_response_probe()
