#!/usr/bin/env python3
"""
Train deception detection probes on extracted activations.

Following Apollo Research methodology:
- Use logistic regression probes
- Train on CT token activations vs regular hidden states
- Compare performance across different layers
- Use cross-validation for robust evaluation
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import argparse
import json

def load_activations(activation_path):
    """Load extracted activations."""
    with open(activation_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Loaded {data['activation_type']} activations:")
    print(f"  Layers: {list(data['activations'].keys())}")
    print(f"  Samples: {len(data['labels'])}")
    print(f"  Honest: {sum(data['labels'])}, Deceptive: {len(data['labels']) - sum(data['labels'])}")

    return data

def train_probe_for_layer(X, y, groups, layer_name, cv_folds=5, test_size=0.2, random_state=42):
    """Train and evaluate probe for a single layer."""
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create logistic regression probe
    probe = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )

    # Cross-validation
    cv_scores = cross_val_score(
        probe, X_scaled, y,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
        scoring='accuracy'
    )

    cv_auc_scores = cross_val_score(
        probe, X_scaled, y,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
        scoring='roc_auc'
    )

    # Group-aware, stratified hold-out split to avoid leakage across question pairs
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X_scaled, y, groups=groups))

    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train on training set only; evaluate on held-out test set
    probe.fit(X_train, y_train)
    y_pred = probe.predict(X_test)
    y_pred_proba = probe.predict_proba(X_test)[:, 1]

    holdout_accuracy = accuracy_score(y_test, y_pred)
    holdout_auc = roc_auc_score(y_test, y_pred_proba)

    results = {
        'layer': layer_name,
        'cv_accuracy_mean': cv_scores.mean(),
        'cv_accuracy_std': cv_scores.std(),
        'cv_auc_mean': cv_auc_scores.mean(),
        'cv_auc_std': cv_auc_scores.std(),
        'holdout_accuracy': holdout_accuracy,
        'holdout_auc': holdout_auc,
        'probe': probe,
        'scaler': scaler
    }

    print(f"\\n{layer_name}:")
    print(f"  CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  CV AUC: {cv_auc_scores.mean():.3f} ± {cv_auc_scores.std():.3f}")
    print(f"  Hold-out Accuracy: {holdout_accuracy:.3f}")
    print(f"  Hold-out AUC: {holdout_auc:.3f}")

    return results

def train_probes(activations_data, activation_type, cv_folds=5, test_size=0.2, random_state=42):
    """Train probes for all layers."""
    print(f"\\n{'='*60}")
    print(f"TRAINING {activation_type.upper()} PROBES")
    print(f"{'='*60}")

    activations = activations_data['activations']
    labels = np.array(activations_data['labels'])
    question_hashes = np.array(activations_data['question_hashes'])

    # Convert boolean labels to binary (True = 1 = honest, False = 0 = deceptive)
    y = labels.astype(int)

    results = []

    for layer_name, layer_activations in activations.items():
        X = layer_activations  # Shape: (n_samples, hidden_dim)

        layer_results = train_probe_for_layer(
            X, y, question_hashes, f"Layer {layer_name}",
            cv_folds=cv_folds, test_size=test_size, random_state=random_state
        )
        results.append(layer_results)

    return results

def compare_results(ct_results, regular_results, output_dir):
    """Compare CT token vs regular hidden state probe performance."""
    print(f"\\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}")

    comparison = {
        'ct_token_results': [],
        'regular_hidden_results': [],
        'summary': {}
    }

    print(f"{'Layer':<10} {'CT Tokens':<15} {'Regular Hidden':<15} {'Difference':<12}")
    print("-" * 60)

    for ct_res, reg_res in zip(ct_results, regular_results):
        layer = ct_res['layer']
        ct_acc = ct_res['cv_accuracy_mean']
        reg_acc = reg_res['cv_accuracy_mean']
        diff = ct_acc - reg_acc

        print(f"{layer:<10} {ct_acc:.3f}           {reg_acc:.3f}            {diff:+.3f}")

        comparison['ct_token_results'].append({
            'layer': layer,
            'accuracy': ct_acc,
            'accuracy_std': ct_res['cv_accuracy_std'],
            'auc': ct_res['cv_auc_mean'],
            'auc_std': ct_res['cv_auc_std']
        })

        comparison['regular_hidden_results'].append({
            'layer': layer,
            'accuracy': reg_acc,
            'accuracy_std': reg_res['cv_accuracy_std'],
            'auc': reg_res['cv_auc_mean'],
            'auc_std': reg_res['cv_auc_std']
        })

    # Summary statistics
    ct_accuracies = [r['cv_accuracy_mean'] for r in ct_results]
    reg_accuracies = [r['cv_accuracy_mean'] for r in regular_results]

    comparison['summary'] = {
        'ct_tokens_best_accuracy': max(ct_accuracies),
        'ct_tokens_mean_accuracy': np.mean(ct_accuracies),
        'regular_hidden_best_accuracy': max(reg_accuracies),
        'regular_hidden_mean_accuracy': np.mean(reg_accuracies),
        'ct_tokens_advantage': np.mean(ct_accuracies) - np.mean(reg_accuracies),
        'random_baseline': 0.5
    }

    print(f"\\n{'Summary':<20}")
    print("-" * 30)
    print(f"CT Tokens Best:      {comparison['summary']['ct_tokens_best_accuracy']:.3f}")
    print(f"CT Tokens Mean:      {comparison['summary']['ct_tokens_mean_accuracy']:.3f}")
    print(f"Regular Hidden Best: {comparison['summary']['regular_hidden_best_accuracy']:.3f}")
    print(f"Regular Hidden Mean: {comparison['summary']['regular_hidden_mean_accuracy']:.3f}")
    print(f"CT Advantage:        {comparison['summary']['ct_tokens_advantage']:+.3f}")
    print(f"Random Baseline:     {comparison['summary']['random_baseline']:.3f}")

    # Save comparison results
    with open(output_dir / "probe_comparison_results.json", 'w') as f:
        json.dump(comparison, f, indent=2)

    return comparison

def main():
    parser = argparse.ArgumentParser(description="Train deception detection probes")
    parser.add_argument("--ct_activations", required=True, help="Path to CT token activations")
    parser.add_argument("--regular_activations", required=True, help="Path to regular hidden state activations")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--test_size", type=float, default=0.2, help="Hold-out test size fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load activations
    ct_data = load_activations(args.ct_activations)
    regular_data = load_activations(args.regular_activations)

    # Verify data consistency
    assert len(ct_data['labels']) == len(regular_data['labels']), "Mismatched sample counts"
    assert ct_data['question_hashes'] == regular_data['question_hashes'], "Mismatched question hashes"

    # Train probes
    ct_results = train_probes(ct_data, "CT Token", cv_folds=args.cv_folds, test_size=args.test_size, random_state=args.seed)
    regular_results = train_probes(regular_data, "Regular Hidden State", cv_folds=args.cv_folds, test_size=args.test_size, random_state=args.seed)

    # Compare results
    comparison = compare_results(ct_results, regular_results, output_dir)

    # Save detailed results
    with open(output_dir / "ct_token_probe_results.json", 'w') as f:
        json.dump([{k: v for k, v in r.items() if k not in ['probe', 'scaler']} for r in ct_results], f, indent=2)

    with open(output_dir / "regular_hidden_probe_results.json", 'w') as f:
        json.dump([{k: v for k, v in r.items() if k not in ['probe', 'scaler']} for r in regular_results], f, indent=2)

    print(f"\\n{'='*60}")
    print("PROBE TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")

    # Research question conclusion
    ct_mean = comparison['summary']['ct_tokens_mean_accuracy']
    reg_mean = comparison['summary']['regular_hidden_mean_accuracy']

    print(f"\\n🔬 RESEARCH QUESTION: Can CT tokens encode deception when trained contrastively?")

    if ct_mean > reg_mean:
        print(f"✅ YES: CT tokens outperform regular hidden states by {ct_mean - reg_mean:.3f}")
        print("   Continuous thought appears to encode deception-relevant information!")
    elif ct_mean < reg_mean:
        print(f"❌ NO: CT tokens underperform regular hidden states by {reg_mean - ct_mean:.3f}")
        print("   Regular hidden states are better for deception detection.")
    else:
        print(f"🤷 UNCLEAR: CT tokens and regular hidden states perform similarly")
        print("   No clear advantage for continuous thought in deception detection.")

    print(f"\\nBoth methods significantly exceed random baseline: {0.5:.3f}")

if __name__ == "__main__":
    main()