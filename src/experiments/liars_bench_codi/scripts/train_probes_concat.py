"""
Train probes on CONCATENATED continuous thought tokens.

Instead of training separate probes for each token, this concatenates all 6 tokens
into a single feature vector: [tok0, tok1, tok2, tok3, tok4, tok5].

Hypothesis: Individual tokens are weak, but combining them might create stronger signal.

Author: Claude Code
Date: 2025-10-28
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score


def train_concat_probe(X, y, layer_name, random_seed=42):
    """Train probe on concatenated tokens with proper train/test split."""

    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=y
    )

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train with CV on train set only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    clf = LogisticRegressionCV(
        Cs=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        cv=cv,
        scoring='accuracy',
        random_state=random_seed,
        max_iter=2000,
        n_jobs=-1
    )

    clf.fit(X_train_scaled, y_train)

    # Evaluate on both train and test
    y_train_pred = clf.predict(X_train_scaled)
    y_test_pred = clf.predict(X_test_scaled)
    y_test_proba = clf.predict_proba(X_test_scaled)[:, 1]

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auroc = roc_auc_score(y_test, y_test_proba)

    test_cm = confusion_matrix(y_test, y_test_pred)

    return {
        'layer': layer_name,
        'strategy': 'concatenation',
        'feature_dim': X.shape[1],
        'random_seed': random_seed,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'cv_accuracy': float(clf.scores_[1].mean()),
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
        'test_auroc': float(test_auroc),
        'overfitting_gap': float(train_acc - test_acc),
        'test_confusion_matrix': test_cm.tolist(),
        'best_C': float(clf.C_[0]) if hasattr(clf, 'C_') else None
    }


def main():
    print("=" * 80)
    print("TOKEN CONCATENATION PROBES - Proper Train/Test Split")
    print("=" * 80)
    print("\nStrategy: Concatenate all 6 continuous thought tokens")
    print("Feature vector: [tok0, tok1, tok2, tok3, tok4, tok5]")
    print("Hypothesis: Combining tokens creates stronger deception signal\n")

    script_dir = Path(__file__).parent
    data_path = script_dir.parent / "data" / "processed" / "probe_dataset_gpt2_clean.json"

    # Load clean data
    print(f"[1/4] Loading clean dataset...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    samples = data['samples']

    print(f"  Total samples: {len(samples)}")
    print(f"  Honest: {data['n_honest']}")
    print(f"  Deceptive: {data['n_deceptive']}")

    # Train probes for each layer
    print(f"\n[2/4] Training concatenation probes...")

    layers = ['layer_4', 'layer_8', 'layer_11']
    random_seeds = [42, 123, 456]  # Multiple seeds for robustness

    all_results = []

    for layer in layers:
        print(f"\n{layer.upper()}:")
        print(f"{'='*60}")

        # Extract and concatenate all 6 tokens
        X = []
        y = []

        for sample in samples:
            # Concatenate all 6 tokens from this layer
            tokens = sample['thoughts'][layer]  # List of 6 x 768-dim vectors
            concatenated = np.concatenate(tokens)  # (4608,) vector
            X.append(concatenated)
            y.append(1 if sample['is_honest'] else 0)

        X = np.array(X)
        y = np.array(y)

        print(f"  Feature shape: {X.shape} (768 x 6 tokens)")

        # Train with multiple seeds
        seed_results = []
        for seed in random_seeds:
            result = train_concat_probe(X, y, layer, random_seed=seed)
            seed_results.append(result)

        # Average across seeds
        avg_result = {
            'layer': layer,
            'strategy': 'concatenation',
            'test_accuracy_mean': np.mean([r['test_accuracy'] for r in seed_results]),
            'test_accuracy_std': np.std([r['test_accuracy'] for r in seed_results]),
            'test_auroc_mean': np.mean([r['test_auroc'] for r in seed_results]),
            'test_f1_mean': np.mean([r['test_f1'] for r in seed_results]),
            'overfitting_gap_mean': np.mean([r['overfitting_gap'] for r in seed_results]),
            'results_by_seed': seed_results
        }

        all_results.append(avg_result)

        # Print results
        print(f"  Test Acc:  {avg_result['test_accuracy_mean']*100:.2f}% ± {avg_result['test_accuracy_std']*100:.2f}%")
        print(f"  Test AUROC: {avg_result['test_auroc_mean']:.3f}")
        print(f"  Test F1:    {avg_result['test_f1_mean']:.3f}")
        print(f"  Overfit:    {avg_result['overfitting_gap_mean']*100:+.2f}pp")

    # Summary comparison
    print(f"\n[3/4] Comparison with baselines...")
    print(f"\n{'Method':<35} {'Test Acc':<15} {'AUROC':<10} {'F1':<10}")
    print(f"{'-'*70}")

    for result in all_results:
        print(f"{result['layer'] + ' (concat)':<35} "
              f"{result['test_accuracy_mean']*100:>6.2f}% ± {result['test_accuracy_std']*100:.2f}%  "
              f"{result['test_auroc_mean']:>6.3f}   "
              f"{result['test_f1_mean']:>6.3f}")

    print(f"{'-'*70}")

    # Baselines
    linear_best = 0.4883  # Best single-token linear probe
    response_tokens = 0.705

    print(f"{'Best single-token linear probe':<35} {linear_best*100:>6.2f}%         {'0.465':<10} {'N/A':<10}")
    print(f"{'Response tokens':<35} {response_tokens*100:>6.2f}%         {'0.777':<10} {'0.642':<10}")

    # Find best concatenation result
    best_result = max(all_results, key=lambda x: x['test_accuracy_mean'])

    print(f"\n{'='*80}")
    print(f"BEST CONCATENATION: {best_result['layer']}")
    print(f"{'='*80}")
    print(f"  Test Accuracy: {best_result['test_accuracy_mean']*100:.2f}% ± {best_result['test_accuracy_std']*100:.2f}%")
    print(f"  Test AUROC:    {best_result['test_auroc_mean']:.3f}")
    print(f"  Test F1:       {best_result['test_f1_mean']:.3f}")

    improvement_vs_single = (best_result['test_accuracy_mean'] - linear_best) * 100
    improvement_vs_response = (best_result['test_accuracy_mean'] - response_tokens) * 100

    print(f"\n  vs Best single token:  {improvement_vs_single:+.2f} percentage points")
    print(f"  vs Response tokens:    {improvement_vs_response:+.2f} percentage points")

    if best_result['test_accuracy_mean'] > response_tokens:
        print(f"\n  ✅ WINNER: Concatenation BEATS response tokens!")
    elif best_result['test_accuracy_mean'] > linear_best:
        print(f"\n  ✅ IMPROVEMENT: Concatenation beats single tokens!")
    elif abs(best_result['test_accuracy_mean'] - 0.5) < 0.05:
        print(f"\n  ❌ RANDOM: Concatenation performs at chance level")
    else:
        print(f"\n  ❌ NO IMPROVEMENT: Concatenation doesn't help")

    # Detailed analysis of best result
    best_seed_result = best_result['results_by_seed'][0]
    cm = best_seed_result['test_confusion_matrix']

    print(f"\n  Confusion Matrix (seed=42):")
    print(f"                Predicted")
    print(f"                Deceptive  Honest")
    print(f"  Actual:")
    print(f"    Deceptive      {cm[0][0]:3d}       {cm[0][1]:3d}")
    print(f"    Honest         {cm[1][0]:3d}       {cm[1][1]:3d}")

    deceptive_recall = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
    honest_recall = cm[1][1] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) > 0 else 0
    print(f"    Deceptive Recall: {deceptive_recall*100:.1f}%")
    print(f"    Honest Recall:    {honest_recall*100:.1f}%")

    print(f"\n{'='*80}")

    # Save results
    print(f"\n[4/4] Saving results...")

    output_dir = script_dir.parent / "results"
    output_file = output_dir / "probe_results_concat_gpt2.json"

    final_results = {
        'model': 'gpt2',
        'probe_type': 'concatenation',
        'strategy': 'concatenate_all_6_tokens',
        'dataset': 'probe_dataset_gpt2_clean.json',
        'dataset_size': len(samples),
        'train_test_split': '80/20',
        'random_seeds': random_seeds,
        'summary': {
            'best_layer': best_result['layer'],
            'best_test_accuracy': float(best_result['test_accuracy_mean']),
            'best_test_auroc': float(best_result['test_auroc_mean']),
            'improvement_over_single_token': float(improvement_vs_single),
            'improvement_over_response': float(improvement_vs_response)
        },
        'all_layers': all_results,
        'baselines': {
            'single_token_linear': linear_best,
            'response_tokens': response_tokens
        }
    }

    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"  ✅ Saved: {output_file}")

    return all_results


if __name__ == "__main__":
    main()
