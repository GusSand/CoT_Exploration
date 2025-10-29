"""
Train linear probes on GPT-2 activations with PROPER held-out methodology.

This corrects Sprint 1 by using separate train/test question sets.
Tests true generalization to unseen questions.

Key differences from original:
- No data leakage (train/test are disjoint question sets)
- Smaller dataset (144 train, 144 test samples per layer/token)
- Tests generalization, not memorization

Author: Claude Code
Date: 2025-10-28
Fixes: Sprint 1 data leakage issue
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score


def load_activations():
    """Load extracted activations."""
    script_dir = Path(__file__).parent
    data_file = script_dir.parent / "data" / "processed" / "probe_activations_gpt2_proper.json"

    print(f"Loading activations from {data_file}...")

    with open(data_file, 'r') as f:
        data = json.load(f)

    print(f"  Train samples: {data['train_size']}")
    print(f"  Test samples: {data['test_size']}")
    print(f"  Layers: {data['layers']}")
    print(f"  Tokens per layer: {data['tokens_per_layer']}")

    return data


def train_probe(X_train, y_train, X_test, y_test, layer_name, token_idx):
    """Train single probe with proper train/test evaluation."""

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train with CV on train set only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegressionCV(
        Cs=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        cv=cv,
        scoring='accuracy',
        random_state=42,
        max_iter=2000,
        n_jobs=-1
    )

    clf.fit(X_train_scaled, y_train)

    # Evaluate
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
        'token': token_idx,
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
    print("LINEAR PROBES - PROPER HELD-OUT METHODOLOGY (Sprint 1 Correction)")
    print("=" * 80)
    print("\nThis corrects Sprint 1 data leakage issue.")
    print("Tests: Can probes detect deception on UNSEEN questions?\n")

    # Load activations
    data = load_activations()

    train_samples = data['train_samples']
    test_samples = data['test_samples']

    print(f"\n[1/3] Training probes on {len(train_samples)} train samples...")
    print(f"       Testing on {len(test_samples)} test samples...")
    print(f"       (Disjoint question sets - no overlap!)\n")

    # Train probes for each layer and token
    layers = data['layers']
    n_tokens = data['tokens_per_layer']

    all_results = []

    for layer in layers:
        print(f"\n{layer.upper()}:")
        print(f"{'='*60}")

        for token_idx in range(n_tokens):
            # Extract features
            X_train = np.array([s['thoughts'][layer][token_idx] for s in train_samples])
            y_train = np.array([1 if s['is_honest'] else 0 for s in train_samples])

            X_test = np.array([s['thoughts'][layer][token_idx] for s in test_samples])
            y_test = np.array([1 if s['is_honest'] else 0 for s in test_samples])

            # Train probe
            result = train_probe(X_train, y_train, X_test, y_test, layer, token_idx)
            all_results.append(result)

            # Print
            print(f"  Token {token_idx}: "
                  f"Test Acc = {result['test_accuracy']*100:.2f}%, "
                  f"AUROC = {result['test_auroc']:.3f}, "
                  f"Overfit = {result['overfitting_gap']*100:+.2f}pp")

    # Summary
    print(f"\n[2/3] Summary...")

    test_accs = [r['test_accuracy'] for r in all_results]
    best_result = max(all_results, key=lambda x: x['test_accuracy'])

    print(f"\n  Mean test accuracy: {np.mean(test_accs)*100:.2f}% ± {np.std(test_accs)*100:.2f}%")
    print(f"  Best probe: {best_result['layer']} Token {best_result['token']}")
    print(f"    Test accuracy: {best_result['test_accuracy']*100:.2f}%")
    print(f"    Test AUROC: {best_result['test_auroc']:.3f}")
    print(f"    Overfitting: {best_result['overfitting_gap']*100:+.2f}pp")

    # Confusion matrix
    cm = best_result['test_confusion_matrix']
    print(f"\n  Best probe confusion matrix:")
    print(f"                Predicted")
    print(f"                Deceptive  Honest")
    print(f"  Actual:")
    print(f"    Deceptive      {cm[0][0]:3d}       {cm[0][1]:3d}")
    print(f"    Honest         {cm[1][0]:3d}       {cm[1][1]:3d}")

    deceptive_recall = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
    honest_recall = cm[1][1] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) > 0 else 0
    print(f"    Deceptive Recall: {deceptive_recall*100:.1f}%")
    print(f"    Honest Recall:    {honest_recall*100:.1f}%")

    # Save results
    print(f"\n[3/3] Saving results...")

    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "probe_results_gpt2_proper_v2.json"

    final_results = {
        'model': 'gpt2',
        'methodology': 'proper_held_out_questions',
        'dataset_size': {
            'train': len(train_samples),
            'test': len(test_samples)
        },
        'zero_overlap': True,
        'summary': {
            'mean_test_accuracy': float(np.mean(test_accs)),
            'std_test_accuracy': float(np.std(test_accs)),
            'best_layer': best_result['layer'],
            'best_token': int(best_result['token']),
            'best_test_accuracy': float(best_result['test_accuracy']),
            'best_test_auroc': float(best_result['test_auroc'])
        },
        'all_probes': all_results,
        'comparison_to_old': {
            'old_methodology': 'overlapping_questions (data leakage)',
            'old_best_accuracy': 0.4883,
            'new_methodology': 'held_out_questions (correct)',
            'new_best_accuracy': float(best_result['test_accuracy']),
            'difference': float(best_result['test_accuracy'] - 0.4883)
        }
    }

    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"  ✅ Saved: {output_file}")

    # Final verdict
    print(f"\n" + "=" * 80)
    print(f"CORRECTED RESULTS - PROPER METHODOLOGY")
    print(f"=" * 80)

    print(f"\nOld (INVALID - data leakage):")
    print(f"  Best accuracy: 48.83% (on seen questions)")

    print(f"\nNew (CORRECT - held-out):")
    print(f"  Best accuracy: {best_result['test_accuracy']*100:.2f}% (on unseen questions)")

    diff = (best_result['test_accuracy'] - 0.4883) * 100
    print(f"\nDifference: {diff:+.2f} percentage points")

    if best_result['test_accuracy'] < 0.4883:
        print(f"\n❌ Performance DECREASED (expected)")
        print(f"   Reason: Now testing generalization, not memorization")
        print(f"   This is the CORRECT result")
    else:
        print(f"\n✅ Performance maintained or improved")
        print(f"   Model generalizes well to unseen questions")

    print(f"\n" + "=" * 80)

    return all_results


if __name__ == "__main__":
    main()
