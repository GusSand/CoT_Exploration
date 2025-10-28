"""
Train probe on response token activations with PROPER held-out methodology.

Tests if response tokens maintain their advantage with correct splits.

Author: Claude Code
Date: 2025-10-28
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score


def load_activations():
    """Load response token activations."""
    script_dir = Path(__file__).parent
    data_file = script_dir.parent / "data" / "processed" / "response_activations_gpt2_proper.json"

    print(f"Loading activations from {data_file}...")

    with open(data_file, 'r') as f:
        data = json.load(f)

    return data


def train_probe(X_train, y_train, X_test, y_test):
    """Train probe on response tokens."""

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train with CV
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
    print("RESPONSE TOKEN PROBE - PROPER HELD-OUT METHODOLOGY")
    print("=" * 80)

    # Load data
    data = load_activations()

    train_samples = data['train_samples']
    test_samples = data['test_samples']

    print(f"\nTrain samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")

    # Extract features
    X_train = np.array([s['response_activation'] for s in train_samples])
    y_train = np.array([1 if s['is_honest'] else 0 for s in train_samples])

    X_test = np.array([s['response_activation'] for s in test_samples])
    y_test = np.array([1 if s['is_honest'] else 0 for s in test_samples])

    print(f"Feature dim: {X_train.shape[1]}")

    # Train probe
    print("\nTraining probe...")
    result = train_probe(X_train, y_train, X_test, y_test)

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\nTrain accuracy: {result['train_accuracy']*100:.2f}%")
    print(f"Test accuracy:  {result['test_accuracy']*100:.2f}%")
    print(f"Test AUROC:     {result['test_auroc']:.3f}")
    print(f"Test F1:        {result['test_f1']:.3f}")
    print(f"Overfitting:    {result['overfitting_gap']*100:+.2f}pp")

    # Confusion matrix
    cm = result['test_confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Deceptive  Honest")
    print(f"  Actual:")
    print(f"    Deceptive      {cm[0][0]:3d}       {cm[0][1]:3d}")
    print(f"    Honest         {cm[1][0]:3d}       {cm[1][1]:3d}")

    deceptive_recall = cm[0][0] / (cm[0][0] + cm[0][1])
    honest_recall = cm[1][1] / (cm[1][0] + cm[1][1])
    print(f"\n  Deceptive Recall: {deceptive_recall*100:.1f}%")
    print(f"  Honest Recall:    {honest_recall*100:.1f}%")

    # Compare to continuous thoughts
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    print(f"\nContinuous thoughts (all probes): 50.00% (random)")
    print(f"Response tokens:                  {result['test_accuracy']*100:.2f}%")
    print(f"Difference:                       {(result['test_accuracy']-0.5)*100:+.2f} percentage points")

    if result['test_accuracy'] > 0.55:
        print(f"\n✅ Response tokens STILL superior to continuous thoughts!")
        print(f"   Even with proper held-out questions, response tokens detect deception.")
    elif result['test_accuracy'] > 0.52:
        print(f"\n⚠️  Response tokens slightly better than random")
        print(f"   Marginal improvement over continuous thoughts")
    else:
        print(f"\n❌ Response tokens also fail with held-out questions")
        print(f"   Deception detection requires question-specific patterns")

    # Save results
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "results"
    output_file = output_dir / "probe_results_response_proper.json"

    final_results = {
        'model': 'gpt2',
        'probe_type': 'response_tokens_mean_pooled',
        'methodology': 'proper_held_out_questions',
        'dataset_size': {
            'train': len(train_samples),
            'test': len(test_samples)
        },
        'results': result,
        'comparison': {
            'continuous_thoughts': 0.50,
            'response_tokens': result['test_accuracy'],
            'improvement': float(result['test_accuracy'] - 0.50)
        }
    }

    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n✅ Saved: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
