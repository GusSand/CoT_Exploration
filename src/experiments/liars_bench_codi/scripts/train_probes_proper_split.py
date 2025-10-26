"""
Train linear probes with PROPER train/test split on CLEAN data.

CRITICAL: This is the definitive probe training script with:
- Clean, deduplicated data (no (question, label) duplicates)
- Proper 80/20 train/test split
- Stratified sampling
- Multiple random seeds for robustness
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score

def train_probe_with_split(X, y, layer_name, token_idx, random_seed=42):
    """Train single probe with proper train/test split."""

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
        'token': token_idx,
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
    print("LINEAR PROBES WITH PROPER TRAIN/TEST SPLIT - CLEAN DATA")
    print("=" * 80)

    script_dir = Path(__file__).parent
    data_path = script_dir.parent / "data" / "processed" / "probe_dataset_gpt2_clean.json"

    # Load clean data
    print(f"\n[1/4] Loading clean dataset...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    samples = data['samples']

    print(f"  Total samples: {len(samples)}")
    print(f"  Honest: {data['n_honest']}")
    print(f"  Deceptive: {data['n_deceptive']}")
    print(f"  Data quality:")
    for key, value in data['data_quality'].items():
        print(f"    {key}: {value}")

    # Train probes for all layers and tokens
    print(f"\n[2/4] Training probes with multiple random seeds...")

    layers = ['layer_4', 'layer_8', 'layer_11']
    tokens = list(range(6))
    random_seeds = [42, 123, 456]  # Multiple seeds for robustness

    all_results = []

    for layer in layers:
        for token in tokens:
            # Extract features
            X = []
            y = []

            for sample in samples:
                activation = sample['thoughts'][layer][token]
                X.append(activation)
                y.append(1 if sample['is_honest'] else 0)

            X = np.array(X)
            y = np.array(y)

            # Train with multiple seeds
            seed_results = []
            for seed in random_seeds:
                result = train_probe_with_split(X, y, layer, token, random_seed=seed)
                seed_results.append(result)

            # Average across seeds
            avg_result = {
                'layer': layer,
                'token': token,
                'test_accuracy_mean': np.mean([r['test_accuracy'] for r in seed_results]),
                'test_accuracy_std': np.std([r['test_accuracy'] for r in seed_results]),
                'test_auroc_mean': np.mean([r['test_auroc'] for r in seed_results]),
                'test_f1_mean': np.mean([r['test_f1'] for r in seed_results]),
                'overfitting_gap_mean': np.mean([r['overfitting_gap'] for r in seed_results]),
                'results_by_seed': seed_results
            }

            all_results.append(avg_result)

            # Print progress
            print(f"  {layer} Token {token}: "
                  f"Test Acc = {avg_result['test_accuracy_mean']*100:.2f}% ± {avg_result['test_accuracy_std']*100:.2f}%, "
                  f"AUROC = {avg_result['test_auroc_mean']:.3f}, "
                  f"Overfit = {avg_result['overfitting_gap_mean']*100:+.2f}pp")

    # Summary
    print(f"\n[3/4] Summary statistics...")

    test_accs = [r['test_accuracy_mean'] for r in all_results]
    best_result = max(all_results, key=lambda x: x['test_accuracy_mean'])

    print(f"  Mean test accuracy: {np.mean(test_accs)*100:.2f}% ± {np.std(test_accs)*100:.2f}%")
    print(f"  Best probe: {best_result['layer']} Token {best_result['token']}")
    print(f"    Test accuracy: {best_result['test_accuracy_mean']*100:.2f}% ± {best_result['test_accuracy_std']*100:.2f}%")
    print(f"    Test AUROC: {best_result['test_auroc_mean']:.3f}")
    print(f"    Overfitting: {best_result['overfitting_gap_mean']*100:+.2f}pp")

    # Detailed best result from one seed
    best_seed_result = best_result['results_by_seed'][0]
    cm = best_seed_result['test_confusion_matrix']
    print(f"\n  Best probe confusion matrix (seed=42):")
    print(f"                Predicted")
    print(f"                Deceptive  Honest")
    print(f"  Actual:")
    print(f"    Deceptive      {cm[0][0]:3d}       {cm[0][1]:3d}")
    print(f"    Honest         {cm[1][0]:3d}       {cm[1][1]:3d}")

    deceptive_recall = cm[0][0] / (cm[0][0] + cm[0][1])
    honest_recall = cm[1][1] / (cm[1][0] + cm[1][1])
    print(f"    Deceptive Recall: {deceptive_recall*100:.1f}%")
    print(f"    Honest Recall:    {honest_recall*100:.1f}%")

    # Save results
    print(f"\n[4/4] Saving results...")

    output_dir = script_dir.parent / "results"
    output_file = output_dir / "probe_results_proper_split_gpt2.json"

    final_results = {
        'model': 'gpt2',
        'dataset': 'probe_dataset_gpt2_clean.json',
        'dataset_size': len(samples),
        'train_test_split': '80/20',
        'random_seeds': random_seeds,
        'summary': {
            'mean_test_accuracy': float(np.mean(test_accs)),
            'std_test_accuracy': float(np.std(test_accs)),
            'best_layer': best_result['layer'],
            'best_token': int(best_result['token']),
            'best_test_accuracy': float(best_result['test_accuracy_mean']),
            'best_test_auroc': float(best_result['test_auroc_mean'])
        },
        'all_probes': all_results
    }

    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"  ✅ Saved: {output_file}")

    # Final verdict
    print(f"\n" + "=" * 80)
    print(f"FINAL RESULTS - PROPER EVALUATION")
    print(f"=" * 80)

    if best_result['test_accuracy_mean'] > 0.90:
        print(f"✅ EXCELLENT: {best_result['test_accuracy_mean']*100:.2f}% test accuracy")
        print(f"   Continuous thoughts encode deception very well!")
    elif best_result['test_accuracy_mean'] > 0.70:
        print(f"✅ GOOD: {best_result['test_accuracy_mean']*100:.2f}% test accuracy")
        print(f"   Continuous thoughts beat response token baseline (70%)")
    elif best_result['test_accuracy_mean'] > 0.60:
        print(f"⚠️  MODERATE: {best_result['test_accuracy_mean']*100:.2f}% test accuracy")
        print(f"   Continuous thoughts encode some deception signal")
    else:
        print(f"❌ POOR: {best_result['test_accuracy_mean']*100:.2f}% test accuracy")
        print(f"   Continuous thoughts do not encode deception well")

    print(f"=" * 80)

    return all_results


if __name__ == "__main__":
    main()
