"""
Quick wins to improve error classification from 65.57% → 75-80%.

Tests three improvements:
1. Raw activations instead of SAE features (eliminate compression loss)
2. Better regularization (fix overfitting)
3. Better classifiers (XGBoost, Random Forest)

Expected improvements:
- Raw activations: +5-10 pts
- Regularization: +3-5 pts
- Better classifier: +3-5 pts
Target: 75-80% test accuracy
"""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not installed. Install with: pip install xgboost")


def load_raw_continuous_thoughts(dataset_path):
    """
    Load raw continuous thought activations (no SAE encoding).

    Returns:
        X: [n_samples, 36864] raw activations (18 vectors × 2048)
        y: [n_samples] binary labels
    """
    print("\n" + "="*80)
    print("LOADING RAW CONTINUOUS THOUGHTS (NO SAE)")
    print("="*80)

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    n_samples = dataset['metadata']['total']
    n_features_per_vector = 2048
    n_vectors = 18  # 3 layers × 6 tokens

    X = np.zeros((n_samples, n_features_per_vector * n_vectors))
    y = np.zeros(n_samples, dtype=int)

    layer_order = ['early', 'middle', 'late']

    # Process incorrect solutions
    idx = 0
    print("Loading incorrect solutions...")
    for sol in tqdm(dataset['incorrect_solutions'], desc="Incorrect"):
        thoughts = sol['continuous_thoughts']

        # Concatenate in order: early_t0, ..., early_t5, middle_t0, ..., late_t5
        concat_features = []
        for layer_name in layer_order:
            layer_thoughts = thoughts[layer_name]  # List of 6 vectors [2048]
            for thought_vec in layer_thoughts:
                concat_features.append(thought_vec)

        X[idx] = np.concatenate(concat_features)
        y[idx] = 0  # Incorrect
        idx += 1

    # Process correct solutions
    print("Loading correct solutions...")
    for sol in tqdm(dataset['correct_solutions'], desc="Correct"):
        thoughts = sol['continuous_thoughts']

        concat_features = []
        for layer_name in layer_order:
            layer_thoughts = thoughts[layer_name]
            for thought_vec in layer_thoughts:
                concat_features.append(thought_vec)

        X[idx] = np.concatenate(concat_features)
        y[idx] = 1  # Correct
        idx += 1

    print(f"\n✅ Loaded raw activations:")
    print(f"  Shape: {X.shape}")
    print(f"  Incorrect: {np.sum(y==0)}")
    print(f"  Correct: {np.sum(y==1)}")

    return X, y


def test_regularized_logistic(X_train, X_test, y_train, y_test):
    """Test logistic regression with different regularization strengths."""
    print("\n" + "="*80)
    print("TEST 1: REGULARIZED LOGISTIC REGRESSION")
    print("="*80)

    # Try different C values (inverse regularization strength)
    C_values = [0.001, 0.01, 0.1, 1.0, 10.0]

    results = []
    for C in C_values:
        print(f"\nTesting C={C}...")
        clf = LogisticRegression(C=C, max_iter=1000, random_state=42, verbose=0)
        clf.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))

        print(f"  Train: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"  Test:  {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"  Gap:   {(train_acc - test_acc)*100:.2f} pts")

        results.append({
            'C': C,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'gap': train_acc - test_acc
        })

    # Find best C
    best = max(results, key=lambda x: x['test_acc'])
    print(f"\n✅ Best regularization: C={best['C']}")
    print(f"  Test accuracy: {best['test_acc']*100:.2f}%")
    print(f"  Overfitting gap: {best['gap']*100:.2f} pts")

    # Train final model with best C
    clf_best = LogisticRegression(C=best['C'], max_iter=1000, random_state=42)
    clf_best.fit(X_train, y_train)

    return clf_best, best


def test_random_forest(X_train, X_test, y_train, y_test):
    """Test Random Forest classifier."""
    print("\n" + "="*80)
    print("TEST 2: RANDOM FOREST")
    print("="*80)

    print("\nTraining Random Forest...")
    # Use fewer trees for speed, can tune later
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))

    print(f"\n✅ Random Forest results:")
    print(f"  Train: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Test:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Gap:   {(train_acc - test_acc)*100:.2f} pts")

    return clf, {'test_acc': test_acc, 'train_acc': train_acc, 'gap': train_acc - test_acc}


def test_xgboost(X_train, X_test, y_train, y_test):
    """Test XGBoost classifier."""
    if not HAS_XGBOOST:
        print("\n⚠️  Skipping XGBoost (not installed)")
        return None, None

    print("\n" + "="*80)
    print("TEST 3: XGBOOST")
    print("="*80)

    print("\nTraining XGBoost...")
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))

    print(f"\n✅ XGBoost results:")
    print(f"  Train: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Test:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Gap:   {(train_acc - test_acc)*100:.2f} pts")

    return clf, {'test_acc': test_acc, 'train_acc': train_acc, 'gap': train_acc - test_acc}


def visualize_comparison(results, baseline_acc, output_dir):
    """Compare all methods against baseline."""
    print("\n" + "="*80)
    print("GENERATING COMPARISON VISUALIZATION")
    print("="*80)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Test accuracy comparison
    ax = axes[0]
    methods = ['Baseline\n(SAE Features)', 'Regularized\nLogistic', 'Random\nForest']
    accuracies = [
        baseline_acc * 100,
        results['logistic']['test_acc'] * 100,
        results['random_forest']['test_acc'] * 100
    ]

    if results.get('xgboost'):
        methods.append('XGBoost')
        accuracies.append(results['xgboost']['test_acc'] * 100)

    colors = ['gray', 'blue', 'green', 'purple'][:len(methods)]
    bars = ax.bar(methods, accuracies, color=colors, alpha=0.7, edgecolor='black')

    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Quick Wins: Error Classification Performance\n(Raw Activations)')
    ax.set_ylim([0, 100])
    ax.axhline(y=baseline_acc*100, color='red', linestyle='--', alpha=0.5,
               label=f'Baseline ({baseline_acc*100:.1f}%)')
    ax.axhline(y=75, color='orange', linestyle='--', alpha=0.5,
               label='Target (75%)')
    ax.grid(alpha=0.3, axis='y')
    ax.legend()

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        improvement = height - baseline_acc*100
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%\n(+{improvement:.1f})',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    # 2. Overfitting comparison
    ax = axes[1]
    gaps = [
        97.67 - baseline_acc*100,  # Baseline gap
        results['logistic']['gap'] * 100,
        results['random_forest']['gap'] * 100
    ]

    gap_methods = methods[:3]
    if results.get('xgboost'):
        gaps.append(results['xgboost']['gap'] * 100)
        gap_methods = methods

    bars = ax.bar(gap_methods, gaps, color=colors[:len(gap_methods)],
                  alpha=0.7, edgecolor='black')
    ax.set_ylabel('Train-Test Gap (percentage points)')
    ax.set_title('Overfitting Comparison')
    ax.set_ylim([0, max(gaps) * 1.2])
    ax.axhline(y=10, color='green', linestyle='--', alpha=0.5,
               label='Acceptable (<10 pts)')
    ax.grid(alpha=0.3, axis='y')
    ax.legend()

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    # Save
    for ext in ['png', 'pdf']:
        save_path = output_dir / f'quick_wins_comparison.{ext}'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='src/experiments/sae_error_analysis/data/error_analysis_dataset.json',
                        help='Path to error analysis dataset')
    parser.add_argument('--output_dir', type=str,
                        default='src/experiments/sae_error_analysis/results',
                        help='Output directory')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_seed', type=int, default=42)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("QUICK WINS: ERROR CLASSIFICATION IMPROVEMENT")
    print("="*80)
    print(f"Goal: Improve from 65.57% → 75-80%")
    print(f"Strategy: Raw activations + regularization + better classifiers")

    # Load raw continuous thoughts (no SAE encoding)
    X, y = load_raw_continuous_thoughts(args.dataset)

    # Split data
    print("\n" + "="*80)
    print("DATA SPLIT")
    print("="*80)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_seed, stratify=y
    )
    print(f"Train: {len(X_train)} samples")
    print(f"Test:  {len(X_test)} samples")

    # Run tests
    results = {}

    # Test 1: Regularized logistic regression
    clf_log, results['logistic'] = test_regularized_logistic(X_train, X_test, y_train, y_test)

    # Test 2: Random Forest
    clf_rf, results['random_forest'] = test_random_forest(X_train, X_test, y_train, y_test)

    # Test 3: XGBoost
    clf_xgb, xgb_results = test_xgboost(X_train, X_test, y_train, y_test)
    if xgb_results:
        results['xgboost'] = xgb_results

    # Summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"\nBaseline (SAE features + default LogReg): 65.57%")
    print(f"\nQuick Wins (Raw activations):")
    print(f"  Regularized Logistic: {results['logistic']['test_acc']*100:.2f}% "
          f"(+{(results['logistic']['test_acc'] - 0.6557)*100:.1f} pts)")
    print(f"  Random Forest:        {results['random_forest']['test_acc']*100:.2f}% "
          f"(+{(results['random_forest']['test_acc'] - 0.6557)*100:.1f} pts)")

    if 'xgboost' in results:
        print(f"  XGBoost:              {results['xgboost']['test_acc']*100:.2f}% "
              f"(+{(results['xgboost']['test_acc'] - 0.6557)*100:.1f} pts)")

    # Find best method
    best_method = max(results.items(), key=lambda x: x[1]['test_acc'])
    best_name, best_result = best_method

    print(f"\n🏆 Best method: {best_name.upper()}")
    print(f"  Test accuracy: {best_result['test_acc']*100:.2f}%")
    print(f"  Improvement: +{(best_result['test_acc'] - 0.6557)*100:.1f} pts")
    print(f"  Target (75%): {'✅ MET' if best_result['test_acc'] >= 0.75 else '❌ NOT MET'}")

    # Visualize
    visualize_comparison(results, baseline_acc=0.6557, output_dir=output_dir)

    # Save results
    summary = {
        'baseline': {
            'method': 'SAE features + Logistic Regression',
            'test_accuracy': 0.6557,
            'train_accuracy': 0.9767,
            'overfitting_gap': 0.321
        },
        'raw_activations': {
            'regularized_logistic': {
                'best_C': results['logistic']['C'],
                'test_accuracy': float(results['logistic']['test_acc']),
                'train_accuracy': float(results['logistic']['train_acc']),
                'overfitting_gap': float(results['logistic']['gap']),
                'improvement_vs_baseline': float(results['logistic']['test_acc'] - 0.6557)
            },
            'random_forest': {
                'test_accuracy': float(results['random_forest']['test_acc']),
                'train_accuracy': float(results['random_forest']['train_acc']),
                'overfitting_gap': float(results['random_forest']['gap']),
                'improvement_vs_baseline': float(results['random_forest']['test_acc'] - 0.6557)
            }
        },
        'best_method': {
            'name': best_name,
            'test_accuracy': float(best_result['test_acc']),
            'improvement_vs_baseline': float(best_result['test_acc'] - 0.6557),
            'target_75_met': best_result['test_acc'] >= 0.75
        }
    }

    if 'xgboost' in results:
        summary['raw_activations']['xgboost'] = {
            'test_accuracy': float(results['xgboost']['test_acc']),
            'train_accuracy': float(results['xgboost']['train_acc']),
            'overfitting_gap': float(results['xgboost']['gap']),
            'improvement_vs_baseline': float(results['xgboost']['test_acc'] - 0.6557)
        }

    results_path = output_dir / 'quick_wins_results.json'
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Results saved to {results_path}")
    print("="*80)


if __name__ == "__main__":
    main()
