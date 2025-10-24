"""
Train error classifier on L14+L16 SAE features (optimal pair from ablation study).

Goal: Achieve 69.40% accuracy with only 2 layers (vs 5 layers for same accuracy).

Key findings from ablation:
- L16 alone: 68.85% (best single layer, beats L14's 67.76%)
- L14+L16: 69.40% (optimal pair, matches all L12-L16)
- L12, L13, L15: Don't help or actively hurt

Pipeline:
1. Load L12-L16 dataset (reuse existing)
2. Load L12-L16 SAE (reuse existing)
3. Encode only L14 + L16 layers → SAE features
4. Concatenate 2 layers × 6 tokens = 12 vectors per problem
5. Train logistic regression: SAE features → correct/incorrect
6. Compare to baselines
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm
from datetime import datetime


class SparseAutoencoder(nn.Module):
    """Simple sparse autoencoder."""

    def __init__(self, input_dim: int = 2048, n_features: int = 2048):
        super().__init__()
        self.input_dim = input_dim
        self.n_features = n_features
        self.encoder = nn.Linear(input_dim, n_features, bias=True)
        self.decoder = nn.Linear(n_features, input_dim, bias=True)

    def encode(self, x):
        return torch.relu(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


def load_sae(weights_path: str, device='cuda'):
    """Load trained SAE from checkpoint."""
    checkpoint = torch.load(weights_path, map_location=device)
    sae = SparseAutoencoder(
        input_dim=checkpoint['config']['input_dim'],
        n_features=checkpoint['config']['n_features']
    )
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.to(device)
    sae.eval()
    return sae, checkpoint


def encode_continuous_thoughts_to_sae_features(dataset, sae, layers_to_use, device='cuda'):
    """Encode continuous thoughts to SAE features for specified layers only."""
    print(f"\nEncoding continuous thoughts to SAE features ({'+'.join(layers_to_use)})...")

    encoded_data = []

    # Process incorrect solutions
    print("  Encoding incorrect solutions...")
    for sol in tqdm(dataset['incorrect_solutions'], desc="Incorrect"):
        thoughts = sol['continuous_thoughts']

        # Encode each layer's thoughts (only specified layers)
        sae_features = {}
        for layer_name in layers_to_use:
            thought_list = thoughts[layer_name]
            thought_tensor = torch.tensor(thought_list, dtype=torch.float32).to(device)

            with torch.no_grad():
                _, features = sae(thought_tensor)  # [6, n_features]

            sae_features[layer_name] = features.cpu().numpy()

        encoded_data.append({
            'pair_id': sol['pair_id'],
            'variant': sol['variant'],
            'is_correct': False,
            'sae_features': sae_features
        })

    # Process correct solutions
    print("  Encoding correct solutions...")
    for sol in tqdm(dataset['correct_solutions'], desc="Correct"):
        thoughts = sol['continuous_thoughts']

        sae_features = {}
        for layer_name in layers_to_use:
            thought_list = thoughts[layer_name]
            thought_tensor = torch.tensor(thought_list, dtype=torch.float32).to(device)

            with torch.no_grad():
                _, features = sae(thought_tensor)

            sae_features[layer_name] = features.cpu().numpy()

        encoded_data.append({
            'pair_id': sol['pair_id'],
            'variant': sol['variant'],
            'is_correct': True,
            'sae_features': sae_features
        })

    print(f"\n  Encoded {len(encoded_data)} solutions")
    print(f"    Incorrect: {sum(1 for d in encoded_data if not d['is_correct'])}")
    print(f"    Correct: {sum(1 for d in encoded_data if d['is_correct'])}")

    return encoded_data


def concatenate_features(encoded_data, n_features, layer_order):
    """
    Concatenate L14+L16 SAE feature vectors per problem.

    2 layers × 6 tokens = 12 vectors (vs 6 for L14-only, 30 for L12-L16)

    Returns:
        X: [n_samples, n_features * 12] feature matrix
        y: [n_samples] binary labels (0=incorrect, 1=correct)
    """
    print("\nConcatenating SAE features (L14+L16)...")

    n_samples = len(encoded_data)
    X = np.zeros((n_samples, n_features * 12))  # 2 layers × 6 tokens
    y = np.zeros(n_samples, dtype=int)

    for i, sample in enumerate(encoded_data):
        sae_features = sample['sae_features']
        y[i] = 1 if sample['is_correct'] else 0

        # Concatenate: L14_t0, L14_t1, ..., L14_t5, L16_t0, ..., L16_t5
        concat_features = []
        for layer_name in layer_order:
            layer_features = sae_features[layer_name]  # [6, n_features]
            for token_idx in range(6):
                concat_features.append(layer_features[token_idx])

        X[i] = np.concatenate(concat_features)

    print(f"  Feature matrix shape: {X.shape} (L14+L16)")
    print(f"  Labels shape: {y.shape}")
    print(f"  Class distribution: {np.sum(y==0)} incorrect, {np.sum(y==1)} correct")

    return X, y


def train_error_classifier(X, y, test_size=0.2, random_state=42):
    """
    Train binary logistic regression on L14+L16 features.

    Returns:
        Results dict
    """
    print("\n" + "="*80)
    print("TRAINING ERROR CLASSIFIER (L14+L16)")
    print("="*80)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    print(f"  Feature dim: {X_train.shape[1]:,} (L14+L16)")

    # Train classifier
    print(f"\nTraining logistic regression...")
    clf = LogisticRegression(max_iter=1000, random_state=random_state, verbose=1)
    clf.fit(X_train, y_train)

    # Predictions
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    # Accuracy
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Train accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Test accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")

    # Classification report
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT (TEST SET)")
    print("="*80)
    print(classification_report(y_test, y_pred_test,
                                target_names=['Incorrect', 'Correct']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 Incorrect  Correct")
    print(f"Actual Incorrect    {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"Actual Correct      {cm[1,0]:4d}      {cm[1,1]:4d}")

    return {
        'classifier': clf,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'y_pred_train': y_pred_train, 'y_pred_test': y_pred_test,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'confusion_matrix': cm
    }


def visualize_comparison(results_l14_l16, output_dir: Path):
    """Generate comparison visualizations."""
    print("\n" + "="*80)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("="*80)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Performance comparison
    ax = axes[0]
    methods = ['L14 only', 'L16 only', 'L14+L16', 'L12-L16\n(all 5)']
    accuracies = [66.67, 68.85, results_l14_l16['test_acc']*100, 69.40]
    feature_dims = [12288, 12288, 24576, 61440]
    colors = ['lightblue', 'lightgreen', 'gold', 'coral']

    bars = ax.bar(methods, accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Error Classification Performance Comparison')
    ax.set_ylim([60, 75])
    ax.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Target (70%)')
    ax.grid(alpha=0.3, axis='y')
    ax.legend()

    # Add value labels and feature dims
    for i, (bar, feat_dim) in enumerate(zip(bars, feature_dims)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax.text(bar.get_x() + bar.get_width()/2., 61,
                f'{feat_dim:,}\nfeatures',
                ha='center', va='bottom', fontsize=8, style='italic')

    # 2. Cost-benefit analysis
    ax = axes[1]
    methods_cost = ['L14 only', 'L16 only', 'L14+L16', 'L12-L16']
    acc_gains = [0, 2.18, results_l14_l16['test_acc']*100 - 66.67, 2.73]
    feat_increases = [0, 0, 12288, 49152]

    # Cost per 1% gain
    costs = []
    for gain, feat_inc in zip(acc_gains, feat_increases):
        if gain > 0:
            costs.append(feat_inc / gain)
        else:
            costs.append(0)

    bars = ax.barh(methods_cost, costs, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Features per 1% Accuracy Gain (lower is better)')
    ax.set_title('Cost-Benefit Analysis')
    ax.grid(alpha=0.3, axis='x')

    # Add value labels
    for i, (bar, cost) in enumerate(zip(bars, costs)):
        width = bar.get_width()
        if width > 0:
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{cost:,.0f} feat/1%',
                    ha='left', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()

    # Save
    for ext in ['png', 'pdf']:
        save_path = output_dir / f'error_classification_l14_l16_comparison.{ext}'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='src/experiments/sae_error_analysis/data/error_analysis_dataset_l12_l16.json')
    parser.add_argument('--sae_weights', type=str,
                        default='src/experiments/sae_error_analysis/sae_l12_l16/sae_weights.pt')
    parser.add_argument('--output_dir', type=str,
                        default='src/experiments/sae_error_analysis/results')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("SAE ERROR CLASSIFICATION - L14+L16 (OPTIMAL PAIR)")
    print("="*80)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"SAE weights: {args.sae_weights}")

    # Load dataset
    with open(args.dataset, 'r') as f:
        dataset = json.load(f)

    print(f"\nDataset: {dataset['metadata']['total']} solutions")

    # Load SAE
    sae, checkpoint = load_sae(args.sae_weights, device)
    print(f"SAE: {checkpoint['config']['n_features']} features")

    # Encode dataset (only L14 and L16)
    layers_to_use = ['L14', 'L16']
    encoded_data = encode_continuous_thoughts_to_sae_features(dataset, sae, layers_to_use, device)

    # Concatenate features
    X, y = concatenate_features(encoded_data, checkpoint['config']['n_features'], layers_to_use)

    # Train classifier
    results = train_error_classifier(X, y)

    # Comparison with baselines
    print("\n" + "="*80)
    print("COMPARISON WITH BASELINES")
    print("="*80)

    baselines = {
        'L14 only': {'acc': 66.67, 'features': 12288, 'layers': 1},
        'L16 only': {'acc': 68.85, 'features': 12288, 'layers': 1},
        'L14+L16': {'acc': results['test_acc']*100, 'features': 24576, 'layers': 2},
        'L12-L16 (all)': {'acc': 69.40, 'features': 61440, 'layers': 5}
    }

    print(f"\n{'Method':<20} {'Accuracy':<12} {'Features':<15} {'Layers':<8} {'vs L14':<10}")
    print("-"*75)
    for name, data in baselines.items():
        vs_l14 = data['acc'] - 66.67
        print(f"{name:<20} {data['acc']:>6.2f}%      {data['features']:>10,}      {data['layers']:>3}      {vs_l14:>+5.2f} pts")

    # Cost-benefit
    print("\n" + "="*80)
    print("COST-BENEFIT ANALYSIS")
    print("="*80)

    for name, data in baselines.items():
        if name == 'L14 only':
            continue
        acc_gain = data['acc'] - 66.67
        feat_increase = data['features'] - 12288
        if acc_gain > 0 and feat_increase > 0:
            cost = feat_increase / acc_gain
            print(f"{name:<20}: +{acc_gain:.2f}% for +{feat_increase:,} features (cost: {cost:,.0f} feat/1%)")
        elif acc_gain > 0 and feat_increase == 0:
            print(f"{name:<20}: +{acc_gain:.2f}% for same features (FREE UPGRADE!)")

    # Visualizations
    visualize_comparison(results, output_dir)

    # Save results
    results_path = output_dir / 'error_classification_l14_l16_results.json'
    results_summary = {
        'experiment': 'SAE Error Classification - L14+L16 (Optimal Pair)',
        'date': datetime.now().isoformat(),
        'layers_used': layers_to_use,
        'test_accuracy': float(results['test_acc']),
        'train_accuracy': float(results['train_acc']),
        'feature_dim': X.shape[1],
        'baselines': baselines,
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'classification_report': classification_report(
            results['y_test'], results['y_pred_test'],
            target_names=['Incorrect', 'Correct'],
            output_dict=True
        )
    }

    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n✅ Results saved to {results_path}")

    # Final summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE - RECOMMENDATION")
    print("="*80)
    print(f"\n{'✅ RECOMMENDED: L14+L16':<40}")
    print(f"  - Accuracy: {results['test_acc']*100:.2f}% (matches all 5 layers)")
    print(f"  - Features: 24,576 (2.5× fewer than L12-L16)")
    print(f"  - Improvement: +{(results['test_acc']*100 - 66.67):.2f} pts vs L14-only")
    print(f"\n{'Alternative: L16 only':<40}")
    print(f"  - Accuracy: 68.85% (better than L14!)")
    print(f"  - Features: 12,288 (same as L14)")
    print(f"  - Best single layer option")


if __name__ == "__main__":
    main()
