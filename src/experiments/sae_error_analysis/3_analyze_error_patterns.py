"""
Analyze error patterns and SAE feature specialization.

Goals:
1. Identify which SAE features are most predictive of errors
2. Perform error localization: which layers/tokens predict failures
3. Analyze feature specialization patterns
4. Map features to error characteristics
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import defaultdict


def analyze_feature_importance(encoded_dataset, results_path):
    """
    Analyze which SAE features are most predictive of errors.

    Args:
        encoded_dataset: Loaded from encoded_error_dataset.pt
        results_path: Path to error_classification_results.json

    Returns:
        Dict with feature importance analysis
    """
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)

    # Load classification results
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Note: The classifier's coefficients aren't saved in the JSON
    # We'll need to analyze the features directly from the data
    X = encoded_dataset['X']
    y = encoded_dataset['y']
    n_features_per_vector = 2048  # SAE features
    n_vectors = 18  # 3 layers × 6 tokens

    print(f"Feature matrix shape: {X.shape}")
    print(f"Features per vector: {n_features_per_vector}")
    print(f"Vectors per sample: {n_vectors}")

    # Compute feature statistics for incorrect vs correct
    X_incorrect = X[y == 0]  # Incorrect solutions
    X_correct = X[y == 1]    # Correct solutions

    print(f"\nSample sizes:")
    print(f"  Incorrect: {len(X_incorrect)}")
    print(f"  Correct: {len(X_correct)}")

    # Compute mean activation difference
    mean_incorrect = X_incorrect.mean(axis=0)
    mean_correct = X_correct.mean(axis=0)
    activation_diff = mean_incorrect - mean_correct

    # Compute statistical measures
    std_incorrect = X_incorrect.std(axis=0)
    std_correct = X_correct.std(axis=0)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((std_incorrect**2 + std_correct**2) / 2)
    cohens_d = activation_diff / (pooled_std + 1e-8)

    # Identify top discriminative features
    n_top = 100
    top_indices = np.argsort(np.abs(cohens_d))[::-1][:n_top]

    print(f"\nTop {n_top} discriminative features:")
    print(f"  Max |Cohen's d|: {np.abs(cohens_d[top_indices[0]]):.4f}")
    print(f"  Mean |Cohen's d| (top {n_top}): {np.mean(np.abs(cohens_d[top_indices])):.4f}")

    # Map feature indices to (layer, token, feature_id)
    def decode_feature_index(idx):
        """Map concatenated feature index to (layer, token, feature_id)."""
        vector_idx = idx // n_features_per_vector
        feature_idx = idx % n_features_per_vector

        layer_idx = vector_idx // 6  # 0=early, 1=middle, 2=late
        token_idx = vector_idx % 6

        layer_names = ['early', 'middle', 'late']
        return layer_names[layer_idx], token_idx, feature_idx

    # Analyze distribution across layers and tokens
    layer_counts = defaultdict(int)
    token_counts = defaultdict(int)
    layer_token_counts = defaultdict(int)

    for idx in top_indices:
        layer, token, feat_id = decode_feature_index(idx)
        layer_counts[layer] += 1
        token_counts[token] += 1
        layer_token_counts[(layer, token)] += 1

    print(f"\nTop {n_top} features distribution:")
    print(f"  By layer:")
    for layer in ['early', 'middle', 'late']:
        print(f"    {layer}: {layer_counts[layer]} ({layer_counts[layer]/n_top*100:.1f}%)")

    print(f"  By token:")
    for token in range(6):
        print(f"    Token {token}: {token_counts[token]} ({token_counts[token]/n_top*100:.1f}%)")

    # Return analysis results
    return {
        'activation_diff': activation_diff,
        'cohens_d': cohens_d,
        'top_indices': top_indices,
        'layer_counts': dict(layer_counts),
        'token_counts': dict(token_counts),
        'layer_token_counts': {f"{k[0]}_t{k[1]}": v for k, v in layer_token_counts.items()},
        'mean_incorrect': mean_incorrect,
        'mean_correct': mean_correct,
        'std_incorrect': std_incorrect,
        'std_correct': std_correct
    }


def error_localization_analysis(importance_results, n_top=100):
    """
    Perform error localization: which layers/tokens predict failures.

    Args:
        importance_results: Output from analyze_feature_importance
        n_top: Number of top features to analyze

    Returns:
        Dict with error localization results
    """
    print("\n" + "="*80)
    print("ERROR LOCALIZATION ANALYSIS")
    print("="*80)

    layer_counts = importance_results['layer_counts']
    token_counts = importance_results['token_counts']

    # Compute percentages
    layer_pcts = {k: v/n_top*100 for k, v in layer_counts.items()}
    token_pcts = {k: v/n_top*100 for k, v in token_counts.items()}

    print(f"\nError-predictive features by layer:")
    for layer in ['early', 'middle', 'late']:
        pct = layer_pcts.get(layer, 0)
        bar = '█' * int(pct / 2)
        print(f"  {layer:8s}: {pct:5.1f}% {bar}")

    print(f"\nError-predictive features by token position:")
    for token in range(6):
        pct = token_pcts.get(token, 0)
        bar = '█' * int(pct / 2)
        print(f"  Token {token}: {pct:5.1f}% {bar}")

    # Heatmap data: layer × token
    heatmap_data = np.zeros((3, 6))
    layer_map = {'early': 0, 'middle': 1, 'late': 2}

    layer_token_counts = importance_results['layer_token_counts']
    for key, count in layer_token_counts.items():
        layer_name, token_str = key.rsplit('_t', 1)
        token_idx = int(token_str)
        layer_idx = layer_map[layer_name]
        heatmap_data[layer_idx, token_idx] = count

    print(f"\nError localization heatmap (layer × token):")
    print(f"         T0   T1   T2   T3   T4   T5")
    for i, layer in enumerate(['Early ', 'Middle', 'Late  ']):
        row_str = "  ".join(f"{int(heatmap_data[i, j]):3d}" for j in range(6))
        print(f"  {layer}: {row_str}")

    return {
        'layer_percentages': layer_pcts,
        'token_percentages': token_pcts,
        'heatmap_data': heatmap_data.tolist(),
        'interpretation': {
            'primary_error_layer': max(layer_pcts, key=layer_pcts.get),
            'primary_error_token': max(token_pcts, key=token_pcts.get)
        }
    }


def visualize_error_analysis(importance_results, localization_results, output_dir):
    """Generate comprehensive error analysis visualizations."""
    print("\n" + "="*80)
    print("GENERATING ERROR ANALYSIS VISUALIZATIONS")
    print("="*80)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Cohen's d distribution
    ax = fig.add_subplot(gs[0, :2])
    cohens_d = importance_results['cohens_d']
    ax.hist(cohens_d, bins=100, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel("Cohen's d (Incorrect - Correct)")
    ax.set_ylabel('Count')
    ax.set_title('Feature Discriminability Distribution\n(Effect Size for Incorrect vs Correct)')
    ax.grid(alpha=0.3)

    # 2. Top features effect sizes
    ax = fig.add_subplot(gs[0, 2])
    top_indices = importance_results['top_indices'][:20]
    top_cohens_d = cohens_d[top_indices]
    colors = ['red' if d > 0 else 'green' for d in top_cohens_d]

    ax.barh(range(20), top_cohens_d, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Feature Rank')
    ax.set_xlabel("Cohen's d")
    ax.set_title('Top 20 Discriminative Features\n(Red=Error, Green=Correct)')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='x')

    # 3. Layer distribution
    ax = fig.add_subplot(gs[1, 0])
    layer_pcts = localization_results['layer_percentages']
    layers = ['early', 'middle', 'late']
    values = [layer_pcts.get(l, 0) for l in layers]
    colors_layer = ['lightblue', 'orange', 'lightgreen']

    bars = ax.bar(layers, values, color=colors_layer, alpha=0.7, edgecolor='black')
    ax.set_ylabel('% of Top 100 Features')
    ax.set_title('Error-Predictive Features by Layer')
    ax.set_ylim([0, max(values) * 1.2])
    ax.grid(alpha=0.3, axis='y')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    # 4. Token distribution
    ax = fig.add_subplot(gs[1, 1])
    token_pcts = localization_results['token_percentages']
    tokens = list(range(6))
    values = [token_pcts.get(t, 0) for t in tokens]
    colors_token = plt.cm.viridis(np.linspace(0, 1, 6))

    bars = ax.bar([f'T{t}' for t in tokens], values, color=colors_token,
                  alpha=0.7, edgecolor='black')
    ax.set_ylabel('% of Top 100 Features')
    ax.set_title('Error-Predictive Features by Token')
    ax.set_ylim([0, max(values) * 1.2])
    ax.grid(alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=8)

    # 5. Localization heatmap
    ax = fig.add_subplot(gs[1, 2])
    heatmap_data = np.array(localization_results['heatmap_data'])

    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(6))
    ax.set_xticklabels([f'T{i}' for i in range(6)])
    ax.set_yticks(range(3))
    ax.set_yticklabels(['Early', 'Middle', 'Late'])
    ax.set_title('Error Localization Heatmap\n(Layer × Token)')

    # Add text annotations
    for i in range(3):
        for j in range(6):
            text = ax.text(j, i, f'{int(heatmap_data[i, j])}',
                         ha="center", va="center", color="black", fontsize=10)

    plt.colorbar(im, ax=ax, label='# of Top Features')

    # 6. Activation patterns for top 5 features
    ax = fig.add_subplot(gs[2, :])
    n_features_per_vector = 2048
    top_5_indices = importance_results['top_indices'][:5]

    # Decode feature positions
    feature_labels = []
    for idx in top_5_indices:
        vector_idx = idx // n_features_per_vector
        feature_idx = idx % n_features_per_vector
        layer_idx = vector_idx // 6
        token_idx = vector_idx % 6
        layer_names = ['Early', 'Middle', 'Late']
        feature_labels.append(f"{layer_names[layer_idx]}_T{token_idx}_F{feature_idx}")

    mean_incorrect = importance_results['mean_incorrect'][top_5_indices]
    mean_correct = importance_results['mean_correct'][top_5_indices]

    x = np.arange(len(top_5_indices))
    width = 0.35

    bars1 = ax.bar(x - width/2, mean_incorrect, width, label='Incorrect',
                   color='red', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, mean_correct, width, label='Correct',
                   color='green', alpha=0.7, edgecolor='black')

    ax.set_ylabel('Mean Activation')
    ax.set_xlabel('Feature')
    ax.set_title('Top 5 Error-Predictive Features: Activation Patterns')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)

    # Save
    for ext in ['png', 'pdf']:
        save_path = output_dir / f'error_pattern_analysis.{ext}'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--encoded_dataset', type=str,
                        default='src/experiments/sae_error_analysis/results/encoded_error_dataset.pt',
                        help='Path to encoded dataset')
    parser.add_argument('--results_json', type=str,
                        default='src/experiments/sae_error_analysis/results/error_classification_results.json',
                        help='Path to classification results JSON')
    parser.add_argument('--output_dir', type=str,
                        default='src/experiments/sae_error_analysis/results',
                        help='Output directory')
    parser.add_argument('--n_top', type=int, default=100,
                        help='Number of top features to analyze')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("ERROR PATTERN ANALYSIS")
    print("="*80)
    print(f"Encoded dataset: {args.encoded_dataset}")
    print(f"Results JSON: {args.results_json}")
    print(f"Output: {output_dir}")

    # Load encoded dataset
    print("\nLoading encoded dataset...")
    encoded_dataset = torch.load(args.encoded_dataset, weights_only=False)
    print(f"  X shape: {encoded_dataset['X'].shape}")
    print(f"  y shape: {encoded_dataset['y'].shape}")

    # Analyze feature importance
    importance_results = analyze_feature_importance(
        encoded_dataset, args.results_json
    )

    # Error localization
    localization_results = error_localization_analysis(
        importance_results, n_top=args.n_top
    )

    # Visualize
    visualize_error_analysis(importance_results, localization_results, output_dir)

    # Save analysis results (convert numpy types to Python types for JSON serialization)
    analysis_summary = {
        'n_top_features': args.n_top,
        'layer_distribution': {str(k): float(v) for k, v in localization_results['layer_percentages'].items()},
        'token_distribution': {str(k): float(v) for k, v in localization_results['token_percentages'].items()},
        'localization_heatmap': localization_results['heatmap_data'],
        'interpretation': {
            'primary_error_layer': str(localization_results['interpretation']['primary_error_layer']),
            'primary_error_token': int(localization_results['interpretation']['primary_error_token'])
        },
        'statistics': {
            'max_cohens_d': float(np.max(np.abs(importance_results['cohens_d']))),
            'mean_cohens_d_top100': float(np.mean(
                np.abs(importance_results['cohens_d'][importance_results['top_indices'][:100]])
            )),
            'median_cohens_d': float(np.median(importance_results['cohens_d']))
        }
    }

    results_path = output_dir / 'error_pattern_analysis.json'
    with open(results_path, 'w') as f:
        json.dump(analysis_summary, f, indent=2)

    print(f"\n✅ Analysis complete! Results saved to {results_path}")

    # Print summary
    print("\n" + "="*80)
    print("ERROR PATTERN SUMMARY")
    print("="*80)
    print(f"\nPrimary error-predictive layer: {localization_results['interpretation']['primary_error_layer']}")
    print(f"Primary error-predictive token: T{localization_results['interpretation']['primary_error_token']}")
    print(f"\nMax discriminability (|Cohen's d|): {analysis_summary['statistics']['max_cohens_d']:.4f}")
    print(f"Mean discriminability (top 100): {analysis_summary['statistics']['mean_cohens_d_top100']:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
