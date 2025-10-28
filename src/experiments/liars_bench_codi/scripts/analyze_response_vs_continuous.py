"""
Analyze why response tokens achieve 70.5% while continuous thoughts achieve ~48%.

This script:
1. Loads both continuous thought and response token activations
2. Compares activation statistics (mean, std, sparsity)
3. Computes feature importance from response probe
4. Checks if important response dimensions correlate with continuous thoughts
5. Visualizes activation distributions

Author: Claude Code
Date: 2025-10-28
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr


def compute_activation_stats(activations, label):
    """Compute statistics for activation vectors."""
    activations = np.array(activations)

    stats = {
        'label': label,
        'n_samples': len(activations),
        'feature_dim': activations.shape[1],
        'mean': float(np.mean(activations)),
        'std': float(np.std(activations)),
        'mean_per_dim': np.mean(activations, axis=0),
        'std_per_dim': np.std(activations, axis=0),
        'sparsity': float(np.mean(activations == 0)),
        'magnitude': float(np.mean(np.linalg.norm(activations, axis=1))),
        'max_activation': float(np.max(activations)),
        'min_activation': float(np.min(activations))
    }

    return stats


def train_probe_and_get_weights(X, y, random_seed=42):
    """Train probe and return feature weights."""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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

    # Get feature weights (coefficients)
    weights = clf.coef_[0]  # Shape: (n_features,)

    return weights, clf, scaler


def main():
    print("=" * 80)
    print("RESPONSE vs CONTINUOUS THOUGHTS ANALYSIS")
    print("=" * 80)

    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results"
    results_dir.mkdir(exist_ok=True)

    figures_dir = results_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Load continuous thought activations
    print("\n[1/6] Loading continuous thought activations...")
    continuous_path = script_dir.parent / "data" / "processed" / "probe_dataset_gpt2_clean.json"

    with open(continuous_path, 'r') as f:
        continuous_data = json.load(f)

    # Extract Layer 4 Token 5 (best performing continuous thought probe)
    continuous_activations = []
    continuous_labels = []

    for sample in continuous_data['samples']:
        activation = sample['thoughts']['layer_4'][5]  # Token 5, Layer 4
        continuous_activations.append(activation)
        continuous_labels.append(1 if sample['is_honest'] else 0)

    continuous_activations = np.array(continuous_activations)
    continuous_labels = np.array(continuous_labels)

    print(f"  Continuous thoughts: {continuous_activations.shape}")
    print(f"  Labels: Honest={np.sum(continuous_labels==1)}, Deceptive={np.sum(continuous_labels==0)}")

    # Load response token activations
    print("\n[2/6] Loading response token activations...")
    response_path = script_dir.parent / "data" / "processed" / "response_activations_gpt2.json"

    with open(response_path, 'r') as f:
        response_data = json.load(f)

    response_activations = []
    response_labels = []

    for sample in response_data['samples']:
        # Response activation is already pooled
        activation = sample['response_activation']
        response_activations.append(activation)
        response_labels.append(1 if sample['is_honest'] else 0)

    response_activations = np.array(response_activations)
    response_labels = np.array(response_labels)

    print(f"  Response tokens: {response_activations.shape}")
    print(f"  Labels: Honest={np.sum(response_labels==1)}, Deceptive={np.sum(response_labels==0)}")

    # Compute activation statistics
    print("\n[3/6] Computing activation statistics...")

    # Split by label
    cont_honest = continuous_activations[continuous_labels == 1]
    cont_deceptive = continuous_activations[continuous_labels == 0]
    resp_honest = response_activations[response_labels == 1]
    resp_deceptive = response_activations[response_labels == 0]

    stats_raw = {
        'continuous_honest': compute_activation_stats(cont_honest, 'Continuous (Honest)'),
        'continuous_deceptive': compute_activation_stats(cont_deceptive, 'Continuous (Deceptive)'),
        'response_honest': compute_activation_stats(resp_honest, 'Response (Honest)'),
        'response_deceptive': compute_activation_stats(resp_deceptive, 'Response (Deceptive)')
    }

    # Remove numpy arrays for JSON serialization
    stats = {}
    for key, val in stats_raw.items():
        stats[key] = {k: v for k, v in val.items() if not isinstance(v, np.ndarray)}

    print(f"\n{'Type':<30} {'Mean':<12} {'Std':<12} {'Sparsity':<12} {'Magnitude':<12}")
    print(f"{'-'*78}")
    for key, stat in stats.items():
        print(f"{stat['label']:<30} {stat['mean']:<12.4f} {stat['std']:<12.4f} "
              f"{stat['sparsity']:<12.4f} {stat['magnitude']:<12.4f}")

    # Train probes and get feature weights
    print("\n[4/6] Training probes and extracting feature weights...")

    continuous_weights, cont_clf, cont_scaler = train_probe_and_get_weights(
        continuous_activations, continuous_labels
    )
    response_weights, resp_clf, resp_scaler = train_probe_and_get_weights(
        response_activations, response_labels
    )

    print(f"\n  Continuous weights: {continuous_weights.shape}, range=[{continuous_weights.min():.3f}, {continuous_weights.max():.3f}]")
    print(f"  Response weights: {response_weights.shape}, range=[{response_weights.min():.3f}, {response_weights.max():.3f}]")

    # Analyze feature importance
    print("\n[5/6] Analyzing feature importance...")

    # Top features by absolute weight
    cont_top_indices = np.argsort(np.abs(continuous_weights))[-20:][::-1]
    resp_top_indices = np.argsort(np.abs(response_weights))[-20:][::-1]

    print(f"\n  Top 10 most important dimensions (by |weight|):")
    print(f"\n  CONTINUOUS THOUGHTS:")
    print(f"  {'Rank':<6} {'Dim':<8} {'Weight':<12} {'Mean Δ':<12}")
    print(f"  {'-'*38}")

    for rank, dim in enumerate(cont_top_indices[:10]):
        weight = continuous_weights[dim]
        mean_diff = stats_raw['continuous_honest']['mean_per_dim'][dim] - stats_raw['continuous_deceptive']['mean_per_dim'][dim]
        print(f"  {rank+1:<6} {dim:<8} {weight:<12.4f} {mean_diff:<12.4f}")

    print(f"\n  RESPONSE TOKENS:")
    print(f"  {'Rank':<6} {'Dim':<8} {'Weight':<12} {'Mean Δ':<12}")
    print(f"  {'-'*38}")

    for rank, dim in enumerate(resp_top_indices[:10]):
        weight = response_weights[dim]
        mean_diff = stats_raw['response_honest']['mean_per_dim'][dim] - stats_raw['response_deceptive']['mean_per_dim'][dim]
        print(f"  {rank+1:<6} {dim:<8} {weight:<12.4f} {mean_diff:<12.4f}")

    # Correlation analysis
    print(f"\n  Weight correlation:")
    pearson_corr, pearson_p = pearsonr(continuous_weights, response_weights)
    spearman_corr, spearman_p = spearmanr(continuous_weights, response_weights)

    print(f"    Pearson:  r={pearson_corr:.3f}, p={pearson_p:.3e}")
    print(f"    Spearman: ρ={spearman_corr:.3f}, p={spearman_p:.3e}")

    if abs(pearson_corr) < 0.3:
        print(f"\n    ⚠️  Weak correlation: Response and continuous encode DIFFERENT features!")
    else:
        print(f"\n    ✅ Correlation found: Some shared feature importance")

    # Visualizations
    print("\n[6/6] Creating visualizations...")

    # 1. Weight magnitude comparison
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(np.abs(continuous_weights), bins=50, alpha=0.7, label='Continuous', color='blue')
    plt.hist(np.abs(response_weights), bins=50, alpha=0.7, label='Response', color='red')
    plt.xlabel('|Weight|')
    plt.ylabel('Count')
    plt.title('Feature Weight Magnitude Distribution')
    plt.legend()
    plt.yscale('log')

    plt.subplot(1, 2, 2)
    plt.scatter(continuous_weights, response_weights, alpha=0.3, s=10)
    plt.xlabel('Continuous Thought Weight')
    plt.ylabel('Response Token Weight')
    plt.title(f'Weight Correlation (r={pearson_corr:.3f})')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / 'weight_comparison.png', dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved: {figures_dir / 'weight_comparison.png'}")
    plt.close()

    # 2. Activation distributions
    plt.figure(figsize=(14, 6))

    # Sample a few top dimensions
    top_resp_dims = resp_top_indices[:4]

    for i, dim in enumerate(top_resp_dims):
        plt.subplot(2, 4, i+1)
        plt.hist(cont_honest[:, dim], bins=30, alpha=0.6, label='Honest', color='green')
        plt.hist(cont_deceptive[:, dim], bins=30, alpha=0.6, label='Deceptive', color='orange')
        plt.xlabel(f'Dim {dim}')
        plt.ylabel('Count')
        plt.title(f'Continuous - Top {i+1} Resp Dim')
        plt.legend(fontsize=8)

        plt.subplot(2, 4, i+5)
        plt.hist(resp_honest[:, dim], bins=30, alpha=0.6, label='Honest', color='green')
        plt.hist(resp_deceptive[:, dim], bins=30, alpha=0.6, label='Deceptive', color='orange')
        plt.xlabel(f'Dim {dim}')
        plt.ylabel('Count')
        plt.title(f'Response - Top {i+1} Dim')
        plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(figures_dir / 'activation_distributions.png', dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved: {figures_dir / 'activation_distributions.png'}")
    plt.close()

    # 3. Per-dimension variance comparison
    cont_var = np.var(continuous_activations, axis=0)
    resp_var = np.var(response_activations, axis=0)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(cont_var, bins=50, alpha=0.7, label='Continuous', color='blue')
    plt.hist(resp_var, bins=50, alpha=0.7, label='Response', color='red')
    plt.xlabel('Variance')
    plt.ylabel('Count')
    plt.title('Per-Dimension Variance Distribution')
    plt.legend()
    plt.yscale('log')

    plt.subplot(1, 2, 2)
    sorted_cont_var = np.sort(cont_var)[::-1]
    sorted_resp_var = np.sort(resp_var)[::-1]
    plt.plot(sorted_cont_var[:200], label='Continuous', color='blue')
    plt.plot(sorted_resp_var[:200], label='Response', color='red')
    plt.xlabel('Dimension Rank')
    plt.ylabel('Variance')
    plt.title('Top 200 Dimensions by Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / 'variance_comparison.png', dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved: {figures_dir / 'variance_comparison.png'}")
    plt.close()

    # Save analysis results
    print("\n[7/7] Saving analysis results...")

    analysis_results = {
        'model': 'gpt2',
        'continuous_source': 'layer_4_token_5',
        'response_source': 'mean_pooled',
        'statistics': stats,
        'weight_analysis': {
            'continuous_top_10_dims': cont_top_indices[:10].tolist(),
            'continuous_top_10_weights': continuous_weights[cont_top_indices[:10]].tolist(),
            'response_top_10_dims': resp_top_indices[:10].tolist(),
            'response_top_10_weights': response_weights[resp_top_indices[:10]].tolist(),
            'pearson_correlation': float(pearson_corr),
            'pearson_pvalue': float(pearson_p),
            'spearman_correlation': float(spearman_corr),
            'spearman_pvalue': float(spearman_p)
        },
        'interpretation': {
            'weight_correlation': 'weak' if abs(pearson_corr) < 0.3 else 'moderate' if abs(pearson_corr) < 0.6 else 'strong',
            'shared_features': bool(abs(pearson_corr) > 0.3),
            'response_superior': 'Response tokens encode deception in different/richer feature space than continuous thoughts'
        }
    }

    output_file = results_dir / 'response_vs_continuous_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)

    print(f"  ✅ Saved: {output_file}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n1. ACTIVATION STATISTICS:")
    print(f"   Response tokens have {stats['response_honest']['magnitude']/stats['continuous_honest']['magnitude']:.2f}x larger magnitude")
    print(f"   Response tokens are {(1-stats['response_honest']['sparsity'])/(1-stats['continuous_honest']['sparsity']):.2f}x less sparse")

    print(f"\n2. FEATURE IMPORTANCE:")
    print(f"   Weight correlation: r={pearson_corr:.3f} ({analysis_results['interpretation']['weight_correlation']})")
    if not analysis_results['interpretation']['shared_features']:
        print(f"   ⚠️  Response and continuous encode DIFFERENT features for deception!")

    print(f"\n3. KEY INSIGHT:")
    print(f"   {analysis_results['interpretation']['response_superior']}")

    print(f"\n4. HYPOTHESIS:")
    print(f"   Continuous thoughts may contain deception signal but:")
    print(f"   - It's distributed across many weak dimensions (not sparse)")
    print(f"   - Linear probes can't find the right combination")
    print(f"   - Response tokens aggregate and sharpen this signal during final layers")

    print("\n" + "=" * 80)

    return analysis_results


if __name__ == "__main__":
    main()
