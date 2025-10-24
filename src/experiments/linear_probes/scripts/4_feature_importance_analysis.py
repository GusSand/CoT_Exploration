"""
Story 2.3: Feature Importance via Probe Weights

Analyze which activation dimensions are most important for correctness prediction.

Uses Layer 8, Token 0 (best probe) to extract top-100 highest weight dimensions.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Import probe trainer
import importlib.util
spec = importlib.util.spec_from_file_location("probe_trainer", Path(__file__).parent / "2_probe_trainer.py")
probe_trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe_trainer)
ProbeTrainer = probe_trainer.ProbeTrainer


def analyze_probe_weights(layer: int, token: int, trainer: ProbeTrainer, output_dir: Path):
    """
    Analyze feature importance for a specific probe.

    Args:
        layer: Layer index
        token: Token index
        trainer: ProbeTrainer instance
        output_dir: Directory to save results
    """
    print(f"\n" + "="*60)
    print(f"FEATURE IMPORTANCE ANALYSIS")
    print(f"="*60)
    print(f"Probe: Layer {layer}, Token {token}")
    print(f"="*60)

    # Train probe (without wandb to get weights quickly)
    trainer.use_wandb = False
    result = trainer.train_probe(layer=layer, token=token)

    # Get weights
    weights = result.weights
    n_features = len(weights)

    print(f"\nProbe Statistics:")
    print(f"  Accuracy: {result.accuracy:.4f} [{result.accuracy_ci_lower:.4f}, {result.accuracy_ci_upper:.4f}]")
    print(f"  Number of features: {n_features}")
    print(f"  Weight statistics:")
    print(f"    Mean: {np.mean(weights):.6f}")
    print(f"    Std: {np.std(weights):.6f}")
    print(f"    Min: {np.min(weights):.6f}")
    print(f"    Max: {np.max(weights):.6f}")

    # Get top-100 features by absolute weight magnitude
    abs_weights = np.abs(weights)
    top_indices = np.argsort(abs_weights)[::-1][:100]
    top_weights = weights[top_indices]
    top_abs_weights = abs_weights[top_indices]

    print(f"\nTop 100 Features:")
    print(f"  Indices: {top_indices[:10].tolist()}... (showing first 10)")
    print(f"  Absolute weights range: [{np.min(top_abs_weights):.6f}, {np.max(top_abs_weights):.6f}]")
    print(f"  Mean absolute weight (top 100): {np.mean(top_abs_weights):.6f}")
    print(f"  Mean absolute weight (all): {np.mean(abs_weights):.6f}")

    # Calculate what percentage of total weight magnitude is captured by top 100
    total_magnitude = np.sum(abs_weights)
    top100_magnitude = np.sum(top_abs_weights)
    percentage_captured = (top100_magnitude / total_magnitude) * 100

    print(f"\n  Top 100 features capture {percentage_captured:.2f}% of total weight magnitude")

    # Save top features
    top_features = {
        'layer': layer,
        'token': token,
        'accuracy': float(result.accuracy),
        'n_features': n_features,
        'top_100_indices': top_indices.tolist(),
        'top_100_weights': top_weights.tolist(),
        'top_100_abs_weights': top_abs_weights.tolist(),
        'percentage_magnitude_captured': float(percentage_captured),
        'weight_stats': {
            'mean': float(np.mean(weights)),
            'std': float(np.std(weights)),
            'min': float(np.min(weights)),
            'max': float(np.max(weights))
        }
    }

    top_features_path = output_dir / f"top_features_L{layer}_T{token}.json"
    with open(top_features_path, 'w') as f:
        json.dump(top_features, f, indent=2)

    print(f"\nTop features saved to {top_features_path}")

    # Visualizations
    create_weight_visualizations(weights, top_indices, top_weights, layer, token, output_dir)

    return top_features


def create_weight_visualizations(weights, top_indices, top_weights, layer, token, output_dir):
    """Create visualizations of probe weights."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Weight distribution (histogram)
    ax = axes[0, 0]
    ax.hist(weights, bins=100, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax.set_xlabel('Weight Value', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'Weight Distribution (Layer {layer}, Token {token})', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Top 100 features bar plot
    ax = axes[0, 1]
    colors = ['red' if w < 0 else 'green' for w in top_weights[:50]]
    ax.barh(range(50), top_weights[:50], color=colors, alpha=0.7)
    ax.set_xlabel('Weight Value', fontsize=11)
    ax.set_ylabel('Feature Rank', fontsize=11)
    ax.set_title(f'Top 50 Feature Weights', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')

    # 3. Cumulative magnitude plot
    ax = axes[1, 0]
    abs_weights_sorted = np.sort(np.abs(weights))[::-1]
    cumsum = np.cumsum(abs_weights_sorted)
    cumsum_pct = (cumsum / cumsum[-1]) * 100

    ax.plot(cumsum_pct, linewidth=2)
    ax.axhline(50, color='red', linestyle='--', label='50%')
    ax.axhline(80, color='orange', linestyle='--', label='80%')
    ax.axhline(90, color='green', linestyle='--', label='90%')
    ax.axvline(100, color='purple', linestyle='--', label='Top 100')

    # Find how many features needed for 90%
    idx_90 = np.where(cumsum_pct >= 90)[0][0]
    ax.scatter([idx_90], [90], color='green', s=100, zorder=5)
    ax.annotate(f'{idx_90} features\nfor 90%',
                xy=(idx_90, 90), xytext=(idx_90 + 200, 85),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=9)

    ax.set_xlabel('Number of Features (sorted by |weight|)', fontsize=11)
    ax.set_ylabel('Cumulative % of Total Magnitude', fontsize=11)
    ax.set_title('Cumulative Weight Magnitude', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 500)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Top feature indices scatter
    ax = axes[1, 1]
    ax.scatter(top_indices[:100], np.abs(top_weights),
               c=range(100), cmap='viridis', s=50, alpha=0.7)
    ax.set_xlabel('Feature Index (in 2048-dim space)', fontsize=11)
    ax.set_ylabel('|Weight|', fontsize=11)
    ax.set_title('Top 100 Feature Locations', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    viz_path = output_dir / f"feature_importance_L{layer}_T{token}.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"Visualizations saved to {viz_path}")

    plt.close()


def main():
    """Run feature importance analysis on best probe."""
    # Paths
    project_root = Path(__file__).parent.parent.parent.parent.parent
    dataset_path = project_root / "src/experiments/linear_probes/data/probe_dataset_100.json"
    results_dir = project_root / "src/experiments/linear_probes/results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize trainer
    trainer = ProbeTrainer(str(dataset_path), use_wandb=False)

    # Analyze best probe: Layer 8, Token 0
    print("\n" + "="*60)
    print("ANALYZING BEST PROBE")
    print("="*60)

    top_features = analyze_probe_weights(
        layer=8,
        token=0,
        trainer=trainer,
        output_dir=results_dir
    )

    # Also analyze Layer 14 Token 5 (worst probe) for comparison
    print("\n" + "="*60)
    print("ANALYZING WORST PROBE (for comparison)")
    print("="*60)

    top_features_worst = analyze_probe_weights(
        layer=14,
        token=5,
        trainer=trainer,
        output_dir=results_dir
    )

    # Compare top features between best and worst
    print("\n" + "="*60)
    print("COMPARISON: BEST vs WORST PROBE")
    print("="*60)

    best_indices = set(top_features['top_100_indices'])
    worst_indices = set(top_features_worst['top_100_indices'])

    overlap = best_indices & worst_indices
    overlap_pct = (len(overlap) / 100) * 100

    print(f"\nOverlap in top-100 features: {len(overlap)}/100 ({overlap_pct:.1f}%)")
    print(f"  This suggests {overlap_pct:.1f}% of important features are shared")
    print(f"  {100 - overlap_pct:.1f}% of features differ between probes")

    # Save comparison
    comparison = {
        'best_probe': {'layer': 8, 'token': 0, 'accuracy': top_features['accuracy']},
        'worst_probe': {'layer': 14, 'token': 5, 'accuracy': top_features_worst['accuracy']},
        'overlap_top100': len(overlap),
        'overlap_percentage': float(overlap_pct),
        'shared_indices': list(overlap)
    }

    comparison_path = results_dir / "probe_feature_comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\nComparison saved to {comparison_path}")

    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
