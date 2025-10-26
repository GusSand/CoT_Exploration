"""
Compute Pareto frontier for TopK SAE grid experiment.

Analyzes quality-sparsity tradeoff:
- Quality: Explained Variance (higher is better)
- Sparsity: K value (lower is better = more sparse)

A configuration is Pareto-optimal if no other configuration has both
better quality AND better sparsity.

Usage:
    python pareto_analysis.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_and_merge_results():
    """Load 3 JSON files and merge into single grid."""
    results_dir = Path('src/experiments/topk_grid_pilot/results')

    # Load all 3 files
    data = {}
    for latent_dim in [512, 1024, 2048]:
        json_path = results_dir / f'grid_metrics_latent{latent_dim}.json'
        with open(json_path, 'r') as f:
            loaded = json.load(f)
            data.update(loaded['results'])

    return data


def extract_configurations(data):
    """
    Extract all 12 configurations with their metrics.

    Returns:
        configs: List of (latent_dim, k, explained_variance, feature_death_rate)
    """
    configs = []

    for latent_dim in [512, 1024, 2048]:
        for k in [5, 10, 20, 100]:
            metrics = data[str(latent_dim)][str(k)]
            configs.append({
                'latent_dim': latent_dim,
                'k': k,
                'explained_variance': metrics['explained_variance'],
                'feature_death_rate': metrics['feature_death_rate'],
                'reconstruction_loss': metrics['reconstruction_loss'],
                'mean_activation': metrics['mean_activation'],
            })

    return configs


def compute_pareto_frontier(configs):
    """
    Compute Pareto frontier for quality-sparsity tradeoff.

    Quality: Explained Variance (maximize)
    Sparsity: K value (minimize)

    Returns:
        pareto_configs: Configurations on Pareto frontier
        dominated_configs: Configurations not on Pareto frontier
    """
    pareto_configs = []
    dominated_configs = []

    for config in configs:
        is_dominated = False

        # Check if any other config dominates this one
        for other in configs:
            if other == config:
                continue

            # Other dominates config if:
            # 1. Other has better quality (higher EV) AND
            # 2. Other has better sparsity (lower K)
            better_quality = other['explained_variance'] > config['explained_variance']
            better_sparsity = other['k'] < config['k']

            if better_quality and better_sparsity:
                is_dominated = True
                break

        if is_dominated:
            dominated_configs.append(config)
        else:
            pareto_configs.append(config)

    return pareto_configs, dominated_configs


def plot_pareto_frontier(pareto_configs, dominated_configs, output_path):
    """Visualize Pareto frontier."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot dominated configurations
    if dominated_configs:
        k_vals = [c['k'] for c in dominated_configs]
        ev_vals = [c['explained_variance'] for c in dominated_configs]
        latent_dims = [c['latent_dim'] for c in dominated_configs]

        scatter = ax.scatter(
            k_vals, ev_vals,
            c=latent_dims,
            cmap='viridis',
            s=200,
            alpha=0.3,
            edgecolors='gray',
            linewidths=1,
            label='Dominated'
        )

    # Plot Pareto-optimal configurations
    if pareto_configs:
        k_vals = [c['k'] for c in pareto_configs]
        ev_vals = [c['explained_variance'] for c in pareto_configs]
        latent_dims = [c['latent_dim'] for c in pareto_configs]

        scatter = ax.scatter(
            k_vals, ev_vals,
            c=latent_dims,
            cmap='viridis',
            s=300,
            alpha=1.0,
            edgecolors='red',
            linewidths=3,
            marker='*',
            label='Pareto Optimal'
        )

        # Sort by K for line plot
        sorted_pareto = sorted(pareto_configs, key=lambda x: x['k'])
        k_sorted = [c['k'] for c in sorted_pareto]
        ev_sorted = [c['explained_variance'] for c in sorted_pareto]

        # Draw Pareto frontier line
        ax.plot(k_sorted, ev_sorted, 'r--', linewidth=2, alpha=0.5, label='Pareto Frontier')

    # Add labels for all points
    all_configs = pareto_configs + dominated_configs
    for config in all_configs:
        label = f"d={config['latent_dim']}, K={config['k']}"
        ax.annotate(
            label,
            (config['k'], config['explained_variance']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.7
        )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Dictionary Size (latent_dim)', fontsize=12)

    # Labels and title
    ax.set_xlabel('Sparsity Level (K) - Lower is Better', fontsize=14, fontweight='bold')
    ax.set_ylabel('Explained Variance - Higher is Better', fontsize=14, fontweight='bold')
    ax.set_title('Pareto Frontier: Quality vs Sparsity Tradeoff', fontsize=16, fontweight='bold')

    # Log scale for K axis
    ax.set_xscale('log')
    ax.set_xticks([5, 10, 20, 100])
    ax.set_xticklabels(['5', '10', '20', '100'])

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def main():
    print("=" * 80)
    print("Pareto Frontier Analysis")
    print("=" * 80)
    print()

    # Load results
    print("Loading results...")
    data = load_and_merge_results()
    configs = extract_configurations(data)
    print(f"  Total configurations: {len(configs)}")
    print()

    # Compute Pareto frontier
    print("Computing Pareto frontier...")
    pareto_configs, dominated_configs = compute_pareto_frontier(configs)
    print(f"  Pareto-optimal: {len(pareto_configs)}")
    print(f"  Dominated: {len(dominated_configs)}")
    print()

    # Print Pareto-optimal configurations
    print("Pareto-Optimal Configurations:")
    print("-" * 80)
    print(f"{'Config':<20} {'EV':>8} {'Death%':>8} {'Loss':>10} {'Mean Act':>10}")
    print("-" * 80)

    pareto_sorted = sorted(pareto_configs, key=lambda x: x['k'])
    for config in pareto_sorted:
        config_str = f"d={config['latent_dim']}, K={config['k']}"
        print(f"{config_str:<20} {config['explained_variance']:>8.4f} "
              f"{config['feature_death_rate']:>8.3f} "
              f"{config['reconstruction_loss']:>10.6f} "
              f"{config['mean_activation']:>10.3f}")

    print("-" * 80)
    print()

    # Generate visualization
    print("Generating Pareto frontier plot...")
    output_dir = Path('src/experiments/topk_grid_pilot/results')
    plot_pareto_frontier(
        pareto_configs,
        dominated_configs,
        output_path=output_dir / 'pareto_frontier.png'
    )

    # Save Pareto configs to JSON
    pareto_path = output_dir / 'pareto_optimal_configs.json'
    with open(pareto_path, 'w') as f:
        json.dump({
            'pareto_optimal': pareto_sorted,
            'num_pareto': len(pareto_configs),
            'num_dominated': len(dominated_configs),
            'analysis': {
                'quality_metric': 'explained_variance',
                'sparsity_metric': 'k',
                'quality_direction': 'maximize',
                'sparsity_direction': 'minimize',
            }
        }, f, indent=2)

    print(f"  Saved: {pareto_path}")
    print()

    print("=" * 80)
    print("Pareto analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
