"""
Generate layer × position heatmaps for all quality metrics.

Creates comprehensive visualizations showing how SAE quality varies
across the 16 layers × 6 positions grid.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_all_metrics():
    """Load metrics for all (layer, position) pairs."""
    results_dir = Path('src/experiments/topk_grid_pilot/results')

    all_data = []

    for json_file in sorted(results_dir.glob('grid_metrics_pos*_layer*_latent*.json')):
        with open(json_file, 'r') as f:
            data = json.load(f)

        metadata = data['metadata']
        position = metadata['position']
        layer = metadata['layer']

        for latent_dim_str, k_results in data['results'].items():
            latent_dim = int(latent_dim_str)

            for k_str, metrics in k_results.items():
                k = int(k_str)

                all_data.append({
                    'layer': layer,
                    'position': position,
                    'latent_dim': latent_dim,
                    'k': k,
                    **metrics
                })

    return pd.DataFrame(all_data)


def plot_layer_position_heatmap(df, metric, k_val, latent_dim, title, output_path, cmap='viridis', fmt='.3f'):
    """Create layer × position heatmap for a specific metric."""

    # Filter data
    subset = df[(df['k'] == k_val) & (df['latent_dim'] == latent_dim)]

    # Pivot to create matrix
    pivot = subset.pivot_table(
        values=metric,
        index='layer',
        columns='position',
        aggfunc='mean'
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        cbar_kws={'label': metric.replace('_', ' ').title()},
        ax=ax
    )

    ax.set_title(f'{title}\n(K={k_val}, latent_dim={latent_dim})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Continuous Thought Position', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def plot_all_k_comparison(df, metric, latent_dim, output_path):
    """Create 2x2 subplot showing all 4 K values."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    k_values = [5, 10, 20, 100]

    for idx, k_val in enumerate(k_values):
        ax = axes[idx // 2, idx % 2]

        subset = df[(df['k'] == k_val) & (df['latent_dim'] == latent_dim)]
        pivot = subset.pivot_table(
            values=metric,
            index='layer',
            columns='position',
            aggfunc='mean'
        )

        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=df[metric].min(),
            vmax=df[metric].max(),
            cbar_kws={'label': metric.replace('_', ' ').title()},
            ax=ax
        )

        ax.set_title(f'K={k_val}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Position' if idx >= 2 else '')
        ax.set_ylabel('Layer' if idx % 2 == 0 else '')

    plt.suptitle(f'{metric.replace("_", " ").title()} Across All K Values\n(latent_dim={latent_dim})',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def main():
    print("=" * 80)
    print("Layer × Position Heatmap Generation")
    print("=" * 80)
    print()

    # Load data
    print("Loading all SAE metrics...")
    df = load_all_metrics()
    print(f"  Loaded {len(df)} SAE configurations\n")

    # Create output directory
    output_dir = Path('src/experiments/topk_grid_pilot/results')

    # Generate heatmaps for K=20, d=1024 (balanced config)
    print("Generating individual heatmaps (K=20, d=1024)...")

    plot_layer_position_heatmap(
        df, 'explained_variance', k_val=20, latent_dim=1024,
        title='Explained Variance by Layer and Position',
        output_path=output_dir / 'layer_position_explained_variance.png',
        cmap='RdYlGn'
    )

    plot_layer_position_heatmap(
        df, 'feature_death_rate', k_val=20, latent_dim=1024,
        title='Feature Death Rate by Layer and Position',
        output_path=output_dir / 'layer_position_feature_death.png',
        cmap='RdYlGn_r'
    )

    plot_layer_position_heatmap(
        df, 'mean_activation', k_val=20, latent_dim=1024,
        title='Mean Activation Magnitude by Layer and Position',
        output_path=output_dir / 'layer_position_mean_activation.png',
        cmap='plasma'
    )

    plot_layer_position_heatmap(
        df, 'reconstruction_loss', k_val=20, latent_dim=1024,
        title='Reconstruction Loss by Layer and Position',
        output_path=output_dir / 'layer_position_reconstruction_loss.png',
        cmap='RdYlGn_r'
    )

    # Generate K comparison plots
    print("\nGenerating K comparison plots (d=1024)...")

    plot_all_k_comparison(
        df, 'explained_variance', latent_dim=1024,
        output_path=output_dir / 'layer_position_all_k_ev.png'
    )

    plot_all_k_comparison(
        df, 'feature_death_rate', latent_dim=1024,
        output_path=output_dir / 'layer_position_all_k_death.png'
    )

    print()
    print("=" * 80)
    print("Visualization complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
