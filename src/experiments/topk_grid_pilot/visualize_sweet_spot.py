"""
Generate layer×position heatmaps specifically for K=100, d=512 (sweet spot).

This shows the quality patterns for our optimal configuration across
all 16 layers × 6 positions.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics_for_config(k=100, latent_dim=512):
    """Load metrics for specific K and latent_dim across all layers/positions."""
    results_dir = Path('src/experiments/topk_grid_pilot/results')

    # Initialize storage
    layers = list(range(16))
    positions = list(range(6))

    metrics = {
        'explained_variance': np.zeros((len(layers), len(positions))),
        'feature_death_rate': np.zeros((len(layers), len(positions))),
        'mean_activation': np.zeros((len(layers), len(positions))),
        'reconstruction_loss': np.zeros((len(layers), len(positions))),
    }

    # Load data for each (layer, position) pair
    for layer in layers:
        for position in positions:
            json_path = results_dir / 'data' / f'grid_metrics_pos{position}_layer{layer}_latent{latent_dim}.json'

            if not json_path.exists():
                print(f"Warning: Missing {json_path}")
                continue

            with open(json_path, 'r') as f:
                data = json.load(f)

            # Extract metrics for specific K value
            # JSON structure: results -> latent_dim -> k -> metrics
            if str(latent_dim) in data['results'] and str(k) in data['results'][str(latent_dim)]:
                config_metrics = data['results'][str(latent_dim)][str(k)]
                metrics['explained_variance'][layer, position] = config_metrics['explained_variance']
                metrics['feature_death_rate'][layer, position] = config_metrics['feature_death_rate']
                metrics['mean_activation'][layer, position] = config_metrics['mean_activation']
                metrics['reconstruction_loss'][layer, position] = config_metrics['reconstruction_loss']

    return metrics, layers, positions


def plot_heatmap(data, metric_name, layers, positions, output_path, k, latent_dim, cmap='RdYlGn'):
    """Generate single heatmap for one metric."""
    fig, ax = plt.subplots(figsize=(10, 12))

    # Create heatmap
    im = ax.imshow(data, cmap=cmap, aspect='auto', interpolation='nearest')

    # Set ticks
    ax.set_xticks(np.arange(len(positions)))
    ax.set_yticks(np.arange(len(layers)))
    ax.set_xticklabels(positions)
    ax.set_yticklabels(layers)

    # Labels
    ax.set_xlabel('Position', fontsize=14, fontweight='bold')
    ax.set_ylabel('Layer', fontsize=14, fontweight='bold')
    ax.set_title(f'{metric_name}\n(K={k}, d={latent_dim})',
                 fontsize=16, fontweight='bold', pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric_name, rotation=270, labelpad=20, fontsize=12)

    # Add text annotations
    for i in range(len(layers)):
        for j in range(len(positions)):
            value = data[i, j]
            # Format based on metric
            if 'loss' in metric_name.lower():
                text = f'{value:.3f}'
            elif 'rate' in metric_name.lower() or 'variance' in metric_name.lower():
                text = f'{value:.2f}'
            else:
                text = f'{value:.2f}'

            ax.text(j, i, text, ha="center", va="center",
                   color="black" if 0.3 < data[i, j] < 0.7 else "white",
                   fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def main():
    k = 100
    latent_dim = 512

    print("=" * 80)
    print(f"Generating Sweet Spot Heatmaps (K={k}, d={latent_dim})")
    print("=" * 80)
    print()

    # Load metrics
    print("Loading metrics...")
    metrics, layers, positions = load_metrics_for_config(k=k, latent_dim=latent_dim)
    print(f"  Loaded data for {len(layers)} layers × {len(positions)} positions")
    print()

    # Generate heatmaps
    print("Generating heatmaps...")
    results_dir = Path('src/experiments/topk_grid_pilot/results')
    viz_dir = results_dir / 'viz'

    # Explained Variance
    plot_heatmap(
        metrics['explained_variance'],
        'Explained Variance',
        layers, positions,
        viz_dir / f'sweetspot_k{k}_d{latent_dim}_explained_variance.png',
        k, latent_dim
    )

    # Feature Death Rate (reversed colormap: 0 death = green, high death = red)
    plot_heatmap(
        metrics['feature_death_rate'],
        'Feature Death Rate',
        layers, positions,
        viz_dir / f'sweetspot_k{k}_d{latent_dim}_feature_death.png',
        k, latent_dim,
        cmap='RdYlGn_r'  # Reversed: green = good (low death), red = bad (high death)
    )

    # Mean Activation
    plot_heatmap(
        metrics['mean_activation'],
        'Mean Activation Magnitude',
        layers, positions,
        viz_dir / f'sweetspot_k{k}_d{latent_dim}_mean_activation.png',
        k, latent_dim
    )

    # Reconstruction Loss (reversed colormap: low MSE = green, high MSE = red)
    plot_heatmap(
        metrics['reconstruction_loss'],
        'Reconstruction Loss (MSE)',
        layers, positions,
        viz_dir / f'sweetspot_k{k}_d{latent_dim}_reconstruction_loss.png',
        k, latent_dim,
        cmap='RdYlGn_r'  # Reversed: green = good (low loss), red = bad (high loss)
    )

    print()
    print("=" * 80)
    print("Sweet spot visualization complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
