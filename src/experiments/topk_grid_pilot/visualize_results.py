"""
Generate 5 quality metric heatmaps for TopK SAE grid experiment.

Heatmaps (K × latent_dim):
1. Explained Variance
2. Feature Death Rate
3. Mean Activation Magnitude
4. Max Activation Magnitude
5. Reconstruction Loss

Usage:
    python visualize_results.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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

    # Extract metadata (all files have same metadata)
    with open(results_dir / 'grid_metrics_latent512.json', 'r') as f:
        metadata = json.load(f)['metadata']

    return data, metadata


def create_metric_grid(data, metric_name):
    """
    Create 2D grid for a specific metric.

    Args:
        data: Merged results dict {latent_dim: {k: metrics}}
        metric_name: Name of metric to extract

    Returns:
        grid: (3, 4) numpy array [latent_dims × k_values]
        latent_dims: [512, 1024, 2048]
        k_values: [5, 10, 20, 100]
    """
    latent_dims = [512, 1024, 2048]
    k_values = [5, 10, 20, 100]

    grid = np.zeros((len(latent_dims), len(k_values)))

    for i, latent_dim in enumerate(latent_dims):
        for j, k in enumerate(k_values):
            grid[i, j] = data[str(latent_dim)][str(k)][metric_name]

    return grid, latent_dims, k_values


def plot_heatmap(grid, latent_dims, k_values, metric_name, title, output_path,
                 cmap='viridis', fmt='.3f', vmin=None, vmax=None):
    """Create and save a heatmap for a metric."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create heatmap
    sns.heatmap(
        grid,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=[f'K={k}' for k in k_values],
        yticklabels=[f'{d}' for d in latent_dims],
        cbar_kws={'label': metric_name},
        vmin=vmin,
        vmax=vmax,
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Sparsity Level (K)', fontsize=12)
    ax.set_ylabel('Dictionary Size (latent_dim)', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def main():
    print("=" * 80)
    print("TopK SAE Grid Visualization")
    print("=" * 80)
    print()

    # Load and merge results
    print("Loading results...")
    data, metadata = load_and_merge_results()
    print(f"  Position: {metadata['position']}")
    print(f"  Layer: {metadata['layer']}")
    print(f"  Train samples: {metadata['n_train']}")
    print(f"  Val samples: {metadata['n_val']}")
    print()

    # Create output directory
    output_dir = Path('src/experiments/topk_grid_pilot/results')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate 5 heatmaps
    print("Generating heatmaps...")

    # 1. Explained Variance
    grid, latent_dims, k_values = create_metric_grid(data, 'explained_variance')
    plot_heatmap(
        grid, latent_dims, k_values,
        metric_name='Explained Variance',
        title='Explained Variance (Higher = Better Reconstruction)',
        output_path=output_dir / 'heatmap_explained_variance.png',
        cmap='RdYlGn',  # Red (low) to Green (high)
        fmt='.3f',
        vmin=0.0,
        vmax=1.0
    )

    # 2. Feature Death Rate
    grid, _, _ = create_metric_grid(data, 'feature_death_rate')
    plot_heatmap(
        grid, latent_dims, k_values,
        metric_name='Feature Death Rate',
        title='Feature Death Rate (Lower = Better Feature Utilization)',
        output_path=output_dir / 'heatmap_feature_death_rate.png',
        cmap='RdYlGn_r',  # Green (low) to Red (high)
        fmt='.3f',
        vmin=0.0,
        vmax=1.0
    )

    # 3. Mean Activation Magnitude
    grid, _, _ = create_metric_grid(data, 'mean_activation')
    plot_heatmap(
        grid, latent_dims, k_values,
        metric_name='Mean Activation',
        title='Mean Activation Magnitude (Feature Strength)',
        output_path=output_dir / 'heatmap_mean_activation.png',
        cmap='plasma',
        fmt='.2f'
    )

    # 4. Max Activation Magnitude
    grid, _, _ = create_metric_grid(data, 'max_activation')
    plot_heatmap(
        grid, latent_dims, k_values,
        metric_name='Max Activation',
        title='Max Activation Magnitude (Peak Feature Strength)',
        output_path=output_dir / 'heatmap_max_activation.png',
        cmap='inferno',
        fmt='.2f'
    )

    # 5. Reconstruction Loss
    grid, _, _ = create_metric_grid(data, 'reconstruction_loss')
    plot_heatmap(
        grid, latent_dims, k_values,
        metric_name='Reconstruction Loss (MSE)',
        title='Reconstruction Loss (Lower = Better)',
        output_path=output_dir / 'heatmap_reconstruction_loss.png',
        cmap='RdYlGn_r',  # Green (low) to Red (high)
        fmt='.4f'
    )

    print()
    print("=" * 80)
    print("Visualization complete!")
    print("=" * 80)

    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 80)

    for metric in ['explained_variance', 'feature_death_rate', 'mean_activation',
                   'max_activation', 'reconstruction_loss']:
        grid, _, _ = create_metric_grid(data, metric)
        print(f"{metric:25s}: min={grid.min():.4f}, max={grid.max():.4f}, mean={grid.mean():.4f}")


if __name__ == '__main__':
    main()
