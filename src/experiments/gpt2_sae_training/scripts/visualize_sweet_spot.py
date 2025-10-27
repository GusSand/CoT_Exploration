"""
Create layer×position heatmaps for GPT-2 sweet spot (d=512, K=150).

Generates:
1. Reconstruction Loss heatmap
2. Feature Death Rate heatmap
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_metrics():
    """Load metrics from all 72 SAEs."""
    metrics_path = Path("src/experiments/gpt2_sae_training/results/sweet_spot_all/sweet_spot_metrics_all.json")

    with open(metrics_path, 'r') as f:
        data = json.load(f)

    results = data['results']

    # Build matrices: 12 layers × 6 positions
    recon_loss_matrix = np.zeros((12, 6))
    death_rate_matrix = np.zeros((12, 6))

    for layer in range(12):
        for position in range(6):
            layer_str = str(layer)
            pos_str = str(position)

            metrics = results[layer_str][pos_str]

            recon_loss_matrix[layer, position] = metrics['reconstruction_loss']
            death_rate_matrix[layer, position] = metrics['feature_death_rate']

    return recon_loss_matrix, death_rate_matrix


def create_heatmap(matrix, title, cmap, vmin=None, vmax=None):
    """Create a single heatmap."""
    fig, ax = plt.subplots(figsize=(8, 10))

    sns.heatmap(
        matrix,
        annot=True,
        fmt='.3f',
        cmap=cmap,
        cbar_kws={'label': title},
        xticklabels=[f'Pos {i}' for i in range(6)],
        yticklabels=[f'L{i}' for i in range(12)],
        vmin=vmin,
        vmax=vmax,
        ax=ax
    )

    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title(f'GPT-2 Sweet Spot (d=512, K=150): {title}', fontsize=14, fontweight='bold')

    plt.tight_layout()

    return fig


def main():
    print("="*80)
    print("GENERATING SWEET SPOT VISUALIZATIONS")
    print("="*80)

    # Load metrics
    print("\n[1/3] Loading metrics from 72 SAEs...")
    recon_loss_matrix, death_rate_matrix = load_metrics()

    print(f"  Reconstruction Loss range: {recon_loss_matrix.min():.3f} - {recon_loss_matrix.max():.3f}")
    print(f"  Feature Death Rate range: {death_rate_matrix.min():.3f} - {death_rate_matrix.max():.3f}")

    # Create output directory
    output_dir = Path("src/experiments/gpt2_sae_training/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create reconstruction loss heatmap
    print("\n[2/3] Creating reconstruction loss heatmap...")
    fig1 = create_heatmap(
        recon_loss_matrix,
        title='Reconstruction Loss',
        cmap='YlOrRd',  # Yellow-Orange-Red (lower is better)
        vmin=0,
        vmax=None
    )

    recon_path = output_dir / 'gpt2_sweet_spot_reconstruction_loss.png'
    fig1.savefig(recon_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {recon_path}")
    plt.close(fig1)

    # Create feature death rate heatmap
    print("\n[3/3] Creating feature death rate heatmap...")
    fig2 = create_heatmap(
        death_rate_matrix,
        title='Feature Death Rate',
        cmap='RdYlGn_r',  # Red-Yellow-Green reversed (lower is better)
        vmin=0,
        vmax=1
    )

    death_path = output_dir / 'gpt2_sweet_spot_feature_death_rate.png'
    fig2.savefig(death_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {death_path}")
    plt.close(fig2)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"  Reconstruction Loss: {recon_path}")
    print(f"  Feature Death Rate: {death_path}")
    print("="*80)


if __name__ == '__main__':
    main()
