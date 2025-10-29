#!/usr/bin/env python3
"""
Visualize CT (Continuous Thought) Token Attention Patterns for CommonsenseQA

Creates visualizations of the 6×6 attention matrix between CT tokens:
- Layer-wise attention heatmaps
- Hub attention (attention TO CT0 from all other CT tokens)
- Layer-wise attention evolution
- Skip connection patterns
- Attention statistics

Usage:
    python 3_visualize_attention_patterns.py [--n_samples N]

Output:
    ../results/visualizations/
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import argparse


def load_attention_data():
    """Load extracted attention patterns and metadata."""
    results_dir = Path(__file__).parent.parent / 'results'

    # Load averaged attention patterns (6x6 per layer)
    attention_file = results_dir / 'commonsense_attention_patterns_avg.npy'
    attention_avg = np.load(attention_file)  # Shape: (n_layers, 6, 6)

    # Load raw attention patterns (n_examples, n_layers, 6, 6)
    attention_raw_file = results_dir / 'commonsense_attention_patterns_raw.npy'
    attention_raw = np.load(attention_raw_file)  # Shape: (n_examples, n_layers, 6, 6)

    # Load statistics
    stats_file = results_dir / 'commonsense_attention_stats.json'
    with open(stats_file, 'r') as f:
        stats = json.load(f)

    # Load metadata
    metadata_file = results_dir / 'commonsense_attention_metadata.json'
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    return attention_avg, attention_raw, stats, metadata


def plot_attention_heatmap(attention_matrix, ax, title, vmax=0.8):
    """
    Plot a 6×6 attention heatmap.

    Args:
        attention_matrix: 6×6 numpy array
        ax: Matplotlib axis
        title: Plot title
        vmax: Maximum value for color scale
    """
    sns.heatmap(attention_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=['CT0', 'CT1', 'CT2', 'CT3', 'CT4', 'CT5'],
                yticklabels=['CT0', 'CT1', 'CT2', 'CT3', 'CT4', 'CT5'],
                ax=ax, vmin=0, vmax=vmax, cbar_kws={'label': 'Attention Weight'})
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Attending TO', fontsize=9)
    ax.set_ylabel('Attending FROM', fontsize=9)


def visualize_layer_attention_grids(attention_avg, output_dir):
    """Create 6×6 heatmaps for selected layers."""
    print("\nCreating attention heatmaps for key layers...")

    n_layers = attention_avg.shape[0]

    # Select representative layers (early, middle, late)
    layers_to_show = [0, 4, 8, 12, 15]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, layer_idx in enumerate(layers_to_show):
        plot_attention_heatmap(attention_avg[layer_idx], axes[i],
                              f'Layer {layer_idx}', vmax=0.8)

    # Remove unused subplot
    fig.delaxes(axes[-1])

    plt.suptitle('CT-to-CT Attention Patterns Across Layers (CommonsenseQA)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_file = output_dir / 'layer_attention_grids.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def visualize_hub_attention(attention_avg, output_dir):
    """
    Visualize hub attention - how much all CT tokens attend to CT0.

    Shows the global pattern of CT0 as an attention hub.
    """
    print("\nCreating hub attention visualization...")

    n_layers = attention_avg.shape[0]

    # Extract attention TO CT0 (column 0)
    # attention_avg shape: [n_layers, 6, 6]
    # We want: [n_layers, 6] - attention from each CT position TO CT0
    hub_attention = attention_avg[:, :, 0]  # [n_layers, 6]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(hub_attention.T, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=[f'L{i}' for i in range(n_layers)],
                yticklabels=['CT0→CT0', 'CT1→CT0', 'CT2→CT0', 'CT3→CT0', 'CT4→CT0', 'CT5→CT0'],
                ax=ax, vmin=0, vmax=0.8, cbar_kws={'label': 'Avg Attention Weight'})
    ax.set_title('Hub Attention: How Much Each CT Token Attends to CT0\n(CommonsenseQA - Averaged across all examples & heads)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Attention Pattern', fontsize=11)

    plt.tight_layout()
    output_file = output_dir / 'hub_attention_by_layer.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def visualize_ct0_hub_strength(attention_avg, output_dir):
    """
    Visualize CT0 hub strength - sum of attention from CT1-CT5 to CT0 by layer.

    This measures whether CT0 is indeed an attention hub (as claimed in the report).
    """
    print("\nCreating CT0 hub strength visualization...")

    n_layers = attention_avg.shape[0]

    # Extract attention TO CT0 from CT1-CT5
    hub_attention = attention_avg[:, 1:, 0]  # [n_layers, 5] (CT1-CT5 → CT0)

    # Sum across CT1-CT5 to get total hub strength per layer
    hub_strength = np.sum(hub_attention, axis=1)  # [n_layers]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(n_layers), hub_strength, 'o-', linewidth=2, markersize=8, color='#d62728')
    ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Reference (20%)')
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Total Attention from CT1-CT5 to CT0', fontsize=11)
    ax.set_title('CT0 Hub Strength Across Layers\n(Sum of attention from CT1-CT5 to CT0)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, n_layers, 2))
    ax.legend()

    plt.tight_layout()
    output_file = output_dir / 'ct0_hub_strength.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

    # Print statistics
    print(f"\nCT0 Hub Statistics:")
    print(f"  Mean hub strength: {np.mean(hub_strength):.4f}")
    print(f"  Max hub strength: {np.max(hub_strength):.4f} (Layer {np.argmax(hub_strength)})")
    print(f"  Min hub strength: {np.min(hub_strength):.4f} (Layer {np.argmin(hub_strength)})")
    print(f"  Early layers (L0-L5): {np.mean(hub_strength[:6]):.4f}")
    print(f"  Middle layers (L6-L10): {np.mean(hub_strength[6:11]):.4f}")
    print(f"  Late layers (L11-L15): {np.mean(hub_strength[11:]):.4f}")


def visualize_skip_connections(attention_avg, output_dir):
    """
    Visualize skip connections - attention to non-adjacent tokens.

    Skip connection strength for CT_i: sum of attention to CT_j where j < i-1
    """
    print("\nCreating skip connection visualization...")

    n_layers = attention_avg.shape[0]

    # Calculate skip connection strength for each position
    skip_strength = np.zeros((n_layers, 6))

    for ct_from in range(6):
        for ct_to in range(ct_from):
            if ct_to < ct_from - 1:  # Skip connection (not adjacent)
                skip_strength[:, ct_from] += attention_avg[:, ct_from, ct_to]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(skip_strength.T, annot=True, fmt='.3f', cmap='YlGnBu',
                xticklabels=[f'L{i}' for i in range(n_layers)],
                yticklabels=[f'CT{i}' for i in range(6)],
                ax=ax, vmin=0, vmax=0.3, cbar_kws={'label': 'Skip Connection Strength'})
    ax.set_title('Skip Connection Strength by Layer (CommonsenseQA)\n(Attention to non-adjacent previous tokens)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('CT Token', fontsize=11)

    plt.tight_layout()
    output_file = output_dir / 'skip_connections_by_layer.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def visualize_layer_evolution(attention_avg, output_dir):
    """
    Show how attention patterns evolve across layers.
    """
    print("\nCreating layer evolution visualization...")

    # Sample evenly across layers
    layers_to_show = [0, 3, 7, 11, 15]

    fig, axes = plt.subplots(1, len(layers_to_show), figsize=(20, 4))

    for i, layer in enumerate(layers_to_show):
        plot_attention_heatmap(attention_avg[layer], axes[i],
                              f'Layer {layer}', vmax=0.8)

    fig.suptitle('Attention Evolution Across Layers (CommonsenseQA)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_file = output_dir / 'layer_evolution.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def visualize_attention_statistics(attention_avg, attention_raw, output_dir):
    """Create summary statistics visualizations."""
    print("\nCreating attention statistics...")

    n_layers = attention_avg.shape[0]

    # Compute mean attention for each layer (excluding self-attention diagonal)
    layer_means = []
    for layer_idx in range(n_layers):
        # Get off-diagonal elements only
        mask = ~np.eye(6, dtype=bool)
        layer_means.append(attention_avg[layer_idx][mask].mean())
    layer_means = np.array(layer_means)

    # Compute variance across examples for each layer
    layer_vars = np.var(attention_raw, axis=0).mean(axis=(1, 2))  # [n_layers]

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Mean attention by layer (off-diagonal only)
    ax1.plot(range(n_layers), layer_means, 'o-', linewidth=2, markersize=8, color='#1f77b4')
    ax1.set_xlabel('Layer', fontsize=11)
    ax1.set_ylabel('Mean Attention Weight (CT×CT, off-diag)', fontsize=11)
    ax1.set_title('Average CT-to-CT Attention by Layer (CommonsenseQA)',
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, n_layers, 2))

    # Plot 2: Attention variance by layer
    ax2.plot(range(n_layers), layer_vars, 'o-', color='orange', linewidth=2, markersize=8)
    ax2.set_xlabel('Layer', fontsize=11)
    ax2.set_ylabel('Attention Variance (CT×CT)', fontsize=11)
    ax2.set_title('CT-to-CT Attention Variance by Layer (CommonsenseQA)',
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(0, n_layers, 2))

    plt.tight_layout()
    output_file = output_dir / 'attention_statistics.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def visualize_ct0_importance_comparison(attention_avg, output_dir):
    """
    Compare CT0 attention dominance across layers with token importance.

    This visualization helps validate whether CT0's attention hub role
    corresponds to its high ablation impact (13%).
    """
    print("\nCreating CT0 importance comparison...")

    n_layers = attention_avg.shape[0]

    # CT0 as source (how much CT0 attends to others)
    ct0_outgoing = attention_avg[:, 0, :].sum(axis=1)  # [n_layers]

    # CT0 as target (how much others attend to CT0)
    ct0_incoming = attention_avg[:, :, 0].sum(axis=1)  # [n_layers]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(n_layers)
    ax.plot(x, ct0_outgoing, 'o-', linewidth=2, markersize=8,
           label='CT0 Outgoing (CT0→Others)', color='#2ca02c')
    ax.plot(x, ct0_incoming, 's-', linewidth=2, markersize=8,
           label='CT0 Incoming (Others→CT0)', color='#d62728')

    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Total Attention Weight', fontsize=11)
    ax.set_title('CT0 Attention Flow: Hub Analysis\n(CommonsenseQA)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, n_layers, 2))

    plt.tight_layout()
    output_file = output_dir / 'ct0_attention_flow.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

    # Print statistics
    print(f"\nCT0 Attention Flow Statistics:")
    print(f"  Mean outgoing (CT0→Others): {np.mean(ct0_outgoing):.4f}")
    print(f"  Mean incoming (Others→CT0): {np.mean(ct0_incoming):.4f}")
    print(f"  Hub ratio (incoming/outgoing): {np.mean(ct0_incoming)/np.mean(ct0_outgoing):.2f}x")


def main():
    parser = argparse.ArgumentParser(description='Visualize CommonsenseQA CT attention patterns')
    parser.add_argument('--n_samples', type=int, default=100,
                       help='Number of samples (for documentation)')
    args = parser.parse_args()

    print("="*80)
    print("COMMONSENSE QA - CT ATTENTION PATTERN VISUALIZATION")
    print("="*80)

    # Load data
    print(f"\nLoading attention data...")
    attention_avg, attention_raw, stats, metadata = load_attention_data()

    print(f"✓ Loaded attention patterns:")
    print(f"  Averaged shape: {attention_avg.shape}")
    print(f"  Raw shape: {attention_raw.shape}")
    print(f"  {metadata['n_problems']} examples")
    print(f"  {metadata['n_layers']} layers")

    # Create output directory
    output_dir = Path(__file__).parent.parent / 'visualizations'
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create visualizations
    visualize_layer_attention_grids(attention_avg, output_dir)
    visualize_hub_attention(attention_avg, output_dir)
    visualize_ct0_hub_strength(attention_avg, output_dir)
    visualize_ct0_importance_comparison(attention_avg, output_dir)
    visualize_skip_connections(attention_avg, output_dir)
    visualize_layer_evolution(attention_avg, output_dir)
    visualize_attention_statistics(attention_avg, attention_raw, output_dir)

    print("\n" + "="*80)
    print("✓ VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated visualizations:")
    print("  1. layer_attention_grids.png - 6×6 heatmaps for key layers")
    print("  2. hub_attention_by_layer.png - CT0 hub attention across layers")
    print("  3. ct0_hub_strength.png - Total attention to CT0 from CT1-CT5")
    print("  4. ct0_attention_flow.png - CT0 incoming vs outgoing attention")
    print("  5. skip_connections_by_layer.png - Non-adjacent token attention")
    print("  6. layer_evolution.png - Attention pattern evolution")
    print("  7. attention_statistics.png - Mean & variance statistics")


if __name__ == '__main__':
    main()
