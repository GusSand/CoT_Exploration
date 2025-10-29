#!/usr/bin/env python3
"""
Visualize CT (Continuous Thought) Token Attention Patterns

Creates visualizations of the 6×6 attention matrix between CT tokens:
- Average attention heatmaps for critical heads
- Hub attention (attention TO CT0 from all other CT tokens)
- Layer-wise attention evolution
- Skip connection patterns

Usage:
    python visualize_ct_attention.py [--model MODEL]

Output:
    ../results/{model}/visualizations/
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
from pathlib import Path
import argparse

def load_attention_data(model_name='llama'):
    """Load extracted attention patterns and metadata."""
    results_dir = Path(__file__).parent.parent / 'results' / model_name

    # Load attention patterns
    attention_file = results_dir / 'attention_patterns_raw.npy'
    attention = np.load(attention_file)  # Shape: (n_problems, n_layers, n_heads, 6, 6)

    # Load metadata
    metadata_file = results_dir / 'attention_metadata.json'
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Load ranked heads
    ranked_file = results_dir / 'ranked_heads.csv'
    ranked_heads = pd.read_csv(ranked_file)

    return attention, metadata, ranked_heads


def plot_average_attention_heatmap(attention, layer_idx, head_idx, ax, title):
    """
    Plot average 6×6 attention heatmap for a specific head.

    Args:
        attention: Full attention array [n_problems, n_layers, n_heads, 6, 6]
        layer_idx: Layer index
        head_idx: Head index
        ax: Matplotlib axis
        title: Plot title
    """
    # Extract attention for this head across all problems
    head_attention = attention[:, layer_idx, head_idx, :, :]  # [n_problems, 6, 6]

    # Average across problems
    avg_attention = np.mean(head_attention, axis=0)  # [6, 6]

    # Plot heatmap
    sns.heatmap(avg_attention, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=['CT0', 'CT1', 'CT2', 'CT3', 'CT4', 'CT5'],
                yticklabels=['CT0', 'CT1', 'CT2', 'CT3', 'CT4', 'CT5'],
                ax=ax, vmin=0, vmax=1, cbar_kws={'label': 'Attention Weight'})
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Attending TO', fontsize=9)
    ax.set_ylabel('Attending FROM', fontsize=9)


def visualize_critical_heads(attention, ranked_heads, output_dir, n_heads=10):
    """Create 6×6 heatmaps for top N critical heads."""
    print(f"\nCreating attention heatmaps for top {n_heads} heads...")

    # Get top N heads
    top_heads = ranked_heads.head(n_heads)

    # Create figure with subplots
    n_rows = (n_heads + 2) // 3  # 3 plots per row
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    axes = axes.flatten()

    for i, (_, head) in enumerate(top_heads.iterrows()):
        layer_idx = int(head['layer'])
        head_idx = int(head['head'])
        score = head['composite_score']

        title = f"L{layer_idx}H{head_idx}\n(score: {score:.3f})"
        plot_average_attention_heatmap(attention, layer_idx, head_idx, axes[i], title)

    # Remove unused subplots
    for i in range(len(top_heads), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    output_file = output_dir / f'top_{n_heads}_heads_attention.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def visualize_hub_attention(attention, output_dir):
    """
    Visualize hub attention - how much all CT tokens attend to CT0.

    Averages across all problems and all heads to show the global pattern.
    """
    print("\nCreating hub attention visualization...")

    # Extract attention TO CT0 (column 0)
    # attention shape: [n_problems, n_layers, n_heads, 6, 6]
    # We want: [n_layers, 6] - average attention from each CT position TO CT0

    n_problems, n_layers, n_heads, _, _ = attention.shape

    # Average across problems and heads
    avg_attention = np.mean(attention, axis=(0, 2))  # [n_layers, 6, 6]

    # Extract column 0 (attention to CT0)
    hub_attention = avg_attention[:, :, 0]  # [n_layers, 6]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(hub_attention.T, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=[f'L{i}' for i in range(n_layers)],
                yticklabels=['CT0→CT0', 'CT1→CT0', 'CT2→CT0', 'CT3→CT0', 'CT4→CT0', 'CT5→CT0'],
                ax=ax, vmin=0, vmax=0.5, cbar_kws={'label': 'Avg Attention Weight'})
    ax.set_title('Hub Attention: How Much Each CT Token Attends to CT0\n(Averaged across all problems & heads)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Attention Pattern', fontsize=11)

    plt.tight_layout()
    output_file = output_dir / 'hub_attention_by_layer.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def visualize_skip_connections(attention, output_dir):
    """
    Visualize skip connections - attention to non-adjacent tokens.

    Skip connection strength for CT_i: sum of attention to CT_j where j < i-1
    """
    print("\nCreating skip connection visualization...")

    n_problems, n_layers, n_heads, _, _ = attention.shape

    # Calculate skip connection strength for each position
    # Skip for CT_i = sum(attention to CT_j where j < i-1)
    skip_strength = np.zeros((n_layers, 6))

    avg_attention = np.mean(attention, axis=(0, 2))  # [n_layers, 6, 6]

    for ct_from in range(6):
        for ct_to in range(ct_from):
            if ct_to < ct_from - 1:  # Skip connection (not adjacent)
                skip_strength[:, ct_from] += avg_attention[:, ct_from, ct_to]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(skip_strength.T, annot=True, fmt='.3f', cmap='YlGnBu',
                xticklabels=[f'L{i}' for i in range(n_layers)],
                yticklabels=[f'CT{i}' for i in range(6)],
                ax=ax, vmin=0, vmax=0.3, cbar_kws={'label': 'Skip Connection Strength'})
    ax.set_title('Skip Connection Strength by Layer\n(Attention to non-adjacent previous tokens)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('CT Token', fontsize=11)

    plt.tight_layout()
    output_file = output_dir / 'skip_connections_by_layer.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def visualize_layer_evolution(attention, ranked_heads, output_dir):
    """
    Show how attention patterns evolve across layers for the most critical head.
    """
    print("\nCreating layer evolution visualization...")

    # Get the most critical head
    top_head = ranked_heads.iloc[0]
    layer_idx = int(top_head['layer'])
    head_idx = int(top_head['head'])

    # Sample a few layers to show evolution
    layers_to_show = [0, 4, 8, 12, 15]

    fig, axes = plt.subplots(1, len(layers_to_show), figsize=(15, 3))

    for i, layer in enumerate(layers_to_show):
        plot_average_attention_heatmap(attention, layer, head_idx, axes[i],
                                      f'Layer {layer}')

    fig.suptitle(f'Attention Evolution Across Layers (Head {head_idx})',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_file = output_dir / f'layer_evolution_H{head_idx}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def visualize_attention_statistics(attention, output_dir):
    """Create summary statistics visualizations."""
    print("\nCreating attention statistics...")

    n_problems, n_layers, n_heads, _, _ = attention.shape

    # Average attention across problems and heads
    avg_attention = np.mean(attention, axis=(0, 2))  # [n_layers, 6, 6]

    # Compute mean attention for each layer
    layer_means = np.mean(avg_attention, axis=(1, 2))  # [n_layers]

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Mean attention by layer
    ax1.plot(range(n_layers), layer_means, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Layer', fontsize=11)
    ax1.set_ylabel('Mean Attention Weight (CT×CT)', fontsize=11)
    ax1.set_title('Average CT-to-CT Attention by Layer', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, n_layers, 2))

    # Plot 2: Attention variance by layer
    layer_vars = np.var(avg_attention, axis=(1, 2))  # [n_layers]
    ax2.plot(range(n_layers), layer_vars, 'o-', color='orange', linewidth=2, markersize=8)
    ax2.set_xlabel('Layer', fontsize=11)
    ax2.set_ylabel('Attention Variance (CT×CT)', fontsize=11)
    ax2.set_title('CT-to-CT Attention Variance by Layer', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(0, n_layers, 2))

    plt.tight_layout()
    output_file = output_dir / 'attention_statistics.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize CT attention patterns')
    parser.add_argument('--model', type=str, default='llama', choices=['llama', 'gpt2'])
    args = parser.parse_args()

    print("="*80)
    print("CT ATTENTION PATTERN VISUALIZATION")
    print("="*80)

    # Load data
    print(f"\nLoading attention data for {args.model}...")
    attention, metadata, ranked_heads = load_attention_data(args.model)

    print(f"✓ Loaded attention patterns:")
    print(f"  Shape: {attention.shape}")
    print(f"  {metadata['n_problems']} problems")
    print(f"  {metadata['n_layers']} layers")
    print(f"  {metadata['n_heads']} heads")

    # Create output directory
    output_dir = Path(__file__).parent.parent / 'results' / args.model / 'visualizations'
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create visualizations
    visualize_critical_heads(attention, ranked_heads, output_dir, n_heads=10)
    visualize_hub_attention(attention, output_dir)
    visualize_skip_connections(attention, output_dir)
    visualize_layer_evolution(attention, ranked_heads, output_dir)
    visualize_attention_statistics(attention, output_dir)

    print("\n" + "="*80)
    print("✓ VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")


if __name__ == '__main__':
    main()
