#!/usr/bin/env python3
"""
Attention Heatmap Visualizer - Story 1.4

Create heatmap visualizations of attention patterns.

Usage:
    python 4_visualize_heatmaps.py [--model MODEL]

Output:
    ../figures/{model}/1_top_heads_attention.png
    ../figures/{model}/2_attention_by_layer.png
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path


def visualize_heatmaps(model: str = 'llama') -> None:
    """
    Create heatmap visualizations of attention patterns.

    Args:
        model: Model name ('llama' or 'gpt2')
    """
    print("=" * 80)
    print("ATTENTION HEATMAP VISUALIZER - Story 1.4")
    print("=" * 80)

    # Load aggregated attention
    results_dir = Path(__file__).parent.parent / 'results' / model
    figures_dir = Path(__file__).parent.parent / 'figures' / model

    avg_path = results_dir / 'attention_patterns_avg.npy'
    stats_path = results_dir / 'attention_stats.json'

    print(f"\nLoading attention patterns from {avg_path}...")
    attention_avg = np.load(avg_path).astype(np.float32)
    print(f"✓ Loaded: {attention_avg.shape}")

    with open(stats_path, 'r') as f:
        stats = json.load(f)

    n_layers, n_heads, _, _ = attention_avg.shape

    # Figure 1: Top 20 heads heatmaps (4×5 grid)
    print("\nCreating Figure 1: Top 20 heads attention heatmaps...")
    top_20 = stats['top_20_heads']

    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle(f'{model.upper()} - Top 20 Attention Heads (by max attention)',
                 fontsize=16, fontweight='bold')

    for idx, head_info in enumerate(top_20):
        row = idx // 5
        col = idx % 5
        ax = axes[row, col]

        layer = head_info['layer']
        head = head_info['head']
        max_attn = head_info['max_attention']

        # Get attention matrix for this head
        attn = attention_avg[layer, head]  # [6, 6]

        # Create heatmap
        sns.heatmap(attn, annot=True, fmt='.2f', cmap='YlOrRd',
                    vmin=0, vmax=1.0, cbar=True,
                    xticklabels=range(6), yticklabels=range(6),
                    ax=ax, square=True)

        ax.set_title(f'L{layer}H{head} (max={max_attn:.2f})', fontsize=10, fontweight='bold')
        ax.set_xlabel('Source Position', fontsize=9)
        ax.set_ylabel('Dest Position', fontsize=9)

    plt.tight_layout()

    fig1_path = figures_dir / '1_top_heads_attention.png'
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {fig1_path}")
    print(f"  Size: {fig1_path.stat().st_size / 1024:.1f} KB")

    # Figure 2: Attention aggregated by layer
    print("\nCreating Figure 2: Attention by layer (averaged across heads)...")

    # Choose representative layers to visualize
    if n_layers >= 16:
        # LLaMA: show layers 0, 4, 8, 12, 15
        layers_to_show = [0, 4, 8, 12, 15]
        layer_labels = ['L0 (early)', 'L4', 'L8 (middle)', 'L12', 'L15 (late)']
    else:
        # GPT-2: show layers 0, 3, 6, 9, 11
        layers_to_show = [0, 3, 6, 9, 11]
        layer_labels = ['L0 (early)', 'L3', 'L6 (middle)', 'L9', 'L11 (late)']

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f'{model.upper()} - Attention by Layer (averaged across heads)',
                 fontsize=16, fontweight='bold')

    for idx, (layer, label) in enumerate(zip(layers_to_show, layer_labels)):
        ax = axes[idx]

        # Average across all heads for this layer
        attn_layer = attention_avg[layer].mean(axis=0)  # [6, 6]

        # Create heatmap
        sns.heatmap(attn_layer, annot=True, fmt='.3f', cmap='YlOrRd',
                    vmin=0, vmax=0.3, cbar=True,
                    xticklabels=range(6), yticklabels=range(6),
                    ax=ax, square=True)

        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Source Position', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Dest Position', fontsize=10)

    plt.tight_layout()

    fig2_path = figures_dir / '2_attention_by_layer.png'
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {fig2_path}")
    print(f"  Size: {fig2_path.stat().st_size / 1024:.1f} KB")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nVisualizations created:")
    print(f"  1. Top 20 heads: {fig1_path}")
    print(f"  2. Attention by layer: {fig2_path}")

    print(f"\nKey observations from top heads:")
    for i in range(min(3, len(top_20))):
        head_info = top_20[i]
        layer = head_info['layer']
        head = head_info['head']
        max_attn = head_info['max_attention']

        attn = attention_avg[layer, head]
        max_pos = np.unravel_index(attn.argmax(), attn.shape)

        print(f"  {i+1}. L{layer}H{head}:")
        print(f"     Max attention: {max_attn:.3f} at position [{max_pos[0]}→{max_pos[1]}]")

    print("\n" + "=" * 80)
    print("STORY 1.4 COMPLETE ✓")
    print("=" * 80)
    print(f"\nCreated {len(layers_to_show) + 1} visualizations")
    print("\nNext step: Run Story 1.5 to analyze hub positions and flow patterns")
    print("  python 5_analyze_hubs_and_flow.py")


def main():
    parser = argparse.ArgumentParser(description='Visualize attention heatmaps')
    parser.add_argument('--model', type=str, default='llama',
                        choices=['llama', 'gpt2'],
                        help='Model to visualize (llama or gpt2)')
    args = parser.parse_args()

    visualize_heatmaps(model=args.model)


if __name__ == '__main__':
    main()
