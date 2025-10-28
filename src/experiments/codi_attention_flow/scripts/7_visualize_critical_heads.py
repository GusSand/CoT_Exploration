#!/usr/bin/env python3
"""
Critical Heads Visualizer - Story 2.5

Visualize attention patterns of top critical heads identified in Story 2.4.

Usage:
    python 7_visualize_critical_heads.py [--model MODEL] [--top_k K]

Output:
    ../figures/{model}/4_critical_heads_attention.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path


def visualize_critical_heads(model: str = 'llama', top_k: int = 10) -> None:
    """
    Visualize attention patterns of top critical heads.

    Args:
        model: Model name ('llama' or 'gpt2')
        top_k: Number of top heads to visualize
    """
    print("=" * 80)
    print(f"CRITICAL HEADS VISUALIZER - Story 2.5")
    print("=" * 80)

    # Load data
    results_dir = Path(__file__).parent.parent / 'results' / model
    figures_dir = Path(__file__).parent.parent / 'figures' / model
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading data for {model}...")

    # Load attention patterns
    attention_avg = np.load(results_dir / 'attention_patterns_avg.npy').astype(np.float32)
    print(f"✓ Loaded attention: {attention_avg.shape}")

    # Load ranked heads
    df = pd.read_csv(results_dir / 'ranked_heads.csv')
    print(f"✓ Loaded rankings: {len(df)} heads")

    # Get top K heads
    top_heads = df.head(top_k)
    print(f"\nVisualizing top {top_k} heads by composite score...")

    # Create visualization
    n_cols = 5
    n_rows = (top_k + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    fig.suptitle(f'{model.upper()} - Top {top_k} Critical Heads',
                 fontsize=16, fontweight='bold', y=0.995)

    axes = axes.flatten() if top_k > 1 else [axes]

    for idx, (_, row) in enumerate(top_heads.iterrows()):
        layer = int(row['layer'])
        head = int(row['head'])

        # Get attention matrix for this head
        attn = attention_avg[layer, head]  # [6, 6]

        # Create heatmap
        ax = axes[idx]
        sns.heatmap(
            attn,
            ax=ax,
            cmap='YlOrRd',
            vmin=0,
            vmax=1.0,
            annot=True,
            fmt='.2f',
            cbar=True,
            square=True,
            xticklabels=[f'CT{i}' for i in range(6)],
            yticklabels=[f'CT{i}' for i in range(6)],
            cbar_kws={'label': 'Attention'}
        )

        # Title with metrics
        title = f"L{layer}H{head} - {row['functional_type']}\n"
        title += f"Composite={row['composite_score']:.3f} "
        title += f"(F:{row['flow_score']:.2f}, H:{row['hub_score']:.2f}, S:{row['skip_score']:.2f})"
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Source Position', fontsize=9)
        ax.set_ylabel('Destination Position', fontsize=9)

    # Hide unused subplots
    for idx in range(top_k, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    # Save
    output_path = figures_dir / '4_critical_heads_attention.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()

    # Create summary comparison figure
    print("\nCreating summary comparison...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{model.upper()} - Top 3 Critical Heads Comparison',
                 fontsize=14, fontweight='bold')

    for idx in range(min(3, top_k)):
        row = top_heads.iloc[idx]
        layer = int(row['layer'])
        head = int(row['head'])
        attn = attention_avg[layer, head]

        ax = axes[idx]
        sns.heatmap(
            attn,
            ax=ax,
            cmap='YlOrRd',
            vmin=0,
            vmax=1.0,
            annot=True,
            fmt='.2f',
            cbar=True,
            square=True,
            xticklabels=[f'CT{i}' for i in range(6)],
            yticklabels=[f'CT{i}' for i in range(6)],
            cbar_kws={'label': 'Attention'}
        )

        title = f"#{idx+1}: L{layer}H{head}\n{row['functional_type']}"
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Source Position', fontsize=10)
        ax.set_ylabel('Destination Position', fontsize=10)

    plt.tight_layout()

    output_path = figures_dir / '5_top3_critical_heads.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

    # Create metric distribution plot
    print("\nCreating metric distribution...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model.upper()} - Critical Head Metrics Distribution',
                 fontsize=14, fontweight='bold')

    # Plot 1: Hub score distribution
    ax = axes[0, 0]
    top_20 = df.head(20)
    colors = ['#e74c3c' if t == 'Hub Aggregator' else '#3498db'
              for t in top_20['functional_type']]

    ax.barh(range(len(top_20)), top_20['hub_score'], color=colors)
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels([f"L{row['layer']}H{row['head']}"
                        for _, row in top_20.iterrows()], fontsize=8)
    ax.set_xlabel('Hub Score (variance)', fontsize=10)
    ax.set_title('Top 20 Heads - Hub Scores', fontsize=11, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Plot 2: Skip score distribution
    ax = axes[0, 1]
    colors = ['#2ecc71' if t == 'Skip Connection' else '#95a5a6'
              for t in top_20['functional_type']]

    ax.barh(range(len(top_20)), top_20['skip_score'], color=colors)
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels([f"L{row['layer']}H{row['head']}"
                        for _, row in top_20.iterrows()], fontsize=8)
    ax.set_xlabel('Skip Score (avg attn 5→0,1,2)', fontsize=10)
    ax.set_title('Top 20 Heads - Skip Scores', fontsize=11, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Plot 3: Layer distribution
    ax = axes[1, 0]
    layer_counts = top_20['layer_type'].value_counts()
    colors_layer = {'early': '#3498db', 'middle': '#f39c12', 'late': '#e74c3c'}
    ax.bar(range(len(layer_counts)), layer_counts.values,
           color=[colors_layer[t] for t in layer_counts.index])
    ax.set_xticks(range(len(layer_counts)))
    ax.set_xticklabels([t.capitalize() for t in layer_counts.index])
    ax.set_ylabel('Count (Top 20 heads)', fontsize=10)
    ax.set_title('Layer Distribution', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add counts on bars
    for i, count in enumerate(layer_counts.values):
        ax.text(i, count + 0.3, str(count), ha='center', fontsize=10, fontweight='bold')

    # Plot 4: Functional type distribution
    ax = axes[1, 1]
    type_counts = top_20['functional_type'].value_counts()
    colors_func = {'Hub Aggregator': '#e74c3c', 'Skip Connection': '#2ecc71',
                   'Forward Flow': '#3498db', 'Multi-Purpose': '#9b59b6'}
    ax.bar(range(len(type_counts)), type_counts.values,
           color=[colors_func.get(t, '#95a5a6') for t in type_counts.index])
    ax.set_xticks(range(len(type_counts)))
    ax.set_xticklabels(type_counts.index, rotation=45, ha='right')
    ax.set_ylabel('Count (Top 20 heads)', fontsize=10)
    ax.set_title('Functional Type Distribution', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add counts on bars
    for i, count in enumerate(type_counts.values):
        ax.text(i, count + 0.3, str(count), ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    output_path = figures_dir / '6_metric_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

    print("\n" + "=" * 80)
    print("STORY 2.5 COMPLETE ✓")
    print("=" * 80)
    print(f"\nGenerated 3 visualizations:")
    print(f"  1. Top {top_k} critical heads: 4_critical_heads_attention.png")
    print(f"  2. Top 3 comparison: 5_top3_critical_heads.png")
    print(f"  3. Metric distributions: 6_metric_distributions.png")
    print(f"\nTop critical head: L{top_heads.iloc[0]['layer']}H{top_heads.iloc[0]['head']} "
          f"({top_heads.iloc[0]['functional_type']}, score={top_heads.iloc[0]['composite_score']:.3f})")

    if top_k >= len(df):
        print("\nNext step: Run Story 2.6 to compare with GPT-2")
        print("  PYTHONPATH=/home/paperspace/dev/CoT_Exploration/codi:$PYTHONPATH \\")
        print("    python scripts/2_extract_attention_6x6.py --model gpt2")


def main():
    parser = argparse.ArgumentParser(description='Visualize critical heads')
    parser.add_argument('--model', type=str, default='llama',
                        choices=['llama', 'gpt2'],
                        help='Model to visualize (llama or gpt2)')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of top heads to visualize (default: 10)')
    args = parser.parse_args()

    visualize_critical_heads(model=args.model, top_k=args.top_k)


if __name__ == '__main__':
    main()
