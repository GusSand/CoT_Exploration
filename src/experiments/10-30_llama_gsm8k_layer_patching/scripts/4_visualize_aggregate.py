"""
Story 6: Create aggregate analysis and visualizations

This script creates aggregate visualizations across all examples:
1. Mean KL divergence by layer with error bars
2. Heatmap showing individual examples vs layers
3. Critical layers analysis
"""

import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add experiment directory to path
exp_dir = Path(__file__).parent.parent
sys.path.insert(0, str(exp_dir))

import config


def load_results():
    """Load patching results and aggregate statistics"""
    results_path = os.path.join(config.RESULTS_DIR, "patching_results.json")
    aggregate_path = os.path.join(config.RESULTS_DIR, "aggregate_statistics.json")

    with open(results_path, 'r') as f:
        results = json.load(f)

    with open(aggregate_path, 'r') as f:
        aggregate = json.load(f)

    print(f"✓ Loaded results for {len(results)} pairs")
    print(f"✓ Loaded aggregate statistics")

    return results, aggregate


def create_aggregate_line_plot(aggregate, output_dir):
    """
    Create line plot showing mean KL divergence by layer with error bars
    """
    layers = list(range(config.NUM_LAYERS))
    means = [aggregate['layer_kl_means'][str(l)] for l in layers]
    stds = [aggregate['layer_kl_stds'][str(l)] for l in layers]

    fig, ax = plt.subplots(figsize=(config.FIG_WIDTH, config.FIG_HEIGHT))

    # Plot with error bars
    ax.plot(layers, means, marker='o', linewidth=2, markersize=6, label='Mean KL Divergence')
    ax.fill_between(layers,
                     [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     alpha=0.3, label='±1 Std Dev')

    # Highlight critical layers
    critical_layers = [cl['layer'] for cl in aggregate['critical_layers']]
    for cl in critical_layers[:3]:  # Top 3
        ax.axvline(cl, color='red', linestyle='--', alpha=0.5)
        ax.text(cl, max(means) * 0.95, f'L{cl}', ha='center', fontsize=10, color='red', fontweight='bold')

    # Styling
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('KL Divergence', fontsize=12, fontweight='bold')
    ax.set_title('Layer-wise Patching Effect (Aggregate across all pairs)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(-0.5, len(layers) - 0.5)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    output_path = os.path.join(output_dir, "aggregate_kl_by_layer.png")
    plt.savefig(output_path, dpi=config.DPI, bbox_inches='tight')
    plt.close()

    print(f"✓ Created aggregate line plot: {output_path}")
    return output_path


def create_heatmap_matrix(results, output_dir):
    """
    Create heatmap showing all examples vs all layers
    """
    # Build matrix: rows = examples, cols = layers
    num_pairs = len(results)
    num_layers = config.NUM_LAYERS

    matrix = np.zeros((num_pairs, num_layers))

    for i, result in enumerate(results):
        for lr in result['layer_results']:
            matrix[i, lr['layer']] = lr['kl_divergence']

    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(8, num_pairs * 0.15)))

    # Create heatmap
    sns.heatmap(matrix,
                cmap='YlOrRd',
                cbar_kws={'label': 'KL Divergence'},
                ax=ax,
                xticklabels=list(range(num_layers)),
                yticklabels=[f"Pair {r['pair_id']}" for r in results])

    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Example Pair', fontsize=12, fontweight='bold')
    ax.set_title('KL Divergence Heatmap: All Pairs × All Layers', fontsize=14, fontweight='bold')

    plt.tight_layout()

    output_path = os.path.join(output_dir, "heatmap_all_pairs_layers.png")
    plt.savefig(output_path, dpi=config.DPI, bbox_inches='tight')
    plt.close()

    print(f"✓ Created heatmap matrix: {output_path}")
    return output_path


def create_critical_layers_bar_chart(aggregate, output_dir):
    """
    Create bar chart showing critical layers
    """
    critical_layers = aggregate['critical_layers'][:10]  # Top 10

    layers = [cl['layer'] for cl in critical_layers]
    kls = [cl['kl'] for cl in critical_layers]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(range(len(layers)), kls, color='coral', edgecolor='black')

    # Add value labels on bars
    for i, (layer, kl) in enumerate(zip(layers, kls)):
        ax.text(i, kl + max(kls) * 0.01, f'{kl:.3f}', ha='center', va='bottom', fontsize=10)

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f'Layer {l}' for l in layers])
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean KL Divergence', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Critical Layers (by Mean KL Divergence)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = os.path.join(output_dir, "critical_layers_bar_chart.png")
    plt.savefig(output_path, dpi=config.DPI, bbox_inches='tight')
    plt.close()

    print(f"✓ Created critical layers bar chart: {output_path}")
    return output_path


def create_layer_similarity_plot(results, output_dir):
    """
    Create plot showing which layers have minimal effect (clean ≈ corrupted)
    """
    # Compute mean KL for each layer
    layer_kls = {}
    for layer_idx in range(config.NUM_LAYERS):
        kl_values = [r['layer_results'][layer_idx]['kl_divergence'] for r in results]
        layer_kls[layer_idx] = np.mean(kl_values)

    # Sort layers by KL (ascending = most similar)
    sorted_layers = sorted(layer_kls.items(), key=lambda x: x[1])

    layers, kls = zip(*sorted_layers)

    fig, ax = plt.subplots(figsize=(config.FIG_WIDTH, config.FIG_HEIGHT))

    # Color code: green for low KL (similar), red for high KL (different)
    colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(layers)))

    bars = ax.barh(range(len(layers)), kls, color=colors, edgecolor='black')

    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f'Layer {l}' for l in layers])
    ax.set_xlabel('Mean KL Divergence', fontsize=12, fontweight='bold')
    ax.set_ylabel('Layer (sorted by similarity)', fontsize=12, fontweight='bold')
    ax.set_title('Layer Similarity: Low KL = Clean ≈ Corrupted after Patching', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add annotation
    ax.text(max(kls) * 0.5, len(layers) - 1,
            'Low KL → Patching has minimal effect\nHigh KL → Patching significantly changes output',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)

    plt.tight_layout()

    output_path = os.path.join(output_dir, "layer_similarity_analysis.png")
    plt.savefig(output_path, dpi=config.DPI, bbox_inches='tight')
    plt.close()

    print(f"✓ Created layer similarity plot: {output_path}")
    return output_path


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("STORY 6: Create Aggregate Analysis and Visualizations")
    print("="*80 + "\n")

    # Load results
    results, aggregate = load_results()

    # Create output directory
    output_dir = os.path.join(config.RESULTS_DIR, "aggregate_analysis")
    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating aggregate visualizations...\n")

    # Create visualizations
    create_aggregate_line_plot(aggregate, output_dir)
    create_heatmap_matrix(results, output_dir)
    create_critical_layers_bar_chart(aggregate, output_dir)
    create_layer_similarity_plot(results, output_dir)

    print("\n" + "="*80)
    print("✓ AGGREGATE VISUALIZATIONS COMPLETE")
    print("="*80)
    print(f"\nAll visualizations saved to: {output_dir}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
