"""
Visualize multi-layer probe results with heatmap and line plots.

Creates:
1. Heatmap: Layer (y-axis) × Position (x-axis) with accuracy color-coded
2. Line plot: Accuracy by layer for each position
3. Grouped line plot: CT positions vs Question/Answer positions

Author: Claude Code
Date: 2025-10-30
Experiment: Pre-compression deception signal analysis
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def load_probe_results(results_path):
    """Load probe training results."""
    print(f"Loading results from: {results_path}")
    with open(results_path, 'r') as f:
        results = json.load(f)

    print(f"  Model: {results['metadata']['model']}")
    print(f"  Epoch: {results['metadata']['epoch']}")
    print(f"  Total probes: {results['metadata']['total_probes']}")
    print()

    return results


def create_heatmap(results, output_path):
    """
    Create heatmap of probe accuracy by layer and position.

    Args:
        results: Probe results dictionary
        output_path: Where to save the figure
    """
    print("Creating heatmap...")

    layers = results['metadata']['layers']
    positions = results['metadata']['positions']

    # Build accuracy matrix
    accuracy_matrix = np.zeros((len(layers), len(positions)))

    for i, layer_idx in enumerate(layers):
        layer_name = f'layer_{layer_idx}'
        for j, position_name in enumerate(positions):
            accuracy_matrix[i, j] = results[layer_name][position_name]['test_accuracy']

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap
    sns.heatmap(
        accuracy_matrix,
        annot=True,
        fmt='.1%',
        cmap='RdYlGn',
        center=0.50,
        vmin=0.45,
        vmax=0.75,
        xticklabels=positions,
        yticklabels=[f'Layer {l}' for l in layers],
        cbar_kws={'label': 'Probe Accuracy'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )

    # Add vertical lines to separate position groups
    # Separate question | ct0-ct5 | answer
    ax.axvline(x=1, color='black', linewidth=2, linestyle='--', alpha=0.5)
    ax.axvline(x=7, color='black', linewidth=2, linestyle='--', alpha=0.5)

    # Title and labels
    ax.set_title(
        f'Deception Detection Accuracy by Layer and Position (LLaMA-1B, {results["metadata"]["epoch"]})',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel('Token Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model Layer', fontsize=12, fontweight='bold')

    # Rotate x labels for readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved heatmap to: {output_path}")
    plt.close()


def create_line_plot(results, output_path):
    """
    Create line plot showing accuracy by layer for each position.

    Args:
        results: Probe results dictionary
        output_path: Where to save the figure
    """
    print("Creating line plot...")

    layers = results['metadata']['layers']
    positions = results['metadata']['positions']

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot line for each position
    colors = plt.cm.tab10(np.linspace(0, 1, len(positions)))

    for pos_idx, position_name in enumerate(positions):
        accuracies = []
        for layer_idx in layers:
            layer_name = f'layer_{layer_idx}'
            acc = results[layer_name][position_name]['test_accuracy']
            accuracies.append(acc)

        # Different line styles for different position types
        if position_name == 'question_last':
            linestyle = '-'
            linewidth = 2
            marker = 'o'
        elif position_name == 'answer_first':
            linestyle = '-'
            linewidth = 2
            marker = 's'
        else:  # CT positions
            linestyle = '--'
            linewidth = 1.5
            marker = 'x'

        ax.plot(
            layers,
            accuracies,
            label=position_name,
            color=colors[pos_idx],
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            markersize=6
        )

    # Add horizontal reference lines
    ax.axhline(y=0.50, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='Random (50%)')
    ax.axhline(y=0.55, color='orange', linestyle=':', linewidth=1, alpha=0.7, label='Signal threshold (55%)')

    # Labels and title
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probe Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Deception Signal Across Layers (LLaMA-1B, {results["metadata"]["epoch"]})',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Legend
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        fontsize=9,
        framealpha=0.9
    )

    # Set y-axis limits
    ax.set_ylim(0.45, 0.75)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved line plot to: {output_path}")
    plt.close()


def create_grouped_line_plot(results, output_path):
    """
    Create line plot with grouped positions: CT vs Question/Answer.

    Args:
        results: Probe results dictionary
        output_path: Where to save the figure
    """
    print("Creating grouped line plot...")

    layers = results['metadata']['layers']
    positions = results['metadata']['positions']

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group 1: CT positions (ct0-ct5)
    ct_positions = [p for p in positions if p.startswith('ct')]
    ct_accuracies_by_layer = []

    for layer_idx in layers:
        layer_name = f'layer_{layer_idx}'
        layer_ct_accs = [
            results[layer_name][pos]['test_accuracy']
            for pos in ct_positions
        ]
        ct_accuracies_by_layer.append(np.mean(layer_ct_accs))

    # Group 2: Question/Answer positions
    qa_positions = ['question_last', 'answer_first']
    qa_accuracies_by_layer = []

    for layer_idx in layers:
        layer_name = f'layer_{layer_idx}'
        layer_qa_accs = [
            results[layer_name][pos]['test_accuracy']
            for pos in qa_positions
        ]
        qa_accuracies_by_layer.append(np.mean(layer_qa_accs))

    # Plot
    ax.plot(
        layers,
        ct_accuracies_by_layer,
        label='CT Positions (ct0-ct5) - Continuous Thoughts',
        color='steelblue',
        linestyle='--',
        linewidth=2.5,
        marker='o',
        markersize=8
    )

    ax.plot(
        layers,
        qa_accuracies_by_layer,
        label='Question/Answer Positions - Language Space',
        color='darkgreen',
        linestyle='-',
        linewidth=2.5,
        marker='s',
        markersize=8
    )

    # Add horizontal reference lines
    ax.axhline(y=0.50, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='Random (50%)')
    ax.axhline(y=0.55, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='Signal threshold (55%)')

    # Labels and title
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Probe Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Continuous Thought vs Language Space Signal (LLaMA-1B, {results["metadata"]["epoch"]})',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Legend
    ax.legend(
        loc='best',
        fontsize=10,
        framealpha=0.9
    )

    # Set y-axis limits
    ax.set_ylim(0.45, 0.75)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved grouped line plot to: {output_path}")
    plt.close()


def main():
    print("\n" + "=" * 80)
    print("MULTI-LAYER PROBE VISUALIZATION")
    print("Pre-Compression Deception Signal Analysis")
    print("=" * 80)
    print()

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epoch',
        type=str,
        default='5ep',
        help='Epoch identifier (5ep, 10ep, 15ep)'
    )
    args = parser.parse_args()

    epoch_str = args.epoch

    # Paths
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / f"multilayer_probe_results_llama1b_{epoch_str}.json"

    # Load results
    results = load_probe_results(results_path)

    # Create visualizations
    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    print()

    # 1. Heatmap
    heatmap_path = figures_dir / f"multilayer_heatmap_llama1b_{epoch_str}.png"
    create_heatmap(results, heatmap_path)

    # 2. Line plot (all positions)
    lineplot_path = figures_dir / f"multilayer_lineplot_llama1b_{epoch_str}.png"
    create_line_plot(results, lineplot_path)

    # 3. Grouped line plot (CT vs Q/A)
    grouped_path = figures_dir / f"multilayer_grouped_llama1b_{epoch_str}.png"
    create_grouped_line_plot(results, grouped_path)

    print()
    print("=" * 80)
    print("✅ VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"  Figures saved to: {figures_dir}")
    print()
    print("Next step: Statistical analysis with scripts/analyze_multilayer_patterns.py")
    print()


if __name__ == "__main__":
    main()
