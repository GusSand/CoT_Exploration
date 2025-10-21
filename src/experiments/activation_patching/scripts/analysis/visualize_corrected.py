"""
Comprehensive Visualizations for Corrected Activation Patching Experiment

Creates multiple visualizations to clearly show:
1. Case breakdown and filtering logic
2. Recovery rates on target cases
3. Comparison with original (buggy) experiment
4. Layer-by-layer analysis

Usage:
    python visualize_corrected.py --corrected_results results/experiment_results_corrected.json \
                                   --output_dir results/plots/
"""

import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def load_results(filepath):
    """Load experiment results from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_case_breakdown(corrected_results, output_dir):
    """Visualize the case breakdown with filtering logic."""

    config = corrected_results['config']
    summary = corrected_results['summary']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left plot: Filtering flowchart
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.axis('off')
    ax1.set_title('Experimental Design: Case Filtering Logic', fontsize=16, fontweight='bold', pad=20)

    # Boxes
    def draw_box(ax, x, y, width, height, text, color, label_count=None):
        box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(box)

        if label_count:
            ax.text(x + width/2, y + height/2 + 0.3, text, ha='center', va='center',
                   fontsize=12, fontweight='bold')
            ax.text(x + width/2, y + height/2 - 0.3, f'n = {label_count}', ha='center', va='center',
                   fontsize=11, style='italic')
        else:
            ax.text(x + width/2, y + height/2, text, ha='center', va='center',
                   fontsize=12, fontweight='bold')

    def draw_arrow(ax, x1, y1, x2, y2, label=''):
        arrow = ConnectionPatch((x1, y1), (x2, y2), "data", "data",
                               arrowstyle="->", shrinkA=0, shrinkB=0, lw=2.5, color='black')
        ax.add_artist(arrow)
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x + 0.5, mid_y, label, fontsize=10, style='italic', color='darkblue')

    # All 45 pairs
    draw_box(ax1, 3, 10, 4, 1.2, f'All Problem Pairs', '#E8E8E8', config['total_pairs'])

    # Split by clean correctness
    draw_arrow(ax1, 5, 10, 2, 8.5, 'Clean ✗')
    draw_arrow(ax1, 5, 10, 8, 8.5, 'Clean ✓')

    # Invalid (clean wrong)
    draw_box(ax1, 0.5, 7.5, 3, 1, f'Invalid\n(Clean Wrong)', '#FFB3B3', config['invalid_pairs'])

    # Valid (clean correct)
    draw_box(ax1, 6.5, 7.5, 3, 1, f'Valid\n(Clean Correct)', '#B3D9FF', config['valid_pairs'])

    # Further split valid cases
    draw_arrow(ax1, 7.3, 7.5, 6.5, 6, 'Corrupted ✓')
    draw_arrow(ax1, 8.7, 7.5, 9.5, 6, 'Corrupted ✗')

    # Already correct (no intervention needed)
    draw_box(ax1, 5, 5, 3, 1, f'Already Correct\n(No help needed)', '#C2F0C2', summary['corrupted_correct'])

    # TARGET CASES (intervention needed)
    draw_box(ax1, 8.5, 5, 3, 1, f'TARGET CASES\n(Patch these!)', '#FFD700', summary['total_targets'])

    # Annotation
    ax1.text(10, 4.5, '← These 9 cases are used\n   for recovery calculation',
             fontsize=11, color='darkgreen', fontweight='bold', ha='right')

    # Right plot: Bar chart of categories
    categories = ['Total\nPairs', 'Valid\n(Clean ✓)', 'Already\nCorrect', 'TARGET\n(Need Fix)', 'Invalid\n(Clean ✗)']
    counts = [
        config['total_pairs'],
        config['valid_pairs'],
        summary['corrupted_correct'],
        summary['total_targets'],
        config['invalid_pairs']
    ]
    colors = ['#E8E8E8', '#B3D9FF', '#C2F0C2', '#FFD700', '#FFB3B3']

    bars = ax2.bar(categories, counts, color=colors, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Number of Problem Pairs', fontsize=13, fontweight='bold')
    ax2.set_title('Case Distribution', fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylim(0, 50)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'case_breakdown.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: case_breakdown.png")
    plt.close()


def plot_recovery_comparison(corrected_results, original_results, output_dir):
    """Compare recovery rates: original (buggy) vs corrected."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    layers = ['Early\n(L3)', 'Middle\n(L6)', 'Late\n(L11)']

    # Original (buggy) results - handle different format
    original_summary = original_results['summary']
    if 'layer_results' in original_summary:
        # Old format: summary['layer_results']['early']
        layer_results = original_summary['layer_results']
        original_recovery = [
            layer_results['early']['recovery_rate'] * 100,
            layer_results['middle']['recovery_rate'] * 100,
            layer_results['late']['recovery_rate'] * 100
        ]
    else:
        # New format: summary['early']
        original_recovery = [
            original_summary['early']['recovery_rate'] * 100,
            original_summary['middle']['recovery_rate'] * 100,
            original_summary['late']['recovery_rate'] * 100
        ]

    # Corrected results
    corrected_summary = corrected_results['summary']
    corrected_recovery = [
        corrected_summary['early']['recovery_rate'] * 100,
        corrected_summary['middle']['recovery_rate'] * 100,
        corrected_summary['late']['recovery_rate'] * 100
    ]

    # Left: Original (buggy)
    bars1 = ax1.bar(layers, original_recovery, color=['#FF6B6B', '#FF6B6B', '#FF8C8C'],
                    edgecolor='black', linewidth=2)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.set_ylabel('Recovery Rate (%)', fontsize=13, fontweight='bold')
    ax1.set_title('❌ ORIGINAL (Buggy)\nNegative Recovery', fontsize=14, fontweight='bold', color='darkred')
    ax1.set_ylim(-160, 20)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars1, original_recovery):
        ax1.text(bar.get_x() + bar.get_width()/2., val - 5 if val < 0 else val + 5,
                f'{val:.1f}%', ha='center', va='top' if val < 0 else 'bottom',
                fontsize=12, fontweight='bold')

    # Add bug annotation
    ax1.text(0.5, 0.05, '🐛 BUG: Computed recovery on\nALL 45 pairs (including 22\nwhere clean was wrong)',
            transform=ax1.transAxes, fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
            ha='center')

    # Right: Corrected
    bars2 = ax2.bar(layers, corrected_recovery, color=['#90EE90', '#90EE90', '#4CAF50'],
                    edgecolor='black', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.set_ylabel('Recovery Rate (%)', fontsize=13, fontweight='bold')
    ax2.set_title('✓ CORRECTED\nPositive Recovery', fontsize=14, fontweight='bold', color='darkgreen')
    ax2.set_ylim(-20, 80)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars2, corrected_recovery):
        ax2.text(bar.get_x() + bar.get_width()/2., val + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add fix annotation
    ax2.text(0.5, 0.05, '✓ FIX: Computed recovery on\nonly 9 TARGET cases\n(Clean ✓, Corrupted ✗)',
            transform=ax2.transAxes, fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3),
            ha='center')

    plt.tight_layout()
    plt.savefig(output_dir / 'recovery_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: recovery_comparison.png")
    plt.close()


def plot_layer_recovery_detailed(corrected_results, output_dir):
    """Detailed layer-by-layer recovery with counts."""

    summary = corrected_results['summary']

    fig, ax = plt.subplots(figsize=(12, 8))

    layers = ['Early (L3)', 'Middle (L6)', 'Late (L11)']
    layer_keys = ['early', 'middle', 'late']

    # Get counts
    total_targets = summary['total_targets']
    recoveries = [summary[key]['correct_count'] for key in layer_keys]
    recovery_rates = [summary[key]['recovery_rate'] * 100 for key in layer_keys]

    x = np.arange(len(layers))

    # Create stacked bar
    fixed = recoveries
    not_fixed = [total_targets - r for r in recoveries]

    bars1 = ax.bar(x, fixed, label='Recovered (Patched → Correct)', color='#4CAF50', edgecolor='black', linewidth=2)
    bars2 = ax.bar(x, not_fixed, bottom=fixed, label='Not Recovered', color='#FFB3B3', edgecolor='black', linewidth=2)

    # Add recovery rate labels
    for i, (bar, rate, count) in enumerate(zip(bars1, recovery_rates, fixed)):
        ax.text(i, total_targets + 0.5, f'{rate:.1f}%', ha='center', va='bottom',
                fontsize=14, fontweight='bold', color='darkgreen')
        ax.text(i, count/2, f'{count}/{total_targets}', ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')

    ax.set_ylabel('Number of Target Cases', fontsize=13, fontweight='bold')
    ax.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax.set_title('Recovery Performance by Layer\n(on 9 Target Cases: Clean ✓, Corrupted ✗)',
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(layers, fontsize=12)
    ax.set_ylim(0, total_targets + 1.5)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    # Highlight best layer
    best_idx = recovery_rates.index(max(recovery_rates))
    ax.patches[best_idx].set_edgecolor('gold')
    ax.patches[best_idx].set_linewidth(4)

    plt.tight_layout()
    plt.savefig(output_dir / 'layer_recovery_detailed.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: layer_recovery_detailed.png")
    plt.close()


def plot_target_case_matrix(corrected_results, output_dir):
    """Visualize all 9 target cases individually."""

    # Extract target cases
    valid_results = corrected_results['valid_results']
    target_cases = [r for r in valid_results if not r['corrupted']['correct']]

    if len(target_cases) == 0:
        print("⚠️  No target cases to visualize")
        return

    # Create matrix: rows = cases, columns = layers
    fig, ax = plt.subplots(figsize=(10, 8))

    layer_keys = ['early', 'middle', 'late']
    layer_labels = ['Early\n(L3)', 'Middle\n(L6)', 'Late\n(L11)']

    # Build matrix (1 = recovered, 0 = not recovered)
    matrix = []
    case_labels = []
    for i, case in enumerate(target_cases):
        row = [1 if case['patched'][key]['correct'] else 0 for key in layer_keys]
        matrix.append(row)
        case_labels.append(f"Pair {case['pair_id']}")

    matrix = np.array(matrix)

    # Plot heatmap
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(len(layer_labels)))
    ax.set_yticks(np.arange(len(case_labels)))
    ax.set_xticklabels(layer_labels, fontsize=11)
    ax.set_yticklabels(case_labels, fontsize=10)

    # Add text annotations
    for i in range(len(case_labels)):
        for j in range(len(layer_labels)):
            text = ax.text(j, i, '✓' if matrix[i, j] == 1 else '✗',
                          ha="center", va="center", color="black", fontsize=16, fontweight='bold')

    ax.set_title('Recovery Success Matrix\n(9 Target Cases × 3 Layers)',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Problem Pair', fontsize=12, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.set_label('Recovered', rotation=270, labelpad=20, fontsize=11, fontweight='bold')
    cbar.ax.set_yticklabels(['No', 'Yes'])

    # Add summary stats
    totals = matrix.sum(axis=0)
    for j, (label, total) in enumerate(zip(layer_labels, totals)):
        ax.text(j, len(case_labels), f'{int(total)}/9', ha='center', va='center',
               fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'target_case_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: target_case_matrix.png")
    plt.close()


def plot_summary_infographic(corrected_results, output_dir):
    """Create a single-page summary infographic."""

    summary = corrected_results['summary']
    config = corrected_results['config']

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)

    # Title
    fig.suptitle('Activation Patching Experiment - Corrected Results',
                fontsize=18, fontweight='bold', y=0.98)

    # Top row: Key metrics
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')

    metrics_text = f"""
    📊 EXPERIMENT SUMMARY

    Total Pairs: {config['total_pairs']}  |  Valid (Clean ✓): {config['valid_pairs']}  |  Target Cases (Corrupted ✗): {summary['total_targets']}  |  Invalid (Clean ✗): {config['invalid_pairs']}

    🎯 RECOVERY RATES (on {summary['total_targets']} target cases):
       Early (L3): {summary['early']['recovery_rate']*100:.1f}%  ({summary['early']['correct_count']}/{summary['early']['total_count']})
       Middle (L6): {summary['middle']['recovery_rate']*100:.1f}%  ({summary['middle']['correct_count']}/{summary['middle']['total_count']})
       Late (L11): {summary['late']['recovery_rate']*100:.1f}%  ({summary['late']['correct_count']}/{summary['late']['total_count']})  ⭐ BEST
    """

    ax1.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=11,
            family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # Middle left: Case breakdown pie
    ax2 = fig.add_subplot(gs[1, 0])
    valid_breakdown = [summary['corrupted_correct'], summary['total_targets']]
    labels = [f'Already Correct\n{summary["corrupted_correct"]}',
             f'TARGET\n{summary["total_targets"]}']
    colors = ['#C2F0C2', '#FFD700']
    ax2.pie(valid_breakdown, labels=labels, colors=colors, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax2.set_title('Valid Cases Breakdown', fontsize=12, fontweight='bold')

    # Middle center: Recovery bars
    ax3 = fig.add_subplot(gs[1, 1])
    layers = ['Early', 'Middle', 'Late']
    recovery = [summary[k]['recovery_rate'] * 100 for k in ['early', 'middle', 'late']]
    bars = ax3.barh(layers, recovery, color=['#90EE90', '#90EE90', '#4CAF50'], edgecolor='black', linewidth=2)
    ax3.set_xlabel('Recovery Rate (%)', fontsize=10, fontweight='bold')
    ax3.set_title('Layer Recovery Rates', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, 100)
    for bar, val in zip(bars, recovery):
        ax3.text(val + 2, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                va='center', fontsize=10, fontweight='bold')

    # Middle right: Baseline accuracies
    ax4 = fig.add_subplot(gs[1, 2])
    conditions = ['Clean\n(baseline)', 'Corrupted\n(baseline)']
    accuracies = [summary['clean_accuracy'] * 100, summary['corrupted_accuracy'] * 100]
    bars = ax4.bar(conditions, accuracies, color=['#4CAF50', '#FF6B6B'], edgecolor='black', linewidth=2)
    ax4.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
    ax4.set_title('Baseline Performance\n(on valid pairs)', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 110)
    for bar, val in zip(bars, accuracies):
        ax4.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}%',
                ha='center', fontsize=10, fontweight='bold')

    # Bottom: Key findings
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')

    best_layer = max(['early', 'middle', 'late'], key=lambda k: summary[k]['recovery_rate'])
    best_recovery = summary[best_layer]['recovery_rate'] * 100

    findings = f"""
    ✅ KEY FINDINGS:

    1. POSITIVE RECOVERY DETECTED: Patching clean activations into corrupted problems improves accuracy by {best_recovery:.1f}% on target cases

    2. LAYER HIERARCHY: Late layer (L11) shows strongest causal effect ({summary['late']['recovery_rate']*100:.1f}% recovery)

    3. EXPERIMENTAL DESIGN MATTERS:
       - Original experiment: Computed recovery on all 45 pairs → NEGATIVE recovery (-100% to -143%)
       - Corrected experiment: Computed recovery on 9 target cases (Clean ✓, Corrupted ✗) → POSITIVE recovery (44-56%)

    4. CONCLUSION: Continuous thought representations in CODI show CAUSAL involvement in mathematical reasoning
    """

    ax5.text(0.05, 0.5, findings, ha='left', va='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.savefig(output_dir / 'summary_infographic.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: summary_infographic.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corrected_results', type=str,
                       default='results/experiment_results_corrected.json')
    parser.add_argument('--original_results', type=str,
                       default='results/experiment_results.json')
    parser.add_argument('--output_dir', type=str, default='results/plots/')
    args = parser.parse_args()

    # Load results
    print("Loading results...")
    corrected_results = load_results(args.corrected_results)
    original_results = load_results(args.original_results)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating visualizations in {output_dir}...")

    # Generate all plots
    plot_case_breakdown(corrected_results, output_dir)
    plot_recovery_comparison(corrected_results, original_results, output_dir)
    plot_layer_recovery_detailed(corrected_results, output_dir)
    plot_target_case_matrix(corrected_results, output_dir)
    plot_summary_infographic(corrected_results, output_dir)

    print(f"\n✓ All visualizations saved to {output_dir}")
    print("\nGenerated files:")
    print("  1. case_breakdown.png - Shows filtering logic and case distribution")
    print("  2. recovery_comparison.png - Compares original (buggy) vs corrected results")
    print("  3. layer_recovery_detailed.png - Detailed recovery by layer")
    print("  4. target_case_matrix.png - Individual target case results")
    print("  5. summary_infographic.png - Single-page summary")


if __name__ == "__main__":
    main()
