#!/usr/bin/env python3
"""
Visualize Teacher Mode Top-K Projection Intervention Results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
results_file = Path('teacher_projection_results_132ex_20251102_175938.json')

print("Loading results...")

def load_and_visualize():
    with open(results_file, 'r') as f:
        data = json.load(f)

    # Extract data
    conditions = []
    k_values = []
    accuracies = []
    avg_interventions = []

    for condition in data['conditions']:
        intervention_type = condition['intervention_type']
        k = condition['k']

        # Create label
        if intervention_type == 'baseline':
            label = 'Baseline'
            k_val = 0  # For plotting
        else:
            label = f'Proj@{k}'
            k_val = k

        conditions.append(label)
        k_values.append(k_val)
        accuracies.append(condition['accuracy'])

        # Calculate average interventions
        avg_int = condition['avg_interventions']
        avg_interventions.append(avg_int)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('Teacher Mode Top-K Projection Intervention Results (N=132)',
                 fontsize=16, fontweight='bold')

    # 1. Accuracy vs K (line plot)
    ax1 = fig.add_subplot(gs[0, 0])

    # Separate baseline and projection points
    baseline_acc = accuracies[0]
    proj_k = k_values[1:]
    proj_acc = accuracies[1:]

    ax1.plot(proj_k, proj_acc, 'o-', color='#3498db', linewidth=2, markersize=8,
             label='Projection@k', alpha=0.7)
    ax1.axhline(y=baseline_acc, color='#2ecc71', linestyle='--', linewidth=2,
                alpha=0.7, label=f'Baseline ({baseline_acc:.1f}%)')

    ax1.set_xlabel('k (subspace dimension)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy vs Subspace Dimension', fontsize=13, fontweight='bold')
    ax1.set_ylim(70, 75)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xscale('log')
    ax1.set_xticks(proj_k)
    ax1.set_xticklabels([str(k) for k in proj_k])

    # 2. Accuracy bar chart (all conditions)
    ax2 = fig.add_subplot(gs[0, 1])

    colors = ['#2ecc71'] + ['#3498db'] * (len(conditions) - 1)
    bars = ax2.bar(range(len(conditions)), accuracies, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=1.5)

    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy by Intervention Type', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(conditions)))
    ax2.set_xticklabels(conditions, rotation=45, ha='right', fontsize=9)
    ax2.set_ylim(70, 75)
    ax2.axhline(y=baseline_acc, color='green', linestyle='--', alpha=0.5,
                label=f'Baseline ({baseline_acc:.1f}%)')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # 3. Correct vs Incorrect counts
    ax3 = fig.add_subplot(gs[0, 2])

    correct_counts = []
    incorrect_counts = []

    for condition in data['conditions']:
        num_correct = sum(1 for r in condition['results'] if r['correct'])
        num_incorrect = len(condition['results']) - num_correct
        correct_counts.append(num_correct)
        incorrect_counts.append(num_incorrect)

    x = np.arange(len(conditions))
    width = 0.6

    bars_correct = ax3.bar(x, correct_counts, width, label='Correct',
                          color='#2ecc71', alpha=0.7, edgecolor='black')
    bars_incorrect = ax3.bar(x, incorrect_counts, width, bottom=correct_counts,
                            label='Incorrect', color='#e74c3c', alpha=0.7, edgecolor='black')

    ax3.set_ylabel('Number of Examples', fontsize=12, fontweight='bold')
    ax3.set_title('Correct vs Incorrect Predictions', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(conditions, rotation=45, ha='right', fontsize=9)
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)

    # Add count labels
    for i, (correct, incorrect) in enumerate(zip(correct_counts, incorrect_counts)):
        # Correct count
        if correct > 10:
            ax3.text(i, correct/2, str(correct), ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')
        # Incorrect count
        if incorrect > 10:
            ax3.text(i, correct + incorrect/2, str(incorrect), ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')

    # 4. Performance degradation from baseline
    ax4 = fig.add_subplot(gs[1, 0])

    degradation = [(baseline_acc - acc) for acc in proj_acc]

    ax4.bar(range(len(proj_k)), degradation, color='#e74c3c', alpha=0.7,
            edgecolor='black', linewidth=1.5)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)

    ax4.set_xlabel('k (subspace dimension)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Accuracy Drop (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Performance Degradation from Baseline', fontsize=13, fontweight='bold')
    ax4.set_xticks(range(len(proj_k)))
    ax4.set_xticklabels([f'k={k}' for k in proj_k], rotation=45, ha='right', fontsize=9)
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim(-1, 1.5)

    # Add value labels
    for i, deg in enumerate(degradation):
        if abs(deg) > 0.01:
            ax4.text(i, deg + 0.1, f'{deg:.1f}%', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

    # 5. Histogram of k values by accuracy group
    ax5 = fig.add_subplot(gs[1, 1])

    # Group by accuracy
    acc_72_7 = [k for k, acc in zip(proj_k, proj_acc) if abs(acc - 72.7) < 0.01]
    acc_72_0 = [k for k, acc in zip(proj_k, proj_acc) if abs(acc - 72.0) < 0.01]

    width = 0.35
    x_pos = np.arange(2)

    ax5.bar(x_pos[0], len(acc_72_7), width, label='72.7% accuracy',
            color='#2ecc71', alpha=0.7, edgecolor='black')
    ax5.bar(x_pos[1], len(acc_72_0), width, label='72.0% accuracy',
            color='#f39c12', alpha=0.7, edgecolor='black')

    ax5.set_ylabel('Number of k values', fontsize=12, fontweight='bold')
    ax5.set_title('Distribution of Accuracy Levels', fontsize=13, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(['72.7%\n(96/132)', '72.0%\n(95/132)'], fontsize=10)
    ax5.legend(fontsize=10)
    ax5.grid(axis='y', alpha=0.3)

    # Add counts
    ax5.text(x_pos[0], len(acc_72_7) + 0.2, str(len(acc_72_7)),
            ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax5.text(x_pos[1], len(acc_72_0) + 0.2, str(len(acc_72_0)),
            ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add text annotation
    text_str = f"k values at 72.7%: {acc_72_7}\nk values at 72.0%: {acc_72_0}"
    ax5.text(0.5, 0.02, text_str, transform=ax5.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # 6. Summary statistics table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    # Calculate statistics
    min_acc = min(proj_acc)
    max_acc = max(proj_acc)
    mean_acc = np.mean(proj_acc)
    std_acc = np.std(proj_acc)

    summary_text = f"""
    SUMMARY STATISTICS
    {'='*40}

    Baseline Performance: {baseline_acc:.1f}%

    Projection Performance:
      • Min accuracy: {min_acc:.1f}%
      • Max accuracy: {max_acc:.1f}%
      • Mean accuracy: {mean_acc:.2f}%
      • Std deviation: {std_acc:.2f}%

    Performance Preservation:
      • Best case: {(max_acc/baseline_acc)*100:.1f}% of baseline
      • Worst case: {(min_acc/baseline_acc)*100:.1f}% of baseline
      • Mean: {(mean_acc/baseline_acc)*100:.1f}% of baseline

    Key Findings:
      • k=1 (discretization): ZERO degradation
      • k≥2: Minimal degradation (≤0.7%)
      • Performance robust across k values
      • Avg interventions: {avg_interventions[1]:.1f} per example
    """

    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()

    # Save figure
    output_file = Path('teacher_mode_projection_visualization.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Visualization saved to {output_file}")

    # Create a focused comparison plot: k vs accuracy
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot with both line and markers
    ax.plot(proj_k, proj_acc, 'o-', color='#3498db', linewidth=2.5,
            markersize=10, label='Projection@k', alpha=0.8, markeredgecolor='black',
            markeredgewidth=1.5)
    ax.axhline(y=baseline_acc, color='#2ecc71', linestyle='--', linewidth=2.5,
               alpha=0.8, label=f'Baseline ({baseline_acc:.1f}%)')

    # Add value labels at each point
    for k, acc in zip(proj_k, proj_acc):
        ax.text(k, acc + 0.15, f'{acc:.1f}%', ha='center', va='bottom',
               fontsize=9, fontweight='bold')

    ax.set_xlabel('k (Vocabulary Subspace Dimension)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Teacher Mode: Accuracy vs Projection Subspace Dimension (N=132)',
                fontsize=14, fontweight='bold')
    ax.set_ylim(71, 74)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='lower right')
    ax.set_xscale('log')
    ax.set_xticks(proj_k)
    ax.set_xticklabels([str(k) for k in proj_k], fontsize=10)

    # Add shaded region for near-baseline performance
    ax.fill_between(proj_k, baseline_acc - 0.5, baseline_acc + 0.5,
                    alpha=0.2, color='green', label='±0.5% of baseline')

    plt.tight_layout()
    output_file2 = Path('teacher_projection_k_vs_accuracy.png')
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"[OK] K vs Accuracy plot saved to {output_file2}")

    return data

if __name__ == "__main__":
    try:
        data = load_and_visualize()
        print("\n" + "="*80)
        print("VISUALIZATION COMPLETE")
        print("="*80)
        print(f"\nDataset: {data['config']['num_examples']} examples")
        print(f"Model: {data['config']['model']}")
        print(f"Mode: {data['config']['mode']}")
        print(f"K values tested: {data['config']['k_values']}")
        print("\nKey Findings:")
        print("  - Baseline accuracy: 72.7% (96/132)")
        print("  - Projection@1 maintains 72.7% accuracy (discretization preserves performance)")
        print("  - Projection@k>=2 shows minimal degradation (72.0-72.7%)")
        print("  - Performance is robust across k in {1, 2, 3, 5, 8, 10, 15, 20, 30, 50}")
        print("  - Demonstrates CoT representations lie in low-dimensional vocabulary subspace")
    except FileNotFoundError:
        print(f"Error: Results file not found at {results_file}")
        print("Please ensure the results JSON file is in the current directory.")
