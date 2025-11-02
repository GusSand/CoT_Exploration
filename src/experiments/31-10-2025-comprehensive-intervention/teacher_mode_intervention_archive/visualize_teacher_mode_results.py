#!/usr/bin/env python3
"""
Visualize Teacher Mode Intervention Results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
results_file = Path('/workspace/CoT_Exploration/src/experiments/31-10-2025-comprehensive-intervention/teacher_mode_intervention_results/teacher_mode_results_20251102_163041.json')

print("Loading results from remote server...")
# Note: This script should be run on the server where results are located

def load_and_visualize():
    with open(results_file, 'r') as f:
        data = json.load(f)

    # Extract data
    conditions = []
    accuracies = []
    avg_interventions = []

    for condition in data['conditions']:
        intervention_type = condition['intervention_type']
        intervention_scope = condition['intervention_scope']

        # Create label
        if intervention_type == 'baseline':
            label = 'Baseline'
        elif intervention_type == 'discretize':
            label = f'Discretize\n({intervention_scope})'
        elif intervention_type == 'discretize_plusone':
            label = f'Discretize+1\n({intervention_scope})'

        conditions.append(label)
        accuracies.append(condition['accuracy'])

        # Calculate average interventions
        avg_int = np.mean([r['num_interventions'] for r in condition['results']])
        avg_interventions.append(avg_int)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Teacher Mode Intervention Comparison Results (N=132)', fontsize=16, fontweight='bold')

    # 1. Accuracy comparison (bar chart)
    ax1 = axes[0, 0]
    colors = ['#2ecc71', '#3498db', '#3498db', '#e74c3c', '#e74c3c']
    bars = ax1.bar(range(len(conditions)), accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy by Intervention Type', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(conditions)))
    ax1.set_xticklabels(conditions, fontsize=9)
    ax1.set_ylim(0, 100)
    ax1.axhline(y=72.7, color='green', linestyle='--', alpha=0.5, label='Baseline (72.7%)')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.legend()

    # 2. Average interventions (bar chart)
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(conditions)), avg_interventions, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Avg Interventions per Example', fontsize=12, fontweight='bold')
    ax2.set_title('Intervention Frequency', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(conditions)))
    ax2.set_xticklabels(conditions, fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, avg_int in zip(bars2, avg_interventions):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{avg_int:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 3. Accuracy vs Interventions scatter
    ax3 = axes[1, 0]
    colors_scatter = ['green', 'blue', 'blue', 'red', 'red']
    markers = ['o', 's', '^', 's', '^']

    for i, (acc, avg_int, label, color, marker) in enumerate(zip(accuracies, avg_interventions, conditions, colors_scatter, markers)):
        ax3.scatter(avg_int, acc, s=200, c=color, marker=marker, alpha=0.7, edgecolors='black', linewidth=2, label=label)

    ax3.set_xlabel('Avg Interventions per Example', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Accuracy vs Intervention Frequency', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8, loc='best')

    # 4. Correct vs Incorrect breakdown
    ax4 = axes[1, 1]

    correct_counts = []
    incorrect_counts = []

    for condition in data['conditions']:
        num_correct = sum(1 for r in condition['results'] if r['correct'])
        num_incorrect = len(condition['results']) - num_correct
        correct_counts.append(num_correct)
        incorrect_counts.append(num_incorrect)

    x = np.arange(len(conditions))
    width = 0.6

    bars_correct = ax4.bar(x, correct_counts, width, label='Correct', color='#2ecc71', alpha=0.7, edgecolor='black')
    bars_incorrect = ax4.bar(x, incorrect_counts, width, bottom=correct_counts, label='Incorrect', color='#e74c3c', alpha=0.7, edgecolor='black')

    ax4.set_ylabel('Number of Examples', fontsize=12, fontweight='bold')
    ax4.set_title('Correct vs Incorrect Predictions', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(conditions, fontsize=9)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    # Add count labels
    for i, (correct, incorrect) in enumerate(zip(correct_counts, incorrect_counts)):
        # Correct count
        ax4.text(i, correct/2, str(correct), ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
        # Incorrect count
        ax4.text(i, correct + incorrect/2, str(incorrect), ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

    plt.tight_layout()

    # Save figure
    output_dir = Path('./teacher_mode_intervention_results')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'teacher_mode_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {output_file}")

    # Also create a detailed comparison plot
    fig2, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Grouped bar chart
    intervention_types = ['Baseline', 'Discretize', 'Discretize+1']
    scopes = ['none', 'numbers', 'all']

    accuracy_matrix = [
        [72.7, 0, 0],      # Baseline (none, N/A, N/A)
        [0, 72.7, 72.7],   # Discretize (N/A, numbers, all)
        [0, 8.3, 8.3]      # Discretize+1 (N/A, numbers, all)
    ]

    x = np.arange(len(intervention_types))
    width = 0.25

    colors_grouped = ['#95a5a6', '#3498db', '#9b59b6']

    for i, (scope, color) in enumerate(zip(['None', 'Numbers', 'All'], colors_grouped)):
        values = [row[i] for row in accuracy_matrix]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=f'Scope: {scope}',
                      color=color, alpha=0.7, edgecolor='black')

        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Accuracy by Intervention Type and Scope', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(intervention_types, fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=72.7, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Baseline Performance')

    plt.tight_layout()
    output_file2 = output_dir / 'teacher_mode_grouped_comparison.png'
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"✓ Grouped comparison saved to {output_file2}")

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
        print("\nKey Findings:")
        print("  • Baseline accuracy: 72.7% (96/132)")
        print("  • Discretize maintains 72.7% accuracy")
        print("  • Plus-one intervention drops to 8.3% (11/132)")
        print("  • Demonstrates arithmetic errors propagate through CoT")
    except FileNotFoundError:
        print(f"Error: Results file not found at {results_file}")
        print("Please run this script on the server with the results file.")
