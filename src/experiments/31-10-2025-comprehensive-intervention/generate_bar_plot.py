#!/usr/bin/env python3
"""
Generate bar plot comparing all intervention conditions
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(results_file):
    """Load results JSON file"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data

def create_bar_plot(data, dataset_name, output_file):
    """Create bar plot visualization"""

    # Extract condition names and accuracies
    conditions = data['conditions']

    # Create labels and values
    labels = []
    accuracies = []
    colors = []

    for cond in conditions:
        interv_type = cond['intervention_type']
        interv_scope = cond['intervention_scope']
        accuracy = cond['accuracy']

        # Create label
        if interv_scope == 'none':
            label = f"Baseline"
        else:
            label = f"{interv_type}\n({interv_scope})"

        labels.append(label)
        accuracies.append(accuracy)

        # Color code by intervention type
        if interv_type == 'baseline':
            colors.append('#2ecc71')  # Green
        elif interv_type == 'replacement':
            colors.append('#3498db')  # Blue
        elif interv_type in ['zero', 'average']:
            colors.append('#e74c3c')  # Red (ablations)
        elif interv_type == 'minus':
            colors.append('#f39c12')  # Orange
        elif 'discretize' in interv_type:
            colors.append('#9b59b6')  # Purple
        elif 'proj' in interv_type:
            colors.append('#1abc9c')  # Teal
        else:
            colors.append('#95a5a6')  # Gray

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))

    # Create bar plot
    x = np.arange(len(labels))
    bars = ax.bar(x, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Customize plot
    ax.set_xlabel('Intervention Condition', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Intervention Comparison - {dataset_name}\n({data["config"]["num_examples"]} examples, {data["config"]["num_conditions"]} conditions)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add accuracy values on top of bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add horizontal line at baseline
    baseline_acc = accuracies[0]
    ax.axhline(y=baseline_acc, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Baseline: {baseline_acc:.1f}%')
    ax.legend(fontsize=12)

    # Tight layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Bar plot saved to {output_file}")

    # Close figure to free memory
    plt.close()


def main():
    # Check for both datasets
    results_dir = Path('./intervention_comparison_results')

    # Process clean dataset
    clean_file = results_dir / 'full_results_clean_132_examples.json'
    if clean_file.exists():
        print("\nGenerating bar plot for clean dataset...")
        data = load_results(clean_file)
        create_bar_plot(data, "Clean Dataset", results_dir / 'bar_plot_clean.png')

    # Process GSM8K dataset
    gsm8k_file = results_dir / 'full_results_gsm8k_train_132_examples.json'
    if gsm8k_file.exists():
        print("\nGenerating bar plot for GSM8K train...")
        data = load_results(gsm8k_file)
        create_bar_plot(data, "GSM8K Train", results_dir / 'bar_plot_gsm8k_train.png')

    print("\n[OK] All bar plots generated!")


if __name__ == "__main__":
    main()
