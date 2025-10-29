#!/usr/bin/env python3
"""
Error Taxonomy Analysis - Story 1

Classify errors by type (calculation, logic, nonsense) for baseline vs interventions
to understand qualitative behavioral changes.

Usage:
    python 1_analyze_error_taxonomy.py

Input:
    - ../codi_attention_flow/results/llama_baseline.json
    - ../codi_attention_flow/results/llama_attention_pattern_position_0.json
    - ../codi_attention_flow/results/llama_attention_pattern_position_4.json

Output:
    - results/error_taxonomy.json
    - results/error_distributions.png
"""
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent / 'utils'))
from error_classifier import classify_error_batch, get_error_distribution, print_error_summary, ERROR_TAXONOMY


def load_results(model='llama'):
    """Load baseline and intervention results."""
    results_dir = Path(__file__).parent.parent / 'codi_attention_flow' / 'results'

    files = {
        'baseline': results_dir / f'{model}_baseline.json',
        'ct0_blocked': results_dir / f'{model}_attention_pattern_position_0.json',
        'ct4_blocked': results_dir / f'{model}_attention_pattern_position_4.json',
    }

    data = {}
    for condition, filepath in files.items():
        if not filepath.exists():
            print(f"⚠️  Warning: {filepath} not found")
            continue

        with open(filepath) as f:
            data[condition] = json.load(f)
            print(f"✓ Loaded {condition}: {len(data[condition]['results_detail'])} problems")

    return data


def analyze_error_taxonomy(data):
    """Classify errors and compute distributions."""
    print("\n" + "="*80)
    print("CLASSIFYING ERRORS")
    print("="*80)

    classified_data = {}

    for condition, results in data.items():
        print(f"\nClassifying {condition}...")
        results_detail = results['results_detail']

        # Classify errors
        classified = classify_error_batch(results_detail)
        classified_data[condition] = classified

        # Get distribution
        distribution = get_error_distribution(classified)
        print_error_summary(distribution, len(results_detail), condition)

    return classified_data


def compare_error_distributions(classified_data):
    """Compare error distributions across conditions."""
    print("\n" + "="*80)
    print("ERROR TYPE SHIFTS")
    print("="*80)

    conditions = list(classified_data.keys())
    if 'baseline' not in conditions:
        print("⚠️  No baseline data for comparison")
        return

    baseline_dist = get_error_distribution(classified_data['baseline'])
    baseline_total = len(classified_data['baseline'])

    for condition in conditions:
        if condition == 'baseline':
            continue

        condition_dist = get_error_distribution(classified_data[condition])
        condition_total = len(classified_data[condition])

        print(f"\n{condition.upper()} vs BASELINE:")
        print(f"{'Error Type':<20s} {'Baseline':>10s} {'Intervention':>12s} {'Change':>10s}")
        print("-" * 60)

        for error_type in ERROR_TAXONOMY.keys():
            baseline_pct = (baseline_dist[error_type] / baseline_total) * 100
            condition_pct = (condition_dist[error_type] / condition_total) * 100
            change = condition_pct - baseline_pct

            change_str = f"{change:+.1f}%" if abs(change) >= 0.5 else "-"

            print(f"{error_type:<20s} {baseline_pct:>9.1f}% {condition_pct:>11.1f}% {change_str:>10s}")


def visualize_error_distributions(classified_data, output_dir):
    """Create visualization of error distributions."""
    import numpy as np

    # Prepare data for plotting
    conditions = list(classified_data.keys())
    error_types = list(ERROR_TAXONOMY.keys())

    data_matrix = []
    for condition in conditions:
        distribution = get_error_distribution(classified_data[condition])
        total = len(classified_data[condition])
        percentages = [(distribution[et] / total) * 100 for et in error_types]
        data_matrix.append(percentages)

    data_matrix = np.array(data_matrix).T  # Transpose for plotting

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Grouped bar chart
    x = np.arange(len(error_types))
    width = 0.25

    for i, condition in enumerate(conditions):
        offset = width * (i - len(conditions)/2 + 0.5)
        ax1.bar(x + offset, data_matrix[:, i], width, label=condition.replace('_', ' ').title())

    ax1.set_xlabel('Error Type')
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Error Distribution by Condition')
    ax1.set_xticks(x)
    ax1.set_xticklabels(error_types, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Stacked bar chart
    bottom = np.zeros(len(conditions))
    colors = plt.cm.Set3(np.linspace(0, 1, len(error_types)))

    for i, error_type in enumerate(error_types):
        ax2.bar(conditions, data_matrix[i, :], bottom=bottom, label=error_type, color=colors[i])
        bottom += data_matrix[i, :]

    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Error Distribution (Stacked)')
    ax2.set_xticklabels([c.replace('_', '\n') for c in conditions])
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    output_path = output_dir / 'error_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {output_path}")

    plt.close()


def save_results(classified_data, output_dir):
    """Save classified results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full classified data
    output_file = output_dir / 'error_taxonomy_full.json'
    with open(output_file, 'w') as f:
        json.dump(classified_data, f, indent=2)
    print(f"✓ Saved full results: {output_file}")

    # Save summary distributions
    summary = {}
    for condition, results in classified_data.items():
        distribution = get_error_distribution(results)
        total = len(results)
        summary[condition] = {
            'total': total,
            'distribution': distribution,
            'percentages': {k: (v/total)*100 for k, v in distribution.items()}
        }

    summary_file = output_dir / 'error_taxonomy_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary: {summary_file}")


def main():
    print("="*80)
    print("ERROR TAXONOMY ANALYSIS")
    print("="*80)

    # Load data
    data = load_results(model='llama')

    if not data:
        print("\n❌ No data loaded. Please run baseline and intervention experiments first.")
        return

    # Classify errors
    classified_data = analyze_error_taxonomy(data)

    # Compare distributions
    compare_error_distributions(classified_data)

    # Save results
    output_dir = Path(__file__).parent / 'results'
    save_results(classified_data, output_dir)

    # Visualize
    visualize_error_distributions(classified_data, output_dir)

    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
