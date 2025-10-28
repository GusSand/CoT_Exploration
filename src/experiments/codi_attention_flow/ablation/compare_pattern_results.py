#!/usr/bin/env python3
"""
Compare attention pattern ablation results.

Creates visualizations comparing accuracy drops across different patterns.

Usage:
    python compare_pattern_results.py [--model MODEL]
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import argparse

def load_pattern_results(model_name='llama'):
    """Load all pattern ablation results."""
    results_dir = Path(__file__).parent.parent / 'results'

    patterns = [
        'hub_to_ct0',
        'skip_connections',
        'backward',
        'position_0',
        'position_1',
        'position_2',
        'position_3',
        'position_4',
        'position_5'
    ]

    results = []
    for pattern in patterns:
        result_file = results_dir / f'{model_name}_attention_pattern_{pattern}.json'
        if result_file.exists():
            with open(result_file, 'r') as f:
                data = json.load(f)
                results.append({
                    'pattern': pattern,
                    'accuracy': data['accuracy'],
                    'accuracy_drop': data['accuracy_drop'],
                    'n_correct': data['n_correct'],
                    'n_problems': data['n_problems']
                })

    return pd.DataFrame(results)


def visualize_pattern_comparison(df, output_path):
    """Create comparison visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Sort by accuracy drop
    df_sorted = df.sort_values('accuracy_drop', ascending=False)

    # Plot 1: Accuracy by pattern
    colors = ['#d62728' if x > 0.10 else '#ff7f0e' if x > 0.05 else '#2ca02c'
              for x in df_sorted['accuracy_drop']]

    ax1.barh(df_sorted['pattern'], df_sorted['accuracy'] * 100, color=colors, alpha=0.7)
    ax1.axvline(59, color='black', linestyle='--', linewidth=2, label='Baseline (59%)')
    ax1.set_xlabel('Accuracy (%)', fontsize=12)
    ax1.set_ylabel('Ablation Pattern', fontsize=12)
    ax1.set_title('Accuracy by Attention Pattern Ablation', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)

    # Format pattern names
    pattern_names = {
        'hub_to_ct0': 'Hub (no CT0)',
        'skip_connections': 'Skip Connections',
        'backward': 'Backward (causal)',
        'position_0': 'Position 0',
        'position_1': 'Position 1',
        'position_2': 'Position 2',
        'position_3': 'Position 3',
        'position_4': 'Position 4',
        'position_5': 'Position 5'
    }
    ax1.set_yticklabels([pattern_names.get(p, p) for p in df_sorted['pattern']])

    # Plot 2: Accuracy drop
    ax2.barh(df_sorted['pattern'], df_sorted['accuracy_drop'] * 100, color=colors, alpha=0.7)
    ax2.set_xlabel('Accuracy Drop from Baseline (%)', fontsize=12)
    ax2.set_ylabel('')
    ax2.set_title('Impact of Pattern Ablation', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.set_yticklabels([pattern_names.get(p, p) for p in df_sorted['pattern']])

    # Add text annotations
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        ax2.text(row['accuracy_drop'] * 100 + 0.5, i, f"{row['accuracy_drop']*100:.1f}%",
                va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison: {output_path}")


def print_summary_table(df):
    """Print summary table."""
    print("\n" + "="*80)
    print("ATTENTION PATTERN ABLATION SUMMARY")
    print("="*80)
    print(f"\nBaseline accuracy: 59.00%")
    print(f"Test problems: {df.iloc[0]['n_problems']}")
    print()

    # Sort by accuracy drop
    df_sorted = df.sort_values('accuracy_drop', ascending=False)

    print(f"{'Pattern':<25} {'Accuracy':>10} {'Drop':>8} {'Correct':>10}")
    print("-" * 80)
    for _, row in df_sorted.iterrows():
        pattern_display = row['pattern'].replace('_', ' ').title()
        print(f"{pattern_display:<25} {row['accuracy']*100:>9.2f}% "
              f"{row['accuracy_drop']*100:>7.1f}% "
              f"{row['n_correct']:>4}/{row['n_problems']:<4}")

    print()
    print("Key Findings:")
    print(f"  • Most critical pattern: {df_sorted.iloc[0]['pattern']} "
          f"({df_sorted.iloc[0]['accuracy_drop']*100:.1f}% drop)")
    print(f"  • Least critical pattern: {df_sorted.iloc[-1]['pattern']} "
          f"({df_sorted.iloc[-1]['accuracy_drop']*100:.1f}% drop)")
    print(f"  • Mean accuracy drop: {df['accuracy_drop'].mean()*100:.1f}%")
    print()


def main():
    parser = argparse.ArgumentParser(description='Compare pattern ablation results')
    parser.add_argument('--model', type=str, default='llama', choices=['llama', 'gpt2'])
    args = parser.parse_args()

    # Load results
    df = load_pattern_results(args.model)

    if df.empty:
        print("No pattern results found. Run ablation experiments first.")
        return

    # Print summary
    print_summary_table(df)

    # Create visualization
    output_path = Path(__file__).parent.parent / 'results' / f'{args.model}_pattern_comparison.png'
    visualize_pattern_comparison(df, output_path)


if __name__ == '__main__':
    main()
