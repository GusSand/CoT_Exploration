#!/usr/bin/env python3
"""
Analysis and Visualization

Combines ablation and attention results to answer:
- RQ1: Which tokens are most important?
- RQ2: Does attention correlate with importance?

Generates 4 key figures:
1. Bar chart: Importance by token position
2. Heatmap: Problems × tokens importance matrix
3. Scatter: Attention vs importance correlation
4. By-position subplots: Per-token correlation analysis

Usage:
    python 3_analyze_and_visualize.py [--test_mode]
"""
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def load_results(test_mode=False):
    """Load ablation and attention results."""
    if test_mode:
        ablation_file = Path(__file__).parent.parent / 'results' / 'token_ablation_results_test.json'
        attention_file = Path(__file__).parent.parent / 'results' / 'attention_weights_test.json'
    else:
        ablation_file = Path(__file__).parent.parent / 'results' / 'token_ablation_results_100.json'
        attention_file = Path(__file__).parent.parent / 'results' / 'attention_weights_100.json'

    with open(ablation_file, 'r') as f:
        ablation_results = json.load(f)

    with open(attention_file, 'r') as f:
        attention_results = json.load(f)

    return ablation_results, attention_results


def prepare_dataframe(ablation_results, attention_results):
    """Combine ablation and attention into a unified dataframe."""
    rows = []

    # Create lookup for attention data
    attention_lookup = {r['problem_id']: r for r in attention_results if 'error' not in r}

    for problem in ablation_results:
        if 'error' in problem or not problem['baseline']['correct']:
            continue  # Skip errors and baseline failures

        problem_id = problem['problem_id']
        difficulty = problem['difficulty']

        if problem_id not in attention_lookup:
            continue  # Skip if no attention data

        attention_data = attention_lookup[problem_id]

        # Extract data for each token
        for token_pos in range(6):
            ablation = problem['token_ablations'][token_pos]
            importance = ablation['importance']

            # Get attention at layer 14 (late layer, closest to answer)
            attn_layer14 = attention_data['attention_by_layer']['layer_14']
            attention_weight = attn_layer14['continuous_token_attention'][token_pos]

            # Also get middle and early layer attention
            attn_layer8 = attention_data['attention_by_layer']['layer_8']
            attn_layer4 = attention_data['attention_by_layer']['layer_4']

            rows.append({
                'problem_id': problem_id,
                'difficulty': difficulty,
                'token_pos': token_pos,
                'importance': importance,
                'attention_layer_14': attention_weight,
                'attention_layer_8': attn_layer8['continuous_token_attention'][token_pos],
                'attention_layer_4': attn_layer4['continuous_token_attention'][token_pos]
            })

    df = pd.DataFrame(rows)
    return df


def create_visualizations(df, output_dir):
    """Create all 4 visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================
    # Figure 1: Importance by Token Position
    # ========================================
    fig, ax = plt.subplots(figsize=(10, 6))

    importance_by_pos = df.groupby('token_pos')['importance'].agg(['mean', 'sem'])

    ax.bar(range(6), importance_by_pos['mean'], yerr=importance_by_pos['sem'],
           capsize=5, color='steelblue', alpha=0.7)
    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Importance Score (Fraction causing failure)', fontsize=12)
    ax.set_title('Continuous Thought Token Importance', fontsize=14, fontweight='bold')
    ax.set_xticks(range(6))
    ax.set_xticklabels([f'Token {i}' for i in range(6)])
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (mean_val, sem_val) in enumerate(zip(importance_by_pos['mean'], importance_by_pos['sem'])):
        ax.text(i, mean_val + sem_val + 0.02, f'{mean_val:.2f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / '1_importance_by_position.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / '1_importance_by_position.png', dpi=150, bbox_inches='tight')
    print("✓ Saved Figure 1: Importance by position")

    # ========================================
    # Figure 2: Heatmap (Problems × Tokens)
    # ========================================
    fig, ax = plt.subplots(figsize=(10, 8))

    pivot = df.pivot_table(index='problem_id', columns='token_pos', values='importance', aggfunc='mean')

    sns.heatmap(pivot, cmap='RdYlGn', center=0.5, vmin=0, vmax=1,
                cbar_kws={'label': 'Critical (1=failure, 0=robust)'},
                ax=ax, linewidths=0.5, linecolor='gray')

    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Problem ID', fontsize=12)
    ax.set_title('Token Importance Matrix (Problems × Tokens)', fontsize=14, fontweight='bold')
    ax.set_xticklabels([f'T{i}' for i in range(6)])

    plt.tight_layout()
    plt.savefig(output_dir / '2_importance_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / '2_importance_heatmap.png', dpi=150, bbox_inches='tight')
    print("✓ Saved Figure 2: Importance heatmap")

    # ========================================
    # Figure 3: Attention vs Importance Scatter
    # ========================================
    fig, ax = plt.subplots(figsize=(8, 6))

    # Use layer 14 attention (late layer)
    x = df['attention_layer_14']
    y = df['importance']

    ax.scatter(x, y, alpha=0.6, s=50, c='steelblue', edgecolors='black', linewidth=0.5)

    # Compute correlation
    corr, pval = pearsonr(x, y)

    # Add regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Linear fit')

    ax.set_xlabel('Attention Weight (Layer 14)', fontsize=12)
    ax.set_ylabel('Importance Score', fontsize=12)
    ax.set_title(f'Attention vs Importance (r={corr:.3f}, p={pval:.3e})', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / '3_attention_vs_importance.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / '3_attention_vs_importance.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved Figure 3: Attention vs importance (r={corr:.3f})")

    # ========================================
    # Figure 4: By-Position Subplots
    # ========================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for token_pos in range(6):
        ax = axes[token_pos]
        subset = df[df['token_pos'] == token_pos]

        x = subset['attention_layer_14']
        y = subset['importance']

        ax.scatter(x, y, alpha=0.6, s=40, c='steelblue', edgecolors='black', linewidth=0.5)

        # Correlation
        if len(x) > 1:
            corr, pval = pearsonr(x, y)
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

            ax.set_title(f'Token {token_pos} (r={corr:.3f}, p={pval:.2e})', fontsize=11, fontweight='bold')
        else:
            ax.set_title(f'Token {token_pos}', fontsize=11, fontweight='bold')

        ax.set_xlabel('Attention Weight', fontsize=10)
        ax.set_ylabel('Importance', fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig(output_dir / '4_correlation_by_position.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / '4_correlation_by_position.png', dpi=150, bbox_inches='tight')
    print("✓ Saved Figure 4: Correlation by position")


def generate_summary_stats(df, output_file):
    """Generate summary statistics."""
    stats = {}

    # Overall correlation
    corr_l14, pval_l14 = pearsonr(df['attention_layer_14'], df['importance'])
    corr_l8, pval_l8 = pearsonr(df['attention_layer_8'], df['importance'])
    corr_l4, pval_l4 = pearsonr(df['attention_layer_4'], df['importance'])

    stats['overall_correlation'] = {
        'layer_14': {'r': float(corr_l14), 'p': float(pval_l14)},
        'layer_8': {'r': float(corr_l8), 'p': float(pval_l8)},
        'layer_4': {'r': float(corr_l4), 'p': float(pval_l4)}
    }

    # Per-token statistics
    stats['per_token'] = {}
    for token_pos in range(6):
        subset = df[df['token_pos'] == token_pos]
        corr, pval = pearsonr(subset['attention_layer_14'], subset['importance'])

        stats['per_token'][f'token_{token_pos}'] = {
            'importance_mean': float(subset['importance'].mean()),
            'importance_std': float(subset['importance'].std()),
            'attention_mean': float(subset['attention_layer_14'].mean()),
            'attention_std': float(subset['attention_layer_14'].std()),
            'correlation': float(corr),
            'p_value': float(pval)
        }

    # By difficulty
    stats['by_difficulty'] = {}
    for diff in df['difficulty'].unique():
        subset = df[df['difficulty'] == diff]
        corr, pval = pearsonr(subset['attention_layer_14'], subset['importance'])

        stats['by_difficulty'][diff] = {
            'num_problems': len(subset) // 6,
            'mean_importance': float(subset['importance'].mean()),
            'correlation': float(corr),
            'p_value': float(pval)
        }

    # Save
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"✓ Saved summary statistics")

    return stats


def print_summary(stats):
    """Print key findings."""
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    print("\n1. OVERALL ATTENTION-IMPORTANCE CORRELATION:")
    for layer, data in stats['overall_correlation'].items():
        print(f"   {layer}: r={data['r']:.3f}, p={data['p']:.3e}")

    print("\n2. TOKEN-LEVEL IMPORTANCE (mean ± std):")
    for token_id, data in stats['per_token'].items():
        print(f"   {token_id}: {data['importance_mean']:.3f} ± {data['importance_std']:.3f} "
              f"(corr={data['correlation']:.3f})")

    print("\n3. DIFFICULTY STRATIFICATION:")
    for diff, data in stats['by_difficulty'].items():
        print(f"   {diff}: n={data['num_problems']} problems, "
              f"importance={data['mean_importance']:.3f}, "
              f"corr={data['correlation']:.3f}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', action='store_true', help='Analyze 10-problem test results')
    args = parser.parse_args()

    print("=" * 80)
    print("ANALYSIS AND VISUALIZATION")
    print("=" * 80)

    # Load results
    print("\nLoading results...")
    ablation_results, attention_results = load_results(test_mode=args.test_mode)
    print(f"✓ Loaded {len(ablation_results)} ablation results")
    print(f"✓ Loaded {len(attention_results)} attention results")

    # Prepare dataframe
    print("\nPreparing analysis dataframe...")
    df = prepare_dataframe(ablation_results, attention_results)
    print(f"✓ Created dataframe with {len(df)} rows ({len(df)//6} problems × 6 tokens)")

    # Create visualizations
    print("\nGenerating visualizations...")
    output_dir = Path(__file__).parent.parent / 'figures'
    create_visualizations(df, output_dir)

    # Generate statistics
    print("\nComputing summary statistics...")
    stats_file = Path(__file__).parent.parent / 'results' / 'summary_statistics.json'
    stats = generate_summary_stats(df, stats_file)

    # Print summary
    print_summary(stats)

    print(f"\n✓ All outputs saved to {Path(__file__).parent.parent}")
    print("  - Figures: figures/")
    print("  - Statistics: results/summary_statistics.json")


if __name__ == "__main__":
    main()
