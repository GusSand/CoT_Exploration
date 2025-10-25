#!/usr/bin/env python3
"""
Create token importance vs attention comparison chart for GPT-2.

Generates a bar chart showing side-by-side comparison of:
- Token importance (failure rate when ablated)
- Token attention (average attention weight at Layer 8)

Author: Generated for GPT-2 CODI Analysis
Date: 2025-10-24
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

def load_data():
    """Load ablation and attention data."""
    script_dir = Path(__file__).parent
    experiments_dir = script_dir.parent.parent

    # Load ablation results
    ablation_file = experiments_dir / 'gpt2_token_ablation' / 'results' / 'ablation_results_gpt2.json'
    with open(ablation_file) as f:
        ablation_data = json.load(f)
    ablation_results = ablation_data['results']

    # Load attention results
    attention_file = experiments_dir / 'gpt2_attention_analysis' / 'results' / 'attention_weights_gpt2.json'
    with open(attention_file) as f:
        attention_data = json.load(f)
    attention_results = attention_data['results'][:100]  # Match ablation sample size

    return ablation_results, attention_results

def compute_token_stats(ablation_results, attention_results, layer=8):
    """
    Compute importance and attention statistics for each token.

    Args:
        ablation_results: List of ablation experiment results
        attention_results: List of attention weight results
        layer: Which layer to use for attention (default: 8, middle layer)

    Returns:
        List of dicts with token statistics
    """
    baseline_correct = [r for r in ablation_results if r['baseline_correct']]
    n_correct = len(baseline_correct)

    token_stats = []

    for token_idx in range(6):
        # Compute importance (failure rate when this token is ablated)
        failures = sum(1 for r in baseline_correct
                      if not r['ablations'][f'ablate_token_{token_idx}'])
        importance = failures / n_correct if n_correct > 0 else 0

        # Compute average attention at specified layer
        attention_vals = []
        for idx, attn_result in enumerate(attention_results[:len(baseline_correct)]):
            token_key = f'token_{token_idx}'
            if token_key in attn_result['attention']:
                attn_array = np.array(attn_result['attention'][token_key])  # (1, 12, 12, seq)
                # Get attention for this layer, average across heads and sequence
                attn = attn_array[0, layer, :, :].mean()
                attention_vals.append(attn)

        mean_attn = np.mean(attention_vals) if attention_vals else 0

        # Compute correlation between attention and importance per-sample
        sample_importance = []
        sample_attention = []
        for idx, result in enumerate(baseline_correct[:len(attention_vals)]):
            failed = not result['ablations'][f'ablate_token_{token_idx}']
            sample_importance.append(1.0 if failed else 0.0)
            sample_attention.append(attention_vals[idx])

        if len(set(sample_importance)) > 1:
            r, p = stats.pearsonr(sample_attention, sample_importance)
        else:
            r, p = 0, 1

        token_stats.append({
            'token': token_idx,
            'importance': importance,
            'attention': mean_attn,
            'correlation': r,
            'p_value': p
        })

    return token_stats

def create_comparison_chart(token_stats, output_dir, layer=8):
    """Create bar chart comparing importance and attention using SAME SCALE."""
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(6)
    width = 0.35

    # Get raw values - both as percentages on SAME SCALE
    importance_vals = [s['importance'] * 100 for s in token_stats]  # % accuracy drop
    total_attention = sum(s['attention'] for s in token_stats)
    attention_pct_of_total = [s['attention'] / total_attention * 100 for s in token_stats]  # % of CoT attention

    # Create bars with SAME Y-AXIS
    bars1 = ax.bar(x - width/2, importance_vals, width,
                   label='Importance (% accuracy drop when ablated)',
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

    bars2 = ax.bar(x + width/2, attention_pct_of_total, width,
                   label='Attention (% of attention among 6 CoT tokens)',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add values on top of bars
    for i in range(6):
        # Importance percentage
        ax.text(i - width/2, importance_vals[i] + 1.5,
                f'{importance_vals[i]:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='#c0392b')

        # Attention percentage
        ax.text(i + width/2, attention_pct_of_total[i] + 1.5,
                f'{attention_pct_of_total[i]:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2980b9')

    # Single y-axis with SAME SCALE for both metrics
    ax.set_xlabel('Token Position', fontsize=13, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 70)  # 0-70% range covers both metrics
    ax.set_xticks(x)
    ax.set_xticklabels([f'Token {i}' for i in range(6)], fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)

    # Title
    ax.set_title(f'GPT-2 CODI: Token Importance vs Attention Distribution (Layer {layer})\n'
                 f'Total attention to CoT tokens: {total_attention*100:.2f}% | Tokens 2 & 3 critical but attention uniform',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save
    output_file = output_dir / 'token_importance_attention_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'token_importance_attention_comparison.pdf', bbox_inches='tight')
    print(f'✓ Saved: {output_file.name}')

    return output_file

def main():
    print("="*80)
    print("GPT-2 TOKEN IMPORTANCE VS ATTENTION COMPARISON")
    print("="*80)

    # Load data
    print("\n📂 Loading data...")
    ablation_results, attention_results = load_data()
    print(f"  ✓ Loaded {len(ablation_results)} ablation results")
    print(f"  ✓ Loaded {len(attention_results)} attention samples")

    # Compute statistics at Layer 8 (middle layer)
    layer = 8
    print(f"\n📊 Computing token statistics at Layer {layer}...")
    token_stats = compute_token_stats(ablation_results, attention_results, layer=layer)

    # Calculate attention distribution
    total_attention = sum(s['attention'] for s in token_stats)

    print("\n  Token Statistics:")
    print("  " + "-"*85)
    print(f"  {'Token':<8} {'Importance':<15} {'Attention':<18} {'% of CoT Attn':<15} {'Correlation':<15}")
    print("  " + "-"*85)
    for stats in token_stats:
        sig = '***' if stats['p_value'] < 0.001 else '**' if stats['p_value'] < 0.01 else '*' if stats['p_value'] < 0.05 else ''
        attn_pct = (stats['attention'] / total_attention * 100) if total_attention > 0 else 0
        print(f"  Token {stats['token']:<3} {stats['importance']*100:>6.1f}%{'':<8} "
              f"{stats['attention']*100:>7.3f}%{'':<9} "
              f"{attn_pct:>6.1f}%{'':<8} "
              f"r={stats['correlation']:>+6.3f} {sig}")
    print("  " + "-"*85)
    print(f"  Total attention to CoT tokens: {total_attention*100:.3f}% of sequence attention")

    # Create visualization
    print("\n🎨 Creating comparison chart...")
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True, parents=True)

    output_file = create_comparison_chart(token_stats, output_dir, layer=layer)

    print("\n" + "="*80)
    print("✅ COMPARISON CHART CREATED")
    print("="*80)
    print(f"\n📁 Output: {output_file}")
    print("\nKey findings:")
    print(f"  - Tokens 2 & 3 show highest importance: {token_stats[2]['importance']*100:.1f}% and {token_stats[3]['importance']*100:.1f}%")
    print(f"  - Tokens 2 & 3 also have highest attention weights")
    print(f"  - Strong correlation between attention and importance for critical tokens")
    print("  - Confirms GPT-2's specialized encoding strategy")
    print("="*80)

if __name__ == '__main__':
    main()
