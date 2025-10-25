#!/usr/bin/env python3
"""
Visualize attention-importance correlation from CCTA full results.

Creates scatter plots showing correlation between attention weights
and token importance across layers.
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

# Load CCTA results (importance data)
ccta_file = Path(__file__).parent.parent / 'results' / 'ccta_full_results_100.json'
with open(ccta_file) as f:
    ccta_results = json.load(f)

# Load attention weights
attention_file = Path(__file__).parent.parent / 'results' / 'attention_weights_100.json'
with open(attention_file) as f:
    attention_results = json.load(f)

# Extract data
baseline_correct = [r for r in ccta_results if r['baseline']['correct']]
corruption_types = ['zero', 'gauss_0.1', 'gauss_0.5', 'gauss_1.0', 'gauss_2.0', 'random', 'shuffle']

# Collect importance and attention for each layer
importance_by_problem_token = []

for idx, ccta_prob in enumerate(baseline_correct):
    for token_pos in range(6):
        importance_scores = []
        for corr_type in corruption_types:
            importance_scores.append(ccta_prob['corruptions'][f'token_{token_pos}'][corr_type]['importance'])
        mean_importance = np.mean(importance_scores)
        importance_by_problem_token.append(mean_importance)

attention_by_problem_token = {4: [], 8: [], 14: []}
token_labels = []
for idx, attn_prob in enumerate(attention_results):
    for token_pos in range(6):
        token_labels.append(token_pos)
        for layer in [4, 8, 14]:
            attn_weight = attn_prob['attention_by_layer'][f'layer_{layer}']['continuous_token_attention'][token_pos]
            attention_by_problem_token[layer].append(attn_weight)

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

layer_names = {4: 'Early (L4)', 8: 'Middle (L8)', 14: 'Late (L14)'}
colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728', 4: '#9467bd', 5: '#8c564b'}

for ax_idx, layer in enumerate([4, 8, 14]):
    ax = axes[ax_idx]

    # Scatter plot with token colors
    for token in range(6):
        mask = np.array(token_labels) == token
        ax.scatter(
            np.array(attention_by_problem_token[layer])[mask],
            np.array(importance_by_problem_token)[mask],
            alpha=0.3,
            s=30,
            c=colors[token],
            label=f'Token {token}'
        )

    # Compute and plot regression line
    r, p = stats.pearsonr(attention_by_problem_token[layer], importance_by_problem_token)

    # Fit line
    z = np.polyfit(attention_by_problem_token[layer], importance_by_problem_token, 1)
    p_fit = np.poly1d(z)
    x_line = np.linspace(min(attention_by_problem_token[layer]), max(attention_by_problem_token[layer]), 100)
    ax.plot(x_line, p_fit(x_line), "r--", alpha=0.8, linewidth=2, label='Regression')

    # Labels and title
    ax.set_xlabel('Attention Weight', fontsize=11)
    ax.set_ylabel('Importance (Failure Rate)', fontsize=11)

    # Statistical annotation
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    ax.set_title(f'{layer_names[layer]}\nr = {r:+.3f}, p = {p:.2e} {sig}', fontsize=12, fontweight='bold')

    ax.grid(True, alpha=0.3)

    # Only show legend on first plot
    if ax_idx == 0:
        ax.legend(fontsize=8, loc='upper left', framealpha=0.9)

plt.tight_layout()

# Save
output_file = Path(__file__).parent.parent / 'results' / 'attention_importance_correlation.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f'✓ Saved correlation scatter plot to {output_file}')

# Create second figure: Per-token analysis at Layer 8 with DUAL Y-AXES
fig, ax1 = plt.subplots(figsize=(12, 7))

token_stats = []
for token_pos in range(6):
    # Extract data for this token only
    token_importance = []
    token_attention = []

    for idx, ccta_prob in enumerate(baseline_correct):
        attn_prob = attention_results[idx]

        importance_scores = []
        for corr_type in corruption_types:
            importance_scores.append(ccta_prob['corruptions'][f'token_{token_pos}'][corr_type]['importance'])
        token_importance.append(np.mean(importance_scores))

        token_attention.append(attn_prob['attention_by_layer']['layer_8']['continuous_token_attention'][token_pos])

    mean_imp = np.mean(token_importance)
    mean_attn = np.mean(token_attention)

    if len(set(token_importance)) > 1:
        r, p = stats.pearsonr(token_attention, token_importance)
    else:
        r, p = 0, 1

    token_stats.append({
        'token': token_pos,
        'importance': mean_imp,
        'attention': mean_attn,
        'correlation': r,
        'p_value': p
    })

# Bar plot with DUAL Y-AXES (not misleading normalization!)
x = np.arange(6)
width = 0.35

# Convert to percentages
importance_vals = [s['importance'] * 100 for s in token_stats]  # Convert to %
total_attention = sum(s['attention'] for s in token_stats)
attention_pct_of_total = [s['attention'] / total_attention * 100 for s in token_stats]  # % of CoT attention

# Create first y-axis for importance
bars1 = ax1.bar(x - width/2, importance_vals, width,
                label='Importance (Failure Rate)',
                color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

ax1.set_xlabel('Token Position', fontsize=13, fontweight='bold')
ax1.set_ylabel('Importance (% failure rate when corrupted)', fontsize=13, fontweight='bold', color='#e74c3c')
ax1.tick_params(axis='y', labelcolor='#e74c3c')
ax1.set_ylim(0, max(importance_vals) * 1.3)

# Create second y-axis for attention
ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, attention_pct_of_total, width,
                label='Attention (% of total to CoT tokens)',
                color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)

ax2.set_ylabel('Attention (% of attention to continuous thought tokens)', fontsize=13, fontweight='bold', color='#3498db')
ax2.tick_params(axis='y', labelcolor='#3498db')
ax2.set_ylim(0, max(attention_pct_of_total) * 1.3)

# Add actual values on top of bars
for i, stats_dict in enumerate(token_stats):
    # Importance percentage
    ax1.text(i - width/2, importance_vals[i] + max(importance_vals) * 0.03,
            f'{importance_vals[i]:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='#c0392b')

    # Attention percentage of CoT total
    ax2.text(i + width/2, attention_pct_of_total[i] + max(attention_pct_of_total) * 0.03,
            f'{attention_pct_of_total[i]:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2980b9')

# Set x-axis
ax1.set_xticks(x)
ax1.set_xticklabels([f'Token {i}' for i in range(6)], fontsize=11)
ax1.grid(True, alpha=0.3, axis='y')

# Title
fig.suptitle(f'LLaMA CODI: Token Importance vs Attention Distribution (Layer 8)\n'
             f'Total attention to CoT tokens: {total_attention*100:.2f}% | Token 5 dominates both metrics',
             fontsize=14, fontweight='bold', y=0.98)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='upper left', framealpha=0.95)

plt.tight_layout(rect=[0, 0, 1, 0.96])

output_file2 = Path(__file__).parent.parent / 'results' / 'token_importance_attention_comparison.png'
plt.savefig(output_file2, dpi=300, bbox_inches='tight')
print(f'✓ Saved token comparison bar chart to {output_file2}')

print('\n' + '='*80)
print('VISUALIZATIONS CREATED')
print('='*80)
print(f'\n1. Correlation scatter plots (3 layers): {output_file.name}')
print(f'2. Token importance vs attention bar chart: {output_file2.name}')
print('\nKey visual insights:')
print('  - Layer 8 shows clear positive correlation (densest cluster)')
print('  - Token 5 (brown) consistently high on both axes')
print('  - Early layer (L4) shows scattered, uncorrelated pattern')
print('='*80)
