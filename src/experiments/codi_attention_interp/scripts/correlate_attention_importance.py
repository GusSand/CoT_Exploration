#!/usr/bin/env python3
"""
Quick correlation analysis: Does attention predict importance?

Correlates attention weights with token importance from CCTA experiment.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

# Load CCTA results (importance data)
ccta_file = Path(__file__).parent.parent / 'results' / 'ccta_full_results_100.json'
with open(ccta_file) as f:
    ccta_results = json.load(f)

# Load attention weights
attention_file = Path(__file__).parent.parent / 'results' / 'attention_weights_100.json'
with open(attention_file) as f:
    attention_results = json.load(f)

print('='*80)
print('ATTENTION-IMPORTANCE CORRELATION ANALYSIS')
print('='*80)

# Extract importance scores (averaged across all corruption methods)
baseline_correct = [r for r in ccta_results if r['baseline']['correct']]
corruption_types = ['zero', 'gauss_0.1', 'gauss_0.5', 'gauss_1.0', 'gauss_2.0', 'random', 'shuffle']

print(f'\nDataset: {len(baseline_correct)} problems with correct baseline')

# For each problem, compute per-token importance (averaged across corruption methods)
importance_by_problem_token = []

for idx, ccta_prob in enumerate(baseline_correct):
    for token_pos in range(6):
        # Compute mean importance across all corruption methods
        importance_scores = []
        for corr_type in corruption_types:
            importance_scores.append(ccta_prob['corruptions'][f'token_{token_pos}'][corr_type]['importance'])
        mean_importance = np.mean(importance_scores)
        importance_by_problem_token.append(mean_importance)

# Get attention for each layer
attention_by_problem_token = {4: [], 8: [], 14: []}
for idx, attn_prob in enumerate(attention_results):
    for token_pos in range(6):
        for layer in [4, 8, 14]:
            attn_weight = attn_prob['attention_by_layer'][f'layer_{layer}']['continuous_token_attention'][token_pos]
            attention_by_problem_token[layer].append(attn_weight)

# Compute correlations for each layer
print('\n' + '='*80)
print('CORRELATION RESULTS (600 data points: 100 problems × 6 tokens)')
print('='*80)

for layer in [4, 8, 14]:
    r, p = stats.pearsonr(attention_by_problem_token[layer], importance_by_problem_token)

    significance = ''
    if p < 0.001:
        significance = '⭐⭐⭐ Highly significant'
    elif p < 0.01:
        significance = '⭐⭐ Very significant'
    elif p < 0.05:
        significance = '⭐ Significant'
    else:
        significance = '❌ Not significant'

    print(f'\nLayer {layer:2d} ({"early" if layer==4 else "middle" if layer==8 else "late"}):')
    print(f'  Correlation (r): {r:+.4f}')
    print(f'  P-value:         {p:.6f}')
    print(f'  Result:          {significance}')

# Per-token correlation
print('\n' + '='*80)
print('PER-TOKEN CORRELATION (Layer 8 - Middle Layer)')
print('='*80)

print('\nToken  Importance  Attention   Correlation  P-value  ')
print('-'*80)

for token_pos in range(6):
    # Extract importance and attention for this specific token across all problems
    token_importance = []
    token_attention = []

    for idx, ccta_prob in enumerate(baseline_correct):
        attn_prob = attention_results[idx]

        # Mean importance across corruption methods
        importance_scores = []
        for corr_type in corruption_types:
            importance_scores.append(ccta_prob['corruptions'][f'token_{token_pos}'][corr_type]['importance'])
        token_importance.append(np.mean(importance_scores))

        # Attention at layer 8
        token_attention.append(attn_prob['attention_by_layer']['layer_8']['continuous_token_attention'][token_pos])

    mean_imp = np.mean(token_importance)
    mean_attn = np.mean(token_attention)

    if len(set(token_importance)) > 1:  # Can only correlate if there's variance
        r, p = stats.pearsonr(token_attention, token_importance)
        print(f'Token {token_pos}   {100*mean_imp:>6.1f}%     {mean_attn:>7.4f}      {r:+.4f}      {p:.4f}')
    else:
        print(f'Token {token_pos}   {100*mean_imp:>6.1f}%     {mean_attn:>7.4f}      (no variance)')

# Summary
print('\n' + '='*80)
print('SUMMARY')
print('='*80)

layer8_r, layer8_p = stats.pearsonr(attention_by_problem_token[8], importance_by_problem_token)

if layer8_p < 0.05:
    print('\n✅ VALIDATED: Middle layer attention significantly predicts token importance!')
    print(f'   r = {layer8_r:+.4f}, p = {layer8_p:.6f}')
    print('   This confirms that attention patterns are mechanistically meaningful.')
else:
    print('\n❌ NOT VALIDATED: Middle layer attention does NOT predict importance.')
    print(f'   r = {layer8_r:+.4f}, p = {layer8_p:.6f}')
    print('   Attention may not be a reliable indicator of causal importance.')

print('\n' + '='*80)
