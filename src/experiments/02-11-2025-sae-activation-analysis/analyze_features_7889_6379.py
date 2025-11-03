#!/usr/bin/env python3
"""
Extract number correlations for features 7889 and 6379
"""

import json
from pathlib import Path
from collections import defaultdict

print("="*80)
print("ANALYZING FEATURES 7889 AND 6379: NUMBER CORRELATIONS")
print("="*80)

# Load Phase 1 full data
print("\n[1/3] Loading Phase 1 full data (214MB, may take a moment)...")
phase1_path = Path('/workspace/CoT_Exploration/src/experiments/02-11-2025-sae-activation-analysis/phase1_sae_activations_full.json')

with open(phase1_path, 'r') as f:
    phase1 = json.load(f)

print(f"    Loaded {phase1['n_examples']} examples")

# Get late layer results
results = phase1['results']
print(f"    Results type: {type(results)}")
print(f"    Number of result entries: {len(results) if isinstance(results, list) else 'N/A'}")

# The structure is a list of dicts, each with layer data
# Find late layer data
late_data = None
for layer_result in results:
    if 'late' in layer_result:
        late_data = layer_result['late']
        break

if late_data is None:
    print("ERROR: Could not find late layer data")
    exit(1)

print(f"    Late layer data found")

# Extract feature co-occurrence data
print("\n[2/3] Extracting token co-occurrences...")

# late_data should be a dict with feature IDs as keys
if '7889' not in late_data or '6379' not in late_data:
    print("ERROR: Features 7889 or 6379 not found in late layer data")
    print(f"Available keys sample: {list(late_data.keys())[:10]}")
    exit(1)

feat_7889_data = late_data['7889']
feat_6379_data = late_data['6379']

# Function to compute number correlations
def analyze_feature_numbers(feature_data, feature_id, total_fires):
    """Analyze number token correlations for a feature"""

    # Get token counts
    token_counts = feature_data.get('tokens', {})

    # Filter for numbers only
    number_correlations = []

    for token, count in token_counts.items():
        # Check if token is a number (digit or space+digit)
        token_stripped = token.strip()
        if token_stripped.isdigit():
            precision = count / total_fires if total_fires > 0 else 0
            number_correlations.append({
                'token': token,
                'number': token_stripped,
                'count': count,
                'precision': precision
            })

    # Sort by precision
    number_correlations.sort(key=lambda x: x['precision'], reverse=True)

    return number_correlations, token_counts


# Analyze Feature 7889
print(f"\n{'='*80}")
print(f"FEATURE 7889 (Rank 53/8192, Frequency 26.2%)")
print(f"{'='*80}")

feat_7889_fires = 11753  # From earlier analysis
num_corr_7889, all_tokens_7889 = analyze_feature_numbers(feat_7889_data, 7889, feat_7889_fires)

print(f"\nTotal fires: {feat_7889_fires:,}")
print(f"Unique tokens co-occurring: {len(all_tokens_7889):,}")
print(f"Number tokens found: {len(num_corr_7889)}")

print(f"\nTop 20 NUMBER correlations:")
print(f"{'Rank':<6} {'Number':<10} {'Token':<15} {'Precision':<12} {'Count':<8}")
print("-"*60)

for rank, corr in enumerate(num_corr_7889[:20], 1):
    print(f"{rank:<6} {corr['number']:<10} {repr(corr['token']):<15} {corr['precision']*100:>6.2f}%     {corr['count']:>6,}")

# Show top non-number tokens for context
print(f"\nTop 15 ALL tokens (for context):")
sorted_all = sorted(all_tokens_7889.items(), key=lambda x: x[1], reverse=True)[:15]
for token, count in sorted_all:
    precision = count / feat_7889_fires * 100
    print(f"  {repr(token):>15s}: {count:>5,} ({precision:5.2f}%)")


# Analyze Feature 6379
print(f"\n{'='*80}")
print(f"FEATURE 6379 (Rank 15/8192, Frequency 62.9%)")
print(f"{'='*80}")

feat_6379_fires = 28220  # From earlier analysis
num_corr_6379, all_tokens_6379 = analyze_feature_numbers(feat_6379_data, 6379, feat_6379_fires)

print(f"\nTotal fires: {feat_6379_fires:,}")
print(f"Unique tokens co-occurring: {len(all_tokens_6379):,}")
print(f"Number tokens found: {len(num_corr_6379)}")

print(f"\nTop 20 NUMBER correlations:")
print(f"{'Rank':<6} {'Number':<10} {'Token':<15} {'Precision':<12} {'Count':<8}")
print("-"*60)

for rank, corr in enumerate(num_corr_6379[:20], 1):
    print(f"{rank:<6} {corr['number']:<10} {repr(corr['token']):<15} {corr['precision']*100:>6.2f}%     {corr['count']:>6,}")

# Show top non-number tokens for context
print(f"\nTop 15 ALL tokens (for context):")
sorted_all = sorted(all_tokens_6379.items(), key=lambda x: x[1], reverse=True)[:15]
for token, count in sorted_all:
    precision = count / feat_6379_fires * 100
    print(f"  {repr(token):>15s}: {count:>5,} ({precision:5.2f}%)")


# Save results
print(f"\n[3/3] Saving results...")
output = {
    'features_analyzed': [7889, 6379],
    'features': {
        '7889': {
            'rank': 53,
            'frequency': 0.262121,
            'total_fires': feat_7889_fires,
            'unique_tokens': len(all_tokens_7889),
            'number_correlations': num_corr_7889[:30],
            'top_all_tokens': [{'token': t, 'count': c, 'precision': c/feat_7889_fires}
                               for t, c in sorted_all[:20]]
        },
        '6379': {
            'rank': 15,
            'frequency': 0.629377,
            'total_fires': feat_6379_fires,
            'unique_tokens': len(all_tokens_6379),
            'number_correlations': num_corr_6379[:30],
            'top_all_tokens': [{'token': t, 'count': c, 'precision': c/feat_6379_fires}
                               for t, c in sorted_all[:20]]
        }
    }
}

output_path = Path('./features_7889_6379_analysis.json')
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Results saved to: {output_path}")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
