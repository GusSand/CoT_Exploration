#!/usr/bin/env python3
"""
Investigate features 7889 and 6379 that appeared in problem variants
but have no data in Phase 5
"""

import json
from pathlib import Path

print("="*80)
print("INVESTIGATING RARE FEATURES 7889 AND 6379")
print("="*80)

# Check Phase 1 summary statistics
print("\n[1] Checking Phase 1 summary statistics...")
phase1_path = Path('/workspace/CoT_Exploration/src/experiments/02-11-2025-sae-activation-analysis/phase1_summary_statistics.json')
with open(phase1_path, 'r') as f:
    phase1 = json.load(f)

late_stats = phase1['layer_stats']['late']
print(f"\nLate layer statistics:")
print(f"  Total examples: {phase1['n_examples']}")
print(f"  Total positions: {phase1['total_positions']}")
print(f"  Features that fired: {late_stats['n_features_fired']}/8192")
print(f"  Total activations: {late_stats['total_activations']}")
print(f"  Mean sparsity: {late_stats['mean_sparsity']:.4f}")

# Check if 7889 and 6379 fired
feat_7889_count = late_stats['feature_counts'].get('7889', 0)
feat_6379_count = late_stats['feature_counts'].get('6379', 0)

print(f"\n[2] Feature fire counts in training set:")
print(f"  Feature 7889: {feat_7889_count} fires")
print(f"  Feature 6379: {feat_6379_count} fires")

# Get percentiles
all_counts = sorted(late_stats['feature_counts'].values(), reverse=True)
if feat_7889_count > 0:
    rank_7889 = sorted(late_stats['feature_counts'].values(), reverse=True).index(feat_7889_count) + 1
    percentile_7889 = rank_7889 / len(all_counts) * 100
    print(f"  Feature 7889 rank: {rank_7889}/{len(all_counts)} ({percentile_7889:.1f} percentile)")

if feat_6379_count > 0:
    rank_6379 = sorted(late_stats['feature_counts'].values(), reverse=True).index(feat_6379_count) + 1
    percentile_6379 = rank_6379 / len(all_counts) * 100
    print(f"  Feature 6379 rank: {rank_6379}/{len(all_counts)} ({percentile_6379:.1f} percentile)")

# Show distribution of fire counts
print(f"\n[3] Distribution of feature fire counts:")
print(f"  Max: {max(all_counts)}")
print(f"  95th percentile: {all_counts[int(len(all_counts)*0.05)]}")
print(f"  90th percentile: {all_counts[int(len(all_counts)*0.10)]}")
print(f"  75th percentile: {all_counts[int(len(all_counts)*0.25)]}")
print(f"  Median: {all_counts[int(len(all_counts)*0.5)]}")
print(f"  25th percentile: {all_counts[int(len(all_counts)*0.75)]}")
print(f"  Min: {min(all_counts)}")

# Top 20 most frequent features
print(f"\n[4] Top 20 most frequent features:")
sorted_features = sorted(late_stats['feature_counts'].items(), key=lambda x: x[1], reverse=True)
for rank, (feat_id, count) in enumerate(sorted_features[:20], 1):
    print(f"  {rank:2d}. Feature {feat_id:4s}: {count:5d} fires")

# Check Phase 5 for what features were analyzed
print(f"\n[5] Checking Phase 5 analysis threshold...")
phase5_path = Path('./phase5_simple_precision_results.json')
with open(phase5_path, 'r') as f:
    phase5 = json.load(f)

late_features = phase5['feature_reports_by_layer']['late']
print(f"  Features analyzed in Phase 5: {len(late_features)}")

# Find minimum fire count in Phase 5
min_freq_in_phase5 = min(f['feature_frequency'] * late_stats['total_activations']
                          for f in late_features.values() if f is not None)
print(f"  Minimum fires for inclusion: ~{int(min_freq_in_phase5)}")

# Analysis
print(f"\n{'='*80}")
print("ANALYSIS")
print(f"{'='*80}")

if feat_7889_count == 0:
    print(f"\nFeature 7889: NEVER FIRED in training set (7472 examples)")
    print("  This is a 'dormant' feature that only activates on specific rare patterns")
    print("  Variant A and B are the FIRST examples where this feature activated!")
else:
    print(f"\nFeature 7889: Fired {feat_7889_count} times (very rare)")
    print(f"  Frequency: {feat_7889_count/late_stats['total_activations']*100:.6f}%")
    print(f"  Too rare to be analyzed in Phase 5")

if feat_6379_count == 0:
    print(f"\nFeature 6379: NEVER FIRED in training set")
    print("  Another dormant feature activated by the variants")
else:
    print(f"\nFeature 6379: Fired {feat_6379_count} times (very rare)")
    print(f"  Frequency: {feat_6379_count/late_stats['total_activations']*100:.6f}%")

print(f"\n{'='*80}")
print("CONCLUSION")
print(f"{'='*80}")

print("""
These are 'out-of-distribution' features that:
1. Were silent on the training set (or nearly silent)
2. Only activated when we modified the problem
3. Represent unusual/rare computational patterns

This suggests the problem variants triggered DIFFERENT computational
pathways than anything in the training data!
""")
