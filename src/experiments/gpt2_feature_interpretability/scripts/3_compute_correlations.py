"""
Compute feature-token correlations using chi-squared tests.

For each feature:
1. Build contingency table: feature active/inactive × token present/absent
2. Perform chi-squared test (p < 0.01)
3. Calculate enrichment score (how much more likely is token when feature active)
4. Filter significant correlations

Monosemantic criteria:
- p-value < 0.01 (statistically significant)
- Enrichment ≥ 2.0 (token 2× more likely when feature active)
- Feature activates in ≥20 samples (minimum data)

Output: gpt2_feature_token_correlations.json
"""

import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from scipy.stats import chi2_contingency
from collections import defaultdict


def compute_contingency_table(feature_active, token_present):
    """
    Build 2×2 contingency table.

    Args:
        feature_active: Boolean array (N,) - feature is active
        token_present: Boolean array (N,) - token is present

    Returns:
        [[a, b], [c, d]] where:
          a = feature active, token present
          b = feature active, token absent
          c = feature inactive, token present
          d = feature inactive, token absent
    """
    a = np.sum(feature_active & token_present)
    b = np.sum(feature_active & ~token_present)
    c = np.sum(~feature_active & token_present)
    d = np.sum(~feature_active & ~token_present)

    return np.array([[a, b], [c, d]])


def compute_enrichment(table):
    """
    Compute enrichment score.

    Enrichment = P(token | feature active) / P(token | feature inactive)
                = (a / (a+b)) / (c / (c+d))

    Returns:
        enrichment score (float)
    """
    a, b = table[0]
    c, d = table[1]

    if (a + b) == 0 or (c + d) == 0:
        return 0.0

    p_token_given_active = a / (a + b)
    p_token_given_inactive = c / (c + d)

    if p_token_given_inactive == 0:
        return float('inf') if p_token_given_active > 0 else 1.0

    return p_token_given_active / p_token_given_inactive


def analyze_all_features():
    """Compute correlations for all features."""
    print("="*80)
    print("COMPUTING FEATURE-TOKEN CORRELATIONS")
    print("="*80)

    # Load features
    print("\n[1/3] Loading features...")
    features_data = torch.load(
        'src/experiments/gpt2_feature_interpretability/data/gpt2_extracted_features.pt',
        weights_only=False
    )

    all_features = features_data['features']  # Dict[(layer, pos)] -> tensor (N, 512)
    metadata = features_data['metadata']

    # Load CoT tokens
    print("[2/3] Loading CoT tokens...")
    with open('src/experiments/gpt2_feature_interpretability/data/gpt2_cot_tokens.json', 'r') as f:
        token_data = json.load(f)

    token_to_problems = token_data['token_to_problems']
    problem_to_tokens = token_data['problem_to_tokens']
    problem_to_tokens = {int(k): v for k, v in problem_to_tokens.items()}  # Convert keys to int

    print(f"  Tokens: {len(token_to_problems)}")

    # Convert problem IDs to indices for efficient lookup
    problem_ids = np.array(metadata['problem_ids'])

    # Build token presence matrix for faster lookup
    print("\nBuilding token presence matrix...")
    token_presence = {}
    for token, prob_list in tqdm(token_to_problems.items(), desc="Processing tokens"):
        presence = np.isin(problem_ids, prob_list)
        token_presence[token] = presence

    # Analyze correlations
    print("\n[3/3] Computing correlations...")

    # Criteria
    MIN_ACTIVATIONS = 20
    P_VALUE_THRESHOLD = 0.01
    ENRICHMENT_THRESHOLD = 2.0

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    total_features_analyzed = 0
    total_correlations_found = 0

    for (layer, position), features_tensor in tqdm(all_features.items(), desc="Processing SAEs"):
        # Get mask for this layer-position
        layer_mask = (np.array(metadata['layers']) == layer) & (np.array(metadata['positions']) == position)

        for feature_id in range(features_tensor.shape[1]):
            # Get feature activations
            feature_acts = features_tensor[:, feature_id].numpy()
            feature_active = feature_acts != 0

            # Skip if too few activations
            num_active = feature_active.sum()
            if num_active < MIN_ACTIVATIONS:
                continue

            total_features_analyzed += 1

            # Test correlations with all tokens
            feature_correlations = []

            for token, token_present in token_presence.items():
                # Get token presence for this layer-position samples
                token_present_subset = token_present[layer_mask]

                # Build contingency table
                table = compute_contingency_table(feature_active, token_present_subset)

                # Skip if table is degenerate
                if table.min() == 0 or table.sum() < 10:
                    continue

                # Chi-squared test
                try:
                    chi2, p_value, dof, expected = chi2_contingency(table)

                    if p_value < P_VALUE_THRESHOLD:
                        enrichment = compute_enrichment(table)

                        if enrichment >= ENRICHMENT_THRESHOLD:
                            feature_correlations.append({
                                'token': token,
                                'p_value': float(p_value),
                                'enrichment': float(enrichment),
                                'chi2': float(chi2),
                                'active_with_token': int(table[0, 0]),
                                'active_without_token': int(table[0, 1]),
                                'inactive_with_token': int(table[1, 0]),
                                'inactive_without_token': int(table[1, 1]),
                            })
                            total_correlations_found += 1

                except (ValueError, ZeroDivisionError):
                    continue

            # Store if any significant correlations found
            if feature_correlations:
                # Sort by enrichment (descending)
                feature_correlations.sort(key=lambda x: x['enrichment'], reverse=True)

                results[layer][position][feature_id] = {
                    'num_activations': int(num_active),
                    'activation_rate': float(num_active / len(feature_active)),
                    'num_correlations': len(feature_correlations),
                    'correlations': feature_correlations[:20],  # Keep top 20
                }

    # Summary
    num_interpretable_features = sum(
        len(results[l][p]) for l in results for p in results[l]
    )

    print(f"\n✓ Analysis complete!")
    print(f"  Features analyzed: {total_features_analyzed:,}")
    print(f"  Interpretable features: {num_interpretable_features:,} ({num_interpretable_features/36864*100:.1f}%)")
    print(f"  Total correlations found: {total_correlations_found:,}")
    print(f"  Avg correlations per interpretable feature: {total_correlations_found/max(num_interpretable_features, 1):.1f}")

    # Save results
    output_path = Path('src/experiments/gpt2_feature_interpretability/data/gpt2_feature_token_correlations.json')
    print(f"\nSaving to: {output_path}")

    output_data = {
        'metadata': {
            'total_features': 36864,  # 72 SAEs × 512 features
            'features_analyzed': total_features_analyzed,
            'interpretable_features': num_interpretable_features,
            'interpretability_rate': num_interpretable_features / 36864,
            'total_correlations': total_correlations_found,
            'criteria': {
                'min_activations': MIN_ACTIVATIONS,
                'p_value_threshold': P_VALUE_THRESHOLD,
                'enrichment_threshold': ENRICHMENT_THRESHOLD,
            }
        },
        'correlations': results,  # correlations[layer][position][feature_id]
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"✓ Saved ({size_mb:.1f} MB)")

    print("\n" + "="*80)
    print("CORRELATION ANALYSIS COMPLETE!")
    print("="*80)
    print(f"  Interpretable features: {num_interpretable_features:,} / 36,864 ({num_interpretable_features/36864*100:.1f}%)")
    print(f"  Output: {output_path}")
    print("="*80)


if __name__ == '__main__':
    analyze_all_features()
