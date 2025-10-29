#!/usr/bin/env python3
"""
Problem Sensitivity Analysis - Story 4

Identify which problem characteristics correlate with intervention impact.

Usage:
    python 2_analyze_problem_sensitivity.py

Input:
    - GSM8K dataset (for problem features)
    - Error taxonomy results (for impact scores)

Output:
    - results/problem_sensitivity.json
    - results/problem_sensitivity_plots.png
"""
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent / 'utils'))
from feature_extractors import extract_features, print_feature_summary


def load_gsm8k_with_features(n_problems=1319):
    """Load GSM8K and extract features."""
    print("Loading GSM8K dataset and extracting features...")
    dataset = load_dataset('gsm8k', 'main', split='test')
    problems = dataset.select(range(n_problems))

    features_list = []
    for i, problem in enumerate(problems):
        features = extract_features(problem['question'], problem['answer'])
        features['problem_id'] = i
        features_list.append(features)

    print(f"✓ Extracted features for {len(features_list)} problems")
    return features_list


def load_error_taxonomy_results():
    """Load classified error results."""
    results_dir = Path(__file__).parent / 'results'
    taxonomy_file = results_dir / 'error_taxonomy_full.json'

    if not taxonomy_file.exists():
        print(f"⚠️  Error taxonomy file not found: {taxonomy_file}")
        print("Please run 1_analyze_error_taxonomy.py first")
        return None

    with open(taxonomy_file) as f:
        data = json.load(f)

    print(f"✓ Loaded error taxonomy results")
    return data


def compute_impact_scores(error_taxonomy):
    """
    Compute intervention impact for each problem.

    Impact = baseline_correct - intervention_correct (binary per problem)
    """
    baseline_results = error_taxonomy.get('baseline', [])
    ct0_results = error_taxonomy.get('ct0_blocked', [])
    ct4_results = error_taxonomy.get('ct4_blocked', [])

    n_problems = min(len(baseline_results), len(ct0_results), len(ct4_results))

    impacts = []
    for i in range(n_problems):
        baseline_correct = baseline_results[i]['correct']
        ct0_correct = ct0_results[i]['correct']
        ct4_correct = ct4_results[i]['correct']

        impact = {
            'problem_id': i,
            'baseline_correct': baseline_correct,
            'ct0_correct': ct0_correct,
            'ct4_correct': ct4_correct,
            'ct0_impact': int(baseline_correct) - int(ct0_correct),  # 1 if broke, 0 if no change, -1 if fixed
            'ct4_impact': int(baseline_correct) - int(ct4_correct),
            'ct0_error_type': ct0_results[i]['error_type'],
            'ct4_error_type': ct4_results[i]['error_type'],
        }

        impacts.append(impact)

    print(f"✓ Computed impact scores for {n_problems} problems")
    return impacts


def merge_features_and_impacts(features_list, impacts):
    """Merge problem features with impact scores."""
    merged = []

    for i in range(len(impacts)):
        if i < len(features_list):
            merged_item = {**features_list[i], **impacts[i]}
            merged.append(merged_item)

    print(f"✓ Merged {len(merged)} records")
    return merged


def analyze_correlations(data):
    """Analyze correlations between features and intervention impact."""
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)

    # Numeric features to analyze
    numeric_features = ['n_tokens', 'n_operations', 'n_operation_types', 'max_number']

    # Compute correlations with CT0 impact
    print("\nCT0 Impact Correlations:")
    print(f"{'Feature':<20s} {'Pearson r':>12s} {'p-value':>10s} {'Spearman ρ':>12s}")
    print("-" * 60)

    ct0_impacts = [d['ct0_impact'] for d in data]

    correlations = {}
    for feature in numeric_features:
        feature_values = [d[feature] for d in data]

        pearson_r, pearson_p = pearsonr(feature_values, ct0_impacts)
        spearman_rho, spearman_p = spearmanr(feature_values, ct0_impacts)

        correlations[feature] = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_rho': spearman_rho,
            'spearman_p': spearman_p
        }

        sig = "***" if pearson_p < 0.001 else ("**" if pearson_p < 0.01 else ("*" if pearson_p < 0.05 else ""))

        print(f"{feature:<20s} {pearson_r:>11.3f}{sig} {pearson_p:>10.4f} {spearman_rho:>12.3f}")

    return correlations


def analyze_by_problem_type(data):
    """Analyze intervention impact by problem type."""
    print("\n" + "="*80)
    print("IMPACT BY PROBLEM TYPE")
    print("="*80)

    # Boolean features
    boolean_features = ['multi_step', 'has_division', 'has_multiplication', 'has_fractions']

    for feature in boolean_features:
        true_group = [d for d in data if d[feature]]
        false_group = [d for d in data if not d[feature]]

        true_ct0_impact = np.mean([d['ct0_impact'] for d in true_group])
        false_ct0_impact = np.mean([d['ct0_impact'] for d in false_group])

        print(f"\n{feature}:")
        print(f"  {feature}=True  (n={len(true_group):4d}): CT0 impact = {true_ct0_impact:.3f}")
        print(f"  {feature}=False (n={len(false_group):4d}): CT0 impact = {false_ct0_impact:.3f}")
        print(f"  Difference: {true_ct0_impact - false_ct0_impact:+.3f}")


def identify_high_impact_problems(data, threshold=0.5):
    """Identify problems with high CT0 impact."""
    high_impact = [d for d in data if d['ct0_impact'] > threshold]
    low_impact = [d for d in data if abs(d['ct0_impact']) <= threshold]

    print("\n" + "="*80)
    print(f"HIGH IMPACT PROBLEMS (CT0 impact > {threshold})")
    print("="*80)

    print(f"\nHigh impact: {len(high_impact)} problems")
    print(f"Low impact: {len(low_impact)} problems")

    if high_impact:
        print("\nHigh Impact Problem Characteristics:")
        for feature in ['n_tokens', 'n_operations', 'n_operation_types', 'max_number']:
            high_values = [d[feature] for d in high_impact]
            low_values = [d[feature] for d in low_impact]
            print(f"  {feature:20s}: High={np.mean(high_values):6.1f}, Low={np.mean(low_values):6.1f}")

    return high_impact, low_impact


def visualize_correlations(data, correlations, output_dir):
    """Create visualizations of problem sensitivity."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    numeric_features = ['n_tokens', 'n_operations', 'n_operation_types', 'max_number']

    for idx, feature in enumerate(numeric_features):
        ax = axes[idx // 2, idx % 2]

        x = [d[feature] for d in data]
        y = [d['ct0_impact'] for d in data]

        ax.scatter(x, y, alpha=0.5, s=30)
        ax.set_xlabel(feature.replace('_', ' ').title())
        ax.set_ylabel('CT0 Impact (baseline - ct0)')
        ax.set_title(f'{feature} vs CT0 Impact\nPearson r={correlations[feature]["pearson_r"]:.3f}')
        ax.grid(alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

        # Add trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(sorted(x), p(sorted(x)), "r--", alpha=0.8, linewidth=2)

    plt.tight_layout()

    output_path = output_dir / 'problem_sensitivity_scatter.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved scatter plots: {output_path}")

    plt.close()

    # Create bar chart for boolean features
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    boolean_features = ['multi_step', 'has_division', 'has_multiplication', 'has_fractions']
    true_impacts = []
    false_impacts = []

    for feature in boolean_features:
        true_group = [d['ct0_impact'] for d in data if d[feature]]
        false_group = [d['ct0_impact'] for d in data if not d[feature]]
        true_impacts.append(np.mean(true_group))
        false_impacts.append(np.mean(false_group))

    x = np.arange(len(boolean_features))
    width = 0.35

    ax.bar(x - width/2, true_impacts, width, label='Feature=True')
    ax.bar(x + width/2, false_impacts, width, label='Feature=False')

    ax.set_ylabel('Mean CT0 Impact')
    ax.set_title('CT0 Impact by Problem Type')
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('_', '\n') for f in boolean_features])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    output_path = output_dir / 'problem_sensitivity_types.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved type comparison: {output_path}")

    plt.close()


def save_results(data, correlations, high_impact, low_impact, output_dir):
    """Save analysis results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full data
    output_file = output_dir / 'problem_sensitivity_full.json'
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ Saved full data: {output_file}")

    # Save summary
    summary = {
        'correlations': correlations,
        'high_impact_problems': {
            'count': len(high_impact),
            'mean_features': {
                'n_tokens': np.mean([d['n_tokens'] for d in high_impact]) if high_impact else 0,
                'n_operations': np.mean([d['n_operations'] for d in high_impact]) if high_impact else 0,
            }
        },
        'low_impact_problems': {
            'count': len(low_impact),
            'mean_features': {
                'n_tokens': np.mean([d['n_tokens'] for d in low_impact]) if low_impact else 0,
                'n_operations': np.mean([d['n_operations'] for d in low_impact]) if low_impact else 0,
            }
        }
    }

    summary_file = output_dir / 'problem_sensitivity_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary: {summary_file}")


def main():
    print("="*80)
    print("PROBLEM SENSITIVITY ANALYSIS")
    print("="*80)

    # Load GSM8K with features
    features_list = load_gsm8k_with_features(n_problems=1319)

    # Load error taxonomy
    error_taxonomy = load_error_taxonomy_results()
    if not error_taxonomy:
        return

    # Compute impact scores
    impacts = compute_impact_scores(error_taxonomy)

    # Merge data
    data = merge_features_and_impacts(features_list, impacts)

    # Analyze correlations
    correlations = analyze_correlations(data)

    # Analyze by problem type
    analyze_by_problem_type(data)

    # Identify high/low impact problems
    high_impact, low_impact = identify_high_impact_problems(data, threshold=0.5)

    # Save results
    output_dir = Path(__file__).parent / 'results'
    save_results(data, correlations, high_impact, low_impact, output_dir)

    # Visualize
    visualize_correlations(data, correlations, output_dir)

    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
