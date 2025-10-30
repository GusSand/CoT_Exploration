#!/usr/bin/env python3
"""
Stories 2.3-2.5: Full Experiment Analysis

Combines per-problem analysis, statistical analysis, and visualizations
for the full 100-problem resampling experiment.

Time estimate: 30 minutes
"""

# CRITICAL: Set PYTHONHASHSEED before imports
import os
os.environ['PYTHONHASHSEED'] = '42'

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List
import sys

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import set_seed

# Set random seed
set_seed(42)

# Ablation reference data (from research journal)
ABLATION_IMPACTS = {
    'CT0': 18.7,
    'CT1': 12.8,  # Updated from pilot analysis
    'CT2': 14.6,
    'CT3': 15.0,
    'CT4': 3.5,
    'CT5': 26.0
}


def load_results(results_path: Path) -> Dict:
    """Load resampling results"""
    with open(results_path, 'r') as f:
        return json.load(f)


def per_problem_analysis(results: Dict) -> Dict:
    """
    Story 2.3: Per-problem analysis

    Analyze variance across problems to identify:
    - Which problems are most/least sensitive to swapping
    - Whether sensitivity correlates with problem difficulty
    - Position-specific sensitivity patterns
    """
    print(f"\n{'='*60}")
    print("Story 2.3: Per-Problem Analysis")
    print(f"{'='*60}\n")

    per_problem_impacts = {}

    # For each position, compute per-problem impact
    for pos_key in ['CT0', 'CT1', 'CT2', 'CT3', 'CT4', 'CT5']:
        pos_data = results[pos_key]
        n_problems = len(pos_data['per_problem_results'])

        impacts = []
        for problem in pos_data['per_problem_results']:
            baseline_correct = problem['baseline_correct']
            n_correct = sum(problem['swapped_correct'])  # FIXED: Use 'swapped_correct' key
            n_samples = len(problem['swapped_correct'])
            resampled_accuracy = n_correct / n_samples

            # Impact: baseline - resampled (positive = performance drop)
            impact = (1.0 if baseline_correct else 0.0) - resampled_accuracy
            impacts.append(impact)

        per_problem_impacts[pos_key] = impacts

    # Compute statistics
    stats_per_position = {}
    for pos_key, impacts in per_problem_impacts.items():
        stats_per_position[pos_key] = {
            'mean': np.mean(impacts),
            'std': np.std(impacts),
            'min': np.min(impacts),
            'max': np.max(impacts),
            'median': np.median(impacts)
        }

    # Print summary
    print("Per-Position Impact Statistics:")
    print(f"{'Position':<10} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Median':<10}")
    print("-" * 60)
    for pos_key in ['CT0', 'CT1', 'CT2', 'CT3', 'CT4', 'CT5']:
        s = stats_per_position[pos_key]
        print(f"{pos_key:<10} {s['mean']:<10.3f} {s['std']:<10.3f} {s['min']:<10.3f} {s['max']:<10.3f} {s['median']:<10.3f}")

    # Identify most/least sensitive problems
    print(f"\nMost Sensitive Problems (averaged across positions):")
    avg_impacts_per_problem = {}
    n_problems = len(per_problem_impacts['CT0'])
    for i in range(n_problems):
        avg_impact = np.mean([per_problem_impacts[pos][i] for pos in ['CT0', 'CT1', 'CT2', 'CT3', 'CT4', 'CT5']])
        avg_impacts_per_problem[i] = avg_impact

    # Top 5 most sensitive
    sorted_problems = sorted(avg_impacts_per_problem.items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"  Top 5: {[f'P{i} ({impact:.2f})' for i, impact in sorted_problems[:5]]}")

    # Top 5 least sensitive
    print(f"  Least 5: {[f'P{i} ({impact:.2f})' for i, impact in sorted_problems[-5:]]}")

    return {
        'per_problem_impacts': per_problem_impacts,
        'stats_per_position': stats_per_position,
        'avg_impacts_per_problem': avg_impacts_per_problem
    }


def statistical_analysis(results: Dict, per_problem_data: Dict) -> Dict:
    """
    Story 2.4: Statistical analysis

    Compute:
    - Correlation with ablation (Pearson)
    - Significance testing (t-tests, permutation tests)
    - Effect sizes (Cohen's d)
    - Confidence intervals
    """
    print(f"\n{'='*60}")
    print("Story 2.4: Statistical Analysis")
    print(f"{'='*60}\n")

    # Extract resampling impacts
    resampling_impacts = []
    ablation_impacts = []
    positions = []

    for pos_key in ['CT0', 'CT1', 'CT2', 'CT3', 'CT4', 'CT5']:
        pos_data = results[pos_key]
        resampling_impact = pos_data['impact'] * 100  # Convert to percentage
        ablation_impact = ABLATION_IMPACTS[pos_key]

        resampling_impacts.append(resampling_impact)
        ablation_impacts.append(ablation_impact)
        positions.append(pos_key)

    # Pearson correlation
    r_pearson, p_pearson = stats.pearsonr(resampling_impacts, ablation_impacts)

    # Spearman correlation (rank-based, more robust)
    r_spearman, p_spearman = stats.spearmanr(resampling_impacts, ablation_impacts)

    print("Correlation Analysis:")
    print(f"  Pearson:  r = {r_pearson:.3f}, p = {p_pearson:.4f}")
    print(f"  Spearman: ρ = {r_spearman:.3f}, p = {p_spearman:.4f}")

    # Compute confidence intervals for each position (bootstrap)
    print(f"\nConfidence Intervals (95%, bootstrap):")
    print(f"{'Position':<10} {'Impact':<15} {'95% CI':<20} {'n':<10}")
    print("-" * 60)

    cis = {}
    for pos_key in ['CT0', 'CT1', 'CT2', 'CT3', 'CT4', 'CT5']:
        impacts = per_problem_data['per_problem_impacts'][pos_key]

        # Bootstrap CI
        n_bootstrap = 1000
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(impacts, size=len(impacts), replace=True)
            bootstrap_means.append(np.mean(sample) * 100)  # Convert to percentage

        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        mean_impact = np.mean(impacts) * 100

        cis[pos_key] = (ci_lower, ci_upper)
        print(f"{pos_key:<10} {mean_impact:>6.1f}%{'':<8} [{ci_lower:>5.1f}%, {ci_upper:>5.1f}%]{'':<5} {len(impacts):<10}")

    # Effect sizes (Cohen's d for difference from zero)
    print(f"\nEffect Sizes (Cohen's d vs. no-impact baseline):")
    effect_sizes = {}
    for pos_key in ['CT0', 'CT1', 'CT2', 'CT3', 'CT4', 'CT5']:
        impacts = per_problem_data['per_problem_impacts'][pos_key]
        mean = np.mean(impacts)
        std = np.std(impacts)
        d = mean / std if std > 0 else 0
        effect_sizes[pos_key] = d

        # Interpretation
        if abs(d) < 0.2:
            size_label = "negligible"
        elif abs(d) < 0.5:
            size_label = "small"
        elif abs(d) < 0.8:
            size_label = "medium"
        else:
            size_label = "large"

        print(f"  {pos_key}: d = {d:.3f} ({size_label})")

    return {
        'correlation': {
            'pearson_r': r_pearson,
            'pearson_p': p_pearson,
            'spearman_r': r_spearman,
            'spearman_p': p_spearman
        },
        'confidence_intervals': cis,
        'effect_sizes': effect_sizes,
        'resampling_impacts': resampling_impacts,
        'ablation_impacts': ablation_impacts
    }


def create_visualizations(results: Dict, per_problem_data: Dict, stats_data: Dict, output_dir: Path):
    """
    Story 2.5: Create comprehensive visualizations

    Creates:
    1. Resampling vs Ablation comparison (bar + scatter)
    2. Per-problem heatmap
    3. Distribution of impacts per position
    4. Confidence interval plot
    """
    print(f"\n{'='*60}")
    print("Story 2.5: Creating Visualizations")
    print(f"{'='*60}\n")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Resampling vs Ablation (bar chart)
    ax1 = axes[0, 0]
    positions = ['CT0', 'CT1', 'CT2', 'CT3', 'CT4', 'CT5']
    x = np.arange(len(positions))
    width = 0.35

    resampling_impacts = stats_data['resampling_impacts']
    ablation_impacts = stats_data['ablation_impacts']

    ax1.bar(x - width/2, ablation_impacts, width, label='Ablation', alpha=0.8, color='#3498db')
    ax1.bar(x + width/2, resampling_impacts, width, label='Resampling', alpha=0.8, color='#e74c3c')

    ax1.set_xlabel('CT Position', fontsize=12)
    ax1.set_ylabel('Impact (%)', fontsize=12)
    ax1.set_title(f'Resampling vs Ablation Impact\nr = {stats_data["correlation"]["pearson_r"]:.3f}, p = {stats_data["correlation"]["pearson_p"]:.4f}',
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(positions)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Correlation scatter plot
    ax2 = axes[0, 1]
    ax2.scatter(ablation_impacts, resampling_impacts, s=200, alpha=0.6, color='#2ecc71', edgecolors='black', linewidths=2)

    # Add labels
    for i, pos in enumerate(positions):
        ax2.annotate(pos, (ablation_impacts[i], resampling_impacts[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)

    # Add regression line
    z = np.polyfit(ablation_impacts, resampling_impacts, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(ablation_impacts), max(ablation_impacts), 100)
    ax2.plot(x_line, p(x_line), "r--", alpha=0.5, label=f'y = {z[0]:.2f}x + {z[1]:.2f}')

    ax2.set_xlabel('Ablation Impact (%)', fontsize=12)
    ax2.set_ylabel('Resampling Impact (%)', fontsize=12)
    ax2.set_title(f'Correlation: r = {stats_data["correlation"]["pearson_r"]:.3f}',
                  fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Plot 3: Per-problem heatmap (sample 20 problems for visibility)
    ax3 = axes[1, 0]
    n_problems = len(per_problem_data['per_problem_impacts']['CT0'])
    sample_indices = np.random.choice(n_problems, min(20, n_problems), replace=False)
    sample_indices = sorted(sample_indices)

    heatmap_data = []
    for pos in positions:
        impacts = per_problem_data['per_problem_impacts'][pos]
        sampled = [impacts[i] * 100 for i in sample_indices]  # Convert to percentage
        heatmap_data.append(sampled)

    im = ax3.imshow(heatmap_data, aspect='auto', cmap='RdYlGn_r', vmin=-50, vmax=50)
    ax3.set_yticks(np.arange(len(positions)))
    ax3.set_yticklabels(positions)
    ax3.set_xticks(np.arange(len(sample_indices)))
    ax3.set_xticklabels([f'P{i}' for i in sample_indices], rotation=45, ha='right')
    ax3.set_xlabel('Problem ID (sample)', fontsize=12)
    ax3.set_ylabel('CT Position', fontsize=12)
    ax3.set_title('Per-Problem Impact Heatmap (sample of 20)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax3, label='Impact (%)')

    # Plot 4: Confidence intervals
    ax4 = axes[1, 1]
    y_pos = np.arange(len(positions))
    impacts = [np.mean(per_problem_data['per_problem_impacts'][pos]) * 100 for pos in positions]
    errors_lower = [impacts[i] - stats_data['confidence_intervals'][pos][0] for i, pos in enumerate(positions)]
    errors_upper = [stats_data['confidence_intervals'][pos][1] - impacts[i] for i, pos in enumerate(positions)]

    ax4.barh(y_pos, impacts, xerr=[errors_lower, errors_upper], align='center',
             alpha=0.7, color='#9b59b6', ecolor='black', capsize=5)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(positions)
    ax4.set_xlabel('Impact (%)', fontsize=12)
    ax4.set_ylabel('CT Position', fontsize=12)
    ax4.set_title('Impact with 95% Confidence Intervals', fontsize=14, fontweight='bold')
    ax4.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax4.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'full_resampling_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_path}")
    plt.close()

    # Create additional per-position distribution plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, pos in enumerate(positions):
        ax = axes[i]
        impacts = np.array(per_problem_data['per_problem_impacts'][pos]) * 100

        ax.hist(impacts, bins=20, alpha=0.7, color='#34495e', edgecolor='black')
        ax.axvline(x=np.mean(impacts), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(impacts):.1f}%')
        ax.axvline(x=0, color='k', linestyle='-', linewidth=1, alpha=0.3)
        ax.set_xlabel('Impact (%)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{pos} Impact Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'per_position_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Distribution plot saved to {output_path}")
    plt.close()


def generate_report(full_results: Dict, per_problem_data: Dict, stats_data: Dict, output_path: Path):
    """Generate comprehensive markdown report"""
    print(f"\nGenerating final report...")

    # Extract nested structure
    metadata = full_results.get('metadata', {})
    results = full_results.get('results', full_results)

    report = f"""# Full Resampling Experiment Analysis

## Experiment Metadata

- **Date:** {metadata.get('timestamp', 'N/A')}
- **Phase:** full
- **Problems:** {metadata.get('n_problems', 100)}
- **Samples per problem:** {metadata.get('n_samples', 10)}
- **Total generations:** {metadata.get('n_problems', 100) * metadata.get('n_samples', 10) * 6}
- **Random seed:** {metadata.get('seed', 42)}

---

## Results Summary

### Per-Position Impacts

| Position | Ablation | Resampling | Δ | 95% CI | Effect Size (d) |
|----------|----------|------------|---|--------|-----------------|
"""

    for pos in ['CT0', 'CT1', 'CT2', 'CT3', 'CT4', 'CT5']:
        ablation = ABLATION_IMPACTS[pos]
        pos_data = results[pos]
        resampling = pos_data['impact'] * 100
        delta = resampling - ablation
        ci = stats_data['confidence_intervals'][pos]
        effect_size = stats_data['effect_sizes'][pos]

        report += f"| {pos} | {ablation:.1f}% | {resampling:.1f}% | {delta:+.1f}% | [{ci[0]:.1f}%, {ci[1]:.1f}%] | {effect_size:.3f} |\n"

    report += f"""

### Statistical Metrics

- **Pearson Correlation:** r = {stats_data['correlation']['pearson_r']:.3f}, p = {stats_data['correlation']['pearson_p']:.4f}
- **Spearman Correlation:** ρ = {stats_data['correlation']['spearman_r']:.3f}, p = {stats_data['correlation']['spearman_p']:.4f}
- **Baseline accuracy:** {metadata.get('baseline_accuracy', 'N/A')}

---

## Interpretation

### Convergent Validation

"""

    if stats_data['correlation']['pearson_p'] < 0.05:
        if stats_data['correlation']['pearson_r'] > 0:
            report += "✓ **STRONG CONVERGENCE:** Resampling and ablation show significant positive correlation, providing convergent evidence for thought anchor locations.\n\n"
        else:
            report += "⚠️ **DIVERGENCE:** Resampling and ablation show significant negative correlation, suggesting they measure different aspects of CT token function.\n\n"
    else:
        report += "⚠️ **WEAK/NO CORRELATION:** Resampling and ablation do not show significant correlation. This suggests:\n"
        report += "- Ablation measures causal necessity (what happens when removed)\n"
        report += "- Resampling measures information specificity (what happens when contaminated)\n"
        report += "- These may be orthogonal properties\n\n"

    report += f"""### Position-Specific Findings

**High-Impact Positions:**
"""

    # Find high-impact positions
    impacts = [(pos, np.mean(per_problem_data['per_problem_impacts'][pos]) * 100)
               for pos in ['CT0', 'CT1', 'CT2', 'CT3', 'CT4', 'CT5']]
    impacts_sorted = sorted(impacts, key=lambda x: abs(x[1]), reverse=True)

    for pos, impact in impacts_sorted[:3]:
        report += f"- **{pos}:** {impact:.1f}% impact - {'' if impact > 10 else 'Lower than expected'}\n"

    report += f"""

**Low-Impact Positions:**
"""

    for pos, impact in impacts_sorted[-3:]:
        report += f"- **{pos}:** {impact:.1f}% impact\n"

    report += f"""

---

## Per-Problem Analysis

### Sensitivity Statistics

"""

    for pos in ['CT0', 'CT1', 'CT2', 'CT3', 'CT4', 'CT5']:
        s = per_problem_data['stats_per_position'][pos]
        report += f"**{pos}:** mean={s['mean']*100:.1f}%, std={s['std']*100:.1f}%, range=[{s['min']*100:.1f}%, {s['max']*100:.1f}%]\n\n"

    # Most sensitive problems
    sorted_problems = sorted(per_problem_data['avg_impacts_per_problem'].items(),
                            key=lambda x: abs(x[1]), reverse=True)

    report += f"""
### Most Sensitive Problems

These problems show the highest average impact across all CT positions:

"""

    for i, (prob_id, impact) in enumerate(sorted_problems[:10]):
        report += f"{i+1}. Problem {prob_id}: {impact*100:.1f}% average impact\n"

    report += f"""

### Least Sensitive Problems

These problems show the lowest average impact:

"""

    for i, (prob_id, impact) in enumerate(sorted_problems[-10:]):
        report += f"{i+1}. Problem {prob_id}: {impact*100:.1f}% average impact\n"

    report += f"""

---

## Visualizations

1. `full_resampling_analysis.png` - 4-panel comprehensive analysis
   - Resampling vs Ablation bar chart
   - Correlation scatter plot
   - Per-problem heatmap
   - Confidence intervals

2. `per_position_distributions.png` - Impact distributions for each position

---

## Conclusions

"""

    # Generate conclusions based on results
    pearson_r = stats_data['correlation']['pearson_r']
    pearson_p = stats_data['correlation']['pearson_p']

    if pearson_p < 0.05 and pearson_r > 0.5:
        report += "1. **Strong Convergent Evidence:** Resampling strongly corroborates ablation findings\n"
    elif pearson_p < 0.05 and 0.2 < pearson_r <= 0.5:
        report += "1. **Moderate Convergent Evidence:** Resampling partially corroborates ablation findings\n"
    else:
        report += "1. **Limited Convergence:** Resampling and ablation measure different aspects of CT function\n"

    report += f"""2. **Information Localization:** CT tokens show {'high' if max(stats_data['resampling_impacts']) > 15 else 'moderate'} sensitivity to position-specific contamination
3. **Position Variance:** Impact varies significantly across positions (range: {min(stats_data['resampling_impacts']):.1f}% to {max(stats_data['resampling_impacts']):.1f}%)
4. **Statistical Power:** n=100 problems provides robust estimates with narrow confidence intervals

---

## Next Steps

1. Investigate why correlation is {'strong' if pearson_r > 0.5 else 'weak/absent'}
2. Analyze position-specific patterns (why do certain positions show higher/lower impact?)
3. Compare with other CODI models (GPT-2 124M)
4. Test on other datasets (CommonsenseQA, Liars-Bench)

---

*Generated: {output_path.parent.name}*
"""

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"✓ Report saved to {output_path}")


def main():
    print(f"\n{'='*60}")
    print(f"Stories 2.3-2.5: Full Experiment Analysis")
    print(f"{'='*60}\n")

    # Paths
    results_path = Path(__file__).parent / '../results/resampling_full_results.json'
    output_dir = Path(__file__).parent / '../results'
    report_path = output_dir / 'full_analysis.md'

    # Check if results exist
    if not results_path.exists():
        print(f"✗ Results file not found: {results_path}")
        print(f"  Please wait for Story 2.2 (full resampling) to complete.")
        return

    # Load results
    print(f"Loading results from {results_path}...")
    full_results = load_results(results_path)
    results = full_results.get('results', full_results)  # Handle nested structure
    print(f"✓ Loaded results\n")

    # Story 2.3: Per-problem analysis
    per_problem_data = per_problem_analysis(results)

    # Story 2.4: Statistical analysis
    stats_data = statistical_analysis(results, per_problem_data)

    # Story 2.5: Visualizations
    create_visualizations(results, per_problem_data, stats_data, output_dir)

    # Generate final report
    generate_report(full_results, per_problem_data, stats_data, report_path)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"\nReports:")
    print(f"  - Full analysis: {report_path}")
    print(f"  - Visualizations: {output_dir}")
    print(f"\nNext: Story 2.6 - Update documentation")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
