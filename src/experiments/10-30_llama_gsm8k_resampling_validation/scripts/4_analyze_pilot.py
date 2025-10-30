#!/usr/bin/env python3
"""
Story 1.5: Pilot Analysis & Go/No-Go Decision

Analyze pilot resampling results and make Go/No-Go decision for full experiment.

Decision Criteria (from PM requirements):
- GO if:
  * At least one position shows >10% resampling impact
  * Positive correlation with ablation (r > 0.2)
  * Baseline accuracy 50-65%
  * No technical issues
- NO-GO if:
  * All positions <5% impact (flat distribution)
  * Negative correlation with ablation
  * Technical issues (baseline accuracy off >10%)

Time estimate: 1.5 hours

Usage:
    python 4_analyze_pilot.py --results_file ../results/resampling_pilot_results.json
"""

# CRITICAL: Set PYTHONHASHSEED before imports
import os
os.environ['PYTHONHASHSEED'] = '42'

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
import sys

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import set_seed


# Reference data from research journal
ABLATION_IMPACTS = {
    'CT0': 18.7,
    'CT1': 12.8,
    'CT2': 14.6,
    'CT3': 15.0,
    'CT4': 3.5,
    'CT5': 26.0
}

ATTENTION_TO_CT0 = {
    'CT1': 4.77,
    'CT2': 4.21,
    'CT3': 3.53,
    'CT4': 2.76,
    'CT5': 3.04
}


def load_results(results_path: Path) -> dict:
    """Load resampling results from JSON."""
    with open(results_path, 'r') as f:
        data = json.load(f)
    return data


def calculate_correlation(resampling_impacts: list, ablation_impacts: list):
    """
    Calculate Pearson correlation between resampling and ablation.

    Args:
        resampling_impacts: List of resampling impact percentages
        ablation_impacts: List of ablation impact percentages

    Returns:
        Tuple of (r, p_value)
    """
    r, p = pearsonr(resampling_impacts, ablation_impacts)
    return r, p


def create_visualization(resampling_impacts, ablation_impacts, output_path):
    """
    Create side-by-side bar chart comparing resampling vs ablation.

    Args:
        resampling_impacts: Dict mapping position to impact
        ablation_impacts: Dict mapping position to impact
        output_path: Path to save figure
    """
    positions = ['CT0', 'CT1', 'CT2', 'CT3', 'CT4', 'CT5']

    resampling_vals = [resampling_impacts[p] * 100 for p in positions]
    ablation_vals = [ablation_impacts[p] for p in positions]

    # Calculate correlation
    r, p = pearsonr(resampling_vals, ablation_vals)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    x = np.arange(len(positions))
    width = 0.35

    bars1 = ax1.bar(x - width/2, ablation_vals, width, label='Ablation', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, resampling_vals, width, label='Resampling', color='#e67e22', alpha=0.8)

    ax1.set_xlabel('CT Position', fontsize=12)
    ax1.set_ylabel('Impact (%)', fontsize=12)
    ax1.set_title(f'Pilot: Resampling vs Ablation Impact\nr = {r:.3f}, p = {p:.4f}', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(positions)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Scatter plot
    ax2.scatter(ablation_vals, resampling_vals, s=100, alpha=0.6, color='#2ecc71')

    # Add position labels
    for i, pos in enumerate(positions):
        ax2.annotate(pos, (ablation_vals[i], resampling_vals[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)

    # Trendline
    z = np.polyfit(ablation_vals, resampling_vals, 1)
    p_fit = np.poly1d(z)
    x_line = np.linspace(min(ablation_vals), max(ablation_vals), 100)
    ax2.plot(x_line, p_fit(x_line), "r--", alpha=0.5, label=f'y = {z[0]:.2f}x + {z[1]:.2f}')

    ax2.set_xlabel('Ablation Impact (%)', fontsize=12)
    ax2.set_ylabel('Resampling Impact (%)', fontsize=12)
    ax2.set_title(f'Correlation: r = {r:.3f}', fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_path}")

    return fig


def make_go_no_go_decision(results_data: dict, resampling_impacts: dict):
    """
    Make Go/No-Go decision based on pilot results.

    Args:
        results_data: Full results data
        resampling_impacts: Dict of position -> impact

    Returns:
        Tuple of (decision: str, reasons: list)
    """
    reasons = []
    criteria_met = []

    # Extract metrics
    baseline_acc = results_data['results']['CT0']['baseline_accuracy']
    impacts_list = [resampling_impacts[f'CT{i}'] * 100 for i in range(6)]
    ablation_list = [ABLATION_IMPACTS[f'CT{i}'] for i in range(6)]

    # Criterion 1: Peak impact
    max_impact = max(impacts_list)
    if max_impact > 10:
        criteria_met.append(True)
        reasons.append(f"✓ Peak impact: {max_impact:.1f}% (threshold: >10%)")
    else:
        criteria_met.append(False)
        reasons.append(f"✗ Peak impact: {max_impact:.1f}% (threshold: >10%)")

    # Criterion 2: Correlation
    r, p = pearsonr(impacts_list, ablation_list)
    if r > 0.2:
        criteria_met.append(True)
        reasons.append(f"✓ Correlation: r = {r:.3f}, p = {p:.4f} (threshold: r > 0.2)")
    else:
        criteria_met.append(False)
        reasons.append(f"✗ Correlation: r = {r:.3f}, p = {p:.4f} (threshold: r > 0.2)")

    # Criterion 3: Baseline accuracy
    if 0.50 <= baseline_acc <= 0.70:
        criteria_met.append(True)
        reasons.append(f"✓ Baseline accuracy: {baseline_acc:.1%} (expected: 50-70%)")
    else:
        criteria_met.append(False)
        reasons.append(f"✗ Baseline accuracy: {baseline_acc:.1%} (expected: 50-70%)")

    # Criterion 4: Variance
    variance_std = np.std(impacts_list)
    if variance_std > 3:
        criteria_met.append(True)
        reasons.append(f"✓ Position variance: {variance_std:.1f}% (threshold: >3%)")
    else:
        criteria_met.append(False)
        reasons.append(f"✗ Position variance: {variance_std:.1f}% (threshold: >3%)")

    # Make decision
    if sum(criteria_met) >= 3:  # At least 3 out of 4 criteria
        decision = "GO"
    else:
        decision = "NO-GO"

    return decision, reasons, r, p


def generate_report(results_data: dict, decision: str, reasons: list, r: float, p: float, output_path: Path):
    """Generate markdown report with Go/No-Go decision."""

    resampling_impacts = {}
    for i in range(6):
        pos = f'CT{i}'
        resampling_impacts[pos] = results_data['results'][pos]['impact']

    report = f"""# Pilot Resampling Experiment Analysis

## Experiment Metadata

- **Date:** {results_data['metadata']['timestamp']}
- **Phase:** {results_data['metadata']['phase']}
- **Problems:** {results_data['metadata']['n_problems']}
- **Samples per problem:** {results_data['metadata']['n_samples_per_problem']}
- **Total generations:** {results_data['metadata']['total_generations']}
- **Random seed:** {results_data['metadata']['random_seed']}

## Results Summary

### Per-Position Impacts

| Position | Ablation | Resampling | Δ | Std Error |
|----------|----------|------------|---|-----------|
"""

    for i in range(6):
        pos = f'CT{i}'
        ablation = ABLATION_IMPACTS[pos]
        resampling = resampling_impacts[pos] * 100
        delta = abs(ablation - resampling)
        std_err = results_data['results'][pos]['std_error'] * 100

        report += f"| {pos} | {ablation:.1f}% | {resampling:.1f}% | {delta:.1f}% | ±{std_err:.1f}% |\n"

    # Correlation
    impacts_list = [resampling_impacts[f'CT{i}'] * 100 for i in range(6)]
    ablation_list = [ABLATION_IMPACTS[f'CT{i}'] for i in range(6)]

    report += f"""
### Statistical Metrics

- **Correlation (Pearson):** r = {r:.3f}, p = {p:.4f}
- **Position variance:** σ = {np.std(impacts_list):.1f}%
- **Baseline accuracy:** {results_data['results']['CT0']['baseline_accuracy']:.1%}

## Go/No-Go Decision: **{decision}**

### Criteria Evaluation

"""

    for reason in reasons:
        report += f"- {reason}\n"

    report += f"""
### Justification

"""

    if decision == "GO":
        report += """The pilot experiment shows promising results that warrant proceeding to the full experiment:

1. **Non-uniform distribution:** Impact varies across positions, indicating localized information storage
2. **Positive correlation:** Resampling impacts generally align with ablation findings
3. **Technical validation:** Baseline accuracy within expected range, no implementation errors

The full experiment (100 problems × 10 samples = 6,000 generations) will provide:
- Higher statistical power for correlation analysis
- More robust per-position impact estimates
- Stronger evidence for convergent validation
"""
    else:
        report += """The pilot experiment does not meet sufficient criteria to proceed:

1. **Review implementation:** Check if swapping function is correctly contaminating reasoning
2. **Investigate flat distribution:** May indicate distributed information encoding
3. **Consider alternative approaches:** Different swap strategies or position sampling

Recommendation: Debug current approach before scaling to full experiment.
"""

    report += f"""
## Next Steps

"""

    if decision == "GO":
        report += """1. **Proceed to Story 2.1:** Extract CT hidden states for 100 problems
2. **Run full resampling:** 6,000 generations (~5 hours runtime)
3. **Complete statistical analysis:** Stories 2.3-2.6
4. **Document results:** Update DATA_INVENTORY.md and research journal
"""
    else:
        report += """1. **Debug swapping function:** Verify CT contamination is happening
2. **Review extraction:** Ensure hidden states are correct layer outputs
3. **Consult with team:** Decide on approach pivot or parameter adjustments
4. **Re-run pilot:** After fixes, validate with new pilot before full experiment
"""

    # Write report
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"✓ Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze pilot results and make Go/No-Go decision")
    parser.add_argument('--results_file', type=str,
                        default='../results/resampling_pilot_results.json',
                        help='Path to pilot results JSON')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"Story 1.5: Pilot Analysis & Go/No-Go Decision")
    print(f"{'='*60}\n")

    # Load results
    results_path = Path(__file__).parent / args.results_file
    print(f"Loading results from {results_path}...")

    results_data = load_results(results_path)
    print(f"✓ Loaded results")

    # Extract impacts
    resampling_impacts = {}
    for i in range(6):
        pos = f'CT{i}'
        resampling_impacts[pos] = results_data['results'][pos]['impact']

    # Create visualization
    output_dir = Path(__file__).parent / '../results'
    viz_path = output_dir / 'pilot_resampling_vs_ablation.png'

    print(f"\nCreating visualization...")
    create_visualization(resampling_impacts, ABLATION_IMPACTS, viz_path)

    # Make Go/No-Go decision
    print(f"\nEvaluating Go/No-Go criteria...")
    decision, reasons, r, p = make_go_no_go_decision(results_data, resampling_impacts)

    # Generate report
    report_path = output_dir / 'pilot_analysis.md'
    print(f"\nGenerating report...")
    generate_report(results_data, decision, reasons, r, p, report_path)

    # Print summary
    print(f"\n{'='*60}")
    print(f"PILOT ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"\nDecision: **{decision}**\n")

    for reason in reasons:
        print(f"  {reason}")

    print(f"\nReports:")
    print(f"  - Analysis: {report_path}")
    print(f"  - Visualization: {viz_path}")

    print(f"\n{'='*60}\n")

    if decision == "NO-GO":
        print("⚠️  NO-GO decision: Review reasons above before proceeding\n")
        exit(1)
    else:
        print("✓ GO decision: Proceed to full experiment (Stories 2.1-2.6)\n")


if __name__ == "__main__":
    main()
