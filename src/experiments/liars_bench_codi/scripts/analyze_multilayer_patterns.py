"""
Statistical analysis of multi-layer probe results.

Analyzes:
1. Where signal degrades (which layer)
2. Early vs late layer performance
3. CT vs Question/Answer performance
4. Statistical significance tests
5. Effect sizes

Outputs:
- Statistical analysis JSON
- Key findings summary
- Recommendations for continued training

Author: Claude Code
Date: 2025-10-30
Experiment: Pre-compression deception signal analysis
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
import argparse


def load_probe_results(results_path):
    """Load probe training results."""
    print(f"Loading results from: {results_path}")
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results


def analyze_signal_degradation(results):
    """Identify where signal degrades across layers."""
    print("\n" + "=" * 80)
    print("SIGNAL DEGRADATION ANALYSIS")
    print("=" * 80)

    layers = results['metadata']['layers']
    positions = results['metadata']['positions']

    # Focus on CT positions
    ct_positions = [p for p in positions if p.startswith('ct')]

    signal_threshold = 0.55  # Accuracy > 55% indicates signal
    random_baseline = 0.50

    analysis = {
        'signal_threshold': signal_threshold,
        'random_baseline': random_baseline,
        'degradation_points': {}
    }

    print(f"\nSignal Threshold: {signal_threshold:.0%}")
    print(f"Random Baseline: {random_baseline:.0%}")
    print()

    # For each CT position, find where signal drops below threshold
    for position_name in ct_positions:
        accuracies = []
        for layer_idx in layers:
            layer_name = f'layer_{layer_idx}'
            acc = results[layer_name][position_name]['test_accuracy']
            accuracies.append((layer_idx, acc))

        # Find first layer where accuracy drops below threshold
        signal_lost_at = None
        for layer_idx, acc in accuracies:
            if acc < signal_threshold:
                signal_lost_at = layer_idx
                break

        # Find if signal ever existed (above threshold in early layers)
        has_signal = any(acc >= signal_threshold for _, acc in accuracies[:3])  # Check first 3 layers

        analysis['degradation_points'][position_name] = {
            'has_early_signal': has_signal,
            'signal_lost_at_layer': signal_lost_at,
            'accuracies_by_layer': {str(layer): acc for layer, acc in accuracies}
        }

        status = "No signal" if not has_signal else f"Signal lost at layer {signal_lost_at}" if signal_lost_at else "Signal maintained"
        print(f"  {position_name:10s}: {status}")

    return analysis


def compare_early_vs_late_layers(results):
    """Compare performance in early vs late layers."""
    print("\n" + "=" * 80)
    print("EARLY VS LATE LAYER COMPARISON")
    print("=" * 80)

    layers = results['metadata']['layers']
    positions = results['metadata']['positions']
    ct_positions = [p for p in positions if p.startswith('ct')]

    # Define early (0, 3, 6) and late (9, 12, 15) layers
    early_layers = [l for l in layers if l <= 6]
    late_layers = [l for l in layers if l >= 9]

    print(f"\nEarly layers: {early_layers}")
    print(f"Late layers: {late_layers}")
    print()

    # Collect accuracies for CT positions
    early_accs = []
    late_accs = []

    for position_name in ct_positions:
        for layer_idx in early_layers:
            layer_name = f'layer_{layer_idx}'
            early_accs.append(results[layer_name][position_name]['test_accuracy'])

        for layer_idx in late_layers:
            layer_name = f'layer_{layer_idx}'
            late_accs.append(results[layer_name][position_name]['test_accuracy'])

    early_mean = np.mean(early_accs)
    early_std = np.std(early_accs)
    late_mean = np.mean(late_accs)
    late_std = np.std(late_accs)

    # T-test
    t_stat, p_value = stats.ttest_ind(early_accs, late_accs)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(early_accs) + np.var(late_accs)) / 2)
    cohens_d = (early_mean - late_mean) / pooled_std if pooled_std > 0 else 0

    print(f"CT Positions (ct0-ct5):")
    print(f"  Early layers: {early_mean:.1%} ± {early_std:.1%}")
    print(f"  Late layers:  {late_mean:.1%} ± {late_std:.1%}")
    print(f"  Difference:   {early_mean - late_mean:+.1%}")
    print(f"  T-test:       t={t_stat:.3f}, p={p_value:.4f}")
    print(f"  Cohen's d:    {cohens_d:.3f}")
    print()

    if p_value < 0.05:
        print(f"  ✅ Statistically significant difference (p < 0.05)")
    else:
        print(f"  ⚠️  Not statistically significant (p >= 0.05)")

    return {
        'early_layers': early_layers,
        'late_layers': late_layers,
        'early_mean': float(early_mean),
        'early_std': float(early_std),
        'late_mean': float(late_mean),
        'late_std': float(late_std),
        'difference': float(early_mean - late_mean),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'significant': bool(p_value < 0.05)
    }


def compare_ct_vs_qa_positions(results):
    """Compare CT positions vs Question/Answer positions."""
    print("\n" + "=" * 80)
    print("CT VS QUESTION/ANSWER COMPARISON")
    print("=" * 80)

    layers = results['metadata']['layers']
    positions = results['metadata']['positions']

    ct_positions = [p for p in positions if p.startswith('ct')]
    qa_positions = ['question_last', 'answer_first']

    # Collect accuracies
    ct_accs = []
    qa_accs = []

    for layer_idx in layers:
        layer_name = f'layer_{layer_idx}'

        for pos in ct_positions:
            ct_accs.append(results[layer_name][pos]['test_accuracy'])

        for pos in qa_positions:
            qa_accs.append(results[layer_name][pos]['test_accuracy'])

    ct_mean = np.mean(ct_accs)
    ct_std = np.std(ct_accs)
    qa_mean = np.mean(qa_accs)
    qa_std = np.std(qa_accs)

    # T-test
    t_stat, p_value = stats.ttest_ind(ct_accs, qa_accs)

    # Effect size
    pooled_std = np.sqrt((np.var(ct_accs) + np.var(qa_accs)) / 2)
    cohens_d = (qa_mean - ct_mean) / pooled_std if pooled_std > 0 else 0

    print(f"\nContinuous Thought (CT) positions: {ct_mean:.1%} ± {ct_std:.1%}")
    print(f"Question/Answer (Q/A) positions:  {qa_mean:.1%} ± {qa_std:.1%}")
    print(f"Difference (Q/A - CT):              {qa_mean - ct_mean:+.1%}")
    print(f"T-test:                             t={t_stat:.3f}, p={p_value:.4f}")
    print(f"Cohen's d:                          {cohens_d:.3f}")
    print()

    if p_value < 0.05:
        print(f"✅ Q/A positions significantly {'better' if qa_mean > ct_mean else 'worse'} than CT (p < 0.05)")
    else:
        print(f"⚠️  No significant difference (p >= 0.05)")

    return {
        'ct_mean': float(ct_mean),
        'ct_std': float(ct_std),
        'qa_mean': float(qa_mean),
        'qa_std': float(qa_std),
        'difference': float(qa_mean - ct_mean),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'significant': bool(p_value < 0.05)
    }


def generate_key_findings(results, degradation, early_late, ct_qa):
    """Generate key findings statement."""
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()

    findings = []

    # Finding 1: Signal existence
    ct_positions = [k for k in degradation['degradation_points'].keys()]
    has_signal_count = sum(
        1 for pos in ct_positions
        if degradation['degradation_points'][pos]['has_early_signal']
    )

    if has_signal_count > 0:
        findings.append(
            f"1. Deception signal EXISTS in early layers for {has_signal_count}/{len(ct_positions)} CT positions"
        )
    else:
        findings.append(
            f"1. NO deception signal detected in CT positions (all at chance level ~50%)"
        )

    # Finding 2: Signal degradation
    if early_late['significant']:
        findings.append(
            f"2. Signal DEGRADES from early to late layers: "
            f"{early_late['early_mean']:.1%} → {early_late['late_mean']:.1%} "
            f"({early_late['difference']:+.1%}, p={early_late['p_value']:.4f})"
        )
    else:
        findings.append(
            f"2. No significant signal change across layers (p={early_late['p_value']:.4f})"
        )

    # Finding 3: CT vs Q/A comparison
    if ct_qa['significant']:
        better_space = "Language space (Q/A)" if ct_qa['qa_mean'] > ct_qa['ct_mean'] else "Continuous thought (CT)"
        findings.append(
            f"3. {better_space} PRESERVES signal better: "
            f"Q/A {ct_qa['qa_mean']:.1%} vs CT {ct_qa['ct_mean']:.1%} "
            f"(Δ={ct_qa['difference']:+.1%}, p={ct_qa['p_value']:.4f})"
        )
    else:
        findings.append(
            f"3. No significant difference between CT and Q/A positions"
        )

    # Finding 4: Training recommendation
    overall_mean = (early_late['early_mean'] + early_late['late_mean']) / 2
    if overall_mean > 0.55:
        findings.append(
            f"4. RECOMMENDATION: Signal is detectable (mean {overall_mean:.1%}). "
            f"5 epochs may be sufficient for this analysis."
        )
    elif overall_mean > 0.52:
        findings.append(
            f"4. RECOMMENDATION: Weak signal detected (mean {overall_mean:.1%}). "
            f"Consider training 5 more epochs (total 10) to strengthen signal."
        )
    else:
        findings.append(
            f"4. RECOMMENDATION: Very weak signal (mean {overall_mean:.1%}). "
            f"Continue training to 10-15 epochs for stronger representations."
        )

    for finding in findings:
        print(f"  {finding}")
        print()

    return findings


def main():
    print("\n" + "=" * 80)
    print("MULTI-LAYER PATTERN ANALYSIS")
    print("Statistical Analysis of Deception Signal")
    print("=" * 80)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epoch',
        type=str,
        default='5ep',
        help='Epoch identifier (5ep, 10ep, 15ep)'
    )
    args = parser.parse_args()

    epoch_str = args.epoch

    # Paths
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results"

    results_path = results_dir / f"multilayer_probe_results_llama1b_{epoch_str}.json"

    # Load results
    results = load_probe_results(results_path)

    # Perform analyses
    degradation = analyze_signal_degradation(results)
    early_late = compare_early_vs_late_layers(results)
    ct_qa = compare_ct_vs_qa_positions(results)

    # Generate key findings
    key_findings = generate_key_findings(results, degradation, early_late, ct_qa)

    # Save statistical analysis
    analysis_output = {
        'epoch': epoch_str,
        'signal_degradation': degradation,
        'early_vs_late': early_late,
        'ct_vs_qa': ct_qa,
        'key_findings': key_findings
    }

    output_path = results_dir / f"multilayer_statistical_analysis_llama1b_{epoch_str}.json"

    print("\n" + "=" * 80)
    print("SAVING ANALYSIS")
    print("=" * 80)
    print(f"  Output: {output_path}")
    print()

    with open(output_path, 'w') as f:
        json.dump(analysis_output, f, indent=2)

    print("=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
