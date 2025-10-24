#!/usr/bin/env python3
"""
Analyze Threshold Degradation Results

Generates:
- Degradation curve (accuracy vs # corrupted tokens)
- Statistical test of 67% threshold claim
- Critical token identification
- Publication-ready figures

Usage:
    python 2_analyze_threshold.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def load_results(results_file):
    """Load threshold experiment results."""
    with open(results_file, 'r') as f:
        return json.load(f)


def compute_degradation_curve(results):
    """
    Compute accuracy degradation curve by corruption level.

    Returns:
        dict: {corruption_method: {level: {mean, std, count}}}
    """
    curves = {}

    # Get unique corruption methods
    methods = set()
    for problem in results:
        if 'error' not in problem:
            for corruption in problem['corruptions']:
                methods.add(corruption['corruption_method'])

    # Compute statistics for each method and level
    for method in methods:
        curves[method] = {}

        for level in range(1, 7):
            # Collect accuracies for this level and method
            accuracies = []

            for problem in results:
                if 'error' in problem or not problem['baseline']['correct']:
                    continue

                for corruption in problem['corruptions']:
                    if (corruption['corruption_level'] == level and
                        corruption['corruption_method'] == method):
                        accuracies.append(1 if corruption['correct'] else 0)

            if accuracies:
                curves[method][level] = {
                    'mean': np.mean(accuracies),
                    'std': np.std(accuracies),
                    'count': len(accuracies),
                    'sem': stats.sem(accuracies) if len(accuracies) > 1 else 0
                }

    return curves


def identify_critical_tokens(results):
    """
    Identify which tokens are most critical using skip tests (level 4).

    Returns:
        dict: {corruption_method: {skip_token: accuracy}}
    """
    critical_tokens = {}

    # Get unique corruption methods
    methods = set()
    for problem in results:
        if 'error' not in problem:
            for corruption in problem['corruptions']:
                methods.add(corruption['corruption_method'])

    for method in methods:
        critical_tokens[method] = {}

        # Analyze level 4 (skip tests) - corrupt 4 tokens, skip 1
        for skip_token in range(6):
            skip_label = f'skip_{skip_token}'

            accuracies = []
            for problem in results:
                if 'error' in problem or not problem['baseline']['correct']:
                    continue

                for corruption in problem['corruptions']:
                    if (corruption['corruption_level'] == 4 and
                        corruption['position_label'] == skip_label and
                        corruption['corruption_method'] == method):
                        accuracies.append(1 if corruption['correct'] else 0)

            if accuracies:
                critical_tokens[method][skip_token] = {
                    'mean_accuracy': np.mean(accuracies),
                    'count': len(accuracies),
                    'interpretation': 'CRITICAL' if np.mean(accuracies) > 0.5 else 'non-critical'
                }

    return critical_tokens


def test_67_percent_threshold(results):
    """
    Test if corrupting 4/6 tokens (67%) causes catastrophic failure.

    Returns:
        dict: Statistical test results
    """
    # Get accuracy at level 4 (67% corruption)
    level_4_accuracies = []
    baseline_accuracies = []

    for problem in results:
        if 'error' in problem:
            continue

        baseline_correct = problem['baseline']['correct']
        baseline_accuracies.append(1 if baseline_correct else 0)

        if baseline_correct:
            # Get all level 4 results for this problem
            level_4_results = [
                corruption['correct']
                for corruption in problem['corruptions']
                if corruption['corruption_level'] == 4
            ]

            if level_4_results:
                # Average across all level 4 configurations
                level_4_accuracies.append(np.mean(level_4_results))

    baseline_acc = np.mean(baseline_accuracies)
    level_4_acc = np.mean(level_4_accuracies)

    # Statistical test
    t_stat, p_value = stats.ttest_rel(baseline_accuracies, level_4_accuracies)

    # Effect size (Cohen's d)
    diff = baseline_acc - level_4_acc
    pooled_std = np.sqrt((np.var(baseline_accuracies) + np.var(level_4_accuracies)) / 2)
    cohens_d = diff / pooled_std if pooled_std > 0 else 0

    return {
        'baseline_accuracy': baseline_acc,
        'level_4_accuracy': level_4_acc,
        'accuracy_drop': diff,
        'relative_drop': diff / baseline_acc if baseline_acc > 0 else 0,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'catastrophic_failure': level_4_acc < 0.2,  # Define catastrophic as <20% accuracy
        'interpretation': 'CATASTROPHIC' if level_4_acc < 0.2 else 'Degraded but functional'
    }


def plot_degradation_curve(curves, output_dir):
    """Plot accuracy degradation curve."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'zero': '#E74C3C', 'gauss_1.0': '#3498DB'}
    labels = {'zero': 'Zero Ablation', 'gauss_1.0': 'Gaussian σ=1.0'}

    for method, curve in curves.items():
        levels = sorted(curve.keys())
        means = [curve[level]['mean'] for level in levels]
        sems = [curve[level]['sem'] for level in levels]

        ax.plot(levels, means, 'o-', color=colors.get(method, 'gray'),
                label=labels.get(method, method), linewidth=2, markersize=8)
        ax.fill_between(levels,
                        [m - s for m, s in zip(means, sems)],
                        [m + s for m, s in zip(means, sems)],
                        alpha=0.2, color=colors.get(method, 'gray'))

    # Mark 67% threshold (level 4)
    ax.axvline(x=4, color='red', linestyle='--', alpha=0.5, linewidth=2,
               label='67% Threshold (4/6 tokens)')

    # Mark catastrophic failure line
    ax.axhline(y=0.2, color='orange', linestyle=':', alpha=0.5, linewidth=2,
               label='Catastrophic Failure (<20%)')

    ax.set_xlabel('Number of Tokens Corrupted', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax.set_title('Threshold Degradation: Accuracy vs Token Corruption',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(1, 7))
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)

    plt.tight_layout()

    # Save both PDF and PNG
    for ext in ['pdf', 'png']:
        output_file = output_dir / f'degradation_curve.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")

    plt.close()


def plot_critical_tokens(critical_tokens, output_dir):
    """Plot critical token identification from skip tests."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    methods = list(critical_tokens.keys())
    positions = list(range(6))

    for idx, method in enumerate(methods):
        ax = axes[idx]

        accuracies = [
            critical_tokens[method].get(pos, {}).get('mean_accuracy', 0)
            for pos in positions
        ]

        colors = ['#27AE60' if acc > 0.5 else '#E74C3C' for acc in accuracies]

        bars = ax.bar(positions, accuracies, color=colors, alpha=0.7, edgecolor='black')

        # Add threshold line
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1,
                   label='50% Threshold')

        ax.set_xlabel('Token Position (skipped)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (when this token is kept)', fontsize=12, fontweight='bold')
        ax.set_title(f'Critical Tokens: {method}', fontsize=13, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()

        # Annotate bars
        for pos, acc in enumerate(accuracies):
            label = 'CRITICAL' if acc > 0.5 else ''
            if label:
                ax.text(pos, acc + 0.05, label, ha='center', fontsize=9,
                        fontweight='bold', color='green')

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        output_file = output_dir / f'critical_tokens.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")

    plt.close()


def analyze_threshold_results():
    """Main analysis function."""
    # Paths
    results_file = Path(__file__).parent.parent / 'results' / 'threshold_test_10.json'
    output_dir = Path(__file__).parent.parent / 'figures'
    stats_file = Path(__file__).parent.parent / 'results' / 'threshold_analysis.json'

    print("=" * 80)
    print("THRESHOLD ANALYSIS")
    print("=" * 80)

    # Load results
    print(f"\nLoading results from {results_file}...")
    results = load_results(results_file)
    print(f"Loaded {len(results)} problems")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute degradation curves
    print("\nComputing degradation curves...")
    curves = compute_degradation_curve(results)

    print("\nDegradation by level:")
    for method, curve in curves.items():
        print(f"\n  {method}:")
        for level in sorted(curve.keys()):
            stats = curve[level]
            print(f"    Level {level}: {stats['mean']:.1%} ± {stats['sem']:.1%} "
                  f"(n={stats['count']})")

    # Test 67% threshold
    print("\nTesting 67% threshold (4/6 corruption)...")
    threshold_test = test_67_percent_threshold(results)

    print(f"  Baseline accuracy: {threshold_test['baseline_accuracy']:.1%}")
    print(f"  Level 4 accuracy: {threshold_test['level_4_accuracy']:.1%}")
    print(f"  Accuracy drop: {threshold_test['accuracy_drop']:.1%}")
    print(f"  Relative drop: {threshold_test['relative_drop']:.1%}")
    print(f"  P-value: {threshold_test['p_value']:.4f}")
    print(f"  Cohen's d: {threshold_test['cohens_d']:.2f}")
    print(f"  Result: {threshold_test['interpretation']}")

    # Identify critical tokens
    print("\nIdentifying critical tokens from skip tests...")
    critical_tokens = identify_critical_tokens(results)

    for method, tokens in critical_tokens.items():
        print(f"\n  {method}:")
        for pos in sorted(tokens.keys()):
            token_stats = tokens[pos]
            print(f"    Token {pos}: {token_stats['mean_accuracy']:.1%} "
                  f"({token_stats['interpretation']})")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_degradation_curve(curves, output_dir)
    plot_critical_tokens(critical_tokens, output_dir)

    # Save statistics (convert numpy types to Python types for JSON serialization)
    def convert_to_python(obj):
        """Recursively convert numpy types to Python types."""
        if isinstance(obj, dict):
            return {k: convert_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python(v) for v in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    analysis_stats = {
        'degradation_curves': {
            method: {str(level): stats for level, stats in curve.items()}
            for method, curve in curves.items()
        },
        'threshold_test_67_percent': convert_to_python(threshold_test),
        'critical_tokens': {
            method: {str(pos): convert_to_python(stats) for pos, stats in tokens.items()}
            for method, tokens in critical_tokens.items()
        }
    }

    with open(stats_file, 'w') as f:
        json.dump(analysis_stats, f, indent=2)

    print(f"\n✓ Statistics saved to {stats_file}")
    print("=" * 80)


if __name__ == "__main__":
    analyze_threshold_results()
