#!/usr/bin/env python3
"""
Analyze Token Enhancement Results

Generates:
- Enhancement heatmap (position × multiplier)
- Optimal multiplier identification
- Statistical tests for critical positions
- Publication-ready figures

Usage:
    python 4_analyze_enhancement.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11


def load_results(results_file):
    """Load enhancement experiment results."""
    with open(results_file, 'r') as f:
        return json.load(f)


def compute_enhancement_matrix(results):
    """
    Compute accuracy matrix: position × multiplier.

    Returns:
        dict: {position: {multiplier: {mean, std, count}}}
    """
    matrix = {}

    for position in range(6):
        matrix[position] = {}

        # Get unique multipliers
        multipliers = set()
        for problem in results:
            if 'error' not in problem:
                for enhancement in problem['enhancements']:
                    if enhancement['position'] == position:
                        multipliers.add(enhancement['multiplier'])

        for multiplier in sorted(multipliers):
            accuracies = []

            for problem in results:
                if 'error' in problem:
                    continue

                for enhancement in problem['enhancements']:
                    if (enhancement['position'] == position and
                        enhancement['multiplier'] == multiplier):
                        accuracies.append(1 if enhancement['correct'] else 0)

            if accuracies:
                matrix[position][multiplier] = {
                    'mean': np.mean(accuracies),
                    'std': np.std(accuracies),
                    'count': len(accuracies),
                    'sem': stats.sem(accuracies) if len(accuracies) > 1 else 0
                }

    return matrix


def identify_optimal_multipliers(matrix):
    """
    Identify optimal enhancement multiplier for each position.

    Returns:
        dict: {position: {optimal_multiplier, max_accuracy}}
    """
    optimal = {}

    for position in range(6):
        if position not in matrix:
            continue

        best_mult = None
        best_acc = -1

        for multiplier, stats in matrix[position].items():
            if stats['mean'] > best_acc:
                best_acc = stats['mean']
                best_mult = multiplier

        optimal[position] = {
            'optimal_multiplier': best_mult,
            'max_accuracy': best_acc
        }

    return optimal


def test_position_criticality(results):
    """
    Test if any positions are significantly more enhancement-responsive.

    Returns:
        dict: Statistical tests comparing positions
    """
    # Get baseline accuracy for each problem
    baseline_accs = {}
    for problem in results:
        if 'error' not in problem:
            baseline_accs[problem['problem_id']] = problem['baseline']['correct']

    # For each position, compute average enhancement effect
    position_effects = {}

    for position in range(6):
        effects = []

        for problem in results:
            if 'error' in problem:
                continue

            problem_id = problem['problem_id']
            baseline = baseline_accs.get(problem_id, False)

            # Average accuracy across all multipliers for this position
            position_accs = [
                enhancement['correct']
                for enhancement in problem['enhancements']
                if enhancement['position'] == position
            ]

            if position_accs:
                avg_acc = np.mean(position_accs)
                # Effect = difference from baseline
                effect = avg_acc - (1 if baseline else 0)
                effects.append(effect)

        position_effects[position] = {
            'mean_effect': np.mean(effects) if effects else 0,
            'std_effect': np.std(effects) if effects else 0,
            'effects': effects
        }

    # ANOVA test: Are enhancement effects different across positions?
    effect_groups = [
        position_effects[pos]['effects']
        for pos in range(6)
        if position_effects[pos]['effects']
    ]

    if len(effect_groups) > 1:
        f_stat, p_value = stats.f_oneway(*effect_groups)
    else:
        f_stat, p_value = None, None

    # Pairwise comparisons (post-hoc)
    pairwise = {}
    for i in range(6):
        for j in range(i + 1, 6):
            if (position_effects[i]['effects'] and
                position_effects[j]['effects']):
                t_stat, p_val = stats.ttest_ind(
                    position_effects[i]['effects'],
                    position_effects[j]['effects']
                )
                pairwise[f'{i}_vs_{j}'] = {
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'significant': p_val < 0.05
                }

    return {
        'position_effects': {
            pos: {
                'mean_effect': data['mean_effect'],
                'std_effect': data['std_effect']
            }
            for pos, data in position_effects.items()
        },
        'anova': {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05 if p_value is not None else None
        },
        'pairwise_comparisons': pairwise
    }


def plot_enhancement_heatmap(matrix, output_dir):
    """Plot enhancement heatmap: position × multiplier."""
    # Prepare data for heatmap
    positions = sorted(matrix.keys())
    multipliers = sorted(set(
        mult for pos_data in matrix.values() for mult in pos_data.keys()
    ))

    # Create accuracy matrix
    acc_matrix = np.zeros((len(positions), len(multipliers)))

    for i, pos in enumerate(positions):
        for j, mult in enumerate(multipliers):
            if mult in matrix[pos]:
                acc_matrix[i, j] = matrix[pos][mult]['mean']

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(acc_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0, vmax=1.0, center=0.5,
                xticklabels=[f'{m:.1f}x' for m in multipliers],
                yticklabels=[f'Token {p}' for p in positions],
                cbar_kws={'label': 'Accuracy'},
                ax=ax)

    ax.set_xlabel('Enhancement Multiplier', fontsize=13, fontweight='bold')
    ax.set_ylabel('Token Position', fontsize=13, fontweight='bold')
    ax.set_title('Token Enhancement Effects: Accuracy by Position × Multiplier',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        output_file = output_dir / f'enhancement_heatmap.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")

    plt.close()


def plot_position_effects(position_stats, output_dir):
    """Plot enhancement responsiveness by position."""
    fig, ax = plt.subplots(figsize=(10, 6))

    positions = sorted(position_stats['position_effects'].keys())
    effects = [position_stats['position_effects'][pos]['mean_effect'] for pos in positions]
    stds = [position_stats['position_effects'][pos]['std_effect'] for pos in positions]

    colors = ['#3498DB' if abs(eff) < 0.05 else '#27AE60' if eff > 0 else '#E74C3C'
              for eff in effects]

    bars = ax.bar(positions, effects, yerr=stds, color=colors, alpha=0.7,
                  edgecolor='black', capsize=5)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Token Position', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean Enhancement Effect', fontsize=13, fontweight='bold')
    ax.set_title('Enhancement Responsiveness by Token Position',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(positions)
    ax.grid(True, alpha=0.3, axis='y')

    # Add ANOVA result
    if position_stats['anova']['p_value'] is not None:
        anova_text = f"ANOVA: F={position_stats['anova']['f_statistic']:.2f}, "
        anova_text += f"p={position_stats['anova']['p_value']:.4f}"
        if position_stats['anova']['significant']:
            anova_text += " (SIGNIFICANT)"
        ax.text(0.02, 0.98, anova_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        output_file = output_dir / f'enhancement_effects.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")

    plt.close()


def analyze_enhancement_results():
    """Main analysis function."""
    # Paths
    results_file = Path(__file__).parent.parent / 'results' / 'enhancement_test_10.json'
    output_dir = Path(__file__).parent.parent / 'figures'
    stats_file = Path(__file__).parent.parent / 'results' / 'enhancement_analysis.json'

    print("=" * 80)
    print("ENHANCEMENT ANALYSIS")
    print("=" * 80)

    # Load results
    print(f"\nLoading results from {results_file}...")
    results = load_results(results_file)
    print(f"Loaded {len(results)} problems")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute enhancement matrix
    print("\nComputing enhancement matrix...")
    matrix = compute_enhancement_matrix(results)

    print("\nAccuracy by position and multiplier:")
    for position in sorted(matrix.keys()):
        print(f"\n  Token {position}:")
        for multiplier in sorted(matrix[position].keys()):
            stats = matrix[position][multiplier]
            print(f"    {multiplier:.1f}x: {stats['mean']:.1%} ± {stats['sem']:.1%}")

    # Identify optimal multipliers
    print("\nIdentifying optimal multipliers...")
    optimal = identify_optimal_multipliers(matrix)

    for position, data in sorted(optimal.items()):
        print(f"  Token {position}: {data['optimal_multiplier']:.1f}x "
              f"(accuracy: {data['max_accuracy']:.1%})")

    # Test position criticality
    print("\nTesting position criticality...")
    position_stats = test_position_criticality(results)

    print("\nEnhancement effects by position:")
    for pos in sorted(position_stats['position_effects'].keys()):
        data = position_stats['position_effects'][pos]
        print(f"  Token {pos}: {data['mean_effect']:+.3f} ± {data['std_effect']:.3f}")

    if position_stats['anova']['p_value'] is not None:
        print(f"\nANOVA test:")
        print(f"  F-statistic: {position_stats['anova']['f_statistic']:.2f}")
        print(f"  P-value: {position_stats['anova']['p_value']:.4f}")
        print(f"  Significant: {position_stats['anova']['significant']}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_enhancement_heatmap(matrix, output_dir)
    plot_position_effects(position_stats, output_dir)

    # Save statistics
    analysis_stats = {
        'enhancement_matrix': {
            str(pos): {
                str(mult): stats
                for mult, stats in mult_data.items()
            }
            for pos, mult_data in matrix.items()
        },
        'optimal_multipliers': {
            str(pos): data
            for pos, data in optimal.items()
        },
        'position_criticality': position_stats
    }

    with open(stats_file, 'w') as f:
    # Convert numpy types to Python types
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

    # Apply conversion
        json.dump(convert_to_python(analysis_stats), f, indent=2)

    print(f"\n✓ Statistics saved to {stats_file}")
    print("=" * 80)


if __name__ == "__main__":
    analyze_enhancement_results()
