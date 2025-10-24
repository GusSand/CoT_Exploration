#!/usr/bin/env python3
"""
Combined Token Criticality Analysis

Synthesizes corruption and enhancement experiments to create comprehensive
token criticality ranking.

Key questions:
- Which tokens are most critical overall?
- Do corruption and enhancement agree on critical tokens?
- How do findings compare to CCTA results and paper claims?

Usage:
    python 5_combined_analysis.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11


def load_results():
    """Load all experiment results."""
    base_dir = Path(__file__).parent.parent / 'results'

    with open(base_dir / 'threshold_test_10.json', 'r') as f:
        threshold = json.load(f)

    with open(base_dir / 'enhancement_test_10.json', 'r') as f:
        enhancement = json.load(f)

    with open(base_dir / 'threshold_analysis.json', 'r') as f:
        threshold_stats = json.load(f)

    with open(base_dir / 'enhancement_analysis.json', 'r') as f:
        enhancement_stats = json.load(f)

    return {
        'threshold_results': threshold,
        'enhancement_results': enhancement,
        'threshold_stats': threshold_stats,
        'enhancement_stats': enhancement_stats
    }


def compute_corruption_criticality(threshold_results):
    """
    Compute token criticality from corruption experiments.

    Uses level=1 (single token corruption) to measure importance.

    Returns:
        dict: {position: criticality_score}
    """
    criticality = {}

    for position in range(6):
        # Count how often corrupting this token causes failure
        failures = []

        for problem in threshold_results:
            if 'error' in problem or not problem['baseline']['correct']:
                continue

            for corruption in problem['corruptions']:
                if (corruption['corruption_level'] == 1 and
                    corruption['positions'] == [position]):
                    failures.append(1 if corruption['importance'] else 0)

        if failures:
            criticality[position] = {
                'failure_rate': np.mean(failures),
                'count': len(failures)
            }

    return criticality


def compute_enhancement_responsiveness(enhancement_results):
    """
    Compute token enhancement responsiveness.

    Measures how much each position benefits from amplification.

    Returns:
        dict: {position: responsiveness_score}
    """
    responsiveness = {}

    for position in range(6):
        # Compute mean enhancement effect across all multipliers
        effects = []

        for problem in enhancement_results:
            if 'error' in problem:
                continue

            baseline = problem['baseline']['correct']

            position_enhancements = [
                e for e in problem['enhancements']
                if e['position'] == position
            ]

            for enhancement in position_enhancements:
                # Effect = change from baseline
                effect = (1 if enhancement['correct'] else 0) - (1 if baseline else 0)
                effects.append(effect)

        if effects:
            responsiveness[position] = {
                'mean_effect': np.mean(effects),
                'std_effect': np.std(effects),
                'count': len(effects)
            }

    return responsiveness


def create_criticality_ranking(corruption_criticality, enhancement_responsiveness):
    """
    Create comprehensive token criticality ranking.

    Combines corruption vulnerability and enhancement responsiveness.

    Returns:
        dict: Ranked tokens with combined scores
    """
    rankings = {}

    for position in range(6):
        # Get corruption criticality (higher = more critical)
        corruption_score = corruption_criticality.get(position, {}).get('failure_rate', 0)

        # Get enhancement responsiveness (higher = more responsive to enhancement)
        enhancement_score = enhancement_responsiveness.get(position, {}).get('mean_effect', 0)

        # Combined score: corruption criticality indicates importance
        # We use corruption as primary indicator
        combined_score = corruption_score

        rankings[position] = {
            'corruption_criticality': corruption_score,
            'enhancement_responsiveness': enhancement_score,
            'combined_score': combined_score
        }

    # Sort by combined score
    sorted_positions = sorted(rankings.keys(),
                             key=lambda p: rankings[p]['combined_score'],
                             reverse=True)

    return rankings, sorted_positions


def test_convergent_validity(corruption_criticality, enhancement_responsiveness):
    """
    Test if corruption and enhancement measures correlate (convergent validity).

    Returns:
        dict: Correlation statistics
    """
    positions = list(range(6))

    corruption_scores = [
        corruption_criticality.get(p, {}).get('failure_rate', 0)
        for p in positions
    ]

    enhancement_scores = [
        enhancement_responsiveness.get(p, {}).get('mean_effect', 0)
        for p in positions
    ]

    # Pearson correlation
    r_pearson, p_pearson = stats.pearsonr(corruption_scores, enhancement_scores)

    # Spearman correlation (rank-based)
    r_spearman, p_spearman = stats.spearmanr(corruption_scores, enhancement_scores)

    return {
        'pearson': {
            'r': r_pearson,
            'p_value': p_pearson,
            'significant': p_pearson < 0.05
        },
        'spearman': {
            'r': r_spearman,
            'p_value': p_spearman,
            'significant': p_spearman < 0.05
        },
        'interpretation': 'Convergent' if abs(r_pearson) > 0.5 else 'Divergent'
    }


def plot_combined_ranking(rankings, sorted_positions, output_dir):
    """Plot combined token criticality ranking."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    positions = list(range(6))

    # Plot 1: Corruption criticality
    ax = axes[0]
    corruption_scores = [rankings[p]['corruption_criticality'] for p in positions]
    colors = sns.color_palette("Reds", len(positions))
    bars = ax.bar(positions, corruption_scores, color=colors, edgecolor='black', alpha=0.7)

    ax.set_xlabel('Token Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Failure Rate (Criticality)', fontsize=12, fontweight='bold')
    ax.set_title('Corruption Vulnerability\n(Higher = More Critical)', fontsize=13, fontweight='bold')
    ax.set_xticks(positions)
    ax.set_ylim(0, max(corruption_scores) * 1.2 if corruption_scores else 1)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Enhancement responsiveness
    ax = axes[1]
    enhancement_scores = [rankings[p]['enhancement_responsiveness'] for p in positions]
    colors_enh = ['#27AE60' if s > 0 else '#E74C3C' if s < 0 else '#95A5A6'
                   for s in enhancement_scores]
    bars = ax.bar(positions, enhancement_scores, color=colors_enh, edgecolor='black', alpha=0.7)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Token Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Enhancement Effect', fontsize=12, fontweight='bold')
    ax.set_title('Enhancement Responsiveness\n(+/- from baseline)', fontsize=13, fontweight='bold')
    ax.set_xticks(positions)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Combined ranking
    ax = axes[2]
    combined_scores = [rankings[p]['combined_score'] for p in positions]

    # Color by rank
    rank_colors = sns.color_palette("viridis", len(positions))
    rank_map = {pos: i for i, pos in enumerate(sorted_positions)}
    colors_combined = [rank_colors[rank_map[p]] for p in positions]

    bars = ax.bar(positions, combined_scores, color=colors_combined, edgecolor='black', alpha=0.7)

    ax.set_xlabel('Token Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Criticality Score', fontsize=12, fontweight='bold')
    ax.set_title('Overall Token Criticality\n(Corruption-based ranking)', fontsize=13, fontweight='bold')
    ax.set_xticks(positions)
    ax.set_ylim(0, max(combined_scores) * 1.2 if combined_scores else 1)
    ax.grid(True, alpha=0.3, axis='y')

    # Add rank annotations
    for pos in positions:
        rank = rank_map[pos] + 1
        ax.text(pos, combined_scores[pos] + 0.02, f'#{rank}',
                ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        output_file = output_dir / f'combined_ranking.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")

    plt.close()


def plot_convergent_validity(corruption_criticality, enhancement_responsiveness,
                            convergence_stats, output_dir):
    """Plot correlation between corruption and enhancement."""
    fig, ax = plt.subplots(figsize=(8, 8))

    positions = list(range(6))

    corruption_scores = [
        corruption_criticality.get(p, {}).get('failure_rate', 0)
        for p in positions
    ]

    enhancement_scores = [
        enhancement_responsiveness.get(p, {}).get('mean_effect', 0)
        for p in positions
    ]

    # Scatter plot
    ax.scatter(corruption_scores, enhancement_scores, s=200, alpha=0.6,
               c=positions, cmap='viridis', edgecolors='black', linewidths=2)

    # Add labels
    for i, pos in enumerate(positions):
        ax.text(corruption_scores[i] + 0.01, enhancement_scores[i] + 0.005,
                f'T{pos}', fontsize=11, fontweight='bold')

    # Fit line
    if len(corruption_scores) > 1:
        z = np.polyfit(corruption_scores, enhancement_scores, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(corruption_scores), max(corruption_scores), 100)
        ax.plot(x_line, p(x_line), 'r--', alpha=0.5, linewidth=2, label='Linear fit')

    # Add correlation stats
    r = convergence_stats['pearson']['r']
    p_val = convergence_stats['pearson']['p_value']
    text = f"Pearson r = {r:.3f}\np = {p_val:.4f}\n"
    text += f"Interpretation: {convergence_stats['interpretation']}"

    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('Corruption Criticality (Failure Rate)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Enhancement Responsiveness (Mean Effect)', fontsize=13, fontweight='bold')
    ax.set_title('Convergent Validity: Corruption vs Enhancement',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        output_file = output_dir / f'convergent_validity.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")

    plt.close()


def run_combined_analysis():
    """Main combined analysis function."""
    output_dir = Path(__file__).parent.parent / 'figures'
    stats_file = Path(__file__).parent.parent / 'results' / 'combined_analysis.json'

    print("=" * 80)
    print("COMBINED TOKEN CRITICALITY ANALYSIS")
    print("=" * 80)

    # Load all results
    print("\nLoading experiment results...")
    data = load_results()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute corruption criticality
    print("\nComputing corruption criticality...")
    corruption_criticality = compute_corruption_criticality(data['threshold_results'])

    for pos in sorted(corruption_criticality.keys()):
        score = corruption_criticality[pos]['failure_rate']
        print(f"  Token {pos}: {score:.1%} failure rate")

    # Compute enhancement responsiveness
    print("\nComputing enhancement responsiveness...")
    enhancement_responsiveness = compute_enhancement_responsiveness(data['enhancement_results'])

    for pos in sorted(enhancement_responsiveness.keys()):
        score = enhancement_responsiveness[pos]['mean_effect']
        print(f"  Token {pos}: {score:+.3f} mean effect")

    # Create combined ranking
    print("\nCreating combined criticality ranking...")
    rankings, sorted_positions = create_criticality_ranking(
        corruption_criticality, enhancement_responsiveness
    )

    print("\nToken Criticality Ranking (Most to Least Critical):")
    for rank, pos in enumerate(sorted_positions, 1):
        data = rankings[pos]
        print(f"  #{rank} Token {pos}:")
        print(f"      Corruption: {data['corruption_criticality']:.1%}")
        print(f"      Enhancement: {data['enhancement_responsiveness']:+.3f}")
        print(f"      Combined: {data['combined_score']:.3f}")

    # Test convergent validity
    print("\nTesting convergent validity...")
    convergence_stats = test_convergent_validity(
        corruption_criticality, enhancement_responsiveness
    )

    print(f"  Pearson r: {convergence_stats['pearson']['r']:.3f} "
          f"(p={convergence_stats['pearson']['p_value']:.4f})")
    print(f"  Spearman r: {convergence_stats['spearman']['r']:.3f} "
          f"(p={convergence_stats['spearman']['p_value']:.4f})")
    print(f"  Interpretation: {convergence_stats['interpretation']}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_combined_ranking(rankings, sorted_positions, output_dir)
    plot_convergent_validity(corruption_criticality, enhancement_responsiveness,
                            convergence_stats, output_dir)

    # Save combined statistics
    combined_stats = {
        'corruption_criticality': {
            str(pos): data
            for pos, data in corruption_criticality.items()
        },
        'enhancement_responsiveness': {
            str(pos): data
            for pos, data in enhancement_responsiveness.items()
        },
        'rankings': {
            str(pos): data
            for pos, data in rankings.items()
        },
        'sorted_positions': [int(p) for p in sorted_positions],
        'convergent_validity': convergence_stats
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
        json.dump(convert_to_python(combined_stats), f, indent=2)

    print(f"\n✓ Combined statistics saved to {stats_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_combined_analysis()
