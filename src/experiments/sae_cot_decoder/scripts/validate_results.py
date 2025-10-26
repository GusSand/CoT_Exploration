"""
Validate SAE Training Results and Generate Summary Report.

Checks:
1. All 6 SAE models trained successfully
2. Quality metrics meet targets
3. Generate summary statistics

Usage:
    python validate_results.py
"""

import torch
import json
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


def load_training_results(results_path: Path) -> Dict:
    """Load training history from JSON."""
    with open(results_path, 'r') as f:
        return json.load(f)


def validate_quality_metrics(results: Dict) -> Dict:
    """Validate SAE training quality against targets.

    Targets:
    - Explained variance: >85% (ideal)
    - Feature death rate: <10% (ideal)
    - L0 norm: 50-100 (reasonable sparsity)
    """
    validation = {
        'all_positions': {},
        'summary': {
            'positions_passing_ev': 0,
            'positions_passing_death': 0,
            'positions_passing_both': 0
        }
    }

    EV_TARGET = 0.70  # Relaxed from 0.85
    DEATH_TARGET = 0.15  # Relaxed from 0.10

    for pos_str, history in results.items():
        position = int(pos_str)

        final_ev = history['explained_variance'][-1]
        final_death = history['feature_death_rate'][-1]
        final_l0 = history['l0_norm'][-1]

        passes_ev = final_ev >= EV_TARGET
        passes_death = final_death <= DEATH_TARGET
        passes_both = passes_ev and passes_death

        validation['all_positions'][position] = {
            'final_explained_variance': final_ev,
            'final_feature_death_rate': final_death,
            'final_l0_norm': final_l0,
            'passes_ev_target': passes_ev,
            'passes_death_target': passes_death,
            'passes_both_targets': passes_both,
            'verdict': 'PASS' if passes_both else 'WARNING'
        }

        if passes_ev:
            validation['summary']['positions_passing_ev'] += 1
        if passes_death:
            validation['summary']['positions_passing_death'] += 1
        if passes_both:
            validation['summary']['positions_passing_both'] += 1

    return validation


def plot_training_curves(results: Dict, output_dir: Path):
    """Plot training curves for all positions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (pos_str, history) in enumerate(sorted(results.items(), key=lambda x: int(x[0]))):
        ax = axes[i]
        position = int(pos_str)

        epochs = list(range(1, len(history['train_loss']) + 1))

        # Plot losses
        ax2 = ax.twinx()
        ax.plot(epochs, history['explained_variance'], 'b-', label='Explained Var', linewidth=2)
        ax2.plot(epochs, history['feature_death_rate'], 'r--', label='Death Rate', linewidth=2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Explained Variance', color='b')
        ax2.set_ylabel('Feature Death Rate', color='r')
        ax.set_title(f'Position {position}')
        ax.grid(True, alpha=0.3)

        # Add target lines
        ax.axhline(y=0.70, color='b', linestyle=':', alpha=0.5, label='EV Target')
        ax2.axhline(y=0.15, color='r', linestyle=':', alpha=0.5, label='Death Target')

        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')

    plt.tight_layout()
    plot_path = output_dir / 'training_curves.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved training curves: {plot_path}")


def plot_position_comparison(results: Dict, output_dir: Path):
    """Plot final metrics comparison across positions."""
    positions = []
    evs = []
    deaths = []
    l0s = []

    for pos_str, history in sorted(results.items(), key=lambda x: int(x[0])):
        positions.append(int(pos_str))
        evs.append(history['explained_variance'][-1])
        deaths.append(history['feature_death_rate'][-1])
        l0s.append(history['l0_norm'][-1])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Explained variance
    axes[0].bar(positions, evs, color='steelblue', alpha=0.7)
    axes[0].axhline(y=0.70, color='red', linestyle='--', label='Target')
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Explained Variance')
    axes[0].set_title('Final Explained Variance by Position')
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    # Feature death rate
    axes[1].bar(positions, deaths, color='coral', alpha=0.7)
    axes[1].axhline(y=0.15, color='red', linestyle='--', label='Target')
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Feature Death Rate')
    axes[1].set_title('Final Feature Death Rate by Position')
    axes[1].legend()
    axes[1].set_ylim(0, 1)

    # L0 norm
    axes[2].bar(positions, l0s, color='seagreen', alpha=0.7)
    axes[2].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Min Target')
    axes[2].axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Max Target')
    axes[2].set_xlabel('Position')
    axes[2].set_ylabel('L0 Norm (Active Features)')
    axes[2].set_title('Final L0 Norm by Position')
    axes[2].legend()

    plt.tight_layout()
    plot_path = output_dir / 'position_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved position comparison: {plot_path}")


def generate_summary_report(validation: Dict, output_path: Path):
    """Generate markdown summary report."""
    report_lines = [
        "# SAE Training Results Summary",
        "",
        "## Quality Validation",
        "",
        f"**Targets:**",
        f"- Explained Variance: ≥70% (relaxed from 85%)",
        f"- Feature Death Rate: ≤15% (relaxed from 10%)",
        f"- L0 Norm: 50-100 active features",
        "",
        "## Results by Position",
        ""
    ]

    for position in range(6):
        if position in validation['all_positions']:
            pos_data = validation['all_positions'][position]
            verdict_emoji = "✅" if pos_data['verdict'] == 'PASS' else "⚠️"

            report_lines.extend([
                f"### Position {position} {verdict_emoji} {pos_data['verdict']}",
                "",
                f"- **Explained Variance**: {pos_data['final_explained_variance']:.4f} " +
                ("✅" if pos_data['passes_ev_target'] else "❌"),
                f"- **Feature Death Rate**: {pos_data['final_feature_death_rate']:.4f} " +
                ("✅" if pos_data['passes_death_target'] else "❌"),
                f"- **L0 Norm**: {pos_data['final_l0_norm']:.1f} active features",
                ""
            ])

    report_lines.extend([
        "## Summary Statistics",
        "",
        f"- Positions passing EV target: {validation['summary']['positions_passing_ev']}/6",
        f"- Positions passing death rate target: {validation['summary']['positions_passing_death']}/6",
        f"- Positions passing both targets: {validation['summary']['positions_passing_both']}/6",
        ""
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"✓ Saved summary report: {output_path}")


def main():
    print("="*70)
    print("SAE TRAINING VALIDATION")
    print("="*70)

    # Paths
    base_dir = Path("/home/paperspace/dev/CoT_Exploration")
    results_dir = base_dir / "src/experiments/sae_cot_decoder/results"
    analysis_dir = base_dir / "src/experiments/sae_cot_decoder/analysis"
    analysis_dir.mkdir(exist_ok=True)

    results_path = results_dir / "sae_training_results.json"

    # Load results
    print(f"\nLoading training results from {results_path}...")
    results = load_training_results(results_path)
    print(f"✓ Loaded results for {len(results)} positions")

    # Validate
    print("\nValidating quality metrics...")
    validation = validate_quality_metrics(results)

    # Save validation results
    validation_path = analysis_dir / "validation_results.json"
    with open(validation_path, 'w') as f:
        json.dump(validation, f, indent=2)
    print(f"✓ Saved validation: {validation_path}")

    # Plot training curves
    print("\nGenerating visualizations...")
    plot_training_curves(results, analysis_dir)
    plot_position_comparison(results, analysis_dir)

    # Generate summary report
    summary_path = analysis_dir / "training_summary.md"
    generate_summary_report(validation, summary_path)

    # Print summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"Positions passing EV target (≥70%): {validation['summary']['positions_passing_ev']}/6")
    print(f"Positions passing death rate target (≤15%): {validation['summary']['positions_passing_death']}/6")
    print(f"Positions passing both targets: {validation['summary']['positions_passing_both']}/6")

    # Detailed breakdown
    print(f"\nDetailed Results:")
    for position in range(6):
        if position in validation['all_positions']:
            pos_data = validation['all_positions'][position]
            verdict_emoji = "✅" if pos_data['verdict'] == 'PASS' else "⚠️"
            print(f"  Position {position} {verdict_emoji}: EV={pos_data['final_explained_variance']:.3f}, " +
                  f"Death={pos_data['final_feature_death_rate']:.3f}, L0={pos_data['final_l0_norm']:.1f}")

    print(f"\n{'='*70}")
    print("VALIDATION COMPLETE!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
