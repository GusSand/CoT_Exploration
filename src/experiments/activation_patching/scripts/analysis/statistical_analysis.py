"""
Statistical Analysis for Activation Patching Experiment

Provides honest statistical assessment of results including:
- Significance tests (binomial, Fisher's exact)
- Confidence intervals (Wilson method)
- Power analysis
- Required sample size calculations
- Effect sizes

Usage:
    python statistical_analysis.py --results results/experiment_results_corrected.json
"""

import json
import argparse
from typing import Dict, List
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


class StatisticalAnalyzer:
    """Perform rigorous statistical analysis on activation patching results."""

    def __init__(self, results_path: str):
        """Load results and extract key metrics."""
        with open(results_path, 'r') as f:
            self.results = json.load(f)

        self.summary = self.results['summary']
        self.n_total = self.summary['total_valid']
        self.n_targets = self.summary['total_targets']

        # Extract recovery counts
        self.layers = ['early', 'middle', 'late']
        self.recovery_counts = {
            layer: self.summary[layer]['correct_count']
            for layer in self.layers
        }

    def binomial_test(self, k: int, n: int, p0: float = 0.5,
                      alternative: str = 'greater') -> Dict:
        """Perform binomial test against null hypothesis p=p0.

        Args:
            k: Number of successes
            n: Number of trials
            p0: Null hypothesis probability
            alternative: 'greater', 'less', or 'two-sided'

        Returns:
            Dict with p_value, significant, observed_rate
        """
        p_value = stats.binom_test(k, n, p=p0, alternative=alternative)

        return {
            'k': k,
            'n': n,
            'p0': p0,
            'observed_rate': k / n,
            'p_value': p_value,
            'significant_05': p_value < 0.05,
            'significant_01': p_value < 0.01,
            'alternative': alternative
        }

    def confidence_interval(self, k: int, n: int, alpha: float = 0.05) -> Dict:
        """Calculate confidence interval using Wilson method.

        Args:
            k: Number of successes
            n: Number of trials
            alpha: Significance level (default 0.05 for 95% CI)

        Returns:
            Dict with lower, upper, width, and includes_50 flag
        """
        ci = proportion_confint(k, n, alpha=alpha, method='wilson')

        return {
            'k': k,
            'n': n,
            'observed_rate': k / n,
            'confidence_level': 1 - alpha,
            'lower': ci[0],
            'upper': ci[1],
            'width': ci[1] - ci[0],
            'includes_50': ci[0] <= 0.5 <= ci[1],
            'includes_0': ci[0] <= 0.0 <= ci[1]
        }

    def cohens_h(self, p1: float, p2: float) -> float:
        """Calculate Cohen's h effect size for two proportions.

        Args:
            p1: First proportion
            p2: Second proportion (usually null hypothesis)

        Returns:
            Cohen's h (standardized difference)
        """
        return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

    def required_sample_size(self, p1: float, p2: float,
                            alpha: float = 0.05, power: float = 0.8) -> int:
        """Calculate required sample size for given effect and power.

        Args:
            p1: Expected proportion (alternative hypothesis)
            p2: Null hypothesis proportion
            alpha: Significance level
            power: Desired statistical power

        Returns:
            Required sample size (integer)
        """
        h = self.cohens_h(p1, p2)

        # Z-scores for alpha and power
        z_alpha = stats.norm.ppf(1 - alpha/2)  # Two-tailed
        z_beta = stats.norm.ppf(power)

        # Sample size formula
        n = ((z_alpha + z_beta) / h) ** 2

        return int(np.ceil(n))

    def analyze_layer(self, layer_name: str) -> Dict:
        """Complete statistical analysis for one layer.

        Args:
            layer_name: 'early', 'middle', or 'late'

        Returns:
            Dict with all statistical tests and metrics
        """
        k = self.recovery_counts[layer_name]
        n = self.n_targets

        return {
            'layer': layer_name,
            'observed': {
                'k': k,
                'n': n,
                'rate': k / n
            },
            'binomial_test_vs_50': self.binomial_test(k, n, p0=0.5),
            'binomial_test_vs_0': self.binomial_test(k, n, p0=0.0, alternative='greater'),
            'confidence_interval_95': self.confidence_interval(k, n, alpha=0.05),
            'confidence_interval_99': self.confidence_interval(k, n, alpha=0.01),
            'effect_size_vs_0': self.cohens_h(k/n, 0.0),
            'effect_size_vs_50': self.cohens_h(k/n, 0.5)
        }

    def sample_size_analysis(self) -> Dict:
        """Calculate required sample sizes for various scenarios.

        Returns:
            Dict with required n for different effect sizes and baselines
        """
        scenarios = []

        # Scenario 1: Detect current observed effect (55.6% vs 0%)
        observed_late = self.recovery_counts['late'] / self.n_targets
        n_vs_0 = self.required_sample_size(observed_late, 0.0)
        scenarios.append({
            'scenario': 'Current effect vs no recovery (55.6% vs 0%)',
            'p1': observed_late,
            'p2': 0.0,
            'required_n': n_vs_0,
            'current_n': self.n_targets,
            'need_multiplier': n_vs_0 / self.n_targets
        })

        # Scenario 2: Detect vs random chance (55.6% vs 50%)
        n_vs_50 = self.required_sample_size(observed_late, 0.5)
        scenarios.append({
            'scenario': 'Current effect vs random chance (55.6% vs 50%)',
            'p1': observed_late,
            'p2': 0.5,
            'required_n': n_vs_50,
            'current_n': self.n_targets,
            'need_multiplier': n_vs_50 / self.n_targets
        })

        # Scenario 3: Detect 70% recovery vs 0%
        n_70_vs_0 = self.required_sample_size(0.7, 0.0)
        scenarios.append({
            'scenario': 'Strong effect (70% vs 0%)',
            'p1': 0.7,
            'p2': 0.0,
            'required_n': n_70_vs_0,
            'current_n': self.n_targets,
            'need_multiplier': n_70_vs_0 / self.n_targets
        })

        # Scenario 4: Detect 70% vs 30%
        n_70_vs_30 = self.required_sample_size(0.7, 0.3)
        scenarios.append({
            'scenario': 'Moderate effect (70% vs 30%)',
            'p1': 0.7,
            'p2': 0.3,
            'required_n': n_70_vs_30,
            'current_n': self.n_targets,
            'need_multiplier': n_70_vs_30 / self.n_targets
        })

        return {'scenarios': scenarios}

    def generate_report(self) -> str:
        """Generate comprehensive statistical report.

        Returns:
            Formatted text report
        """
        report = []
        report.append("="*80)
        report.append("STATISTICAL ANALYSIS REPORT - Activation Patching Experiment")
        report.append("="*80)

        # Sample size summary
        report.append(f"\n📊 SAMPLE SIZE:")
        report.append(f"  Total problem pairs: {self.results['config']['total_pairs']}")
        report.append(f"  Valid pairs (clean ✓): {self.n_total}")
        report.append(f"  Target cases (clean ✓, corrupted ✗): {self.n_targets}")

        # Per-layer analysis
        for layer in self.layers:
            analysis = self.analyze_layer(layer)

            report.append(f"\n{'='*80}")
            report.append(f"LAYER: {layer.upper()}")
            report.append(f"{'='*80}")

            # Observed results
            obs = analysis['observed']
            report.append(f"\n  Observed: {obs['k']}/{obs['n']} = {obs['rate']*100:.1f}%")

            # Binomial test vs 50%
            bt50 = analysis['binomial_test_vs_50']
            sig_marker = "✓ SIGNIFICANT" if bt50['significant_05'] else "✗ NOT significant"
            report.append(f"\n  Test vs random chance (50%):")
            report.append(f"    p-value = {bt50['p_value']:.4f}  {sig_marker} (α=0.05)")

            # Confidence intervals
            ci95 = analysis['confidence_interval_95']
            ci99 = analysis['confidence_interval_99']
            report.append(f"\n  95% Confidence Interval: [{ci95['lower']*100:.1f}%, {ci95['upper']*100:.1f}%]")
            report.append(f"    Width: {ci95['width']*100:.1f} percentage points")
            report.append(f"    Includes 50% (random chance)? {'YES ⚠️' if ci95['includes_50'] else 'NO ✓'}")

            # Effect sizes
            report.append(f"\n  Effect Size (Cohen's h):")
            report.append(f"    vs 0% baseline: h = {analysis['effect_size_vs_0']:.3f}")
            report.append(f"    vs 50% baseline: h = {analysis['effect_size_vs_50']:.3f}")

        # Sample size analysis
        report.append(f"\n{'='*80}")
        report.append("REQUIRED SAMPLE SIZE ANALYSIS")
        report.append(f"{'='*80}")
        report.append("\nFor 80% power and α=0.05 (two-tailed):\n")

        ss_analysis = self.sample_size_analysis()
        for scenario in ss_analysis['scenarios']:
            report.append(f"  {scenario['scenario']}:")
            report.append(f"    Required n: {scenario['required_n']} target cases")
            report.append(f"    Current n: {scenario['current_n']} target cases")
            report.append(f"    Need {scenario['need_multiplier']:.1f}x more data")
            report.append("")

        # Overall verdict
        report.append(f"{'='*80}")
        report.append("STATISTICAL VERDICT")
        report.append(f"{'='*80}")

        # Check if any layer is significant
        any_sig = any(
            self.analyze_layer(layer)['binomial_test_vs_50']['significant_05']
            for layer in self.layers
        )

        if any_sig:
            report.append("\n✓ At least one layer shows statistically significant recovery")
        else:
            report.append("\n❌ NO layers show statistically significant recovery (p ≥ 0.05)")

        # Check CIs
        late_analysis = self.analyze_layer('late')
        if late_analysis['confidence_interval_95']['includes_50']:
            report.append("❌ Confidence intervals include 50% (cannot reject random chance)")

        # Sample size verdict
        report.append(f"❌ Sample size (n={self.n_targets}) is inadequate for robust conclusions")

        # Recommendations
        min_n = min(s['required_n'] for s in ss_analysis['scenarios'])
        report.append(f"\n✓ Minimum recommended n: {min_n} target cases")
        report.append("✓ Results are SUGGESTIVE but not statistically conclusive")
        report.append("✓ Suitable for pilot study / hypothesis generation")
        report.append("✓ NOT suitable for publication without replication at larger n")

        report.append(f"\n{'='*80}")

        return "\n".join(report)

    def save_report(self, output_path: str):
        """Save statistical report to file."""
        report = self.generate_report()
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"✓ Statistical report saved to {output_path}")

    def plot_results(self, output_dir: str):
        """Generate statistical visualization plots."""
        import matplotlib.pyplot as plt
        from pathlib import Path

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: Confidence intervals
        fig, ax = plt.subplots(figsize=(10, 6))

        y_pos = np.arange(len(self.layers))
        colors = ['#90EE90', '#90EE90', '#4CAF50']

        for i, layer in enumerate(self.layers):
            analysis = self.analyze_layer(layer)
            obs = analysis['observed']['rate']
            ci = analysis['confidence_interval_95']

            # Plot point estimate
            ax.scatter(obs * 100, i, s=200, color=colors[i],
                      edgecolor='black', linewidth=2, zorder=3, label=f'{layer.capitalize()}')

            # Plot CI
            ax.plot([ci['lower']*100, ci['upper']*100], [i, i],
                   color=colors[i], linewidth=4, zorder=2)

            # Add text
            ax.text(ci['upper']*100 + 3, i, f"{obs*100:.1f}%",
                   va='center', fontsize=11, fontweight='bold')

        # Reference lines
        ax.axvline(x=50, color='red', linestyle='--', linewidth=2,
                  label='Random Chance (50%)', alpha=0.7)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5,
                  label='No Recovery (0%)', alpha=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'{l.capitalize()} (L{3+3*i})' for i, l in enumerate(self.layers)])
        ax.set_xlabel('Recovery Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Recovery Rates with 95% Confidence Intervals\n(n={self.n_targets} target cases)',
                    fontsize=14, fontweight='bold')
        ax.set_xlim(-10, 100)
        ax.grid(axis='x', alpha=0.3)
        ax.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(output_dir / 'confidence_intervals.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: confidence_intervals.png")
        plt.close()

        # Plot 2: Sample size requirements
        fig, ax = plt.subplots(figsize=(12, 6))

        ss_analysis = self.sample_size_analysis()
        scenarios = ss_analysis['scenarios']

        x_pos = np.arange(len(scenarios))
        required_ns = [s['required_n'] for s in scenarios]
        labels = [s['scenario'] for s in scenarios]

        bars = ax.bar(x_pos, required_ns, color='steelblue', edgecolor='black', linewidth=2)

        # Add current n line
        ax.axhline(y=self.n_targets, color='red', linestyle='--', linewidth=2.5,
                  label=f'Current n = {self.n_targets}')

        # Add value labels
        for bar, val in zip(bars, required_ns):
            ax.text(bar.get_x() + bar.get_width()/2, val + 5, f'n={val}',
                   ha='center', fontsize=10, fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
        ax.set_ylabel('Required Sample Size (n)', fontsize=12, fontweight='bold')
        ax.set_title('Required Sample Size for 80% Power (α=0.05)',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'sample_size_requirements.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: sample_size_requirements.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Statistical analysis of activation patching results')
    parser.add_argument('--results', type=str,
                       default='results/experiment_results_corrected.json',
                       help='Path to experiment results JSON')
    parser.add_argument('--output_dir', type=str,
                       default='results/statistical_analysis/',
                       help='Output directory for report and plots')
    args = parser.parse_args()

    # Run analysis
    analyzer = StatisticalAnalyzer(args.results)

    # Generate and save report
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    report_path = os.path.join(args.output_dir, 'statistical_report.txt')
    analyzer.save_report(report_path)

    # Generate plots
    analyzer.plot_results(args.output_dir)

    # Print report to console
    print("\n" + analyzer.generate_report())

    print(f"\n✓ Analysis complete! Files saved to {args.output_dir}")


if __name__ == "__main__":
    main()
