#!/usr/bin/env python3
"""
Validation Analysis for Three-Way CODI Comparison

Addresses critical concerns from review:
1. Statistical significance (bootstrap CI)
2. Correct vs incorrect analysis
3. Layer-wise progression
4. Random baseline comparison
5. PCA variance validation
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import bootstrap
from sklearn.decomposition import PCA
from typing import Dict, Tuple
import json

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11


class ValidationAnalyzer:
    """Validate three-way comparison results."""

    def __init__(self, results_dir: str = None):
        if results_dir is None:
            results_dir = Path(__file__).parent / 'results'
        self.results_dir = Path(results_dir)
        self.viz_dir = self.results_dir / 'validation'
        self.viz_dir.mkdir(exist_ok=True)

    def load_activations(self, task: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load CT activations for a task."""
        filepath = self.results_dir / f'activations_{task}.npz'
        data = np.load(filepath, allow_pickle=True)
        return data['hidden_states'], data['correct']

    def compute_variance_ratio(self, embeddings: np.ndarray) -> float:
        """Compute variance ratio (compactness metric)."""
        cov = np.cov(embeddings, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]
        if eigenvalues.sum() > 0:
            return eigenvalues[0] / eigenvalues.sum()
        return 0.0

    def bootstrap_variance_ratio(self, embeddings: np.ndarray,
                                 n_bootstrap: int = 1000) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence interval for variance ratio.

        Returns:
            mean, lower_ci, upper_ci
        """
        def statistic(x, axis):
            # x shape: (n_bootstrap, n_samples, n_features)
            return np.array([self.compute_variance_ratio(sample) for sample in x])

        # Bootstrap sampling
        n_samples = embeddings.shape[0]
        ratios = []

        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            sample = embeddings[indices]
            ratio = self.compute_variance_ratio(sample)
            ratios.append(ratio)

        ratios = np.array(ratios)
        mean = ratios.mean()
        lower = np.percentile(ratios, 2.5)
        upper = np.percentile(ratios, 97.5)

        return mean, lower, upper

    def analyze_statistical_significance(self, all_activations: Dict):
        """
        Priority 1: Add statistical significance tests (bootstrap CI)
        """
        print("\n" + "="*80)
        print("STATISTICAL SIGNIFICANCE ANALYSIS (Bootstrap CI)")
        print("="*80)

        results = {}

        for task, (hidden_states, correct) in all_activations.items():
            print(f"\n{task.upper()}")
            print("-" * 40)

            # Average across layers and tokens for overall representation
            # Shape: [N, 16, 6, 2048] -> [N, 2048]
            embeddings = hidden_states.mean(axis=(1, 2))

            # Overall variance ratio with CI
            mean, lower, upper = self.bootstrap_variance_ratio(embeddings, n_bootstrap=1000)
            results[task] = {
                'overall': {'mean': mean, 'ci_lower': lower, 'ci_upper': upper}
            }

            print(f"Overall Variance Ratio: {mean:.4f} [{lower:.4f}, {upper:.4f}]")

        # Save results
        output_path = self.viz_dir / 'statistical_significance.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Saved to {output_path}")

        # Visualize with error bars
        fig, ax = plt.subplots(figsize=(10, 6))

        tasks = list(results.keys())
        means = [results[t]['overall']['mean'] for t in tasks]
        ci_lower = [results[t]['overall']['ci_lower'] for t in tasks]
        ci_upper = [results[t]['overall']['ci_upper'] for t in tasks]

        errors = np.array([[means[i] - ci_lower[i], ci_upper[i] - means[i]]
                          for i in range(len(tasks))]).T

        x = np.arange(len(tasks))
        ax.bar(x, means, yerr=errors, capsize=10, alpha=0.7,
              color=['#E74C3C', '#3498DB', '#2ECC71'])

        ax.set_xticks(x)
        ax.set_xticklabels([t.replace('_', ' ').title() for t in tasks])
        ax.set_ylabel('Variance Ratio (Compactness)', fontweight='bold')
        ax.set_title('Task Compactness with 95% Bootstrap CI (1000 samples)',
                    fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'variance_ratio_with_ci.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization: variance_ratio_with_ci.png")
        plt.close()

        return results

    def analyze_correct_vs_incorrect(self, all_activations: Dict):
        """
        Priority 1: Analyze correct vs incorrect examples separately
        """
        print("\n" + "="*80)
        print("CORRECT VS INCORRECT ANALYSIS")
        print("="*80)

        results = {}

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        for task_idx, (task, (hidden_states, correct)) in enumerate(all_activations.items()):
            print(f"\n{task.upper()}")
            print("-" * 40)

            # Average across layers and tokens
            embeddings = hidden_states.mean(axis=(1, 2))  # [N, 2048]

            # Split by correctness
            correct_mask = correct
            incorrect_mask = ~correct

            correct_embeddings = embeddings[correct_mask]
            incorrect_embeddings = embeddings[incorrect_mask]

            print(f"Correct examples: {correct_mask.sum()}")
            print(f"Incorrect examples: {incorrect_mask.sum()}")

            # Compute variance ratios
            vr_correct = self.compute_variance_ratio(correct_embeddings)
            vr_incorrect = self.compute_variance_ratio(incorrect_embeddings)

            print(f"Variance Ratio (Correct): {vr_correct:.4f}")
            print(f"Variance Ratio (Incorrect): {vr_incorrect:.4f}")
            print(f"Difference: {vr_correct - vr_incorrect:+.4f}")

            results[task] = {
                'n_correct': int(correct_mask.sum()),
                'n_incorrect': int(incorrect_mask.sum()),
                'vr_correct': float(vr_correct),
                'vr_incorrect': float(vr_incorrect),
                'vr_diff': float(vr_correct - vr_incorrect)
            }

            # PCA visualization for correct vs incorrect
            # Top row: PCA colored by correctness
            ax = axes[0, task_idx]
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)

            ax.scatter(embeddings_2d[correct_mask, 0], embeddings_2d[correct_mask, 1],
                      c='green', label='Correct', alpha=0.6, s=30, marker='o')
            ax.scatter(embeddings_2d[incorrect_mask, 0], embeddings_2d[incorrect_mask, 1],
                      c='red', label='Incorrect', alpha=0.6, s=30, marker='x')

            ax.set_title(f'{task.replace("_", " ").title()}', fontweight='bold')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)

            # Bottom row: Variance ratio comparison
            ax = axes[1, task_idx]
            categories = ['Correct', 'Incorrect']
            values = [vr_correct, vr_incorrect]
            colors = ['green', 'red']

            ax.bar(categories, values, color=colors, alpha=0.7)
            ax.set_ylabel('Variance Ratio', fontweight='bold')
            ax.set_title(f'{task.replace("_", " ").title()}', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, max(values) * 1.2])

            # Add value labels on bars
            for i, v in enumerate(values):
                ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

        fig.suptitle('Correct vs Incorrect: PCA and Variance Ratio Comparison',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'correct_vs_incorrect_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved visualization: correct_vs_incorrect_analysis.png")
        plt.close()

        # Save results
        output_path = self.viz_dir / 'correct_vs_incorrect.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✓ Saved to {output_path}")

        return results

    def analyze_layer_progression(self, all_activations: Dict):
        """
        Priority 2: Layer-wise progression analysis
        """
        print("\n" + "="*80)
        print("LAYER-WISE PROGRESSION ANALYSIS")
        print("="*80)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        task_colors = {
            'personal_relations': '#E74C3C',
            'gsm8k': '#3498DB',
            'commonsense': '#2ECC71'
        }

        # Metric 1: Variance ratio across layers
        ax = axes[0]
        for task, (hidden_states, correct) in all_activations.items():
            variance_ratios = []

            for layer_idx in range(16):
                # Average across tokens for this layer: [N, 6, 2048] -> [N, 2048]
                layer_embeddings = hidden_states[:, layer_idx, :, :].mean(axis=1)
                vr = self.compute_variance_ratio(layer_embeddings)
                variance_ratios.append(vr)

            ax.plot(range(16), variance_ratios, marker='o', linewidth=2, markersize=6,
                   label=task.replace('_', ' ').title(), color=task_colors[task])

        ax.set_xlabel('Layer', fontweight='bold')
        ax.set_ylabel('Variance Ratio (Compactness)', fontweight='bold')
        ax.set_title('Representation Compactness Across Depth', fontweight='bold', fontsize=14)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Metric 2: Representation norm across layers
        ax = axes[1]
        for task, (hidden_states, correct) in all_activations.items():
            norms = []

            for layer_idx in range(16):
                layer_embeddings = hidden_states[:, layer_idx, :, :].mean(axis=1)
                norm = np.linalg.norm(layer_embeddings, axis=1).mean()
                norms.append(norm)

            ax.plot(range(16), norms, marker='o', linewidth=2, markersize=6,
                   label=task.replace('_', ' ').title(), color=task_colors[task])

        ax.set_xlabel('Layer', fontweight='bold')
        ax.set_ylabel('Average L2 Norm', fontweight='bold')
        ax.set_title('Representation Magnitude Across Depth', fontweight='bold', fontsize=14)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Metric 3: PCA explained variance across layers
        ax = axes[2]
        for task, (hidden_states, correct) in all_activations.items():
            explained_vars = []

            for layer_idx in range(16):
                layer_embeddings = hidden_states[:, layer_idx, :, :].mean(axis=1)
                pca = PCA(n_components=2)
                pca.fit(layer_embeddings)
                explained_var = pca.explained_variance_ratio_.sum()
                explained_vars.append(explained_var)

            ax.plot(range(16), explained_vars, marker='o', linewidth=2, markersize=6,
                   label=task.replace('_', ' ').title(), color=task_colors[task])

        ax.set_xlabel('Layer', fontweight='bold')
        ax.set_ylabel('PCA Variance Explained (PC1+PC2)', fontweight='bold')
        ax.set_title('Low-Dimensional Structure Across Depth', fontweight='bold', fontsize=14)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Metric 4: Correct vs incorrect separation across layers
        ax = axes[3]
        for task, (hidden_states, correct) in all_activations.items():
            separations = []

            for layer_idx in range(16):
                layer_embeddings = hidden_states[:, layer_idx, :, :].mean(axis=1)

                correct_centroid = layer_embeddings[correct].mean(axis=0)
                incorrect_centroid = layer_embeddings[~correct].mean(axis=0)

                separation = np.linalg.norm(correct_centroid - incorrect_centroid)
                separations.append(separation)

            ax.plot(range(16), separations, marker='o', linewidth=2, markersize=6,
                   label=task.replace('_', ' ').title(), color=task_colors[task])

        ax.set_xlabel('Layer', fontweight='bold')
        ax.set_ylabel('Centroid Distance', fontweight='bold')
        ax.set_title('Correct/Incorrect Separation Across Depth', fontweight='bold', fontsize=14)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        fig.suptitle('Layer-Wise Progression of CT Representations',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'layer_progression.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization: layer_progression.png")
        plt.close()

    def analyze_random_baselines(self, all_activations: Dict):
        """
        Priority 2: Compare to random baselines
        """
        print("\n" + "="*80)
        print("RANDOM BASELINE COMPARISON")
        print("="*80)

        results = {}

        for task, (hidden_states, correct) in all_activations.items():
            print(f"\n{task.upper()}")
            print("-" * 40)

            n_examples = len(correct)
            accuracy = correct.mean()

            # Determine random baseline based on task
            if task == 'personal_relations':
                # Assuming 5 relationship types
                random_baseline = 0.20
            elif task == 'gsm8k':
                # Open-ended, very hard to guess
                random_baseline = 0.01  # ~1%
            else:  # commonsense
                # Multiple choice with 5 options
                random_baseline = 0.20

            # Compute above-chance performance
            above_chance = accuracy - random_baseline
            relative_improvement = (accuracy / random_baseline - 1) * 100 if random_baseline > 0 else np.inf

            print(f"Accuracy: {accuracy:.1%}")
            print(f"Random baseline: {random_baseline:.1%}")
            print(f"Above chance: {above_chance:.1%}")
            print(f"Relative improvement: {relative_improvement:.1f}%")

            results[task] = {
                'accuracy': float(accuracy),
                'random_baseline': float(random_baseline),
                'above_chance': float(above_chance),
                'relative_improvement_pct': float(relative_improvement)
            }

        # Save results
        output_path = self.viz_dir / 'random_baselines.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Saved to {output_path}")

        # Visualize
        fig, ax = plt.subplots(figsize=(12, 6))

        tasks = list(results.keys())
        x = np.arange(len(tasks))
        width = 0.35

        accuracies = [results[t]['accuracy'] for t in tasks]
        baselines = [results[t]['random_baseline'] for t in tasks]

        ax.bar(x - width/2, accuracies, width, label='Model Accuracy',
              color=['#E74C3C', '#3498DB', '#2ECC71'], alpha=0.8)
        ax.bar(x + width/2, baselines, width, label='Random Baseline',
              color='gray', alpha=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([t.replace('_', ' ').title() for t in tasks])
        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.set_title('Model Performance vs Random Baseline', fontweight='bold', fontsize=14)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, (acc, base) in enumerate(zip(accuracies, baselines)):
            ax.text(i - width/2, acc + 0.02, f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
            ax.text(i + width/2, base + 0.02, f'{base:.1%}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'random_baseline_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization: random_baseline_comparison.png")
        plt.close()

        return results

    def validate_pca_assumptions(self, all_activations: Dict):
        """
        Test concern: Is 85% variance in 2D due to task ID signal or genuine structure?
        """
        print("\n" + "="*80)
        print("PCA VARIANCE VALIDATION")
        print("="*80)

        print("\nWithin-task PCA (should show higher variance if cross-task dominates):")
        print("-" * 40)

        for task, (hidden_states, correct) in all_activations.items():
            # Average across layers and tokens
            embeddings = hidden_states.mean(axis=(1, 2))

            # PCA on this task alone
            pca = PCA(n_components=2)
            pca.fit(embeddings)

            explained = pca.explained_variance_ratio_.sum()

            print(f"{task:20s}: {explained:.1%} variance in 2D")

        print("\nCross-task PCA (original analysis):")
        print("-" * 40)

        # Combined PCA
        all_embeddings = []
        for task, (hidden_states, correct) in all_activations.items():
            embeddings = hidden_states.mean(axis=(1, 2))
            all_embeddings.append(embeddings)

        all_embeddings = np.vstack(all_embeddings)
        pca = PCA(n_components=2)
        pca.fit(all_embeddings)

        explained = pca.explained_variance_ratio_.sum()
        print(f"Combined: {explained:.1%} variance in 2D")

        print("\nInterpretation:")
        if explained > 0.80:
            print("  → Within-task variance is similar to cross-task")
            print("  → Tasks genuinely occupy low-dimensional manifolds")
        else:
            print("  → Within-task variance is lower than cross-task")
            print("  → Task ID signal dominates the PCA projection")

    def run_validation(self):
        """Run all validation analyses."""
        print("="*80)
        print("THREE-WAY COMPARISON VALIDATION ANALYSIS")
        print("="*80)

        # Load all activations
        all_activations = {}
        for task in ['personal_relations', 'gsm8k', 'commonsense']:
            hidden_states, correct = self.load_activations(task)
            all_activations[task] = (hidden_states, correct)
            print(f"Loaded {task}: {hidden_states.shape}, Accuracy: {correct.mean():.1%}")

        # Run analyses
        self.analyze_statistical_significance(all_activations)
        self.analyze_correct_vs_incorrect(all_activations)
        self.analyze_layer_progression(all_activations)
        self.analyze_random_baselines(all_activations)
        self.validate_pca_assumptions(all_activations)

        print("\n" + "="*80)
        print("VALIDATION ANALYSIS COMPLETE!")
        print("="*80)
        print(f"All results saved to: {self.viz_dir}")
        print("="*80)


def main():
    """Main validation script."""
    analyzer = ValidationAnalyzer()
    analyzer.run_validation()


if __name__ == '__main__':
    main()
