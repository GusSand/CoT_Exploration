#!/usr/bin/env python3
"""
Story 6: Layer-wise Divergence Analysis
Compute quantitative metrics comparing CT representations across tasks.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance
from typing import Dict, Tuple
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11


class DivergenceAnalyzer:
    """Analyze divergence between CT representations across tasks."""

    def __init__(self, results_dir: str = None):
        if results_dir is None:
            results_dir = Path(__file__).parent / 'results'
        self.results_dir = Path(results_dir)
        self.viz_dir = self.results_dir / 'visualizations'
        self.viz_dir.mkdir(exist_ok=True)

    def load_activations(self, task: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load CT activations for a task.

        Returns:
            hidden_states: [N, 16, 6, 2048]
            correct: [N] boolean array
        """
        filepath = self.results_dir / f'activations_{task}.npz'
        data = np.load(filepath, allow_pickle=True)

        return data['hidden_states'], data['correct']

    def compute_centroid_distance(self, task1_embeddings: np.ndarray,
                                  task2_embeddings: np.ndarray) -> float:
        """
        Compute Euclidean distance between task centroids.

        Args:
            task1_embeddings: [N1, hidden_dim]
            task2_embeddings: [N2, hidden_dim]

        Returns:
            distance: float
        """
        centroid1 = task1_embeddings.mean(axis=0)
        centroid2 = task2_embeddings.mean(axis=0)

        distance = np.linalg.norm(centroid1 - centroid2)

        return distance

    def compute_cosine_similarity(self, task1_embeddings: np.ndarray,
                                   task2_embeddings: np.ndarray) -> float:
        """
        Compute average cosine similarity between task centroids.

        Args:
            task1_embeddings: [N1, hidden_dim]
            task2_embeddings: [N2, hidden_dim]

        Returns:
            similarity: float in [0, 1]
        """
        centroid1 = task1_embeddings.mean(axis=0)
        centroid2 = task2_embeddings.mean(axis=0)

        # Cosine similarity = 1 - cosine distance
        similarity = 1 - cosine(centroid1, centroid2)

        return similarity

    def compute_variance_ratio(self, task_embeddings: np.ndarray) -> float:
        """
        Compute ratio of explained variance (compactness metric).

        Higher values = more compact/focused representations.

        Args:
            task_embeddings: [N, hidden_dim]

        Returns:
            variance_ratio: float
        """
        # Compute covariance
        cov = np.cov(task_embeddings, rowvar=False)

        # Get eigenvalues
        eigenvalues = np.linalg.eigvalsh(cov)

        # Sort descending
        eigenvalues = np.sort(eigenvalues)[::-1]

        # Ratio of top eigenvalue to sum
        if eigenvalues.sum() > 0:
            ratio = eigenvalues[0] / eigenvalues.sum()
        else:
            ratio = 0.0

        return ratio

    def analyze_layer_wise_divergence(self, all_activations: Dict):
        """
        Compute layer-wise divergence metrics.

        For each layer:
        - Centroid distance between tasks
        - Cosine similarity between tasks
        - Variance ratio (compactness) per task
        """
        print("\n" + "="*80)
        print("LAYER-WISE DIVERGENCE ANALYSIS")
        print("="*80)

        task_names = list(all_activations.keys())
        n_layers = 16
        n_tokens = 6

        # Storage for metrics
        results = {
            'centroid_distances': {},
            'cosine_similarities': {},
            'variance_ratios': {}
        }

        # Task pair comparisons
        task_pairs = [
            ('personal_relations', 'gsm8k'),
            ('personal_relations', 'commonsense'),
            ('gsm8k', 'commonsense')
        ]

        # Compute metrics for each layer and token
        for layer_idx in range(n_layers):
            print(f"\nProcessing Layer {layer_idx}...")

            for token_idx in range(n_tokens):
                # Extract embeddings for this layer and token
                task_embeddings = {}

                for task, (hidden_states, correct) in all_activations.items():
                    # hidden_states: [N, 16, 6, 2048]
                    embeddings = hidden_states[:, layer_idx, token_idx, :]  # [N, 2048]
                    task_embeddings[task] = embeddings

                # Compute pairwise distances
                for task1, task2 in task_pairs:
                    pair_name = f"{task1}_vs_{task2}"

                    if pair_name not in results['centroid_distances']:
                        results['centroid_distances'][pair_name] = np.zeros((n_layers, n_tokens))
                        results['cosine_similarities'][pair_name] = np.zeros((n_layers, n_tokens))

                    # Centroid distance
                    dist = self.compute_centroid_distance(
                        task_embeddings[task1],
                        task_embeddings[task2]
                    )
                    results['centroid_distances'][pair_name][layer_idx, token_idx] = dist

                    # Cosine similarity
                    sim = self.compute_cosine_similarity(
                        task_embeddings[task1],
                        task_embeddings[task2]
                    )
                    results['cosine_similarities'][pair_name][layer_idx, token_idx] = sim

                # Compute variance ratios (compactness)
                for task in task_names:
                    if task not in results['variance_ratios']:
                        results['variance_ratios'][task] = np.zeros((n_layers, n_tokens))

                    ratio = self.compute_variance_ratio(task_embeddings[task])
                    results['variance_ratios'][task][layer_idx, token_idx] = ratio

        print("\n  ✓ Computed all metrics!")

        return results

    def visualize_centroid_distances(self, results: Dict):
        """Visualize centroid distances across layers and tokens."""
        print("\nVisualizing centroid distances...")

        task_pairs = list(results['centroid_distances'].keys())

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, pair_name in enumerate(task_pairs):
            ax = axes[idx]
            distances = results['centroid_distances'][pair_name]

            # Heatmap
            im = ax.imshow(distances.T, aspect='auto', cmap='YlOrRd', origin='lower')

            ax.set_xlabel('Layer', fontweight='bold')
            ax.set_ylabel('CT Token', fontweight='bold')
            ax.set_title(pair_name.replace('_', ' ').title().replace(' Vs ', ' vs '),
                        fontweight='bold')

            ax.set_xticks(range(16))
            ax.set_yticks(range(6))
            ax.set_yticklabels([f'CT{i}' for i in range(6)])

            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Euclidean Distance', rotation=270, labelpad=20)

        fig.suptitle('Centroid Distance: Task Separation by Layer and Token',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'centroid_distances.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: centroid_distances.png")
        plt.close()

    def visualize_cosine_similarities(self, results: Dict):
        """Visualize cosine similarities across layers and tokens."""
        print("\nVisualizing cosine similarities...")

        task_pairs = list(results['cosine_similarities'].keys())

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, pair_name in enumerate(task_pairs):
            ax = axes[idx]
            similarities = results['cosine_similarities'][pair_name]

            # Heatmap
            im = ax.imshow(similarities.T, aspect='auto', cmap='RdYlGn', origin='lower',
                          vmin=0, vmax=1)

            ax.set_xlabel('Layer', fontweight='bold')
            ax.set_ylabel('CT Token', fontweight='bold')
            ax.set_title(pair_name.replace('_', ' ').title().replace(' Vs ', ' vs '),
                        fontweight='bold')

            ax.set_xticks(range(16))
            ax.set_yticks(range(6))
            ax.set_yticklabels([f'CT{i}' for i in range(6)])

            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Cosine Similarity', rotation=270, labelpad=20)

        fig.suptitle('Cosine Similarity: Task Alignment by Layer and Token',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'cosine_similarities.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: cosine_similarities.png")
        plt.close()

    def visualize_variance_ratios(self, results: Dict):
        """Visualize variance ratios (compactness) across layers and tokens."""
        print("\nVisualizing variance ratios...")

        tasks = list(results['variance_ratios'].keys())

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, task in enumerate(tasks):
            ax = axes[idx]
            ratios = results['variance_ratios'][task]

            # Heatmap
            im = ax.imshow(ratios.T, aspect='auto', cmap='Blues', origin='lower',
                          vmin=0, vmax=1)

            ax.set_xlabel('Layer', fontweight='bold')
            ax.set_ylabel('CT Token', fontweight='bold')
            ax.set_title(task.replace('_', ' ').title(), fontweight='bold')

            ax.set_xticks(range(16))
            ax.set_yticks(range(6))
            ax.set_yticklabels([f'CT{i}' for i in range(6)])

            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Variance Ratio', rotation=270, labelpad=20)

        fig.suptitle('Variance Ratio: Representation Compactness by Layer and Token',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'variance_ratios.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: variance_ratios.png")
        plt.close()

    def visualize_token_trajectories(self, results: Dict):
        """Plot how metrics evolve across layers for each token."""
        print("\nVisualizing token trajectories...")

        # Average centroid distance across all task pairs
        task_pairs = list(results['centroid_distances'].keys())

        # Average across pairs
        avg_distances = np.mean([results['centroid_distances'][pair] for pair in task_pairs], axis=0)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Centroid distance trajectories
        ax = axes[0]
        for token_idx in range(6):
            ax.plot(range(16), avg_distances[:, token_idx], marker='o', label=f'CT{token_idx}',
                   linewidth=2, markersize=6)

        ax.set_xlabel('Layer', fontweight='bold')
        ax.set_ylabel('Average Centroid Distance', fontweight='bold')
        ax.set_title('Task Separation Across Layers', fontweight='bold', fontsize=14)
        ax.legend(loc='best', framealpha=0.9, ncol=2)
        ax.grid(True, alpha=0.3)

        # Plot 2: Variance ratio trajectories (averaged across tasks)
        ax = axes[1]
        tasks = list(results['variance_ratios'].keys())
        avg_variance = np.mean([results['variance_ratios'][task] for task in tasks], axis=0)

        for token_idx in range(6):
            ax.plot(range(16), avg_variance[:, token_idx], marker='o', label=f'CT{token_idx}',
                   linewidth=2, markersize=6)

        ax.set_xlabel('Layer', fontweight='bold')
        ax.set_ylabel('Average Variance Ratio', fontweight='bold')
        ax.set_title('Representation Compactness Across Layers', fontweight='bold', fontsize=14)
        ax.legend(loc='best', framealpha=0.9, ncol=2)
        ax.grid(True, alpha=0.3)

        fig.suptitle('CT Token Trajectories Through Network Depth',
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'token_trajectories.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: token_trajectories.png")
        plt.close()

    def save_summary_statistics(self, results: Dict, all_activations: Dict):
        """Save summary statistics to JSON."""
        print("\nComputing summary statistics...")

        summary = {
            'task_accuracies': {},
            'avg_centroid_distances': {},
            'avg_cosine_similarities': {},
            'avg_variance_ratios': {}
        }

        # Task accuracies
        for task, (_, correct) in all_activations.items():
            summary['task_accuracies'][task] = float(correct.mean())

        # Average metrics across all layers and tokens
        for pair_name, distances in results['centroid_distances'].items():
            summary['avg_centroid_distances'][pair_name] = float(distances.mean())

        for pair_name, similarities in results['cosine_similarities'].items():
            summary['avg_cosine_similarities'][pair_name] = float(similarities.mean())

        for task, ratios in results['variance_ratios'].items():
            summary['avg_variance_ratios'][task] = float(ratios.mean())

        # Save to JSON
        output_path = self.results_dir / 'divergence_summary.json'
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"  ✓ Saved: divergence_summary.json")

        # Print summary
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print("\nTask Accuracies:")
        for task, acc in summary['task_accuracies'].items():
            print(f"  {task:20s}: {acc:.1%}")

        print("\nAverage Centroid Distances (Task Separation):")
        for pair, dist in summary['avg_centroid_distances'].items():
            print(f"  {pair:40s}: {dist:.2f}")

        print("\nAverage Cosine Similarities (Task Alignment):")
        for pair, sim in summary['avg_cosine_similarities'].items():
            print(f"  {pair:40s}: {sim:.4f}")

        print("\nAverage Variance Ratios (Compactness):")
        for task, ratio in summary['avg_variance_ratios'].items():
            print(f"  {task:20s}: {ratio:.4f}")

        print("="*80)

        return summary

    def run_analysis(self):
        """Run complete divergence analysis."""
        print("="*80)
        print("CT DIVERGENCE ANALYSIS")
        print("="*80)

        # Load all activations
        all_activations = {}
        for task in ['personal_relations', 'gsm8k', 'commonsense']:
            hidden_states, correct = self.load_activations(task)
            all_activations[task] = (hidden_states, correct)
            print(f"Loaded {task}: {hidden_states.shape}, Accuracy: {correct.mean():.1%}")

        # Compute divergence metrics
        results = self.analyze_layer_wise_divergence(all_activations)

        # Generate visualizations
        self.visualize_centroid_distances(results)
        self.visualize_cosine_similarities(results)
        self.visualize_variance_ratios(results)
        self.visualize_token_trajectories(results)

        # Save summary
        summary = self.save_summary_statistics(results, all_activations)

        print("\n" + "="*80)
        print("DIVERGENCE ANALYSIS COMPLETE!")
        print("="*80)
        print(f"All results saved to: {self.results_dir}")
        print("="*80)

        return results, summary


def main():
    """Main analysis script."""
    analyzer = DivergenceAnalyzer()
    analyzer.run_analysis()


if __name__ == '__main__':
    main()
