#!/usr/bin/env python3
"""
Story 3: CT Embedding Visualization
Generate PCA and t-SNE visualizations comparing CT tokens across 3 tasks.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


class EmbeddingVisualizer:
    """Visualize CT embeddings across tasks."""

    def __init__(self, results_dir: str = None):
        if results_dir is None:
            results_dir = Path(__file__).parent / 'results'
        self.results_dir = Path(results_dir)
        self.viz_dir = self.results_dir / 'visualizations'
        self.viz_dir.mkdir(exist_ok=True)

    def load_activations(self, task: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load CT activations for a task.

        Returns:
            hidden_states: [N, 16, 6, 2048]
            correct: [N] boolean array
            metadata: dict with task info
        """
        filepath = self.results_dir / f'activations_{task}.npz'
        print(f"Loading {task} from {filepath}...")

        data = np.load(filepath, allow_pickle=True)

        hidden_states = data['hidden_states']
        correct = data['correct']

        # Extract metadata
        metadata = {
            'task': str(data['task_name']),
            'n_examples': int(data['n_examples']),
            'accuracy': float(data['accuracy']),
            'n_correct': int(data['n_correct'])
        }

        print(f"  Shape: {hidden_states.shape}")
        print(f"  Accuracy: {metadata['n_correct']}/{metadata['n_examples']} = {metadata['accuracy']:.1%}")

        return hidden_states, correct, metadata

    def prepare_embedding_data(self, all_activations: Dict) -> Tuple[np.ndarray, List[str], List[str], List[bool]]:
        """
        Prepare data for dimensionality reduction.

        Strategy: Average across all layers to get a single embedding per CT token.

        Returns:
            embeddings: [N_total_tokens, 2048] - flattened embeddings
            tasks: [N_total_tokens] - task labels
            token_ids: [N_total_tokens] - CT0-CT5 labels
            correct: [N_total_tokens] - correctness labels
        """
        embeddings = []
        tasks = []
        token_ids = []
        correct_labels = []

        for task, (hidden_states, correct, _) in all_activations.items():
            # hidden_states: [N, 16 layers, 6 tokens, 2048]
            # Average across layers: [N, 6, 2048]
            layer_avg = hidden_states.mean(axis=1)

            n_examples, n_tokens, hidden_dim = layer_avg.shape

            # Flatten to [N*6, 2048]
            flat = layer_avg.reshape(-1, hidden_dim)

            embeddings.append(flat)

            # Create labels
            for ex_idx in range(n_examples):
                for tok_idx in range(n_tokens):
                    tasks.append(task)
                    token_ids.append(f"CT{tok_idx}")
                    correct_labels.append(correct[ex_idx])

        embeddings = np.vstack(embeddings)

        print(f"\nPrepared embedding data:")
        print(f"  Total embeddings: {embeddings.shape[0]}")
        print(f"  Embedding dim: {embeddings.shape[1]}")
        print(f"  Tasks: {len(set(tasks))}")
        print(f"  Tokens: {len(set(token_ids))}")

        return embeddings, tasks, token_ids, correct_labels

    def visualize_pca(self, embeddings: np.ndarray, tasks: List[str],
                     token_ids: List[str], correct: List[bool]):
        """Generate PCA visualizations."""
        print("\nGenerating PCA visualizations...")

        # Fit PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)

        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.1%}")

        # Create color maps
        task_colors = {'personal_relations': '#E74C3C', 'gsm8k': '#3498DB', 'commonsense': '#2ECC71'}
        token_colors = {f'CT{i}': plt.cm.viridis(i/5) for i in range(6)}

        # 1. PCA by Task
        fig, ax = plt.subplots(figsize=(12, 8))

        for task in task_colors.keys():
            mask = np.array([t == task for t in tasks])
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=[task_colors[task]], label=task.replace('_', ' ').title(),
                      alpha=0.6, s=50)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title('PCA: CT Embeddings by Task (Layer-Averaged)', fontsize=14, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'pca_by_task.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: pca_by_task.png")
        plt.close()

        # 2. PCA by CT Token Position
        fig, ax = plt.subplots(figsize=(12, 8))

        for token_id in sorted(set(token_ids)):
            mask = np.array([t == token_id for t in token_ids])
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=[token_colors[token_id]], label=token_id,
                      alpha=0.6, s=50)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title('PCA: CT Embeddings by Token Position', fontsize=14, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9, ncol=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'pca_by_token.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: pca_by_token.png")
        plt.close()

        # 3. PCA by Correctness (per task)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, task in enumerate(task_colors.keys()):
            ax = axes[idx]
            task_mask = np.array([t == task for t in tasks])

            # Correct examples
            correct_mask = task_mask & np.array(correct)
            ax.scatter(embeddings_2d[correct_mask, 0], embeddings_2d[correct_mask, 1],
                      c='green', label='Correct', alpha=0.6, s=50, marker='o')

            # Incorrect examples
            incorrect_mask = task_mask & ~np.array(correct)
            ax.scatter(embeddings_2d[incorrect_mask, 0], embeddings_2d[incorrect_mask, 1],
                      c='red', label='Incorrect', alpha=0.6, s=50, marker='x')

            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.set_title(f'{task.replace("_", " ").title()}', fontweight='bold')
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)

        fig.suptitle('PCA: Correct vs Incorrect Examples by Task', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'pca_by_correctness.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: pca_by_correctness.png")
        plt.close()

        return pca, embeddings_2d

    def visualize_tsne(self, embeddings: np.ndarray, tasks: List[str],
                      token_ids: List[str], correct: List[bool]):
        """Generate t-SNE visualizations."""
        print("\nGenerating t-SNE visualizations...")

        # Subsample if too large (t-SNE is slow)
        max_samples = 1800  # 300 examples * 6 tokens = 1800
        if embeddings.shape[0] > max_samples:
            print(f"  Subsampling to {max_samples} points for t-SNE...")
            indices = np.random.choice(embeddings.shape[0], max_samples, replace=False)
            embeddings = embeddings[indices]
            tasks = [tasks[i] for i in indices]
            token_ids = [token_ids[i] for i in indices]
            correct = [correct[i] for i in indices]

        # Fit t-SNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
        embeddings_2d = tsne.fit_transform(embeddings)

        # Create color maps
        task_colors = {'personal_relations': '#E74C3C', 'gsm8k': '#3498DB', 'commonsense': '#2ECC71'}

        # t-SNE by Task
        fig, ax = plt.subplots(figsize=(12, 8))

        for task in task_colors.keys():
            mask = np.array([t == task for t in tasks])
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=[task_colors[task]], label=task.replace('_', ' ').title(),
                      alpha=0.6, s=50)

        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title('t-SNE: CT Embeddings by Task (Layer-Averaged)', fontsize=14, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'tsne_by_task.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: tsne_by_task.png")
        plt.close()

        return tsne, embeddings_2d

    def visualize_layer_progression(self, all_activations: Dict):
        """Visualize how embeddings evolve across layers."""
        print("\nGenerating layer progression visualizations...")

        # For each task, show CT0 progression through layers
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        task_colors = {'personal_relations': '#E74C3C', 'gsm8k': '#3498DB', 'commonsense': '#2ECC71'}

        for task_idx, (task, (hidden_states, correct, _)) in enumerate(all_activations.items()):
            # Extract CT0 across all layers: [N, 16, 2048]
            ct0_layers = hidden_states[:, :, 0, :]  # [N, 16, 2048]

            # For each layer, compute PCA on CT0
            layer_pca_results = []

            for layer_idx in range(16):
                layer_embeddings = ct0_layers[:, layer_idx, :]  # [N, 2048]

                # PCA to 2D
                pca = PCA(n_components=2)
                layer_2d = pca.fit_transform(layer_embeddings)

                layer_pca_results.append(layer_2d)

            # Plot layer 0, 5, 10, 15 for this task
            selected_layers = [0, 5, 10, 15]

            for plot_idx, layer_idx in enumerate(selected_layers):
                ax = axes[plot_idx] if task_idx == 0 else axes[plot_idx]

                if task_idx == 0:
                    ax.set_title(f'Layer {layer_idx}', fontweight='bold')

                layer_2d = layer_pca_results[layer_idx]

                ax.scatter(layer_2d[:, 0], layer_2d[:, 1],
                          c=[task_colors[task]], label=task.replace('_', ' ').title(),
                          alpha=0.6, s=40)

                if plot_idx == 0:
                    ax.set_ylabel('PC2')
                if plot_idx == 3:
                    ax.set_xlabel('PC1')

                ax.grid(True, alpha=0.3)

        # Add legend to last subplot
        axes[5].legend(loc='center', framealpha=0.9)
        axes[5].axis('off')

        fig.suptitle('CT0 Token Evolution Across Layers (PCA per Layer)',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'layer_progression_ct0.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: layer_progression_ct0.png")
        plt.close()

    def generate_all_visualizations(self):
        """Generate all visualization plots."""
        print("="*80)
        print("CT EMBEDDING VISUALIZATION")
        print("="*80)

        # Load all activations
        all_activations = {}
        for task in ['personal_relations', 'gsm8k', 'commonsense']:
            hidden_states, correct, metadata = self.load_activations(task)
            all_activations[task] = (hidden_states, correct, metadata)

        # Prepare data
        embeddings, tasks, token_ids, correct = self.prepare_embedding_data(all_activations)

        # Generate visualizations
        self.visualize_pca(embeddings, tasks, token_ids, correct)
        self.visualize_tsne(embeddings, tasks, token_ids, correct)
        self.visualize_layer_progression(all_activations)

        print("\n" + "="*80)
        print("VISUALIZATION COMPLETE!")
        print("="*80)
        print(f"All plots saved to: {self.viz_dir}")
        print("="*80)


def main():
    """Main visualization script."""
    visualizer = EmbeddingVisualizer()
    visualizer.generate_all_visualizations()


if __name__ == '__main__':
    main()
