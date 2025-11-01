"""
Visualization utilities for attention patterns
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def plot_attention_heatmap(attention_matrix, title, save_path, cmap="YlOrRd", vmin=0, vmax=1):
    """
    Plot a single attention heatmap

    Args:
        attention_matrix: [num_latent, num_latent] attention matrix
        title: Title for the plot
        save_path: Path to save the figure
        cmap: Colormap name
        vmin, vmax: Color scale limits
    """
    plt.figure(figsize=(8, 6))

    # Create heatmap
    sns.heatmap(
        attention_matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt=".3f",
        square=True,
        cbar_kws={'label': 'Attention Weight'},
        xticklabels=[f'Pos {i}' for i in range(attention_matrix.shape[1])],
        yticklabels=[f'Pos {i}' for i in range(attention_matrix.shape[0])]
    )

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('TO Position', fontsize=12)
    plt.ylabel('FROM Position', fontsize=12)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to {save_path}")


def plot_layer_wise_heatmaps(attention_dict, save_dir, cmap="YlOrRd"):
    """
    Plot heatmaps for each layer

    Args:
        attention_dict: Dict mapping layer_idx to [num_latent, num_latent]
        save_dir: Directory to save figures
        cmap: Colormap name
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Find global min/max for consistent color scale
    all_values = np.concatenate([attn.flatten() for attn in attention_dict.values()])
    vmin, vmax = float(all_values.min()), float(all_values.max())

    for layer_idx in sorted(attention_dict.keys()):
        attn = attention_dict[layer_idx]
        title = f'CoT Attention Patterns - Layer {layer_idx}'
        save_path = save_dir / f'layer_{layer_idx:02d}_attention.png'
        plot_attention_heatmap(attn, title, save_path, cmap=cmap, vmin=vmin, vmax=vmax)


def plot_aggregated_attention(attention_matrix, title, save_path, cmap="YlOrRd"):
    """
    Plot aggregated attention (e.g., averaged across layers or examples)

    Args:
        attention_matrix: [num_latent, num_latent] aggregated attention
        title: Title for the plot
        save_path: Path to save the figure
        cmap: Colormap name
    """
    plot_attention_heatmap(attention_matrix, title, save_path, cmap=cmap)


def plot_metrics_comparison(metrics_dict, save_path):
    """
    Plot bar chart comparing different attention metrics

    Args:
        metrics_dict: Dict with aggregated metrics
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Sequential vs Self-Attention
    ax1 = axes[0, 0]
    metrics_to_plot = {
        'Sequential\n(N→N-1)': metrics_dict['sequential_score_mean'],
        'Self-Attention\n(N→N)': metrics_dict['self_attention_mean']
    }
    colors = ['#e74c3c', '#3498db']
    bars = ax1.bar(metrics_to_plot.keys(), metrics_to_plot.values(), color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Mean Attention Weight', fontsize=11)
    ax1.set_title('Sequential vs Self-Attention', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, max(metrics_to_plot.values()) * 1.2])
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)

    # 2. Entropy
    ax2 = axes[0, 1]
    entropy_val = metrics_dict['entropy_mean']
    max_entropy = np.log2(5)  # Maximum entropy for 5 positions
    ax2.bar(['Mean Entropy'], [entropy_val], color='#2ecc71', alpha=0.7, edgecolor='black')
    ax2.axhline(y=max_entropy, color='red', linestyle='--', label=f'Max Entropy ({max_entropy:.2f} bits)', linewidth=2)
    ax2.set_ylabel('Entropy (bits)', fontsize=11)
    ax2.set_title('Attention Distribution Entropy', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, max_entropy * 1.1])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.text(0, entropy_val, f'{entropy_val:.3f}', ha='center', va='bottom', fontsize=10)

    # 3. Forward vs Backward
    ax3 = axes[1, 0]
    fwd_bwd = {
        'Forward\nAttention': metrics_dict['forward_attention_mean'],
        'Backward\nAttention': metrics_dict['backward_attention_mean']
    }
    colors = ['#9b59b6', '#f39c12']
    bars = ax3.bar(fwd_bwd.keys(), fwd_bwd.values(), color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Mean Attention Weight', fontsize=11)
    ax3.set_title('Forward vs Backward Attention', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, max(fwd_bwd.values()) * 1.2])
    ax3.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)

    # 4. Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
    Key Metrics Summary:

    Sequential Score: {metrics_dict['sequential_score_mean']:.4f} ± {metrics_dict['sequential_score_std']:.4f}
    Self-Attention: {metrics_dict['self_attention_mean']:.4f} ± {metrics_dict['self_attention_std']:.4f}

    Entropy: {metrics_dict['entropy_mean']:.3f} ± {metrics_dict['entropy_std']:.3f} bits
    Max Possible: {max_entropy:.3f} bits

    Forward Attention: {metrics_dict['forward_attention_mean']:.4f}
    Backward Attention: {metrics_dict['backward_attention_mean']:.4f}
    Ratio: {metrics_dict.get('forward_backward_ratio_mean', 'N/A')}

    Interpretation:
    • High sequential score → Chain-like processing
    • High entropy → Distributed/parallel processing
    • Low entropy → Focused/sequential processing
    """

    ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics comparison to {save_path}")


def plot_layer_evolution(attention_dict, metric_name, save_path):
    """
    Plot how a specific metric evolves across layers

    Args:
        attention_dict: Dict mapping layer_idx to attention matrix
        metric_name: Name of metric to track ('sequential', 'entropy', etc.)
        save_path: Path to save the figure
    """
    from .attention_metrics import (compute_sequential_score, compute_attention_entropy,
                                     compute_self_attention_score)

    layers = sorted(attention_dict.keys())
    values = []

    for layer_idx in layers:
        attn = attention_dict[layer_idx]
        if metric_name == 'sequential':
            val, _ = compute_sequential_score(attn)
        elif metric_name == 'entropy':
            val, _ = compute_attention_entropy(attn)
        elif metric_name == 'self_attention':
            val, _ = compute_self_attention_score(attn)
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
        values.append(val)

    plt.figure(figsize=(12, 6))
    plt.plot(layers, values, marker='o', linewidth=2, markersize=8, color='#3498db')
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
    plt.title(f'{metric_name.replace("_", " ").title()} Across Layers', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved layer evolution plot to {save_path}")
