"""
Visualization Script
Creates plots from activation patching experiment results and logs to WandB.

Usage:
    python plot_results.py --results ../experiments/activation_patching/results/experiment_results.json \
                           --output plots/ \
                           --wandb_run_id <run_id>
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import wandb


def plot_accuracy_by_layer(summary: dict, output_path: str):
    """Plot accuracy comparison across all conditions.

    Args:
        summary: Summary dict from experiment results
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    layer_results = summary['layer_results']

    conditions = ['Clean', 'Corrupted', 'Early\n(L3)', 'Middle\n(L6)', 'Late\n(L11)']
    accuracies = [
        summary['clean_accuracy'],
        summary['corrupted_accuracy'],
        layer_results['early']['accuracy'],
        layer_results['middle']['accuracy'],
        layer_results['late']['accuracy']
    ]

    colors = ['green', 'red', 'skyblue', 'steelblue', 'darkblue']
    bars = ax.bar(conditions, accuracies, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.1%}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim(0, min(1.0, max(accuracies) * 1.15))
    ax.set_title('Activation Patching: Accuracy by Layer', fontsize=16, fontweight='bold', pad=20)

    # Add horizontal line for clean baseline
    ax.axhline(summary['clean_accuracy'], color='green', linestyle='--',
               alpha=0.4, linewidth=2, label='Clean baseline')

    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_recovery_by_layer(summary: dict, output_path: str):
    """Plot recovery rate comparison across layers.

    Args:
        summary: Summary dict from experiment results
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    layer_results = summary['layer_results']

    layers = ['Early\n(L3)', 'Middle\n(L6)', 'Late\n(L11)']
    recovery_rates = [
        layer_results['early']['recovery_rate'],
        layer_results['middle']['recovery_rate'],
        layer_results['late']['recovery_rate']
    ]

    colors = ['skyblue', 'steelblue', 'darkblue']
    bars = ax.bar(layers, recovery_rates, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, recovery_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.1%}',
                ha='center', va='bottom', fontsize=13, fontweight='bold')

    ax.set_ylabel('Recovery Rate', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(1.0, max(recovery_rates) * 1.2) if recovery_rates else 1.0)
    ax.set_title('Recovery Rate by Layer\n(Higher = Stronger Causal Effect)',
                 fontsize=16, fontweight='bold', pad=20)

    # Add 50% threshold line
    ax.axhline(0.5, color='red', linestyle='--', linewidth=2.5,
               label='50% threshold', alpha=0.7)

    ax.legend(fontsize=12, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_layer_importance(summary: dict, output_path: str):
    """Plot layer importance ranking.

    Args:
        summary: Summary dict from experiment results
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    layer_results = summary['layer_results']

    layer_names = list(layer_results.keys())
    importance = [layer_results[l]['recovery_rate'] for l in layer_names]

    # Sort by importance (descending)
    sorted_idx = np.argsort(importance)[::-1]
    sorted_layers = [layer_names[i].capitalize() + f" (L{[3,6,11][i]})" for i in sorted_idx]
    sorted_importance = [importance[i] for i in sorted_idx]

    colors = ['gold', 'silver', '#CD7F32']  # Gold, silver, bronze
    bars = ax.barh(sorted_layers, sorted_importance, color=colors,
                   edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, sorted_importance):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                f'{val:.1%}',
                ha='left', va='center', fontsize=13, fontweight='bold')

    ax.set_xlabel('Recovery Rate (Causal Importance)', fontsize=14, fontweight='bold')
    ax.set_title('Layer Importance Ranking', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(0, max(1.0, max(sorted_importance) * 1.15) if sorted_importance else 1.0)

    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_summary_text(summary: dict) -> str:
    """Create text summary of results.

    Args:
        summary: Summary dict from experiment results

    Returns:
        Formatted summary text
    """
    layer_results = summary['layer_results']

    text = f"""
ACTIVATION PATCHING EXPERIMENT RESULTS
{'='*60}

BASELINE PERFORMANCE:
  Clean Accuracy:      {summary['clean_accuracy']:.2%}
  Corrupted Accuracy:  {summary['corrupted_accuracy']:.2%}
  Accuracy Drop:       {summary['clean_accuracy'] - summary['corrupted_accuracy']:.2%}

PATCHING RESULTS BY LAYER:
"""

    for layer_name in ['early', 'middle', 'late']:
        layer_idx = {'early': 3, 'middle': 6, 'late': 11}[layer_name]
        acc = layer_results[layer_name]['accuracy']
        recovery = layer_results[layer_name]['recovery_rate']
        text += f"  {layer_name.capitalize():8s} (L{layer_idx:2d}): Accuracy = {acc:.2%}, Recovery = {recovery:.1%}\n"

    # Find best layer
    best_layer = max(layer_results.keys(), key=lambda k: layer_results[k]['recovery_rate'])
    best_recovery = layer_results[best_layer]['recovery_rate']

    text += f"\nBEST LAYER: {best_layer.capitalize()} with {best_recovery:.1%} recovery rate\n"

    # Interpretation
    text += f"\n{'='*60}\n"
    if best_recovery > 0.5:
        text += "INTERPRETATION: Strong causal effect detected!\n"
        text += f"Patching clean activations at {best_layer} layer recovers >50% of accuracy.\n"
        text += "This suggests continuous thoughts are causally involved in reasoning.\n"
    elif best_recovery > 0.3:
        text += "INTERPRETATION: Moderate causal effect detected.\n"
        text += f"Patching shows some recovery ({best_recovery:.1%}), suggesting partial causal role.\n"
    else:
        text += "INTERPRETATION: Weak or no causal effect.\n"
        text += "Continuous thoughts may be epiphenomenal correlates rather than causal.\n"

    text += f"{'='*60}\n"

    return text


def log_to_wandb(results_path: str, plots_dir: str, wandb_run_id: str = None):
    """Log plots to existing WandB run.

    Args:
        results_path: Path to experiment_results.json
        plots_dir: Directory containing plots
        wandb_run_id: WandB run ID to resume (optional)
    """
    # Load results
    with open(results_path, 'r') as f:
        data = json.load(f)

    summary = data['summary']

    # Resume WandB run if ID provided
    if wandb_run_id:
        wandb.init(id=wandb_run_id, resume='must')
    else:
        # Create new run for visualization
        wandb.init(project='codi-activation-patching', name='visualization')

    # Log plots
    plot_files = {
        'plots/accuracy_by_layer': f'{plots_dir}/accuracy_by_layer.png',
        'plots/recovery_by_layer': f'{plots_dir}/recovery_by_layer.png',
        'plots/layer_importance': f'{plots_dir}/layer_importance.png'
    }

    for key, path in plot_files.items():
        if Path(path).exists():
            wandb.log({key: wandb.Image(path)})
            print(f"✓ Logged {key} to WandB")

    # Log summary text
    summary_text = create_summary_text(summary)
    wandb.log({'summary_text': wandb.Html(f"<pre>{summary_text}</pre>")})

    wandb.finish()
    print(f"✓ All plots logged to WandB")


def main():
    parser = argparse.ArgumentParser(description="Visualize activation patching results")
    parser.add_argument('--results', type=str, required=True,
                        help='Path to experiment_results.json')
    parser.add_argument('--output', type=str, default='plots/',
                        help='Output directory for plots')
    parser.add_argument('--wandb_run_id', type=str, default=None,
                        help='WandB run ID to log plots to (optional)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Skip WandB logging')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print(f"Loading results from {args.results}...")
    with open(args.results, 'r') as f:
        data = json.load(f)

    summary = data['summary']

    print(f"\n{'='*60}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*60}\n")

    # Create plots
    plot_accuracy_by_layer(summary, str(output_dir / 'accuracy_by_layer.png'))
    plot_recovery_by_layer(summary, str(output_dir / 'recovery_by_layer.png'))
    plot_layer_importance(summary, str(output_dir / 'layer_importance.png'))

    # Print summary
    summary_text = create_summary_text(summary)
    print(summary_text)

    # Log to WandB
    if not args.no_wandb:
        print(f"\n{'='*60}")
        print("LOGGING TO WANDB")
        print(f"{'='*60}\n")
        log_to_wandb(args.results, str(output_dir), args.wandb_run_id)

    print(f"\n✓ Visualization complete! Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
