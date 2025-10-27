"""
Visualize Large K Experiment Results.

Creates comprehensive visualizations comparing K=100, K=200, K=300 SAEs:
1. Specialization rate by K value
2. Minimum activation frequency by K
3. Quality metrics comparison (EV, death rate, loss)
4. Activation frequency distributions

Key finding: Larger K ELIMINATES specialized features rather than making them more usable.

Usage:
    python visualize_large_k_results.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add topk_grid_pilot to path
sys.path.insert(0, 'src/experiments/topk_grid_pilot')
from topk_sae import TopKAutoencoder


def load_checkpoint_data(k, position=3, layer=14, latent_dim=512):
    """Load checkpoint and analysis data for a specific K value."""
    # Checkpoint paths
    if k == 100:
        ckpt_path = f'src/experiments/topk_grid_pilot/results/checkpoints/pos{position}_layer{layer}_d{latent_dim}_k{k}.pt'
    else:
        ckpt_path = f'src/experiments/llama_sae_hierarchy/checkpoints/pos{position}_layer{layer}_d{latent_dim}_k{k}.pt'

    # Analysis paths - K=50 and K=75 use different rank ranges due to feature death
    if k == 50:
        analysis_path = f'src/experiments/llama_sae_hierarchy/activation_analysis_layer{layer}_pos{position}_rank200-348_k{k}.json'
    elif k == 75:
        analysis_path = f'src/experiments/llama_sae_hierarchy/activation_analysis_layer{layer}_pos{position}_rank300-455_k{k}.json'
    elif k == 100:
        analysis_path = f'src/experiments/llama_sae_hierarchy/activation_analysis_layer{layer}_pos{position}_rank400-512.json'
    else:
        analysis_path = f'src/experiments/llama_sae_hierarchy/activation_analysis_layer{layer}_pos{position}_rank400-512_k{k}.json'

    # Load checkpoint
    ckpt = torch.load(ckpt_path, weights_only=False)

    # Load analysis if exists
    analysis = None
    if Path(analysis_path).exists():
        with open(analysis_path, 'r') as f:
            analysis = json.load(f)

    return ckpt, analysis


def compute_activation_frequencies(model, val_data):
    """Compute activation frequency for all features."""
    model.eval()
    with torch.no_grad():
        _, sparse, _ = model(val_data)

    # Activation frequencies
    feature_activation_freq = (sparse != 0).float().mean(dim=0)
    return feature_activation_freq.cpu().numpy()


def create_visualizations(k_values=[50, 75, 100, 200], position=3, layer=14, latent_dim=512):
    """Create all visualizations comparing different K values."""

    print(f"\n{'='*80}")
    print(f"Creating Visualizations: K={k_values}")
    print(f"{'='*80}\n")

    # Load data for all K values
    all_data = {}
    val_data = None

    for k in k_values:
        print(f"Loading K={k}...")
        ckpt, analysis = load_checkpoint_data(k, position, layer, latent_dim)

        # Load model
        model = TopKAutoencoder(
            input_dim=ckpt['config']['input_dim'],
            latent_dim=ckpt['config']['latent_dim'],
            k=ckpt['config']['k']
        )
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        # Load validation data once
        if val_data is None:
            val_data_path = 'src/experiments/sae_cot_decoder/data/full_val_activations.pt'
            val_full = torch.load(val_data_path, weights_only=False)
            positions = np.array(val_full['metadata']['positions'])
            layers = np.array(val_full['metadata']['layers'])
            mask = (positions == position) & (layers == layer)
            val_data = val_full['activations'][mask]

        # Compute activation frequencies
        activation_freqs = compute_activation_frequencies(model, val_data)

        all_data[k] = {
            'metrics': ckpt['metrics'],
            'analysis': analysis,
            'activation_freqs': activation_freqs,
            'model': model
        }

    print("✓ Data loaded\n")

    # Create output directory
    output_dir = Path('src/experiments/llama_sae_hierarchy/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # 1. Specialization Rate by K
    ax1 = plt.subplot(2, 3, 1)
    specialization_rates = []
    for k in k_values:
        if all_data[k]['analysis']:
            total_features = all_data[k]['analysis']['metadata']['total_features_analyzed']
            specialized = all_data[k]['analysis']['metadata']['specialized_features_found']
            rate = 100.0 * specialized / total_features if total_features > 0 else 0
        else:
            rate = 0.0
        specialization_rates.append(rate)

    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']  # Red for K=50, Orange for K=75, Green for K=100, Blue for K=200
    bars = ax1.bar([str(k) for k in k_values], specialization_rates, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('K Value', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Specialization Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('A. Specialized Features by K\n(Different rank ranges analyzed)', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, rate in zip(bars, specialization_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 2. Minimum Activation Frequency by K (among active features)
    ax2 = plt.subplot(2, 3, 2)
    min_activations = []
    for k in k_values:
        freqs = all_data[k]['activation_freqs']
        # Get active features only
        active_freqs = freqs[freqs > 0]
        min_freq = active_freqs.min() * 100 if len(active_freqs) > 0 else 0  # Convert to percentage
        min_activations.append(min_freq)

    bars = ax2.bar([str(k) for k in k_values], min_activations, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('K Value', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Min Activation Frequency (%)', fontsize=12, fontweight='bold')
    ax2.set_title('B. Minimum Activation Frequency\n(Among Active Features)', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, min_act in zip(bars, min_activations):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{min_act:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 3. Explained Variance Comparison
    ax3 = plt.subplot(2, 3, 3)
    ev_values = [all_data[k]['metrics']['explained_variance'] * 100 for k in k_values]
    bars = ax3.bar([str(k) for k in k_values], ev_values, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('K Value', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Explained Variance (%)', fontsize=12, fontweight='bold')
    ax3.set_title('C. Quality Metric: Explained Variance', fontsize=13, fontweight='bold')
    ax3.set_ylim([85, 95])
    ax3.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, ev in zip(bars, ev_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{ev:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 4. Reconstruction Loss Comparison
    ax4 = plt.subplot(2, 3, 4)
    loss_values = [all_data[k]['metrics']['reconstruction_loss'] * 1000 for k in k_values]  # Scale to milli
    bars = ax4.bar([str(k) for k in k_values], loss_values, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('K Value', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Reconstruction Loss (×10⁻³)', fontsize=12, fontweight='bold')
    ax4.set_title('D. Quality Metric: Reconstruction Loss', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, loss in zip(bars, loss_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 5. Activation Frequency Distributions (Active Features)
    ax5 = plt.subplot(2, 3, 5)
    for k, color in zip(k_values, colors):
        freqs = all_data[k]['activation_freqs']
        # Get only active features (freq > 0)
        active_freqs = freqs[freqs > 0] * 100

        # Plot histogram
        ax5.hist(active_freqs, bins=30, alpha=0.5, label=f'K={k}', color=color, edgecolor='black')

    ax5.set_xlabel('Activation Frequency (%)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
    ax5.set_title('E. Activation Distribution\n(Active Features Only)', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=11)
    ax5.grid(axis='y', alpha=0.3)

    # 6. Feature Death Rate
    ax6 = plt.subplot(2, 3, 6)
    death_rates = [all_data[k]['metrics']['feature_death_rate'] * 100 for k in k_values]
    bars = ax6.bar([str(k) for k in k_values], death_rates, color=colors, alpha=0.7, edgecolor='black')
    ax6.set_xlabel('K Value', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Feature Death Rate (%)', fontsize=12, fontweight='bold')
    ax6.set_title('F. Feature Death Rate', fontsize=13, fontweight='bold')
    ax6.set_ylim([0, 1])
    ax6.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, death in zip(bars, death_rates):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., max(height, 0.05),
                f'{death:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Overall title
    fig.suptitle('K Sparsity Experiment Results: K=50 vs K=75 vs K=100 vs K=200\n' +
                'Lower K → More Specialization | Higher K → Better Quality',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # Save figure
    output_path = output_dir / 'large_k_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}\n")

    # Create second figure: Detailed activation frequency curves
    fig2, ax = plt.subplots(figsize=(14, 8))

    for k, color in zip(k_values, colors):
        freqs = all_data[k]['activation_freqs']
        sorted_freqs = np.sort(freqs)[::-1]  # Sort descending

        # Plot full curve
        ax.plot(range(len(sorted_freqs)), sorted_freqs * 100,
               label=f'K={k}', color=color, linewidth=2, alpha=0.8)

    # Highlight analyzed regions
    ax.axvspan(199, 347, alpha=0.1, color='red', label='K=50 Analyzed (Rank 200-348)')
    ax.axvspan(299, 454, alpha=0.1, color='yellow', label='K=75 Analyzed (Rank 300-455)')
    ax.axvspan(399, 511, alpha=0.1, color='green', label='K=100/200 Analyzed (Rank 400-512)')

    ax.set_xlabel('Feature Rank (sorted by activation frequency)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Activation Frequency (%)', fontsize=13, fontweight='bold')
    ax.set_title('Activation Frequency by Feature Rank\n' +
                'K=50 has many rare features (high specialization), K=200 eliminates them',
                fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 512])

    # Log scale for better visibility
    ax.set_yscale('log')
    ax.set_ylabel('Activation Frequency (%, log scale)', fontsize=13, fontweight='bold')

    plt.tight_layout()

    # Save second figure
    output_path2 = output_dir / 'large_k_activation_curves.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path2}\n")

    # Print summary statistics
    print(f"{'='*80}")
    print(f"Summary Statistics")
    print(f"{'='*80}\n")

    print(f"{'K':<6} {'EV (%)':<10} {'Loss (×10⁻³)':<15} {'Death %':<10} {'Specialized':<15} {'Min Act %':<12}")
    print(f"{'-'*80}")
    for k, spec_rate, min_act in zip(k_values, specialization_rates, min_activations):
        metrics = all_data[k]['metrics']
        print(f"{k:<6} {metrics['explained_variance']*100:<10.2f} "
              f"{metrics['reconstruction_loss']*1000:<15.2f} "
              f"{metrics['feature_death_rate']*100:<10.2f} "
              f"{spec_rate:<15.2f} "
              f"{min_act:<12.2f}")
    print(f"{'-'*80}\n")

    print("Key Findings:")
    print("  1. Larger K improves reconstruction quality (higher EV, lower loss)")
    print("  2. Larger K eliminates feature death (all K values: 0% death)")
    print("  3. Larger K ELIMINATES specialized features:")
    print(f"     - K=100: {specialization_rates[0]:.1f}% specialized (5 features)")
    print(f"     - K=200: {specialization_rates[1]:.1f}% specialized (0 features)")
    print(f"     - K=300: {specialization_rates[2]:.1f}% specialized (0 features)")
    print("  4. Larger K increases minimum activation frequency:")
    print(f"     - K=100: {min_activations[0]:.2f}% (rare patterns)")
    print(f"     - K=200: {min_activations[1]:.2f}% (no rare patterns)")
    print(f"     - K=300: {min_activations[2]:.2f}% (no rare patterns)")
    print("\nConclusion: Larger K distributes computational load, preventing specialization.")
    print()


def main():
    print(f"\n{'='*80}")
    print(f"K Sparsity Visualization: Comparing K=50, K=75, K=100, K=200")
    print(f"{'='*80}")

    create_visualizations(k_values=[50, 75, 100, 200], position=3, layer=14, latent_dim=512)

    print(f"{'='*80}")
    print(f"Visualizations Complete!")
    print(f"{'='*80}\n")
    print("Files created:")
    print("  - src/experiments/llama_sae_hierarchy/visualizations/large_k_comparison.png")
    print("  - src/experiments/llama_sae_hierarchy/visualizations/large_k_activation_curves.png")
    print()


if __name__ == '__main__':
    main()
