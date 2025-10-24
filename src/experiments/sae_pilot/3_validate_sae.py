"""
Story 1.3: Validate SAE Quality

This script computes comprehensive quality metrics for the trained SAE:
- Reconstruction quality (MSE, explained variance)
- Sparsity metrics (L0, L1, % dead features)
- Feature usage statistics

Generates validation report and quality visualizations.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import defaultdict


class SparseAutoencoder(nn.Module):
    """Simple sparse autoencoder - must match training architecture."""

    def __init__(self, input_dim: int = 2048, n_features: int = 8192):
        super().__init__()
        self.input_dim = input_dim
        self.n_features = n_features
        self.encoder = nn.Linear(input_dim, n_features, bias=True)
        self.decoder = nn.Linear(n_features, input_dim, bias=True)

    def encode(self, x):
        return torch.relu(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


def load_sae(weights_path: str, device='cuda'):
    """Load trained SAE from checkpoint."""
    print(f"Loading SAE from {weights_path}...")

    checkpoint = torch.load(weights_path, map_location=device)

    sae = SparseAutoencoder(
        input_dim=checkpoint['config']['input_dim'],
        n_features=checkpoint['config']['n_features']
    )
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.to(device)
    sae.eval()

    print(f"  Input dim: {sae.input_dim}")
    print(f"  Features: {sae.n_features}")
    print(f"  Expansion: {sae.n_features / sae.input_dim:.2f}x")

    return sae, checkpoint


def compute_reconstruction_quality(sae, activations, device='cuda'):
    """Compute reconstruction metrics."""
    print("\nComputing reconstruction quality...")

    with torch.no_grad():
        x = activations.to(device)
        x_recon, features = sae(x)

        # MSE loss
        mse = nn.functional.mse_loss(x_recon, x).item()

        # Explained variance
        var_original = torch.var(x, dim=0).mean().item()
        var_residual = torch.var(x - x_recon, dim=0).mean().item()
        explained_var = 1 - (var_residual / var_original)

        # Cosine similarity
        cosine_sim = nn.functional.cosine_similarity(x, x_recon, dim=1).mean().item()

    print(f"  MSE: {mse:.6f}")
    print(f"  Explained variance: {explained_var:.4f} ({explained_var*100:.2f}%)")
    print(f"  Cosine similarity: {cosine_sim:.4f}")

    return {
        'mse': mse,
        'explained_variance': explained_var,
        'cosine_similarity': cosine_sim
    }


def compute_sparsity_metrics(sae, activations, device='cuda', threshold=1e-3):
    """Compute sparsity metrics."""
    print("\nComputing sparsity metrics...")

    with torch.no_grad():
        x = activations.to(device)
        _, features = sae(x)

        # L0 sparsity (mean active features per vector)
        active_features = (features.abs() > threshold).float()
        l0_per_vector = active_features.sum(dim=1)
        l0_mean = l0_per_vector.mean().item()
        l0_std = l0_per_vector.std().item()

        # L1 norm
        l1_mean = features.abs().sum(dim=1).mean().item()

        # Feature usage (how many vectors activate each feature)
        feature_usage = active_features.sum(dim=0).cpu().numpy()
        n_dead_features = (feature_usage == 0).sum()
        pct_dead = (n_dead_features / sae.n_features) * 100

        # Max activation per feature
        max_activations = features.max(dim=0)[0].cpu().numpy()

    print(f"  L0 (active features/vector): {l0_mean:.2f} ± {l0_std:.2f}")
    print(f"  L0 percentage: {l0_mean/sae.n_features*100:.2f}%")
    print(f"  L1 norm: {l1_mean:.4f}")
    print(f"  Dead features: {n_dead_features}/{sae.n_features} ({pct_dead:.2f}%)")

    return {
        'l0_mean': l0_mean,
        'l0_std': l0_std,
        'l0_percentage': l0_mean / sae.n_features,
        'l1_mean': l1_mean,
        'n_dead_features': int(n_dead_features),
        'pct_dead_features': pct_dead,
        'feature_usage': feature_usage,
        'max_activations': max_activations
    }


def visualize_validation(
    recon_metrics,
    sparsity_metrics,
    output_dir: Path
):
    """Create validation visualizations."""
    print("\nGenerating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Feature usage histogram
    ax = axes[0, 0]
    usage = sparsity_metrics['feature_usage']
    ax.hist(usage[usage > 0], bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Vectors Activating Feature')
    ax.set_ylabel('Number of Features')
    ax.set_title(f'Feature Usage Distribution\n{sparsity_metrics["n_dead_features"]:,} dead features ({sparsity_metrics["pct_dead_features"]:.1f}%)')
    ax.grid(alpha=0.3)

    # 2. Max activation distribution
    ax = axes[0, 1]
    max_acts = sparsity_metrics['max_activations']
    ax.hist(max_acts[max_acts > 0], bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax.set_xlabel('Max Activation Value')
    ax.set_ylabel('Number of Features')
    ax.set_title('Maximum Feature Activations')
    ax.grid(alpha=0.3)

    # 3. Reconstruction quality metrics
    ax = axes[1, 0]
    metrics = [
        ('Explained\nVariance', recon_metrics['explained_variance'] * 100),
        ('Cosine\nSimilarity', recon_metrics['cosine_similarity'] * 100)
    ]
    names, values = zip(*metrics)
    colors = ['green' if v > 90 else 'orange' if v > 70 else 'red' for v in values]
    bars = ax.bar(names, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Reconstruction Quality')
    ax.set_ylim([0, 105])
    ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Good (>90%)')
    ax.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='Fair (>70%)')
    ax.grid(alpha=0.3, axis='y')
    ax.legend()

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    # 4. Sparsity metrics
    ax = axes[1, 1]
    l0_pct = sparsity_metrics['l0_percentage'] * 100
    dead_pct = sparsity_metrics['pct_dead_features']

    metrics_sparse = [
        ('L0\nSparsity', l0_pct),
        ('Active\nFeatures', 100 - dead_pct)
    ]
    names, values = zip(*metrics_sparse)
    colors = ['green', 'blue']
    bars = ax.bar(names, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Sparsity Metrics')
    ax.set_ylim([0, 105])
    ax.grid(alpha=0.3, axis='y')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    # Save
    for ext in ['png', 'pdf']:
        save_path = output_dir / f'sae_validation.{ext}'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sae_weights', type=str,
                        default='src/experiments/sae_pilot/results/sae_weights.pt',
                        help='Path to SAE weights')
    parser.add_argument('--activations', type=str,
                        default='src/experiments/sae_pilot/data/sae_training_activations.pt',
                        help='Path to activations')
    parser.add_argument('--output_dir', type=str,
                        default='src/experiments/sae_pilot/results',
                        help='Output directory')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SAE
    sae, checkpoint = load_sae(args.sae_weights, device)

    # Load activations
    print(f"\nLoading activations from {args.activations}...")
    activation_data = torch.load(args.activations)
    activations = activation_data['activations']
    print(f"  Shape: {activations.shape}")

    # Compute metrics
    recon_metrics = compute_reconstruction_quality(sae, activations, device)
    sparsity_metrics = compute_sparsity_metrics(sae, activations, device)

    # Generate visualizations
    visualize_validation(recon_metrics, sparsity_metrics, output_dir)

    # Save validation report
    validation_report = {
        'sae_config': checkpoint['config'],
        'training_info': checkpoint['training_info'],
        'reconstruction_quality': recon_metrics,
        'sparsity_metrics': {
            'l0_mean': sparsity_metrics['l0_mean'],
            'l0_std': sparsity_metrics['l0_std'],
            'l0_percentage': sparsity_metrics['l0_percentage'],
            'l1_mean': sparsity_metrics['l1_mean'],
            'n_dead_features': sparsity_metrics['n_dead_features'],
            'pct_dead_features': sparsity_metrics['pct_dead_features']
        },
        'verdict': {
            'reconstruction': 'GOOD' if recon_metrics['explained_variance'] > 0.90 else 'FAIR' if recon_metrics['explained_variance'] > 0.70 else 'POOR',
            'sparsity': 'GOOD' if sparsity_metrics['pct_dead_features'] < 10 else 'FAIR' if sparsity_metrics['pct_dead_features'] < 30 else 'POOR'
        }
    }

    report_path = output_dir / 'validation_report.json'
    with open(report_path, 'w') as f:
        json.dump(validation_report, f, indent=2)

    print(f"\n✅ Validation complete! Report saved to {report_path}")
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Reconstruction: {validation_report['verdict']['reconstruction']}")
    print(f"  Explained variance: {recon_metrics['explained_variance']*100:.2f}%")
    print(f"  Cosine similarity: {recon_metrics['cosine_similarity']*100:.2f}%")
    print(f"\nSparsity: {validation_report['verdict']['sparsity']}")
    print(f"  L0: {sparsity_metrics['l0_mean']:.2f} features/vector ({sparsity_metrics['l0_percentage']*100:.2f}%)")
    print(f"  Dead features: {sparsity_metrics['pct_dead_features']:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
