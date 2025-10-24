"""
Story 2.1: Visualize Top-10 SAE Features

This script visualizes the most active/important SAE features to enable
human interpretation of what patterns they capture.

For each top feature, shows:
- Which problems activate it most
- Which operation types activate it
- Which token positions activate it
- Which layers activate it
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import Counter, defaultdict


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
    checkpoint = torch.load(weights_path, map_location=device)
    sae = SparseAutoencoder(
        input_dim=checkpoint['config']['input_dim'],
        n_features=checkpoint['config']['n_features']
    )
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.to(device)
    sae.eval()
    return sae, checkpoint


def extract_all_features(sae, activations, metadata, device='cuda', threshold=1e-3):
    """Extract all feature activations and compute statistics."""
    print("\nExtracting feature activations...")

    with torch.no_grad():
        x = activations.to(device)
        _, features = sae(x)

        # Compute feature statistics
        feature_max_act = features.max(dim=0)[0].cpu().numpy()
        feature_mean_act = features.mean(dim=0).cpu().numpy()
        feature_usage = (features > threshold).sum(dim=0).cpu().numpy()

        # Get activation patterns for each feature
        features_np = features.cpu().numpy()

    print(f"  Features extracted: {features_np.shape}")

    return {
        'activations': features_np,
        'max_activation': feature_max_act,
        'mean_activation': feature_mean_act,
        'usage_count': feature_usage
    }


def analyze_top_features(
    feature_stats,
    metadata,
    n_top=10,
    threshold=1e-3
):
    """Analyze top features by different criteria."""
    print(f"\nAnalyzing top-{n_top} features...")

    # Get top features by different criteria
    top_by_max = np.argsort(feature_stats['max_activation'])[::-1][:n_top]
    top_by_usage = np.argsort(feature_stats['usage_count'])[::-1][:n_top]
    top_by_mean = np.argsort(feature_stats['mean_activation'])[::-1][:n_top]

    # Combine and get unique top features
    top_features = np.unique(np.concatenate([top_by_max, top_by_usage, top_by_mean]))[:n_top]

    print(f"  Selected {len(top_features)} features for analysis")

    # Analyze each feature
    feature_analyses = {}

    for feat_idx in top_features:
        # Get activation pattern
        activations = feature_stats['activations'][:, feat_idx]
        active_mask = activations > threshold

        if not active_mask.any():
            continue

        # Analyze operation types
        op_types = [metadata['operation_types'][i] for i in range(len(activations)) if active_mask[i]]
        op_counts = Counter(op_types)

        # Analyze layers
        layers = [metadata['layers'][i] for i in range(len(activations)) if active_mask[i]]
        layer_counts = Counter(layers)

        # Analyze token positions
        tokens = [metadata['tokens'][i] for i in range(len(activations)) if active_mask[i]]
        token_counts = Counter(tokens)

        # Get top activating instances
        top_indices = np.argsort(activations)[::-1][:5]
        top_activations = [
            {
                'problem_idx': metadata['problems'][i],
                'layer': metadata['layers'][i],
                'token': metadata['tokens'][i],
                'operation': metadata['operation_types'][i],
                'activation': float(activations[i])
            }
            for i in top_indices
        ]

        feature_analyses[int(feat_idx)] = {
            'feature_id': int(feat_idx),
            'max_activation': float(feature_stats['max_activation'][feat_idx]),
            'mean_activation': float(feature_stats['mean_activation'][feat_idx]),
            'usage_count': int(feature_stats['usage_count'][feat_idx]),
            'operation_distribution': dict(op_counts),
            'layer_distribution': dict(layer_counts),
            'token_distribution': dict(token_counts),
            'top_activations': top_activations
        }

    return feature_analyses


def visualize_features(feature_analyses, output_dir: Path):
    """Create visualizations for top features."""
    print(f"\nGenerating visualizations for {len(feature_analyses)} features...")

    # Create subplots grid
    n_features = len(feature_analyses)
    fig, axes = plt.subplots(n_features, 3, figsize=(15, 4*n_features))

    if n_features == 1:
        axes = axes.reshape(1, -1)

    feature_ids = sorted(feature_analyses.keys(),
                         key=lambda f: feature_analyses[f]['max_activation'],
                         reverse=True)

    for row_idx, feat_id in enumerate(feature_ids):
        analysis = feature_analyses[feat_id]

        # 1. Operation type distribution
        ax = axes[row_idx, 0]
        op_dist = analysis['operation_distribution']
        if op_dist:
            labels = list(op_dist.keys())
            values = list(op_dist.values())
            colors = ['red', 'blue', 'green'][:len(labels)]
            ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel('Activation Count')
            ax.set_title(f'Feature {feat_id}: Operation Distribution')
            ax.tick_params(axis='x', rotation=45)

        # 2. Layer distribution
        ax = axes[row_idx, 1]
        layer_dist = analysis['layer_distribution']
        if layer_dist:
            layers = sorted(layer_dist.keys())
            values = [layer_dist[l] for l in layers]
            layer_names = [f'L{l}' for l in layers]
            ax.bar(layer_names, values, color='purple', alpha=0.7, edgecolor='black')
            ax.set_ylabel('Activation Count')
            ax.set_title(f'Feature {feat_id}: Layer Distribution')

        # 3. Token position distribution
        ax = axes[row_idx, 2]
        token_dist = analysis['token_distribution']
        if token_dist:
            tokens = sorted(token_dist.keys())
            values = [token_dist[t] for t in tokens]
            token_names = [f'T{t}' for t in tokens]
            ax.bar(token_names, values, color='orange', alpha=0.7, edgecolor='black')
            ax.set_ylabel('Activation Count')
            ax.set_title(f'Feature {feat_id}: Token Distribution')

    plt.tight_layout()

    # Save
    for ext in ['png', 'pdf']:
        save_path = output_dir / f'top_features_visualization.{ext}'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sae_weights', type=str,
                        default='src/experiments/sae_pilot/results/sae_weights.pt')
    parser.add_argument('--activations', type=str,
                        default='src/experiments/sae_pilot/data/sae_training_activations.pt')
    parser.add_argument('--output_dir', type=str,
                        default='src/experiments/sae_pilot/results')
    parser.add_argument('--n_top', type=int, default=10,
                        help='Number of top features to analyze')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)

    # Load SAE
    print("Loading SAE...")
    sae, checkpoint = load_sae(args.sae_weights, device)

    # Load activations
    print("Loading activations...")
    activation_data = torch.load(args.activations)
    activations = activation_data['activations']
    metadata = activation_data['metadata']

    # Extract features
    feature_stats = extract_all_features(sae, activations, metadata, device)

    # Analyze top features
    feature_analyses = analyze_top_features(
        feature_stats,
        metadata,
        n_top=args.n_top
    )

    # Visualize
    visualize_features(feature_analyses, output_dir)

    # Save analysis
    analysis_path = output_dir / 'feature_analysis.json'
    with open(analysis_path, 'w') as f:
        json.dump(feature_analyses, f, indent=2)

    print(f"\n✅ Feature analysis complete! Saved to {analysis_path}")

    # Print summary
    print("\n" + "="*80)
    print("TOP FEATURES SUMMARY")
    print("="*80)

    for feat_id in sorted(feature_analyses.keys(),
                          key=lambda f: feature_analyses[f]['max_activation'],
                          reverse=True):
        analysis = feature_analyses[feat_id]
        print(f"\nFeature {feat_id}:")
        print(f"  Max activation: {analysis['max_activation']:.4f}")
        print(f"  Usage: {analysis['usage_count']} vectors")
        print(f"  Operation preference: {max(analysis['operation_distribution'].items(), key=lambda x: x[1])}")
        print(f"  Layer preference: {max(analysis['layer_distribution'].items(), key=lambda x: x[1])}")
        print(f"  Token preference: {max(analysis['token_distribution'].items(), key=lambda x: x[1])}")

    print("="*80)


if __name__ == "__main__":
    main()
