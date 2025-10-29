"""
Find interesting (specialized) features in TopK SAE.

Instead of most common features, find:
1. Rare features (5-30% activation frequency)
2. Layer-selective features (high selectivity index)
3. Features with distinctive token correlations
"""

import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

import sys
sys.path.insert(0, 'src/experiments/topk_grid_pilot')
from topk_sae import TopKAutoencoder


def load_sae_and_data(layer=14, position=3, k=100, latent_dim=512):
    """Load SAE model and validation data."""
    print(f"Loading SAE model: Layer {layer}, Position {position}, K={k}, d={latent_dim}")

    # Load model
    ckpt_path = f'src/experiments/topk_grid_pilot/results/checkpoints/pos{position}_layer{layer}_d{latent_dim}_k{k}.pt'
    ckpt = torch.load(ckpt_path, weights_only=False)

    model = TopKAutoencoder(
        input_dim=ckpt['config']['input_dim'],
        latent_dim=ckpt['config']['latent_dim'],
        k=ckpt['config']['k']
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Load validation data
    print("Loading validation data...")
    val_data = torch.load('src/experiments/sae_cot_decoder/data/full_val_activations.pt', weights_only=False)
    positions = np.array(val_data['metadata']['positions'])
    layers = np.array(val_data['metadata']['layers'])

    mask = (positions == position) & (layers == layer)
    activations = val_data['activations'][mask]

    # Extract metadata
    problem_ids = [val_data['metadata']['problem_ids'][i] for i, m in enumerate(mask) if m]
    cot_sequences = [val_data['metadata']['cot_sequences'][i] for i, m in enumerate(mask) if m]

    print(f"Loaded {len(activations)} samples for Layer {layer}, Position {position}")

    return model, activations, problem_ids, cot_sequences


def analyze_all_features(model, activations):
    """Compute activation statistics for ALL features."""
    print("\nExtracting sparse features for all samples...")

    with torch.no_grad():
        _, sparse_features, _ = model(activations)

    print(f"Sparse features shape: {sparse_features.shape}")
    print(f"K (active per sample): {(sparse_features != 0).sum(dim=1).float().mean():.1f}")

    # Compute activation frequency for each feature
    activation_freq = (sparse_features != 0).float().mean(dim=0)

    # Compute mean activation magnitude
    mean_activation = sparse_features.mean(dim=0)

    # Compute activation strength (mean of non-zero activations)
    activation_strength = []
    for feat_id in range(sparse_features.shape[1]):
        activations_feat = sparse_features[:, feat_id]
        nonzero = activations_feat[activations_feat != 0]
        if len(nonzero) > 0:
            activation_strength.append(nonzero.mean().item())
        else:
            activation_strength.append(0.0)
    activation_strength = torch.tensor(activation_strength)

    return sparse_features, activation_freq, mean_activation, activation_strength


def find_specialized_features(activation_freq, n_features=20):
    """
    Find specialized features based on activation frequency.

    Target: Features that are active in 5-30% of samples (specialized but not dead)
    """
    print(f"\n{'='*80}")
    print("FINDING SPECIALIZED FEATURES")
    print(f"{'='*80}")

    # Activation frequency distribution
    freq_values = activation_freq.numpy()

    print("\nActivation Frequency Distribution:")
    print(f"  Min: {freq_values.min():.1%}")
    print(f"  25th percentile: {np.percentile(freq_values, 25):.1%}")
    print(f"  Median: {np.median(freq_values):.1%}")
    print(f"  75th percentile: {np.percentile(freq_values, 75):.1%}")
    print(f"  Max: {freq_values.max():.1%}")

    # Find features in the "specialized" range (5-30%)
    specialized_mask = (activation_freq >= 0.05) & (activation_freq <= 0.30)
    specialized_indices = torch.where(specialized_mask)[0]

    print(f"\nFeatures with 5-30% activation: {len(specialized_indices)}")

    if len(specialized_indices) == 0:
        print("No features in 5-30% range. Expanding search...")
        # Try 1-40% range
        specialized_mask = (activation_freq >= 0.01) & (activation_freq <= 0.40)
        specialized_indices = torch.where(specialized_mask)[0]
        print(f"Features with 1-40% activation: {len(specialized_indices)}")

    if len(specialized_indices) == 0:
        print("Still no specialized features. Using lowest frequency features...")
        # Just take the least common features
        specialized_indices = activation_freq.argsort()[:n_features]
    else:
        # Sample diverse features from this range
        if len(specialized_indices) > n_features:
            # Sort by frequency and take evenly spaced samples
            specialized_indices = specialized_indices[activation_freq[specialized_indices].argsort()]
            step = len(specialized_indices) // n_features
            specialized_indices = specialized_indices[::step][:n_features]

    print(f"\nSelected {len(specialized_indices)} specialized features:")
    for idx in specialized_indices[:10]:
        print(f"  Feature {idx.item()}: {activation_freq[idx].item():.1%} activation")

    return specialized_indices


def main():
    """Find interesting features for visualization."""
    layer = 14
    position = 3
    k = 100
    latent_dim = 512

    # Step 1: Load model and data
    model, activations, problem_ids, cot_sequences = load_sae_and_data(
        layer=layer, position=position, k=k, latent_dim=latent_dim
    )

    # Step 2: Analyze all features
    sparse_features, activation_freq, mean_activation, activation_strength = analyze_all_features(
        model, activations
    )

    # Step 3: Find specialized features
    specialized_feature_ids = find_specialized_features(activation_freq, n_features=20)

    print(f"\n{'='*80}")
    print("SPECIALIZED FEATURES IDENTIFIED")
    print(f"{'='*80}")
    print("\nTop 20 specialized features (by activation frequency):")
    for i, feat_id in enumerate(specialized_feature_ids, 1):
        feat_id = feat_id.item()
        freq = activation_freq[feat_id].item()
        strength = activation_strength[feat_id].item()
        print(f"{i:2d}. Feature {feat_id:3d}: {freq:5.1%} activation, "
              f"mean strength={strength:.3f}")

    # Save feature IDs for next step
    output_path = Path('src/experiments/topk_grid_pilot/visualizations/specialized_features.txt')
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        for feat_id in specialized_feature_ids:
            f.write(f"{feat_id.item()}\n")

    print(f"\n✓ Saved specialized feature IDs to: {output_path}")
    print("\nNext step: Run visualize_features_neuronpedia.py with these feature IDs")

    return specialized_feature_ids


if __name__ == '__main__':
    main()
