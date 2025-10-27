"""
Train TopK SAE Grid for GPT-2

8 configs:
  Very Sparse:    d=192 K=20,40
  Medium Sparse:  d=256 K=30,50,75
  Larger Dict:    d=384 K=75, d=512 K=100,150

Position: 3, Layer: 8 (middle layer)

Usage:
    python train_gpt2_grid.py
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Import TopK SAE from existing code
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "topk_grid_pilot"))
from topk_sae import TopKAutoencoder


def load_gpt2_data(position=3, layer=8):
    """Load Position 3, Layer 8 activations from GPT-2 data."""
    print(f"Loading GPT-2 Position {position}, Layer {layer} data...")

    # Load full dataset
    train_data = torch.load(
        'src/experiments/gpt2_sae_training/data/gpt2_full_train_activations.pt',
        weights_only=False
    )
    val_data = torch.load(
        'src/experiments/gpt2_sae_training/data/gpt2_full_val_activations.pt',
        weights_only=False
    )

    # Extract Position 3, Layer 8
    def extract(data, pos, lyr):
        activations = data['activations']
        positions = np.array(data['metadata']['positions'])
        layers = np.array(data['metadata']['layers'])
        mask = (positions == pos) & (layers == lyr)
        return activations[mask]

    train_acts = extract(train_data, position, layer)
    val_acts = extract(val_data, position, layer)

    print(f"  Train: {train_acts.shape}")
    print(f"  Val: {val_acts.shape}")
    print()

    return train_acts, val_acts


def train_sae(model, train_data, val_data, epochs=25, batch_size=256, lr=1e-3, device='cuda'):
    """Train a single TopK SAE."""
    model = model.to(device)
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    # Data loaders
    train_dataset = TensorDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for (batch,) in train_loader:
            reconstruction, sparse, _ = model(batch)
            loss = model.loss(batch, reconstruction)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch.size(0)

        epoch_loss /= len(train_data)

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch + 1}/{epochs}: Loss={epoch_loss:.6f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        # Validation metrics
        val_reconstruction, val_sparse, val_metrics = model(val_data)
        val_loss = model.loss(val_data, val_reconstruction).item()

        # Explained variance
        residual = val_data - val_reconstruction
        explained_variance = 1 - (residual.var() / val_data.var())

        # Cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(
            val_data,
            val_reconstruction,
            dim=-1
        ).mean()

        # Feature statistics (compute on validation set)
        feature_stats = model.compute_feature_stats(val_sparse)

        # Gather all sparse activations to compute global feature stats
        # For feature death rate, we need to check across ALL validation samples
        train_reconstruction, train_sparse, train_metrics = model(train_data)
        combined_sparse = torch.cat([train_sparse, val_sparse], dim=0)
        global_feature_stats = model.compute_feature_stats(combined_sparse)

    metrics = {
        'explained_variance': explained_variance.item(),
        'reconstruction_loss': val_loss,
        'cosine_similarity': cosine_sim.item(),
        'l0_mean': val_metrics['l0_mean'],
        'l0_std': val_metrics['l0_std'],
        'mean_activation': val_metrics['mean_activation'],
        'max_activation': val_metrics['max_activation'],
        'median_activation': val_metrics['median_activation'],
        'feature_death_rate': global_feature_stats['feature_death_rate'],
        'num_dead_features': global_feature_stats['num_dead_features'],
        'num_active_features': global_feature_stats['num_active_features'],
    }

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_idx', type=int, default=None,
                        help='Train only this config index (0-7) for parallel execution')
    parser.add_argument('--position', type=int, default=3,
                        help='Continuous thought position (default: 3)')
    parser.add_argument('--layer', type=int, default=8,
                        help='Layer to extract from (default: 8, middle layer)')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Training epochs (default: 25)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size (default: 256)')
    args = parser.parse_args()

    # Load data
    train_data, val_data = load_gpt2_data(args.position, args.layer)

    # 8 configs for GPT-2 (input_dim=768)
    configs = [
        # Very Sparse
        (768, 192, 20),   # d=192 (0.25x), K=20 (10.4%)
        (768, 192, 40),   # d=192 (0.25x), K=40 (20.8%)

        # Medium Sparse (sweet spot candidates)
        (768, 256, 30),   # d=256 (0.33x), K=30 (11.7%)
        (768, 256, 50),   # d=256 (0.33x), K=50 (19.5%) ⭐
        (768, 256, 75),   # d=256 (0.33x), K=75 (29.3%)

        # Larger Dictionary
        (768, 384, 75),   # d=384 (0.5x), K=75 (19.5%)
        (768, 512, 100),  # d=512 (0.67x), K=100 (19.5%)
        (768, 512, 150),  # d=512 (0.67x), K=150 (29.3%)
    ]

    if args.config_idx is not None:
        configs_to_train = [configs[args.config_idx]]
        print(f"Training config {args.config_idx} only (parallel mode)\n")
    else:
        configs_to_train = configs
        print(f"Training all 8 configs sequentially\n")

    # Create output directory
    output_dir = Path('src/experiments/gpt2_sae_training/results')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Results dictionary
    all_results = {}

    # Train configs
    for config in configs_to_train:
        input_dim, latent_dim, k = config

        print(f"{'=' * 80}")
        print(f"Training: GPT-2, latent_dim={latent_dim}, K={k}")
        print(f"  Input dim: {input_dim}")
        print(f"  Latent dim: {latent_dim} ({latent_dim/input_dim:.2f}x)")
        print(f"  K: {k} ({k/latent_dim*100:.1f}% sparsity)")
        print(f"{'=' * 80}")

        # Initialize model
        model = TopKAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            k=k
        )

        # Train
        start_time = time.time()
        metrics = train_sae(
            model,
            train_data,
            val_data,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        train_time = time.time() - start_time

        metrics['train_time_sec'] = train_time
        metrics['epochs'] = args.epochs

        # Save checkpoint
        checkpoint_path = output_dir / f'gpt2_pos{args.position}_layer{args.layer}_d{latent_dim}_k{k}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'input_dim': input_dim,
                'latent_dim': latent_dim,
                'k': k,
                'position': args.position,
                'layer': args.layer,
                'model': 'gpt2',
            },
            'metrics': metrics,
        }, checkpoint_path)

        # Store results
        config_key = f'd{latent_dim}_k{k}'
        all_results[config_key] = metrics

        # Print summary
        print(f"\n  Results:")
        print(f"    Explained Variance: {metrics['explained_variance']:.3f}")
        print(f"    Feature Death Rate: {metrics['feature_death_rate']:.3f}")
        print(f"    Mean Activation: {metrics['mean_activation']:.3f}")
        print(f"    Max Activation: {metrics['max_activation']:.3f}")
        print(f"    L0: {metrics['l0_mean']:.1f} ± {metrics['l0_std']:.1f}")
        print(f"    Training Time: {train_time:.1f}s")
        print()

    # Save results
    suffix = f"_config{args.config_idx}" if args.config_idx is not None else "_all"
    results_path = output_dir / f'gpt2_grid_metrics_pos{args.position}_layer{args.layer}{suffix}.json'
    with open(results_path, 'w') as f:
        json.dump({
            'metadata': {
                'model': 'gpt2',
                'position': args.position,
                'layer': args.layer,
                'n_train': len(train_data),
                'n_val': len(val_data),
                'input_dim': 768,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
            },
            'results': all_results,
        }, f, indent=2)

    print(f"{'=' * 80}")
    print(f"Training complete!")
    print(f"Results saved to: {results_path}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
