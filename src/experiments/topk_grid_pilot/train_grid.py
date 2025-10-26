"""
Train TopK SAE Grid: K={5,10,20,100} × latent_dim={512,1024,2048}

Usage:
    # Sequential (all 12 SAEs):
    python train_grid.py

    # Parallel (one latent_dim):
    python train_grid.py --latent_dim 512 &
    python train_grid.py --latent_dim 1024 &
    python train_grid.py --latent_dim 2048 &
    wait
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

from topk_sae import TopKAutoencoder


def load_position_layer_data(position=3, layer=14):
    """Load Position 3, Layer 14 activations from cache."""
    print(f"Loading Position {position}, Layer {layer} data...")

    # Load full dataset
    train_data = torch.load(
        'src/experiments/sae_cot_decoder/data/full_train_activations.pt',
        weights_only=False
    )
    val_data = torch.load(
        'src/experiments/sae_cot_decoder/data/full_val_activations.pt',
        weights_only=False
    )

    # Extract Position 3, Layer 14
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
    parser.add_argument('--latent_dim', type=int, default=None,
                        help='Train only this latent_dim (for parallel execution)')
    parser.add_argument('--position', type=int, default=3,
                        help='Continuous thought position (default: 3)')
    parser.add_argument('--layer', type=int, default=14,
                        help='Layer to extract from (default: 14)')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Training epochs (default: 25)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size (default: 256)')
    args = parser.parse_args()

    # Load data
    train_data, val_data = load_position_layer_data(args.position, args.layer)

    # Grid parameters
    k_values = [5, 10, 20, 100]
    if args.latent_dim is not None:
        latent_dims = [args.latent_dim]
        print(f"Training latent_dim={args.latent_dim} only (parallel mode)\n")
    else:
        latent_dims = [512, 1024, 2048]
        print(f"Training all latent_dims sequentially\n")

    # Create output directory
    output_dir = Path('src/experiments/topk_grid_pilot/results')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Results dictionary
    all_results = {}

    # Train grid
    for latent_dim in latent_dims:
        all_results[latent_dim] = {}

        for k in k_values:
            print(f"{'=' * 80}")
            print(f"Training: latent_dim={latent_dim}, K={k}")
            print(f"{'=' * 80}")

            # Initialize model
            model = TopKAutoencoder(
                input_dim=train_data.shape[1],
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
            checkpoint_path = output_dir / f'pos{args.position}_layer{args.layer}_d{latent_dim}_k{k}.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'input_dim': train_data.shape[1],
                    'latent_dim': latent_dim,
                    'k': k,
                    'position': args.position,
                    'layer': args.layer,
                },
                'metrics': metrics,
            }, checkpoint_path)

            # Store results
            all_results[latent_dim][k] = metrics

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
    results_path = output_dir / f'grid_metrics_pos{args.position}_layer{args.layer}_latent{args.latent_dim if args.latent_dim else "all"}.json'
    with open(results_path, 'w') as f:
        json.dump({
            'metadata': {
                'position': args.position,
                'layer': args.layer,
                'n_train': len(train_data),
                'n_val': len(val_data),
                'input_dim': train_data.shape[1],
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
