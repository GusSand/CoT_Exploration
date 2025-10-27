"""
Train sweet spot config (d=512, K=150) on all layers and positions.

Trains 72 SAEs: 12 layers × 6 positions
For creating layer×position heatmaps.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Import TopK SAE
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "topk_grid_pilot"))
from topk_sae import TopKAutoencoder


def load_gpt2_data(position, layer):
    """Load specific position-layer data."""
    train_data = torch.load(
        'src/experiments/gpt2_sae_training/data/gpt2_full_train_activations.pt',
        weights_only=False
    )
    val_data = torch.load(
        'src/experiments/gpt2_sae_training/data/gpt2_full_val_activations.pt',
        weights_only=False
    )

    def extract(data, pos, lyr):
        activations = data['activations']
        positions = np.array(data['metadata']['positions'])
        layers = np.array(data['metadata']['layers'])
        mask = (positions == pos) & (layers == lyr)
        return activations[mask]

    train_acts = extract(train_data, position, layer)
    val_acts = extract(val_data, position, layer)

    return train_acts, val_acts


def train_sae(model, train_data, val_data, epochs=25, batch_size=256, lr=1e-3, device='cuda'):
    """Train a single TopK SAE."""
    model = model.to(device)
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    train_dataset = TensorDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)

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

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_reconstruction, val_sparse, val_metrics = model(val_data)
        val_loss = model.loss(val_data, val_reconstruction).item()

        residual = val_data - val_reconstruction
        explained_variance = 1 - (residual.var() / val_data.var())

        train_reconstruction, train_sparse, train_metrics = model(train_data)
        combined_sparse = torch.cat([train_sparse, val_sparse], dim=0)
        global_feature_stats = model.compute_feature_stats(combined_sparse)

    metrics = {
        'explained_variance': explained_variance.item(),
        'reconstruction_loss': val_loss,
        'feature_death_rate': global_feature_stats['feature_death_rate'],
        'l0_mean': val_metrics['l0_mean'],
    }

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    print("="*80)
    print("TRAINING SWEET SPOT (d=512, K=150) ON ALL LAYERS×POSITIONS")
    print("="*80)

    # Sweet spot config
    input_dim = 768
    latent_dim = 512
    k = 150

    # Output directory
    output_dir = Path('src/experiments/gpt2_sae_training/results/sweet_spot_all')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Results storage
    all_results = {}

    # Train on all 12 layers × 6 positions
    total = 12 * 6
    count = 0

    for layer in range(12):
        for position in range(6):
            count += 1
            print(f"\n[{count}/{total}] Training Layer {layer}, Position {position}...")

            # Load data
            train_data, val_data = load_gpt2_data(position, layer)

            # Initialize model
            model = TopKAutoencoder(input_dim=input_dim, latent_dim=latent_dim, k=k)

            # Train
            start = time.time()
            metrics = train_sae(model, train_data, val_data, epochs=args.epochs, batch_size=args.batch_size)
            duration = time.time() - start

            print(f"  EV: {metrics['explained_variance']:.3f}, Death: {metrics['feature_death_rate']:.3f}, Time: {duration:.1f}s")

            # Save checkpoint
            checkpoint_path = output_dir / f'gpt2_sweet_spot_pos{position}_layer{layer}.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'input_dim': input_dim,
                    'latent_dim': latent_dim,
                    'k': k,
                    'position': position,
                    'layer': layer,
                    'model': 'gpt2',
                },
                'metrics': metrics,
            }, checkpoint_path)

            # Store results
            if layer not in all_results:
                all_results[layer] = {}
            all_results[layer][position] = metrics

    # Save summary
    summary_path = output_dir / 'sweet_spot_metrics_all.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'config': {
                'input_dim': input_dim,
                'latent_dim': latent_dim,
                'k': k,
                'epochs': args.epochs,
            },
            'results': all_results,
        }, f, indent=2)

    print("\n" + "="*80)
    print("ALL TRAINING COMPLETE!")
    print("="*80)
    print(f"  Trained {count} SAEs")
    print(f"  Results: {summary_path}")
    print("="*80)


if __name__ == '__main__':
    main()
