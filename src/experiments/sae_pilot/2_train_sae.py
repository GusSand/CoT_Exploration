"""
Story 1.2: Train SAE with 512 Features

This script trains a Sparse Autoencoder (SAE) on continuous thought activations
using the sae_lens library.

Configuration:
- Input dim: 2048 (Llama hidden size)
- Features: 512 (dictionary size)
- Sparsity: L1 coefficient to achieve ~1% activation (L0 ~ 5)
- Training data: 10,800 vectors (600 problems × 3 layers × 6 tokens)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb
from pathlib import Path
from datetime import datetime
import json


class SparseAutoencoder(nn.Module):
    """Simple sparse autoencoder for continuous thought interpretation."""

    def __init__(self, input_dim: int = 2048, n_features: int = 512):
        super().__init__()

        self.input_dim = input_dim
        self.n_features = n_features

        # Encoder: input_dim -> n_features
        self.encoder = nn.Linear(input_dim, n_features, bias=True)

        # Decoder: n_features -> input_dim
        self.decoder = nn.Linear(n_features, input_dim, bias=True)

        # Initialize decoder as transpose of encoder (tied weights help)
        self.decoder.weight.data = self.encoder.weight.data.t().clone()

    def encode(self, x):
        """Encode input to sparse features."""
        return torch.relu(self.encoder(x))  # ReLU for non-negative sparse codes

    def decode(self, z):
        """Decode sparse features back to input space."""
        return self.decoder(z)

    def forward(self, x):
        """Forward pass: encode then decode."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


def compute_sparsity_loss(features, l1_coefficient=0.01):
    """Compute L1 sparsity penalty on feature activations."""
    return l1_coefficient * torch.abs(features).sum(dim=1).mean()


def compute_reconstruction_loss(x, x_recon):
    """Compute MSE reconstruction loss."""
    return nn.functional.mse_loss(x_recon, x)


def compute_l0_sparsity(features, threshold=1e-3):
    """Compute L0 sparsity (number of active features per vector)."""
    return (features.abs() > threshold).float().sum(dim=1).mean().item()


def train_sae(
    activations_path: str,
    output_dir: str,
    n_features: int = 512,
    l1_coefficient: float = 0.01,
    batch_size: int = 256,
    n_epochs: int = 20,
    learning_rate: float = 3e-4,
    use_wandb: bool = True
):
    """
    Train sparse autoencoder on continuous thought activations.

    Args:
        activations_path: Path to activation tensor file
        output_dir: Directory to save SAE weights and results
        n_features: Number of features in SAE dictionary
        l1_coefficient: L1 sparsity penalty weight
        batch_size: Training batch size
        n_epochs: Number of training epochs
        learning_rate: Adam learning rate
        use_wandb: Enable WandB logging
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load activations
    print(f"\nLoading activations from {activations_path}...")
    data = torch.load(activations_path)
    activations = data['activations'].to(device)
    metadata = data['metadata']
    info = data['info']

    print(f"  Activations shape: {activations.shape}")
    print(f"  Problems: {info['num_problems']}")
    print(f"  Layers: {info['num_layers']} {info['layer_names']}")
    print(f"  Total vectors: {info['total_vectors']:,}")

    # Create dataloader
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"\n  Batches per epoch: {len(dataloader)}")

    # Initialize SAE
    print(f"\nInitializing SAE...")
    sae = SparseAutoencoder(input_dim=info['hidden_dim'], n_features=n_features)
    sae = sae.to(device)

    print(f"  Input dim: {sae.input_dim}")
    print(f"  Features: {sae.n_features}")
    print(f"  Expansion factor: {sae.n_features / sae.input_dim:.2f}x")
    print(f"  Parameters: {sum(p.numel() for p in sae.parameters()):,}")

    # Optimizer
    optimizer = optim.Adam(sae.parameters(), lr=learning_rate)

    # Initialize WandB
    if use_wandb:
        wandb.init(
            project="sae-pilot",
            name=f"sae_{n_features}feat_l1{l1_coefficient}",
            config={
                "n_features": n_features,
                "input_dim": info['hidden_dim'],
                "l1_coefficient": l1_coefficient,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "learning_rate": learning_rate,
                "n_problems": info['num_problems'],
                "n_vectors": info['total_vectors']
            }
        )

    # Training loop
    print(f"\nTraining SAE for {n_epochs} epochs...")
    print("="*80)

    for epoch in range(n_epochs):
        sae.train()
        epoch_recon_loss = 0.0
        epoch_sparsity_loss = 0.0
        epoch_total_loss = 0.0
        epoch_l0 = 0.0

        for batch_idx, (batch_x,) in enumerate(dataloader):
            batch_x = batch_x.to(device)

            # Forward pass
            x_recon, features = sae(batch_x)

            # Compute losses
            recon_loss = compute_reconstruction_loss(batch_x, x_recon)
            sparsity_loss = compute_sparsity_loss(features, l1_coefficient)
            total_loss = recon_loss + sparsity_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Track metrics
            epoch_recon_loss += recon_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()
            epoch_total_loss += total_loss.item()
            epoch_l0 += compute_l0_sparsity(features)

        # Epoch averages
        n_batches = len(dataloader)
        avg_recon = epoch_recon_loss / n_batches
        avg_sparsity = epoch_sparsity_loss / n_batches
        avg_total = epoch_total_loss / n_batches
        avg_l0 = epoch_l0 / n_batches

        # Print progress
        print(f"Epoch {epoch+1:3d}/{n_epochs} | "
              f"Loss: {avg_total:.4f} | "
              f"Recon: {avg_recon:.4f} | "
              f"Sparsity: {avg_sparsity:.4f} | "
              f"L0: {avg_l0:.2f}")

        # Log to WandB
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "loss/total": avg_total,
                "loss/reconstruction": avg_recon,
                "loss/sparsity": avg_sparsity,
                "sparsity/l0": avg_l0,
            })

    print("="*80)
    print("Training complete!")

    # Save SAE
    sae_save_path = output_path / "sae_weights.pt"
    print(f"\nSaving SAE to {sae_save_path}...")

    torch.save({
        'model_state_dict': sae.state_dict(),
        'config': {
            'input_dim': sae.input_dim,
            'n_features': sae.n_features,
            'l1_coefficient': l1_coefficient
        },
        'training_info': {
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'final_recon_loss': avg_recon,
            'final_l0_sparsity': avg_l0
        }
    }, sae_save_path)

    # Save results summary
    results = {
        'config': {
            'n_features': n_features,
            'input_dim': info['hidden_dim'],
            'l1_coefficient': l1_coefficient,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        },
        'data': {
            'n_problems': info['num_problems'],
            'n_vectors': info['total_vectors'],
            'layers': info['layer_names']
        },
        'final_metrics': {
            'reconstruction_loss': avg_recon,
            'sparsity_loss': avg_sparsity,
            'total_loss': avg_total,
            'l0_sparsity': avg_l0
        },
        'training_time': datetime.now().isoformat()
    }

    results_path = output_path / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_path}")

    if use_wandb:
        wandb.finish()

    print("\n✅ SAE training complete!")

    return sae, results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--activations_path', type=str,
                        default='src/experiments/sae_pilot/data/sae_training_activations.pt',
                        help='Path to activation tensor file')
    parser.add_argument('--output_dir', type=str,
                        default='src/experiments/sae_pilot/results',
                        help='Output directory for SAE weights')
    parser.add_argument('--n_features', type=int, default=512,
                        help='Number of SAE features')
    parser.add_argument('--l1_coefficient', type=float, default=0.01,
                        help='L1 sparsity coefficient')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--n_epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Adam learning rate')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable WandB logging')

    args = parser.parse_args()

    sae, results = train_sae(
        activations_path=args.activations_path,
        output_dir=args.output_dir,
        n_features=args.n_features,
        l1_coefficient=args.l1_coefficient,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        use_wandb=not args.no_wandb
    )


if __name__ == "__main__":
    main()
