"""
Train SAE on L12-L16 activations for error prediction improvement.

Goal: Test if L12-L16 features can achieve 70-75% error prediction accuracy
(vs 66.67% with L14 only).

Configuration (same as refined SAE):
- Input dim: 2048 (Llama hidden size)
- Features: 2048 (1x expansion)
- L1 coefficient: 0.0005 (weak sparsity for better discriminability)
- Training data: 27,420 vectors (914 solutions × 5 layers × 6 tokens)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm


class SparseAutoencoder(nn.Module):
    """Simple sparse autoencoder for continuous thought interpretation."""

    def __init__(self, input_dim: int = 2048, n_features: int = 2048):
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


def compute_sparsity_loss(features, l1_coefficient=0.0005):
    """Compute L1 sparsity penalty on feature activations."""
    return l1_coefficient * torch.abs(features).sum(dim=1).mean()


def compute_reconstruction_loss(x, x_recon):
    """Compute MSE reconstruction loss."""
    return nn.functional.mse_loss(x_recon, x)


def compute_l0_sparsity(features, threshold=1e-3):
    """Compute L0 sparsity (number of active features per vector)."""
    return (features.abs() > threshold).float().sum(dim=1).mean().item()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='src/experiments/sae_error_analysis/data/error_analysis_dataset_l12_l16.json',
                        help='Path to L12-L16 dataset')
    parser.add_argument('--output_dir', type=str,
                        default='src/experiments/sae_error_analysis/sae_l12_l16',
                        help='Output directory for SAE weights')
    parser.add_argument('--n_features', type=int, default=2048,
                        help='Number of SAE features')
    parser.add_argument('--l1_coefficient', type=float, default=0.0005,
                        help='L1 sparsity coefficient')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--n_epochs', type=int, default=25,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Adam learning rate')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("SAE TRAINING - L12-L16 ERROR ANALYSIS")
    print("="*80)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {output_path}")

    # Load dataset
    print("\n" + "="*80)
    print("LOADING L12-L16 DATASET")
    print("="*80)
    with open(args.dataset, 'r') as f:
        dataset = json.load(f)

    print(f"Metadata:")
    print(f"  Total solutions: {dataset['metadata']['total']}")
    print(f"  Correct: {dataset['metadata']['n_correct']}")
    print(f"  Incorrect: {dataset['metadata']['n_incorrect']}")
    print(f"  Layers: {dataset['metadata']['layers']}")
    print(f"  Layer indices: {dataset['metadata']['layer_indices']}")

    # Flatten all activations into single tensor
    print("\n" + "="*80)
    print("PREPARING ACTIVATIONS")
    print("="*80)

    all_activations = []

    # Process incorrect solutions
    for sol in tqdm(dataset['incorrect_solutions'], desc="Incorrect"):
        thoughts = sol['continuous_thoughts']
        for layer_name in dataset['metadata']['layers']:
            for token_vector in thoughts[layer_name]:
                all_activations.append(token_vector)

    # Process correct solutions
    for sol in tqdm(dataset['correct_solutions'], desc="Correct"):
        thoughts = sol['continuous_thoughts']
        for layer_name in dataset['metadata']['layers']:
            for token_vector in thoughts[layer_name]:
                all_activations.append(token_vector)

    # Convert to tensor
    activations = torch.tensor(all_activations, dtype=torch.float32).to(device)

    print(f"\n  Activations shape: {activations.shape}")
    print(f"  Expected: [27420, 2048] (914 solutions × 5 layers × 6 tokens)")
    print(f"  Memory: {activations.element_size() * activations.nelement() / 1e9:.2f} GB")

    # Create dataloader
    dataset_tensor = TensorDataset(activations)
    dataloader = DataLoader(dataset_tensor, batch_size=args.batch_size, shuffle=True)
    print(f"  Batches per epoch: {len(dataloader)}")

    # Initialize SAE
    print("\n" + "="*80)
    print("INITIALIZING SAE")
    print("="*80)
    sae = SparseAutoencoder(input_dim=2048, n_features=args.n_features).to(device)

    print(f"  Input dim: {sae.input_dim}")
    print(f"  Features: {sae.n_features}")
    print(f"  Expansion factor: {sae.n_features / sae.input_dim:.2f}x")
    print(f"  L1 coefficient: {args.l1_coefficient}")
    print(f"  Parameters: {sum(p.numel() for p in sae.parameters()):,}")

    # Optimizer
    optimizer = optim.Adam(sae.parameters(), lr=args.learning_rate)

    # Initialize WandB
    wandb.init(
        project="sae-l12-l16-error-analysis",
        name=f"sae_l12_l16_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "n_features": args.n_features,
            "input_dim": 2048,
            "l1_coefficient": args.l1_coefficient,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "learning_rate": args.learning_rate,
            "n_solutions": dataset['metadata']['total'],
            "n_layers": len(dataset['metadata']['layers']),
            "n_vectors": activations.shape[0]
        }
    )

    # Training loop
    print("\n" + "="*80)
    print(f"TRAINING SAE FOR {args.n_epochs} EPOCHS")
    print("="*80)

    for epoch in range(args.n_epochs):
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
            sparsity_loss = compute_sparsity_loss(features, args.l1_coefficient)
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
        print(f"Epoch {epoch+1:3d}/{args.n_epochs} | "
              f"Loss: {avg_total:.4f} | "
              f"Recon: {avg_recon:.4f} | "
              f"Sparsity: {avg_sparsity:.4f} | "
              f"L0: {avg_l0:.2f}")

        # Log to WandB
        wandb.log({
            "epoch": epoch + 1,
            "loss/total": avg_total,
            "loss/reconstruction": avg_recon,
            "loss/sparsity": avg_sparsity,
            "metrics/l0_sparsity": avg_l0
        })

    # Save SAE weights
    print("\n" + "="*80)
    print("SAVING SAE WEIGHTS")
    print("="*80)

    weights_path = output_path / 'sae_weights.pt'
    torch.save({
        'model_state_dict': sae.state_dict(),
        'config': {
            'input_dim': sae.input_dim,
            'n_features': sae.n_features,
            'l1_coefficient': args.l1_coefficient
        },
        'training_args': vars(args),
        'final_metrics': {
            'reconstruction_loss': avg_recon,
            'sparsity_loss': avg_sparsity,
            'total_loss': avg_total,
            'l0_sparsity': avg_l0
        }
    }, weights_path)

    print(f"  Saved: {weights_path}")

    # Validation on full dataset
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)

    sae.eval()
    with torch.no_grad():
        x_recon, features = sae(activations)

        # Reconstruction metrics
        mse = compute_reconstruction_loss(activations, x_recon).item()

        # Explained variance
        var_original = torch.var(activations).item()
        var_residual = torch.var(activations - x_recon).item()
        explained_var = 1 - (var_residual / var_original)

        # Sparsity metrics
        l0 = compute_l0_sparsity(features)
        dead_features = (features.abs().max(dim=0)[0] < 1e-6).sum().item()
        dead_pct = (dead_features / args.n_features) * 100

        print(f"  MSE: {mse:.4f}")
        print(f"  Explained variance: {explained_var*100:.2f}%")
        print(f"  L0 sparsity: {l0:.2f}")
        print(f"  Dead features: {dead_features}/{args.n_features} ({dead_pct:.1f}%)")

    # Save validation metrics
    validation_results = {
        'reconstruction_mse': float(mse),
        'explained_variance': float(explained_var),
        'l0_sparsity': float(l0),
        'dead_features': int(dead_features),
        'dead_features_pct': float(dead_pct),
        'active_features': int(args.n_features - dead_features)
    }

    results_path = output_path / 'validation_results.json'
    with open(results_path, 'w') as f:
        json.dump(validation_results, f, indent=2)

    print(f"\n  Saved validation results: {results_path}")

    # Log final metrics to wandb
    wandb.log({
        "final/reconstruction_mse": mse,
        "final/explained_variance": explained_var,
        "final/l0_sparsity": l0,
        "final/dead_features_pct": dead_pct
    })

    wandb.finish()

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Next steps:")
    print(f"  1. Encode error dataset with trained SAE")
    print(f"  2. Train error classifier on L12-L16 features")
    print(f"  3. Compare to L14-only baseline (66.67%)")
    print(f"  4. Target: 70-75% accuracy")


if __name__ == "__main__":
    main()
