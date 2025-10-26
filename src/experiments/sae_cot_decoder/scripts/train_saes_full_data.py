"""
Train Position-Specific SAEs on FULL GSM8K Dataset Activations.

Trains 6 independent SAEs (one per continuous thought position) using
the full 7,473 GSM8K training problem dataset (vs original 800 problems).

Usage:
    python train_saes_full_data.py --epochs 50
    python train_saes_full_data.py --epochs 50 --positions 0  # Train only position 0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List

from sae_model import SparseAutoencoder


class PositionDataset(Dataset):
    """Dataset for one continuous thought position across all layers."""

    def __init__(self, activations, metadata, position: int):
        """
        Args:
            activations: All activation tensors (N, hidden_dim)
            metadata: Metadata dict with positions list
            position: Which position to extract (0-5)
        """
        # Filter samples for this position
        position_indices = [
            i for i, pos in enumerate(metadata['positions'])
            if pos == position
        ]

        self.activations = activations[position_indices]
        self.position = position

        # Also store metadata for this position
        self.problem_ids = [metadata['problem_ids'][i] for i in position_indices]
        self.layers = [metadata['layers'][i] for i in position_indices]
        self.cot_sequences = [metadata['cot_sequences'][i] for i in position_indices]

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return self.activations[idx]


def create_position_datasets(data_path: str):
    """Load data and create datasets for each position.

    Args:
        data_path: Path to full_train_activations.pt or full_val_activations.pt

    Returns:
        Dict mapping position → dataset
    """
    print(f"\nLoading full dataset from {data_path}...")
    data = torch.load(data_path, weights_only=False)

    print(f"✓ Loaded {len(data['activations'])} samples")
    print(f"  Problems: {data['config']['num_problems']}")
    print(f"  Layers: {data['config']['num_layers']}")
    print(f"  Positions: {data['config']['num_ct_tokens']}")

    # Create datasets for each position
    datasets = {}
    for position in range(6):
        dataset = PositionDataset(
            data['activations'],
            data['metadata'],
            position
        )
        datasets[position] = dataset
        print(f"  Position {position}: {len(dataset):,} samples")

    return datasets


def train_sae(
    sae: SparseAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    position: int,
    epochs: int = 50,
    device: str = 'cuda',
    save_dir: Path = None
) -> Dict:
    """Train one SAE.

    Args:
        sae: SAE model
        train_loader: Training data loader
        val_loader: Validation data loader
        position: Position ID (0-5)
        epochs: Number of training epochs
        device: Device to train on
        save_dir: Directory to save checkpoints

    Returns:
        Training history dictionary
    """
    optimizer = optim.Adam(sae.parameters(), lr=1e-3, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        'train_loss': [],
        'val_loss': [],
        'explained_variance': [],
        'feature_death_rate': [],
        'l0_norm': []
    }

    best_val_loss = float('inf')
    print(f"\n{'='*70}")
    print(f"Training SAE for Position {position}")
    print(f"{'='*70}")

    for epoch in range(epochs):
        # Training
        sae.train()
        train_losses = []
        train_features = []

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in progress:
            batch = batch.to(device)

            # Forward pass
            reconstruction, features = sae(batch)
            loss_dict = sae.loss(batch, reconstruction, features)

            # Backward pass
            optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            optimizer.step()

            train_losses.append(loss_dict['total_loss'].item())
            train_features.append(features.detach())

            progress.set_postfix({'loss': f"{loss_dict['total_loss'].item():.6f}"})

        # Calculate training metrics
        train_loss = sum(train_losses) / len(train_losses)
        all_train_features = torch.cat(train_features, dim=0)
        train_stats = sae.get_feature_statistics(all_train_features)

        # Evaluation
        sae.eval()
        val_losses = []
        val_features = []
        val_inputs = []
        val_reconstructions = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                reconstruction, features = sae(batch)
                loss_dict = sae.loss(batch, reconstruction, features)

                val_losses.append(loss_dict['total_loss'].item())
                val_features.append(features)
                val_inputs.append(batch)
                val_reconstructions.append(reconstruction)

        # Calculate validation metrics
        val_loss = sum(val_losses) / len(val_losses)
        all_val_features = torch.cat(val_features, dim=0)
        all_val_inputs = torch.cat(val_inputs, dim=0)
        all_val_reconstructions = torch.cat(val_reconstructions, dim=0)

        val_stats = sae.get_feature_statistics(all_val_features)
        explained_var = sae.compute_explained_variance(all_val_inputs, all_val_reconstructions)
        cosine_sim = sae.compute_cosine_similarity(all_val_inputs, all_val_reconstructions)

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['explained_variance'].append(explained_var)
        history['feature_death_rate'].append(val_stats['feature_death_rate'])
        history['l0_norm'].append(val_stats['l0_norm_mean'])

        # Learning rate step
        scheduler.step()

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            print(f"  Explained Var: {explained_var:.4f} | Cosine Sim: {cosine_sim:.4f}")
            print(f"  Death Rate: {val_stats['feature_death_rate']:.4f} | L0: {val_stats['l0_norm_mean']:.1f}")

        # Save checkpoint
        if save_dir and (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': sae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'explained_variance': explained_var
            }
            checkpoint_path = save_dir / f'pos_{position}_epoch_{epoch+1}.pt'
            torch.save(checkpoint, checkpoint_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_dir:
                best_path = save_dir / f'pos_{position}_best.pt'
                torch.save(sae.state_dict(), best_path)

    print(f"\n✓ Training complete for Position {position}")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print(f"  Final explained variance: {explained_var:.4f}")
    print(f"  Final feature death rate: {val_stats['feature_death_rate']:.4f}")

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--positions', type=int, nargs='+', default=list(range(6)), help='Which positions to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4096, help='Batch size')
    args = parser.parse_args()

    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models_full_dataset"
    MODEL_DIR.mkdir(exist_ok=True)

    train_data_path = DATA_DIR / "full_train_activations.pt"
    val_data_path = DATA_DIR / "full_val_activations.pt"

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load datasets
    train_datasets = create_position_datasets(train_data_path)
    val_datasets = create_position_datasets(val_data_path)

    # Training results
    all_results = {}

    # Train SAEs
    for position in args.positions:
        print(f"\n{'='*70}")
        print(f"POSITION {position}")
        print(f"{'='*70}")

        # Create SAE
        sae = SparseAutoencoder(
            input_dim=2048,
            n_features=2048,
            l1_coefficient=0.0005
        ).to(device)

        print(f"SAE parameters: {sae.num_parameters():,}")

        # Create data loaders
        train_loader = DataLoader(
            train_datasets[position],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_datasets[position],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")

        # Train
        history = train_sae(
            sae,
            train_loader,
            val_loader,
            position,
            epochs=args.epochs,
            device=device,
            save_dir=MODEL_DIR
        )

        all_results[position] = history

        # Save final model
        final_path = MODEL_DIR / f'pos_{position}_final.pt'
        torch.save(sae.state_dict(), final_path)
        print(f"✓ Saved final model: {final_path}")

    # Save training results
    results_path = BASE_DIR / 'analysis' / 'sae_training_results_full_data.json'
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        # Convert to JSON-serializable format
        json_results = {}
        for pos, hist in all_results.items():
            json_results[str(pos)] = {k: [float(v) for v in vals] for k, vals in hist.items()}
        json.dump(json_results, f, indent=2)

    print(f"\n✓ Saved training results: {results_path}")

    print(f"\n{'='*70}")
    print("ALL SAE TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Models saved to: {MODEL_DIR}")
    print(f"\nNext step: Compare with baseline (800 problems) and run feature analysis")
    print(f"  python compare_old_vs_new_saes.py")


if __name__ == '__main__':
    main()
