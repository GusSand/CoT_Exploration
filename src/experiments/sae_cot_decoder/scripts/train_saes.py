"""
Train Position-Specific SAEs on Continuous Thought Activations.

Trains 6 independent SAEs (one per continuous thought position) on all 16 layers.

Usage:
    python train_saes.py --parallel  # Train positions 0-2, then 3-5 in parallel
    python train_saes.py             # Train all sequentially
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
import wandb

from sae_model import SparseAutoencoder


class PositionDataset(Dataset):
    """Dataset for one continuous thought position across all layers."""

    def __init__(self, hidden_states, metadata, position: int):
        """
        Args:
            hidden_states: All activation tensors (N, hidden_dim)
            metadata: Metadata dict with positions list
            position: Which position to extract (0-5)
        """
        # Filter samples for this position
        position_indices = [
            i for i, pos in enumerate(metadata['positions'])
            if pos == position
        ]

        self.hidden_states = hidden_states[position_indices]
        self.position = position

        # Also store metadata for this position
        self.problem_ids = [metadata['problem_ids'][i] for i in position_indices]
        self.layers = [metadata['layers'][i] for i in position_indices]
        self.cot_steps = [metadata['cot_steps'][i] for i in position_indices]

    def __len__(self):
        return len(self.hidden_states)

    def __getitem__(self, idx):
        return self.hidden_states[idx]


def create_position_datasets(data_path: str):
    """Load data and create datasets for each position.

    Returns:
        Dict mapping position → (train_dataset, test_dataset)
    """
    print(f"\nLoading enriched data from {data_path}...")
    data = torch.load(data_path, weights_only=False)

    print(f"✓ Loaded {len(data['hidden_states'])} samples")
    print(f"  Samples with CoT: {sum(data['metadata']['has_cot'])}")

    # Create datasets for each position
    datasets = {}
    for position in range(6):
        dataset = PositionDataset(
            data['hidden_states'],
            data['metadata'],
            position
        )
        datasets[position] = dataset
        print(f"  Position {position}: {len(dataset)} samples")

    return datasets


def train_sae(
    sae: SparseAutoencoder,
    train_loader: DataLoader,
    test_loader: DataLoader,
    position: int,
    epochs: int = 50,
    device: str = 'cuda',
    wandb_run=None,
    save_dir: Path = None
) -> Dict:
    """Train one SAE.

    Args:
        sae: SAE model
        train_loader: Training data loader
        test_loader: Test data loader
        position: Position ID (0-5)
        epochs: Number of training epochs
        device: Device to train on
        wandb_run: WandB run for logging
        save_dir: Directory to save checkpoints

    Returns:
        Training history dictionary
    """
    optimizer = optim.Adam(sae.parameters(), lr=1e-3, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        'train_loss': [],
        'test_loss': [],
        'explained_variance': [],
        'feature_death_rate': [],
        'l0_norm': []
    }

    best_test_loss = float('inf')
    print(f"\n{'='*70}")
    print(f"Training SAE for Position {position}")
    print(f"{'='*70}")

    for epoch in range(epochs):
        # Training
        sae.train()
        train_losses = []
        train_features = []

        for batch in train_loader:
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

        # Calculate training metrics
        train_loss = sum(train_losses) / len(train_losses)
        all_train_features = torch.cat(train_features, dim=0)
        train_stats = sae.get_feature_statistics(all_train_features)

        # Evaluation
        sae.eval()
        test_losses = []
        test_features = []
        test_inputs = []
        test_reconstructions = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)

                reconstruction, features = sae(batch)
                loss_dict = sae.loss(batch, reconstruction, features)

                test_losses.append(loss_dict['total_loss'].item())
                test_features.append(features)
                test_inputs.append(batch)
                test_reconstructions.append(reconstruction)

        # Calculate test metrics
        test_loss = sum(test_losses) / len(test_losses)
        all_test_features = torch.cat(test_features, dim=0)
        all_test_inputs = torch.cat(test_inputs, dim=0)
        all_test_reconstructions = torch.cat(test_reconstructions, dim=0)

        test_stats = sae.get_feature_statistics(all_test_features)
        explained_var = sae.compute_explained_variance(all_test_inputs, all_test_reconstructions)
        cosine_sim = sae.compute_cosine_similarity(all_test_inputs, all_test_reconstructions)

        # Update history
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['explained_variance'].append(explained_var)
        history['feature_death_rate'].append(test_stats['feature_death_rate'])
        history['l0_norm'].append(test_stats['l0_norm_mean'])

        # Learning rate step
        scheduler.step()

        # Logging
        if wandb_run:
            wandb_run.log({
                f'pos_{position}/train_loss': train_loss,
                f'pos_{position}/test_loss': test_loss,
                f'pos_{position}/explained_variance': explained_var,
                f'pos_{position}/cosine_similarity': cosine_sim,
                f'pos_{position}/feature_death_rate': test_stats['feature_death_rate'],
                f'pos_{position}/l0_norm': test_stats['l0_norm_mean'],
                f'pos_{position}/active_features': test_stats['active_features'],
                f'pos_{position}/lr': scheduler.get_last_lr()[0],
                'epoch': epoch
            })

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")
            print(f"  Explained Var: {explained_var:.4f} | Cosine Sim: {cosine_sim:.4f}")
            print(f"  Death Rate: {test_stats['feature_death_rate']:.4f} | L0: {test_stats['l0_norm_mean']:.1f}")

        # Save checkpoint
        if save_dir and (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': sae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': test_loss,
                'explained_variance': explained_var
            }
            checkpoint_path = save_dir / f'pos_{position}_epoch_{epoch+1}.pt'
            torch.save(checkpoint, checkpoint_path)

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            if save_dir:
                best_path = save_dir / f'pos_{position}_best.pt'
                torch.save(sae.state_dict(), best_path)

    print(f"\n✓ Training complete for Position {position}")
    print(f"  Best test loss: {best_test_loss:.6f}")
    print(f"  Final explained variance: {explained_var:.4f}")
    print(f"  Final feature death rate: {test_stats['feature_death_rate']:.4f}")

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parallel', action='store_true', help='Train in 2 parallel batches (0-2, then 3-5)')
    parser.add_argument('--positions', type=int, nargs='+', default=list(range(6)), help='Which positions to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4096, help='Batch size')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB logging')
    args = parser.parse_args()

    # Paths
    base_dir = Path("/home/paperspace/dev/CoT_Exploration")
    data_dir = base_dir / "src/experiments/sae_cot_decoder/results"
    model_dir = base_dir / "src/experiments/sae_cot_decoder/models"
    model_dir.mkdir(exist_ok=True)

    train_data_path = data_dir / "enriched_train_data_with_cot.pt"
    test_data_path = data_dir / "enriched_test_data_with_cot.pt"

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize WandB
    wandb_run = None
    if not args.no_wandb:
        wandb_run = wandb.init(
            project="codi-sae-decoder",
            name="position-specific-saes",
            config={
                'n_positions': len(args.positions),
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'n_features': 2048,
                'l1_coefficient': 0.0005,
                'parallel': args.parallel
            }
        )

    # Load datasets
    train_datasets = create_position_datasets(train_data_path)
    test_datasets = create_position_datasets(test_data_path)

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

        test_loader = DataLoader(
            test_datasets[position],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Train
        history = train_sae(
            sae,
            train_loader,
            test_loader,
            position,
            epochs=args.epochs,
            device=device,
            wandb_run=wandb_run,
            save_dir=model_dir
        )

        all_results[position] = history

        # Save final model
        final_path = model_dir / f'pos_{position}_final.pt'
        torch.save(sae.state_dict(), final_path)
        print(f"✓ Saved final model: {final_path}")

    # Save training results
    results_path = data_dir / 'sae_training_results.json'
    with open(results_path, 'w') as f:
        # Convert numpy types to python types for JSON
        json_results = {}
        for pos, hist in all_results.items():
            json_results[str(pos)] = {k: [float(v) for v in vals] for k, vals in hist.items()}
        json.dump(json_results, f, indent=2)

    print(f"\n✓ Saved training results: {results_path}")

    if wandb_run:
        wandb_run.finish()

    print(f"\n{'='*70}")
    print("ALL SAE TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Models saved to: {model_dir}")
    print(f"\nNext step: Run interpretability analysis")


if __name__ == '__main__':
    main()
