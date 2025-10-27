"""
Train Matryoshka SAE on Position 3 continuous thought activations.

Trains hierarchical SAE with 3 levels [512, 1024, 2048] using weighted loss.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
from pathlib import Path
from tqdm import tqdm
import time

from matryoshka_sae import create_matryoshka_sae


def load_position3_data(data_path: str, train_split: float = 0.8):
    """Load Position 3 data and split into train/val.

    Args:
        data_path: Path to position_3_activations.pt
        train_split: Fraction for training (0.8 = 80%)

    Returns:
        Tuple of (train_data, val_data)
    """
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path)

    activations = data['activations']  # (95648, 2048)
    print(f"  Total samples: {activations.shape[0]:,}")
    print(f"  Hidden dim: {activations.shape[1]}")

    # Shuffle and split
    n_samples = activations.shape[0]
    n_train = int(n_samples * train_split)

    # Random shuffle
    indices = torch.randperm(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_data = activations[train_indices]
    val_data = activations[val_indices]

    print(f"  Train: {train_data.shape[0]:,} ({train_data.shape[0]/n_samples*100:.1f}%)")
    print(f"  Val: {val_data.shape[0]:,} ({val_data.shape[0]/n_samples*100:.1f}%)")

    return train_data, val_data


def train_matryoshka_sae(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    levels: list = [512, 1024, 2048],
    level_weights: list = [0.3, 0.3, 0.4],
    l1_coefficient: float = 0.0005,
    batch_size: int = 4096,
    n_epochs: int = 50,
    learning_rate: float = 1e-3,
    device: str = 'cuda',
    save_path: str = None
):
    """Train Matryoshka SAE.

    Args:
        train_data: Training activations
        val_data: Validation activations
        levels: Feature dimensions [512, 1024, 2048]
        level_weights: Reconstruction loss weights [0.3, 0.3, 0.4]
        l1_coefficient: Sparsity penalty
        batch_size: Batch size
        n_epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to train on
        save_path: Path to save best model

    Returns:
        Tuple of (model, training_history)
    """
    print("\n" + "=" * 60)
    print("Training Matryoshka SAE")
    print("=" * 60)

    # Create model
    model = create_matryoshka_sae(
        input_dim=2048,
        levels=levels,
        l1_coefficient=l1_coefficient,
        level_weights=level_weights,
        device=device
    )

    # Create dataloaders
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for now
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler (cosine annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': [],
        'val_metrics': []
    }

    best_val_loss = float('inf')
    best_epoch = 0

    # Training loop
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {n_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Optimizer: Adam")
    print(f"  Scheduler: CosineAnnealingLR")
    print(f"  Device: {device}")

    start_time = time.time()

    for epoch in range(n_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            x = batch[0].to(device)

            optimizer.zero_grad()

            # Forward pass
            reconstructions, features = model(x)

            # Compute loss
            loss_dict = model.loss(x, reconstructions, features)
            loss = loss_dict['total_loss']

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        train_loss /= train_batches

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_metrics_list = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)

                # Forward pass
                reconstructions, features = model(x)

                # Compute loss
                loss_dict = model.loss(x, reconstructions, features)
                loss = loss_dict['total_loss']

                val_loss += loss.item()
                val_batches += 1

                # Compute metrics for last batch
                if val_batches == len(val_loader):
                    metrics = model.compute_metrics(x, reconstructions, features)
                    val_metrics_list.append(metrics)

        val_loss /= val_batches

        # Average validation metrics
        if val_metrics_list:
            val_metrics = val_metrics_list[0]  # Use last batch metrics
        else:
            val_metrics = {}

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)

        # Learning rate step
        scheduler.step()

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"\nEpoch {epoch+1}/{n_epochs} ({elapsed/60:.1f}m)")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

            if val_metrics:
                print(f"  Validation Metrics:")
                for level, level_metrics in val_metrics.items():
                    ev = level_metrics['explained_variance']
                    dr = level_metrics['feature_death_rate']
                    l0 = level_metrics['l0_norm']
                    print(f"    {level}: EV={ev:.1%}, Death={dr:.1%}, L0={l0:.1f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics,
                    'config': {
                        'levels': levels,
                        'level_weights': level_weights,
                        'l1_coefficient': l1_coefficient,
                        'input_dim': 2048
                    }
                }, save_path)

    training_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total time: {training_time/60:.1f} minutes")
    print(f"Best epoch: {best_epoch + 1}")
    print(f"Best val loss: {best_val_loss:.6f}")

    if save_path:
        print(f"Model saved to: {save_path}")

    return model, history


def save_training_metrics(history: dict, save_path: str):
    """Save training history to JSON.

    Args:
        history: Training history dictionary
        save_path: Path to save JSON
    """
    # Convert metrics to serializable format
    serializable_history = {
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'val_metrics': []
    }

    for metrics in history['val_metrics']:
        serializable_metrics = {}
        for level, level_metrics in metrics.items():
            serializable_metrics[level] = level_metrics
        serializable_history['val_metrics'].append(serializable_metrics)

    with open(save_path, 'w') as f:
        json.dump(serializable_history, f, indent=2)

    print(f"Training metrics saved to: {save_path}")


if __name__ == "__main__":
    # Paths
    base_dir = Path(__file__).parent
    data_path = base_dir / "data/position_3_activations.pt"
    model_save_path = base_dir / "models/pos3_hierarchical.pt"
    metrics_save_path = base_dir / "results/training_metrics.json"

    # Create directories
    model_save_path.parent.mkdir(exist_ok=True)
    metrics_save_path.parent.mkdir(exist_ok=True)

    print("=" * 60)
    print("Matryoshka SAE Training - Position 3")
    print("=" * 60)

    # Load data
    train_data, val_data = load_position3_data(str(data_path))

    # Train model
    model, history = train_matryoshka_sae(
        train_data=train_data,
        val_data=val_data,
        levels=[512, 1024, 2048],
        level_weights=[0.3, 0.3, 0.4],
        l1_coefficient=0.0005,
        batch_size=4096,
        n_epochs=50,
        learning_rate=1e-3,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        save_path=str(model_save_path)
    )

    # Save metrics
    save_training_metrics(history, str(metrics_save_path))

    print("\n✓ Training complete!")
