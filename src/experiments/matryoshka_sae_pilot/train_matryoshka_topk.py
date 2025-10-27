"""
Train Matryoshka-TopK SAE on Position 3 continuous thought activations.

This hybrid approach combines:
1. Hierarchical representation (Matryoshka)
2. TopK activation (efficient sparsity)

Goal: Match TopK's superior metrics (87.8% EV, 100% utilization, 0% feature death)
while preserving hierarchical benefits.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json
import time

from matryoshka_topk_sae import MatryoshkaTopKSAE


def train_matryoshka_topk(
    data_path: str,
    output_dir: str,
    levels: list = [128, 256, 512],
    k_values: list = [25, 35, 40],
    level_weights: list = [0.3, 0.3, 0.4],
    batch_size: int = 4096,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    device: str = 'cuda'
):
    """Train Matryoshka-TopK SAE.

    Args:
        data_path: Path to position_3_activations.pt
        output_dir: Directory to save results
        levels: Feature counts per hierarchy level
        k_values: TopK values per level
        level_weights: Reconstruction loss weights
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        device: Device to train on
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Matryoshka-TopK SAE Training")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from {data_path}...")
    data = torch.load(data_path)
    activations = data['activations']

    # Train/val split (80/20)
    n_samples = activations.shape[0]
    n_train = int(n_samples * 0.8)

    train_data = activations[:n_train]
    val_data = activations[n_train:]

    print(f"  Total samples: {n_samples:,}")
    print(f"  Train: {n_train:,}")
    print(f"  Val: {len(val_data):,}")
    print(f"  Input dim: {activations.shape[1]}")

    # Create dataloaders
    train_dataset = TensorDataset(train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_dataset = TensorDataset(val_data)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Initialize model
    print("\nInitializing Matryoshka-TopK SAE...")
    model = MatryoshkaTopKSAE(
        input_dim=activations.shape[1],
        levels=levels,
        k_values=k_values,
        level_weights=level_weights
    ).to(device)

    config = model.get_config()
    print(f"  Levels: {config['levels']}")
    print(f"  K values: {config['k_values']}")
    print(f"  Total features: {config['total_features']}")
    print(f"  Active per sample: {config['total_active_per_sample']}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=learning_rate * 0.1
    )

    # Training loop
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    training_history = []
    best_val_loss = float('inf')

    start_time = time.time()

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_batches = 0

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()

            # Forward
            reconstructions, features = model(batch_x)
            loss, _ = model.loss(batch_x, reconstructions, features)

            # Backward
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        train_loss /= train_batches

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)

                reconstructions, features = model(batch_x)
                loss, _ = model.loss(batch_x, reconstructions, features)

                val_loss += loss.item()
                val_batches += 1

        val_loss /= val_batches

        # Learning rate step
        scheduler.step()

        # Log
        epoch_stats = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': scheduler.get_last_lr()[0]
        }
        training_history.append(epoch_stats)

        # Print every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train: {train_loss:.6f} | "
                  f"Val: {val_loss:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }

            torch.save(checkpoint, output_dir / 'pos3_hierarchical_topk.pt')

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {best_val_loss:.6f}")

    # Save training history
    with open(output_dir / 'training_metrics_topk.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    print(f"\nModel saved to: {output_dir / 'pos3_hierarchical_topk.pt'}")
    print(f"Metrics saved to: {output_dir / 'training_metrics_topk.json'}")

    return model, training_history


def validate_final_model(model, data_path: str, device: str = 'cuda'):
    """Run final validation metrics.

    Args:
        model: Trained model
        data_path: Path to position_3_activations.pt
        device: Device to run on

    Returns:
        Dictionary of validation metrics
    """
    print("\n" + "=" * 60)
    print("Final Validation Metrics")
    print("=" * 60)

    # Load data
    data = torch.load(data_path)
    activations = data['activations']

    # Use last 20% as validation
    n_samples = activations.shape[0]
    n_val = int(n_samples * 0.2)
    val_data = activations[-n_val:].to(device)

    # Compute metrics
    model.eval()
    with torch.no_grad():
        reconstructions, features = model(val_data)
        metrics = model.compute_metrics(val_data, reconstructions, features)

    # Print table
    print(f"\n{'Metric':<25} {'Level 1':<15} {'Level 2':<15} {'Level 3':<15}")
    print("-" * 70)

    # Explained Variance
    ev1 = metrics['level_1']['explained_variance']
    ev2 = metrics['level_2']['explained_variance']
    ev3 = metrics['level_3']['explained_variance']
    print(f"{'Explained Variance':<25} {ev1:>14.1%} {ev2:>14.1%} {ev3:>14.1%}")

    # L0 Norm (should equal K)
    l0_1 = metrics['level_1']['l0_norm']
    l0_2 = metrics['level_2']['l0_norm']
    l0_3 = metrics['level_3']['l0_norm']
    print(f"{'L0 Norm (Active)':<25} {l0_1:>14.1f} {l0_2:>14.1f} {l0_3:>14.1f}")

    # Expected K
    k1 = metrics['level_1']['expected_k']
    k2 = metrics['level_2']['expected_k']
    k3 = metrics['level_3']['expected_k']
    print(f"{'Expected K':<25} {k1:>14,} {k2:>14,} {k3:>14,}")

    # Feature Death Rate
    death1 = metrics['level_1']['feature_death_rate']
    death2 = metrics['level_2']['feature_death_rate']
    death3 = metrics['level_3']['feature_death_rate']
    print(f"{'Feature Death Rate':<25} {death1:>14.1%} {death2:>14.1%} {death3:>14.1%}")

    # Utilization
    util1 = metrics['level_1']['utilization_pct']
    util2 = metrics['level_2']['utilization_pct']
    util3 = metrics['level_3']['utilization_pct']
    print(f"{'Utilization':<25} {util1:>14.1f}% {util2:>14.1f}% {util3:>14.1f}%")

    print("\n" + "=" * 60)
    print("Target Comparison (Goal: Match TopK)")
    print("=" * 60)
    print(f"TopK Baseline:  87.8% EV, 100% utilization, 0% feature death")
    print(f"Matryoshka-TopK L3: {ev3:.1%} EV, {util3:.1f}% utilization, {death3:.1%} feature death")

    return metrics


if __name__ == "__main__":
    # Paths
    base_dir = Path(__file__).parent
    data_path = base_dir / "data/position_3_activations.pt"
    output_dir = base_dir / "models"

    # Training config
    config = {
        'levels': [128, 256, 512],
        'k_values': [25, 35, 40],
        'level_weights': [0.3, 0.3, 0.4],
        'batch_size': 4096,
        'num_epochs': 50,
        'learning_rate': 1e-3
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Train
    model, history = train_matryoshka_topk(
        data_path=str(data_path),
        output_dir=str(output_dir),
        **config
    )

    # Final validation
    metrics = validate_final_model(model, str(data_path))

    # Save validation metrics
    validation_output = {
        'config': config,
        'metrics': metrics
    }

    with open(output_dir / 'validation_metrics_topk.json', 'w') as f:
        json.dump(validation_output, f, indent=2)

    print(f"\nValidation metrics saved to: {output_dir / 'validation_metrics_topk.json'}")
    print("\n✓ Training complete!")
