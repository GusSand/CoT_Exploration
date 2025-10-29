"""
Train TopK SAEs with larger K values to reduce feature death.

Based on feature hierarchy findings, we train K=200 and K=300 with d=512
to test if larger K produces more usable specialized pattern features.

Hypothesis: Larger K → more active features per sample → less feature death
           → more pattern features at usable activation frequencies (>1%)

Usage:
    python train_large_k.py --k 200
    python train_large_k.py --k 300
    python train_large_k.py --k 200 --k 300  # Train both
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add topk_grid_pilot to path
sys.path.insert(0, 'src/experiments/topk_grid_pilot')
from topk_sae import TopKAutoencoder


def load_data(position=3, layer=14):
    """
    Load training and validation data for specified layer/position.

    Returns:
        train_acts: Training activations (N_train, 2048)
        val_acts: Validation activations (N_val, 2048)
    """
    print(f"\n{'='*80}")
    print(f"Loading Data: Layer {layer}, Position {position}")
    print(f"{'='*80}\n")

    # Load full datasets
    print("Loading full train activations...")
    train_data = torch.load(
        'src/experiments/sae_cot_decoder/data/full_train_activations.pt',
        weights_only=False
    )

    print("Loading full val activations...")
    val_data = torch.load(
        'src/experiments/sae_cot_decoder/data/full_val_activations.pt',
        weights_only=False
    )

    # Extract specified position and layer
    def extract(data, pos, lyr):
        activations = data['activations']
        positions = np.array(data['metadata']['positions'])
        layers = np.array(data['metadata']['layers'])
        mask = (positions == pos) & (layers == lyr)
        return activations[mask]

    train_acts = extract(train_data, position, layer)
    val_acts = extract(val_data, position, layer)

    print(f"✓ Train: {train_acts.shape}")
    print(f"✓ Val: {val_acts.shape}")
    print()

    return train_acts, val_acts


def train_sae(model, train_data, val_data, epochs=25, batch_size=256, lr=1e-3,
              device='cuda', verbose=True):
    """
    Train a single TopK SAE.

    Args:
        model: TopKAutoencoder instance
        train_data: Training activations
        val_data: Validation activations
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: 'cuda' or 'cpu'
        verbose: Print training progress

    Returns:
        model: Trained model
        history: Training history dict
    """
    model = model.to(device)
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    # Data loader
    train_dataset = TensorDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_explained_variance': [],
        'val_feature_death_rate': []
    }

    # Training loop
    model.train()

    if verbose:
        print(f"\n{'='*80}")
        print(f"Training: K={model.k}, latent_dim={model.latent_dim}")
        print(f"{'='*80}\n")

    for epoch in range(epochs):
        # Train
        epoch_loss = 0
        num_batches = 0

        for (batch,) in train_loader:
            reconstruction, sparse, _ = model(batch)
            loss = nn.MSELoss()(reconstruction, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches
        history['train_loss'].append(avg_train_loss)

        # Validate
        model.eval()
        with torch.no_grad():
            reconstruction, sparse, _ = model(val_data)
            val_loss = nn.MSELoss()(reconstruction, val_data).item()

            # Compute explained variance
            var_original = torch.var(val_data)
            var_residual = torch.var(val_data - reconstruction)
            explained_variance = 1.0 - (var_residual / var_original)

            # Compute feature death rate
            feature_activations = (sparse != 0).float().sum(dim=0)
            feature_death_rate = (feature_activations == 0).float().mean()

            history['val_loss'].append(val_loss)
            history['val_explained_variance'].append(float(explained_variance))
            history['val_feature_death_rate'].append(float(feature_death_rate))

        model.train()

        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}/{epochs}: "
                  f"Train Loss={avg_train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, "
                  f"EV={explained_variance:.4f}, "
                  f"Death={feature_death_rate:.4f}")

    if verbose:
        print()

    return model, history


def evaluate_sae(model, val_data, device='cuda'):
    """
    Evaluate trained SAE on validation set.

    Returns:
        metrics: Dict with evaluation metrics
    """
    model = model.to(device)
    model.eval()
    val_data = val_data.to(device)

    with torch.no_grad():
        reconstruction, sparse, _ = model(val_data)

        # Reconstruction metrics
        mse_loss = nn.MSELoss()(reconstruction, val_data)
        var_original = torch.var(val_data)
        var_residual = torch.var(val_data - reconstruction)
        explained_variance = 1.0 - (var_residual / var_original)

        # Sparsity metrics
        l0_norm = (sparse != 0).float().sum(dim=-1).mean()
        feature_activations = (sparse != 0).float().sum(dim=0)
        feature_death_rate = (feature_activations == 0).float().mean()

        # Activation statistics
        active_mask = sparse != 0
        mean_activation = sparse[active_mask].abs().mean() if active_mask.any() else 0.0

        metrics = {
            'reconstruction_loss': float(mse_loss),
            'explained_variance': float(explained_variance),
            'l0_norm': float(l0_norm),
            'feature_death_rate': float(feature_death_rate),
            'mean_activation': float(mean_activation),
            'num_active_features': int((feature_activations > 0).sum()),
            'num_dead_features': int((feature_activations == 0).sum())
        }

    return metrics


def save_checkpoint(model, metrics, history, output_dir, position, layer):
    """Save model checkpoint and metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint filename
    k = model.k
    latent_dim = model.latent_dim
    filename = f'pos{position}_layer{layer}_d{latent_dim}_k{k}.pt'
    ckpt_path = output_dir / filename

    # Save checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': model.input_dim,
            'latent_dim': model.latent_dim,
            'k': model.k,
            'position': position,
            'layer': layer
        },
        'metrics': metrics,
        'training_history': history
    }

    torch.save(checkpoint, ckpt_path)

    return ckpt_path


def train_single_config(k, latent_dim, position, layer, output_dir,
                       train_data, val_data, device='cuda'):
    """Train a single SAE configuration."""
    start_time = time.time()

    print(f"\n{'='*80}")
    print(f"Training Configuration: K={k}, d={latent_dim}")
    print(f"{'='*80}")

    # Initialize model
    input_dim = train_data.shape[1]
    model = TopKAutoencoder(input_dim=input_dim, latent_dim=latent_dim, k=k)

    # Train
    model, history = train_sae(
        model, train_data, val_data,
        epochs=25, batch_size=256, lr=1e-3,
        device=device, verbose=True
    )

    # Evaluate
    print("Final evaluation...")
    metrics = evaluate_sae(model, val_data, device=device)

    # Print results
    print(f"\n{'='*80}")
    print(f"Results: K={k}, d={latent_dim}")
    print(f"{'='*80}")
    print(f"  Explained Variance: {metrics['explained_variance']:.4f}")
    print(f"  Feature Death Rate: {metrics['feature_death_rate']:.4f}")
    print(f"  Active Features: {metrics['num_active_features']}/{latent_dim}")
    print(f"  Reconstruction Loss: {metrics['reconstruction_loss']:.6f}")
    print(f"  Mean L0: {metrics['l0_norm']:.1f}")

    # Save checkpoint
    ckpt_path = save_checkpoint(model, metrics, history, output_dir, position, layer)
    print(f"  Saved: {ckpt_path}")

    elapsed = time.time() - start_time
    print(f"  Time: {elapsed:.1f}s")
    print()

    return metrics, history


def main():
    parser = argparse.ArgumentParser(description='Train TopK SAEs with large K')
    parser.add_argument('--k', type=int, nargs='+', default=[200, 300],
                       help='K values to train (default: 200 300)')
    parser.add_argument('--latent_dim', type=int, default=512,
                       help='Dictionary size (default: 512)')
    parser.add_argument('--position', type=int, default=3,
                       help='Token position (default: 3)')
    parser.add_argument('--layer', type=int, default=14,
                       help='Layer index (default: 14)')
    parser.add_argument('--output_dir', type=str,
                       default='src/experiments/llama_sae_hierarchy/checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    print(f"\n{'='*80}")
    print(f"Large K Experiment: Test if K=200,300 produces more usable pattern features")
    print(f"{'='*80}")
    print(f"\nConfiguration:")
    print(f"  K values: {args.k}")
    print(f"  latent_dim: {args.latent_dim}")
    print(f"  Position: {args.position}")
    print(f"  Layer: {args.layer}")
    print(f"  Device: {args.device}")
    print(f"  Output: {args.output_dir}")

    # Load data once
    train_data, val_data = load_data(args.position, args.layer)

    # Train each K value
    all_results = {}

    for k in args.k:
        metrics, history = train_single_config(
            k=k,
            latent_dim=args.latent_dim,
            position=args.position,
            layer=args.layer,
            output_dir=args.output_dir,
            train_data=train_data,
            val_data=val_data,
            device=args.device
        )

        all_results[k] = {
            'metrics': metrics,
            'history': history
        }

    # Save summary
    summary_path = Path(args.output_dir) / f'large_k_summary_layer{args.layer}_pos{args.position}.json'
    summary = {
        'config': {
            'k_values': args.k,
            'latent_dim': args.latent_dim,
            'position': args.position,
            'layer': args.layer,
            'input_dim': int(train_data.shape[1]),
            'train_samples': int(train_data.shape[0]),
            'val_samples': int(val_data.shape[0])
        },
        'results': all_results
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"{'='*80}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*80}\n")

    # Comparison table
    print(f"{'='*80}")
    print(f"Comparison: K=100 (baseline) vs K=200,300")
    print(f"{'='*80}\n")

    # Load K=100 baseline
    baseline_path = 'src/experiments/topk_grid_pilot/results/checkpoints/pos3_layer14_d512_k100.pt'
    if Path(baseline_path).exists():
        baseline = torch.load(baseline_path, weights_only=False)
        baseline_metrics = baseline['metrics']

        print(f"{'K':<6} {'EV':<10} {'Death%':<10} {'Active':<12} {'Loss':<12}")
        print(f"{'-'*60}")
        print(f"100    {baseline_metrics['explained_variance']:<10.4f} "
              f"{baseline_metrics['feature_death_rate']*100:<10.1f} "
              f"{512-int(baseline_metrics['feature_death_rate']*512):<12} "
              f"{baseline_metrics['reconstruction_loss']:<12.6f}")

        for k in args.k:
            m = all_results[k]['metrics']
            print(f"{k:<6} {m['explained_variance']:<10.4f} "
                  f"{m['feature_death_rate']*100:<10.1f} "
                  f"{m['num_active_features']:<12} "
                  f"{m['reconstruction_loss']:<12.6f}")

        print(f"{'-'*60}\n")

        # Analysis
        print("Expected outcomes:")
        print("  ✓ K=200: Lower death rate, more active features")
        print("  ✓ K=300: Even lower death rate, most active features")
        print("  ✓ Higher K → More features at usable frequencies (>1%)")
        print("\nNext step: Run feature analysis to find pattern features\n")
    else:
        print(f"Baseline K=100 not found at {baseline_path}")
        print("Run comparison after baseline training.\n")


if __name__ == '__main__':
    main()
