"""
Training Script for Tuned Lens (Story 3).

This script trains Tuned Lens transformations on pre-extracted CODI activations.
Unlike the standard tuned-lens training loop which runs forward passes through the model,
we train directly on cached activations for efficiency.

Key Features:
- Loads pre-extracted activations (76,800 training samples)
- Trains layer-specific affine transformations
- Uses cross-entropy loss against target tokens
- W&B logging for metrics tracking
- Early stopping based on validation loss
- Saves best model checkpoint

Usage:
    python train.py --config config.yaml
    python train.py --num-epochs 100 --batch-size 64  # Override config
"""

import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Tuple
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_config, setup_logging, set_seed, get_device,
    clear_gpu_cache, format_time
)
from model import create_codi_tuned_lens


class ActivationDataset(Dataset):
    """Dataset for pre-extracted continuous thought activations."""

    def __init__(self, data_path: str):
        """Load pre-extracted activation data.

        Args:
            data_path: Path to .pt file with activations
        """
        print(f"Loading data from: {data_path}")
        data = torch.load(data_path)

        self.hidden_states = data['hidden_states']  # (N, hidden_size)
        self.target_token_ids = data['target_token_ids']  # (N,)
        self.metadata = data['metadata']
        self.config = data['config']

        print(f"Loaded {len(self.hidden_states)} samples")
        print(f"Hidden states shape: {self.hidden_states.shape}")
        print(f"Target tokens shape: {self.target_token_ids.shape}")

    def __len__(self):
        return len(self.hidden_states)

    def __getitem__(self, idx):
        return {
            'hidden_states': self.hidden_states[idx],
            'target_token_ids': self.target_token_ids[idx],
            'layer': self.metadata['layers'][idx],
            'position': self.metadata['positions'][idx],
            'difficulty': self.metadata['difficulties'][idx]
        }


def compute_metrics(logits: torch.Tensor, target_ids: torch.Tensor) -> Dict[str, float]:
    """Compute evaluation metrics.

    Args:
        logits: Model predictions (batch_size, vocab_size)
        target_ids: Ground truth token IDs (batch_size,)

    Returns:
        Dictionary of metrics
    """
    with torch.no_grad():
        # Cross-entropy loss
        loss = F.cross_entropy(logits, target_ids)

        # Top-1 accuracy
        pred_ids = logits.argmax(dim=-1)
        top1_acc = (pred_ids == target_ids).float().mean()

        # Top-5 accuracy
        top5_preds = logits.topk(k=5, dim=-1).indices
        top5_acc = (top5_preds == target_ids.unsqueeze(-1)).any(dim=-1).float().mean()

        return {
            'loss': loss.item(),
            'top1_accuracy': top1_acc.item(),
            'top5_accuracy': top5_acc.item()
        }


def train_epoch(
    model_wrapper,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    logger,
    wandb_enabled: bool = False
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model_wrapper: CODITunedLensWrapper instance
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to use
        logger: Logger instance
        wandb_enabled: Whether to log to W&B

    Returns:
        Dictionary of average training metrics
    """
    model_wrapper.tuned_lens.train()

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch in pbar:
        # Move data to device
        hidden_states = batch['hidden_states'].to(device)
        target_ids = batch['target_token_ids'].to(device)
        layers = batch['layer']

        # Forward pass through tuned lens
        # Group by layer for efficient processing
        layer_losses = []
        for layer_idx in range(model_wrapper.num_layers):
            # Get samples from this layer
            layer_mask = (layers == layer_idx)
            if not layer_mask.any():
                continue

            layer_hidden = hidden_states[layer_mask]
            layer_targets = target_ids[layer_mask]

            # Apply tuned lens transformation
            logits = model_wrapper.tuned_lens(layer_hidden, layer_idx)

            # Compute loss
            loss = F.cross_entropy(logits, layer_targets)
            layer_losses.append(loss)

            # Compute metrics for logging
            with torch.no_grad():
                pred_ids = logits.argmax(dim=-1)
                top1_acc = (pred_ids == layer_targets).float().mean()
                top5_preds = logits.topk(k=5, dim=-1).indices
                top5_acc = (top5_preds == layer_targets.unsqueeze(-1)).any(dim=-1).float().mean()

                total_top1 += top1_acc.item()
                total_top5 += top5_acc.item()

        # Average loss across layers
        if layer_losses:
            batch_loss = torch.stack(layer_losses).mean()

            # Backward pass
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model_wrapper.get_trainable_parameters(), 1.0)
            optimizer.step()

            total_loss += batch_loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': batch_loss.item()})

    # Compute average metrics
    avg_metrics = {
        'train_loss': total_loss / num_batches if num_batches > 0 else 0.0,
        'train_top1_accuracy': total_top1 / (num_batches * model_wrapper.num_layers) if num_batches > 0 else 0.0,
        'train_top5_accuracy': total_top5 / (num_batches * model_wrapper.num_layers) if num_batches > 0 else 0.0
    }

    return avg_metrics


def evaluate(
    model_wrapper,
    eval_loader: DataLoader,
    device: torch.device,
    logger
) -> Dict[str, float]:
    """Evaluate model on validation set.

    Args:
        model_wrapper: CODITunedLensWrapper instance
        eval_loader: Evaluation data loader
        device: Device to use
        logger: Logger instance

    Returns:
        Dictionary of evaluation metrics
    """
    model_wrapper.tuned_lens.eval()

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(eval_loader, desc="Evaluating", leave=False)
        for batch in pbar:
            hidden_states = batch['hidden_states'].to(device)
            target_ids = batch['target_token_ids'].to(device)
            layers = batch['layer']

            # Process each layer
            for layer_idx in range(model_wrapper.num_layers):
                layer_mask = (layers == layer_idx)
                if not layer_mask.any():
                    continue

                layer_hidden = hidden_states[layer_mask]
                layer_targets = target_ids[layer_mask]

                # Apply tuned lens
                logits = model_wrapper.tuned_lens(layer_hidden, layer_idx)

                # Compute metrics
                metrics = compute_metrics(logits, layer_targets)
                total_loss += metrics['loss']
                total_top1 += metrics['top1_accuracy']
                total_top5 += metrics['top5_accuracy']
                num_batches += 1

    # Compute averages
    avg_metrics = {
        'val_loss': total_loss / num_batches if num_batches > 0 else 0.0,
        'val_top1_accuracy': total_top1 / num_batches if num_batches > 0 else 0.0,
        'val_top5_accuracy': total_top5 / num_batches if num_batches > 0 else 0.0
    }

    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Tuned Lens for CODI")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--num-epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode (small dataset)')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override with command line arguments
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.debug:
        config['debug']['enabled'] = True

    # Setup logging
    logger = setup_logging(config)
    logger.info("="*70)
    logger.info("TUNED LENS TRAINING (Story 3)")
    logger.info("="*70)

    # Set random seed
    set_seed(config['data']['random_seed'])
    logger.info(f"Random seed: {config['data']['random_seed']}")

    # Get device
    device = get_device(config)
    logger.info(f"Device: {device}")

    # Load datasets
    logger.info("\nLoading pre-extracted activations...")
    train_data_path = config['output']['train_data_file'].format(
        model=config['model']['name'],
        representation=config['data']['representation']
    )
    test_data_path = config['output']['test_data_file'].format(
        model=config['model']['name'],
        representation=config['data']['representation']
    )

    train_dataset = ActivationDataset(train_data_path)
    test_dataset = ActivationDataset(test_data_path)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['compute']['num_workers'],
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['compute']['num_workers'],
        pin_memory=True
    )

    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    # Create model wrapper
    logger.info("\nInitializing Tuned Lens...")
    model_wrapper = create_codi_tuned_lens(config, device=str(device))
    logger.info(f"Trainable parameters: {model_wrapper.num_parameters():,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model_wrapper.get_trainable_parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Learning rate scheduler
    if config['training'].get('use_lr_schedule', False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs']
        )
    else:
        scheduler = None

    # Initialize W&B
    if config['wandb']['enabled']:
        import wandb
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            name=config['wandb'].get('run_name'),
            config=config,
            tags=config['wandb']['tags']
        )

    # Training loop
    logger.info("\n" + "="*70)
    logger.info("STARTING TRAINING")
    logger.info("="*70)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = config['output']['best_model_file'].format(
        model=config['model']['name'],
        representation=config['data']['representation']
    )

    import time
    start_time = time.time()

    for epoch in range(config['training']['num_epochs']):
        logger.info(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")

        # Train
        train_metrics = train_epoch(
            model_wrapper, train_loader, optimizer, device, logger,
            wandb_enabled=config['wandb']['enabled']
        )

        # Evaluate
        val_metrics = evaluate(model_wrapper, test_loader, device, logger)

        # Log metrics
        logger.info(f"Train Loss: {train_metrics['train_loss']:.4f} | "
                   f"Train Top-1: {train_metrics['train_top1_accuracy']:.2%} | "
                   f"Train Top-5: {train_metrics['train_top5_accuracy']:.2%}")
        logger.info(f"Val Loss: {val_metrics['val_loss']:.4f} | "
                   f"Val Top-1: {val_metrics['val_top1_accuracy']:.2%} | "
                   f"Val Top-5: {val_metrics['val_top5_accuracy']:.2%}")

        # W&B logging
        if config['wandb']['enabled']:
            import wandb
            wandb.log({
                **train_metrics,
                **val_metrics,
                'epoch': epoch,
                'learning_rate': optimizer.param_groups[0]['lr']
            })

        # Learning rate scheduling
        if scheduler:
            scheduler.step()

        # Early stopping
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0

            # Save best model
            if config['training'].get('save_best_only', True):
                model_wrapper.save_tuned_lens(best_model_path)
                logger.info(f" New best model saved (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epochs")

            if patience_counter >= config['training']['early_stopping_patience']:
                logger.info(f"\n  Early stopping triggered after {epoch+1} epochs")
                break

        # Clear cache periodically
        if (epoch + 1) % config['compute'].get('clear_cache_every_n_batches', 10) == 0:
            clear_gpu_cache()

    # Training complete
    elapsed_time = time.time() - start_time
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info(f"Total time: {format_time(elapsed_time)}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best model saved to: {best_model_path}")

    # Final evaluation with best model
    logger.info("\n" + "="*70)
    logger.info("FINAL EVALUATION ON BEST MODEL")
    logger.info("="*70)

    model_wrapper.load_tuned_lens(best_model_path)
    final_metrics = evaluate(model_wrapper, test_loader, device, logger)

    logger.info(f"\nFinal Test Metrics:")
    logger.info(f"  Loss: {final_metrics['val_loss']:.4f}")
    logger.info(f"  Top-1 Accuracy: {final_metrics['val_top1_accuracy']:.2%}")
    logger.info(f"  Top-5 Accuracy: {final_metrics['val_top5_accuracy']:.2%}")

    if config['wandb']['enabled']:
        import wandb
        wandb.log({
            'final_test_loss': final_metrics['val_loss'],
            'final_test_top1_accuracy': final_metrics['val_top1_accuracy'],
            'final_test_top5_accuracy': final_metrics['val_top5_accuracy']
        })
        wandb.finish()

    logger.info("\n Training pipeline complete!")


if __name__ == '__main__':
    main()
