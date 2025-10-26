"""
Train Tuned Lens for CoT Token Alignment (All Layers).

This script trains Tuned Lens transformations to decode continuous thought
positions into their corresponding CoT tokens (instead of arbitrary vocabulary tokens).

Key Differences from standard tuned-lens training:
- Targets are CoT tokens from the reasoning steps
- Each CT position gets assigned specific CoT tokens via uniform split
- We evaluate how well the lens can decode CT → CoT alignment
- Training on ALL 16 layers for better generalization

Usage:
    python train_cot_alignment.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
import numpy as np
import time

# Import our existing utilities
from utils import setup_logging, set_seed, get_device, clear_gpu_cache, format_time
from model import create_codi_tuned_lens


class CoTAlignmentDataset(Dataset):
    """Dataset for CoT token alignment training."""

    def __init__(self, data_path: str):
        """Load CoT alignment data.

        Args:
            data_path: Path to cot_train/test_all_layers.pt file
        """
        print(f"Loading CoT alignment data from: {data_path}")
        data = torch.load(data_path, weights_only=False)

        self.hidden_states = data['hidden_states']  # (N, 2048)
        self.cot_target_token_ids = data['cot_target_token_ids']  # List of lists
        self.metadata = data['metadata']
        self.config = data['config']

        # Extract first CoT token as primary target (for standard cross-entropy)
        self.primary_targets = torch.tensor([
            tokens[0] if len(tokens) > 0 else 0
            for tokens in self.cot_target_token_ids
        ])

        # Statistics
        layer_range = self.config['layer_range']
        num_layers = layer_range[1] - layer_range[0] + 1
        print(f"Loaded {len(self.hidden_states)} samples")
        print(f"Layers: {layer_range[0]}-{layer_range[1]} ({num_layers} layers)")
        print(f"CT positions: {len(set(self.metadata['positions']))}")
        print(f"Average CoT tokens per sample: {np.mean(self.metadata['num_cot_tokens']):.2f}")

    def __len__(self):
        return len(self.hidden_states)

    def __getitem__(self, idx):
        return {
            'hidden_states': self.hidden_states[idx],
            'primary_target': self.primary_targets[idx],
            'layer': self.metadata['layers'][idx],
            'position': self.metadata['positions'][idx],
        }


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Compute evaluation metrics.

    Args:
        logits: Model predictions (batch_size, vocab_size)
        targets: Ground truth token IDs (batch_size,)

    Returns:
        Dictionary of metrics
    """
    with torch.no_grad():
        # Cross-entropy loss
        loss = F.cross_entropy(logits, targets)

        # Top-1 accuracy (exact match)
        pred_ids = logits.argmax(dim=-1)
        top1_acc = (pred_ids == targets).float().mean()

        # Top-5 accuracy
        top5_preds = logits.topk(k=5, dim=-1).indices
        top5_acc = (top5_preds == targets.unsqueeze(-1)).any(dim=-1).float().mean()

        # Top-10 accuracy (for CoT tokens)
        top10_preds = logits.topk(k=10, dim=-1).indices
        top10_acc = (top10_preds == targets.unsqueeze(-1)).any(dim=-1).float().mean()

        return {
            'loss': loss.item(),
            'top1_accuracy': top1_acc.item(),
            'top5_accuracy': top5_acc.item(),
            'top10_accuracy': top10_acc.item(),
        }


def train_epoch(
    model_wrapper,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """Train for one epoch."""
    model_wrapper.tuned_lens.train()

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_top10 = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch in pbar:
        # Move data to device
        hidden_states = batch['hidden_states'].to(device)
        targets = batch['primary_target'].to(device)
        layers = batch['layer']

        # Process each layer in the batch
        layer_losses = []
        batch_top1 = []
        batch_top5 = []
        batch_top10 = []

        for layer_idx in range(model_wrapper.num_layers):
            # Get samples from this layer
            layer_mask = (layers == layer_idx)
            if not layer_mask.any():
                continue

            layer_hidden = hidden_states[layer_mask]
            layer_targets = targets[layer_mask]

            # Forward pass
            logits = model_wrapper.tuned_lens(layer_hidden, layer_idx)

            # Compute loss
            loss = F.cross_entropy(logits, layer_targets)
            layer_losses.append(loss)

            # Compute metrics
            with torch.no_grad():
                pred_ids = logits.argmax(dim=-1)
                top1_acc = (pred_ids == layer_targets).float().mean()
                top5_preds = logits.topk(k=5, dim=-1).indices
                top5_acc = (top5_preds == layer_targets.unsqueeze(-1)).any(dim=-1).float().mean()
                top10_preds = logits.topk(k=10, dim=-1).indices
                top10_acc = (top10_preds == layer_targets.unsqueeze(-1)).any(dim=-1).float().mean()

                batch_top1.append(top1_acc.item())
                batch_top5.append(top5_acc.item())
                batch_top10.append(top10_acc.item())

        # Average loss across layers in batch
        if layer_losses:
            batch_loss = torch.stack(layer_losses).mean()

            # Backward pass
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model_wrapper.get_trainable_parameters(), 1.0)
            optimizer.step()

            total_loss += batch_loss.item()
            total_top1 += np.mean(batch_top1)
            total_top5 += np.mean(batch_top5)
            total_top10 += np.mean(batch_top10)
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': batch_loss.item(), 'top1': f'{np.mean(batch_top1):.2%}'})

    # Compute average metrics
    avg_metrics = {
        'train_loss': total_loss / num_batches,
        'train_top1_accuracy': total_top1 / num_batches,
        'train_top5_accuracy': total_top5 / num_batches,
        'train_top10_accuracy': total_top10 / num_batches,
    }

    return avg_metrics


def evaluate(
    model_wrapper,
    eval_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model_wrapper.tuned_lens.eval()

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_top10 = 0.0
    num_samples = 0

    with torch.no_grad():
        pbar = tqdm(eval_loader, desc="Evaluating", leave=False)
        for batch in pbar:
            hidden_states = batch['hidden_states'].to(device)
            targets = batch['primary_target'].to(device)
            layers = batch['layer']

            # Process each layer in the batch
            for layer_idx in range(model_wrapper.num_layers):
                layer_mask = (layers == layer_idx)
                if not layer_mask.any():
                    continue

                layer_hidden = hidden_states[layer_mask]
                layer_targets = targets[layer_mask]

                # Apply tuned lens
                logits = model_wrapper.tuned_lens(layer_hidden, layer_idx)

                # Compute metrics
                metrics = compute_metrics(logits, layer_targets)
                total_loss += metrics['loss'] * len(layer_targets)
                total_top1 += metrics['top1_accuracy'] * len(layer_targets)
                total_top5 += metrics['top5_accuracy'] * len(layer_targets)
                total_top10 += metrics['top10_accuracy'] * len(layer_targets)
                num_samples += len(layer_targets)

    # Compute averages
    avg_metrics = {
        'val_loss': total_loss / num_samples if num_samples > 0 else 0.0,
        'val_top1_accuracy': total_top1 / num_samples if num_samples > 0 else 0.0,
        'val_top5_accuracy': total_top5 / num_samples if num_samples > 0 else 0.0,
        'val_top10_accuracy': total_top10 / num_samples if num_samples > 0 else 0.0,
    }

    return avg_metrics


def main():
    # Configuration
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-3
    EARLY_STOPPING_PATIENCE = 5

    # Setup
    set_seed(42)
    device = get_device({'compute': {'device': 'cuda'}})

    # Create output directories
    output_dir = Path("results/cot_alignment")
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("models/cot_alignment")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging({'output': {'output_dir': str(output_dir), 'verbose': True}})
    logger.info("="*70)
    logger.info("CoT TOKEN ALIGNMENT TRAINING (All Layers)")
    logger.info("="*70)

    # Load datasets
    logger.info("\nLoading CoT alignment datasets...")
    train_dataset = CoTAlignmentDataset("data/cot_train_all_layers.pt")
    test_dataset = CoTAlignmentDataset("data/cot_test_all_layers.pt")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    # Create model wrapper
    logger.info("\nInitializing Tuned Lens (All 16 layers)...")
    config = {
        'model': {
            'name': 'llama',
            'checkpoint_path': '~/codi_ckpt/llama_gsm8k/',
            'hidden_size': 2048,
            'num_layers': 16,
            'vocab_size': 32000,
        },
        'tuned_lens': {
            'use_layer_norm': True,
            'initialize_near_identity': True,
        }
    }
    model_wrapper = create_codi_tuned_lens(config, device=str(device))
    logger.info(f"Trainable parameters: {model_wrapper.num_parameters():,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model_wrapper.get_trainable_parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Training loop
    logger.info("\n" + "="*70)
    logger.info("STARTING TRAINING")
    logger.info("="*70)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = models_dir / "tuned_lens_all_layers_best.pt"

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        logger.info(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        # Train
        train_metrics = train_epoch(model_wrapper, train_loader, optimizer, device)

        # Evaluate
        val_metrics = evaluate(model_wrapper, test_loader, device)

        # Log metrics
        logger.info(f"Train Loss: {train_metrics['train_loss']:.4f} | "
                   f"Top-1: {train_metrics['train_top1_accuracy']:.2%} | "
                   f"Top-5: {train_metrics['train_top5_accuracy']:.2%} | "
                   f"Top-10: {train_metrics['train_top10_accuracy']:.2%}")
        logger.info(f"Val Loss: {val_metrics['val_loss']:.4f} | "
                   f"Top-1: {val_metrics['val_top1_accuracy']:.2%} | "
                   f"Top-5: {val_metrics['val_top5_accuracy']:.2%} | "
                   f"Top-10: {val_metrics['val_top10_accuracy']:.2%}")

        # Learning rate scheduling
        scheduler.step()

        # Early stopping
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0

            # Save best model
            model_wrapper.save_tuned_lens(str(best_model_path))
            logger.info(f"✓ New best model saved (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epochs")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info(f"\n⏹ Early stopping triggered after {epoch+1} epochs")
                break

        # Clear cache periodically
        if (epoch + 1) % 10 == 0:
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

    model_wrapper.load_tuned_lens(str(best_model_path))
    final_metrics = evaluate(model_wrapper, test_loader, device)

    logger.info(f"\nFinal Test Metrics:")
    logger.info(f"  Loss: {final_metrics['val_loss']:.4f}")
    logger.info(f"  Top-1 Accuracy: {final_metrics['val_top1_accuracy']:.2%}")
    logger.info(f"  Top-5 Accuracy: {final_metrics['val_top5_accuracy']:.2%}")
    logger.info(f"  Top-10 Accuracy: {final_metrics['val_top10_accuracy']:.2%}")

    logger.info("\n✓ CoT alignment training complete!")


if __name__ == '__main__':
    main()
