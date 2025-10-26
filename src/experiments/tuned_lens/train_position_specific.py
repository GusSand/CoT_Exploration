"""
Train Position-Specific Tuned Lens (Lite Version) for CoT Token Alignment.

This script trains position-specific Tuned Lens transformations for critical layers only.
Critical layers (6, 9, 14, 15) have separate transformations for each of the 6 CT positions.

Usage:
    python train_position_specific.py
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Dict
import numpy as np
import time

# Import utilities
from utils import setup_logging, set_seed, get_device, clear_gpu_cache, format_time
from position_specific_model import create_position_specific_tuned_lens
from train_cot_alignment import CoTAlignmentDataset, compute_metrics


def train_epoch_position_specific(
    model_wrapper,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """Train for one epoch with position-specific model."""
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
        positions = batch['position']

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
            layer_positions = positions[layer_mask]

            # Check if this is a critical layer (needs position-specific handling)
            if layer_idx in model_wrapper.critical_layers:
                # Process each position separately for critical layers
                for pos_idx in range(model_wrapper.num_positions):
                    pos_mask = (layer_positions == pos_idx)
                    if not pos_mask.any():
                        continue

                    pos_hidden = layer_hidden[pos_mask]
                    pos_targets = layer_targets[pos_mask]

                    # Forward pass with position-specific transformation
                    logits = model_wrapper.forward(pos_hidden, layer_idx, pos_idx)

                    # Compute loss
                    loss = F.cross_entropy(logits, pos_targets)
                    layer_losses.append(loss)

                    # Compute metrics
                    with torch.no_grad():
                        pred_ids = logits.argmax(dim=-1)
                        top1_acc = (pred_ids == pos_targets).float().mean()
                        top5_preds = logits.topk(k=5, dim=-1).indices
                        top5_acc = (top5_preds == pos_targets.unsqueeze(-1)).any(dim=-1).float().mean()
                        top10_preds = logits.topk(k=10, dim=-1).indices
                        top10_acc = (top10_preds == pos_targets.unsqueeze(-1)).any(dim=-1).float().mean()

                        batch_top1.append(top1_acc.item())
                        batch_top5.append(top5_acc.item())
                        batch_top10.append(top10_acc.item())
            else:
                # Non-critical layer: use position-agnostic transformation
                logits = model_wrapper.forward(layer_hidden, layer_idx, None)

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
        'train_loss': total_loss / num_batches if num_batches > 0 else 0.0,
        'train_top1_accuracy': total_top1 / num_batches if num_batches > 0 else 0.0,
        'train_top5_accuracy': total_top5 / num_batches if num_batches > 0 else 0.0,
        'train_top10_accuracy': total_top10 / num_batches if num_batches > 0 else 0.0,
    }

    return avg_metrics


def evaluate_position_specific(
    model_wrapper,
    eval_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate position-specific model on validation set."""
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
            positions = batch['position']

            # Process each layer
            for layer_idx in range(model_wrapper.num_layers):
                layer_mask = (layers == layer_idx)
                if not layer_mask.any():
                    continue

                layer_hidden = hidden_states[layer_mask]
                layer_targets = targets[layer_mask]
                layer_positions = positions[layer_mask]

                # Check if critical layer
                if layer_idx in model_wrapper.critical_layers:
                    # Process each position separately
                    for pos_idx in range(model_wrapper.num_positions):
                        pos_mask = (layer_positions == pos_idx)
                        if not pos_mask.any():
                            continue

                        pos_hidden = layer_hidden[pos_mask]
                        pos_targets = layer_targets[pos_mask]

                        # Forward pass
                        logits = model_wrapper.forward(pos_hidden, layer_idx, pos_idx)

                        # Compute metrics
                        metrics = compute_metrics(logits, pos_targets)
                        total_loss += metrics['loss'] * len(pos_targets)
                        total_top1 += metrics['top1_accuracy'] * len(pos_targets)
                        total_top5 += metrics['top5_accuracy'] * len(pos_targets)
                        total_top10 += metrics['top10_accuracy'] * len(pos_targets)
                        num_samples += len(pos_targets)
                else:
                    # Non-critical layer
                    logits = model_wrapper.forward(layer_hidden, layer_idx, None)

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
    CRITICAL_LAYERS = [6, 9, 14, 15]  # Lite version: only best-performing layers

    # Setup
    set_seed(42)
    device = get_device({'compute': {'device': 'cuda'}})

    # Create output directories
    output_dir = Path("results/position_specific")
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("models/position_specific")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging({'output': {'output_dir': str(output_dir), 'verbose': True}})
    logger.info("="*70)
    logger.info("POSITION-SPECIFIC TUNED LENS TRAINING (LITE VERSION)")
    logger.info("="*70)
    logger.info(f"Critical layers: {CRITICAL_LAYERS}")
    logger.info(f"Position-specific transforms for: {len(CRITICAL_LAYERS)} layers × 6 positions")

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
    logger.info("\nInitializing Position-Specific Tuned Lens...")
    config = {
        'model': {
            'name': 'llama',
            'checkpoint_path': '~/codi_ckpt/llama_gsm8k/',
            'hidden_size': 2048,
            'num_layers': 16,
            'num_ct_tokens': 6,
            'vocab_size': 32000,
        },
        'tuned_lens': {
            'use_layer_norm': True,
            'initialize_near_identity': True,
            'critical_layers': CRITICAL_LAYERS,
        }
    }
    model_wrapper = create_position_specific_tuned_lens(config, device=str(device))
    logger.info(f"Trainable parameters: {model_wrapper.num_parameters():,}")
    logger.info(f"Critical layers: {model_wrapper.critical_layers}")

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
    best_model_path = models_dir / "position_specific_best.pt"

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        logger.info(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        # Train
        train_metrics = train_epoch_position_specific(model_wrapper, train_loader, optimizer, device)

        # Evaluate
        val_metrics = evaluate_position_specific(model_wrapper, test_loader, device)

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
    final_metrics = evaluate_position_specific(model_wrapper, test_loader, device)

    logger.info(f"\nFinal Test Metrics:")
    logger.info(f"  Loss: {final_metrics['val_loss']:.4f}")
    logger.info(f"  Top-1 Accuracy: {final_metrics['val_top1_accuracy']:.2%}")
    logger.info(f"  Top-5 Accuracy: {final_metrics['val_top5_accuracy']:.2%}")
    logger.info(f"  Top-10 Accuracy: {final_metrics['val_top10_accuracy']:.2%}")

    logger.info("\n✓ Position-specific training complete!")


if __name__ == '__main__':
    main()
