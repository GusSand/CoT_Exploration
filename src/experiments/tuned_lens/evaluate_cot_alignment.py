"""
Evaluate and Visualize CoT Token Alignment Results.

This script analyzes the trained Tuned Lens model on CoT alignment task,
providing detailed breakdowns by layer, position, and sample predictions.

Usage:
    python evaluate_cot_alignment.py
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
import numpy as np
import json
from collections import defaultdict

# Import utilities
from utils import setup_logging, set_seed, get_device
from model import create_codi_tuned_lens
from train_cot_alignment import CoTAlignmentDataset, compute_metrics

# Try to load tokenizer
try:
    from transformers import AutoTokenizer
    MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
except:
    tokenizer = None
    print("Warning: Could not load tokenizer. Token decoding will be limited.")


def evaluate_by_layer(
    model_wrapper,
    eval_loader: DataLoader,
    device: torch.device
) -> Dict[int, Dict[str, float]]:
    """Evaluate model performance broken down by layer.

    Args:
        model_wrapper: Trained Tuned Lens model
        eval_loader: Evaluation data loader
        device: Device to use

    Returns:
        Dictionary mapping layer_idx -> metrics
    """
    model_wrapper.tuned_lens.eval()

    # Track metrics per layer
    layer_metrics = defaultdict(lambda: {
        'total_loss': 0.0,
        'total_top1': 0.0,
        'total_top5': 0.0,
        'total_top10': 0.0,
        'num_samples': 0
    })

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating by layer"):
            hidden_states = batch['hidden_states'].to(device)
            targets = batch['primary_target'].to(device)
            layers = batch['layer']

            # Process each layer
            for layer_idx in range(model_wrapper.num_layers):
                layer_mask = (layers == layer_idx)
                if not layer_mask.any():
                    continue

                layer_hidden = hidden_states[layer_mask]
                layer_targets = targets[layer_mask]

                # Apply tuned lens
                logits = model_wrapper.tuned_lens(layer_hidden, layer_idx)

                # Compute metrics
                loss = F.cross_entropy(logits, layer_targets)
                pred_ids = logits.argmax(dim=-1)
                top1_acc = (pred_ids == layer_targets).float().mean()
                top5_preds = logits.topk(k=5, dim=-1).indices
                top5_acc = (top5_preds == layer_targets.unsqueeze(-1)).any(dim=-1).float().mean()
                top10_preds = logits.topk(k=10, dim=-1).indices
                top10_acc = (top10_preds == layer_targets.unsqueeze(-1)).any(dim=-1).float().mean()

                # Accumulate
                layer_metrics[layer_idx]['total_loss'] += loss.item() * len(layer_targets)
                layer_metrics[layer_idx]['total_top1'] += top1_acc.item() * len(layer_targets)
                layer_metrics[layer_idx]['total_top5'] += top5_acc.item() * len(layer_targets)
                layer_metrics[layer_idx]['total_top10'] += top10_acc.item() * len(layer_targets)
                layer_metrics[layer_idx]['num_samples'] += len(layer_targets)

    # Compute averages
    results = {}
    for layer_idx in range(model_wrapper.num_layers):
        metrics = layer_metrics[layer_idx]
        n = metrics['num_samples']
        if n > 0:
            results[layer_idx] = {
                'loss': metrics['total_loss'] / n,
                'top1_accuracy': metrics['total_top1'] / n,
                'top5_accuracy': metrics['total_top5'] / n,
                'top10_accuracy': metrics['total_top10'] / n,
                'num_samples': n
            }

    return results


def evaluate_by_position(
    model_wrapper,
    eval_loader: DataLoader,
    device: torch.device
) -> Dict[int, Dict[str, float]]:
    """Evaluate model performance broken down by CT position.

    Args:
        model_wrapper: Trained Tuned Lens model
        eval_loader: Evaluation data loader
        device: Device to use

    Returns:
        Dictionary mapping position -> metrics
    """
    model_wrapper.tuned_lens.eval()

    # Track metrics per position
    position_metrics = defaultdict(lambda: {
        'total_loss': 0.0,
        'total_top1': 0.0,
        'total_top5': 0.0,
        'total_top10': 0.0,
        'num_samples': 0
    })

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating by position"):
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

                # Apply tuned lens
                logits = model_wrapper.tuned_lens(layer_hidden, layer_idx)

                # Compute metrics per position
                for pos in range(6):
                    pos_mask = (layer_positions == pos)
                    if not pos_mask.any():
                        continue

                    pos_logits = logits[pos_mask]
                    pos_targets = layer_targets[pos_mask]

                    loss = F.cross_entropy(pos_logits, pos_targets)
                    pred_ids = pos_logits.argmax(dim=-1)
                    top1_acc = (pred_ids == pos_targets).float().mean()
                    top5_preds = pos_logits.topk(k=5, dim=-1).indices
                    top5_acc = (top5_preds == pos_targets.unsqueeze(-1)).any(dim=-1).float().mean()
                    top10_preds = pos_logits.topk(k=10, dim=-1).indices
                    top10_acc = (top10_preds == pos_targets.unsqueeze(-1)).any(dim=-1).float().mean()

                    # Accumulate
                    position_metrics[pos]['total_loss'] += loss.item() * len(pos_targets)
                    position_metrics[pos]['total_top1'] += top1_acc.item() * len(pos_targets)
                    position_metrics[pos]['total_top5'] += top5_acc.item() * len(pos_targets)
                    position_metrics[pos]['total_top10'] += top10_acc.item() * len(pos_targets)
                    position_metrics[pos]['num_samples'] += len(pos_targets)

    # Compute averages
    results = {}
    for pos in range(6):
        metrics = position_metrics[pos]
        n = metrics['num_samples']
        if n > 0:
            results[pos] = {
                'loss': metrics['total_loss'] / n,
                'top1_accuracy': metrics['total_top1'] / n,
                'top5_accuracy': metrics['total_top5'] / n,
                'top10_accuracy': metrics['total_top10'] / n,
                'num_samples': n
            }

    return results


def sample_predictions(
    model_wrapper,
    test_dataset: CoTAlignmentDataset,
    device: torch.device,
    num_samples: int = 20,
    top_k: int = 5
) -> List[Dict]:
    """Generate sample predictions for visualization.

    Args:
        model_wrapper: Trained Tuned Lens model
        test_dataset: Test dataset
        device: Device to use
        num_samples: Number of samples to decode
        top_k: Number of top predictions to show

    Returns:
        List of prediction dictionaries
    """
    model_wrapper.tuned_lens.eval()

    # Sample random indices
    indices = np.random.choice(len(test_dataset), size=num_samples, replace=False)

    predictions = []

    with torch.no_grad():
        for idx in tqdm(indices, desc="Generating sample predictions"):
            sample = test_dataset[int(idx)]

            # Get data
            hidden_state = sample['hidden_states'].unsqueeze(0).to(device)
            target_id = sample['primary_target'].item()
            layer = sample['layer']
            position = sample['position']

            # Apply tuned lens
            logits = model_wrapper.tuned_lens(hidden_state, layer)

            # Get top-k predictions
            topk_logits, topk_ids = logits.topk(k=top_k, dim=-1)
            topk_probs = F.softmax(topk_logits, dim=-1)

            # Decode tokens if tokenizer available
            if tokenizer is not None:
                target_token = tokenizer.decode([target_id])
                predicted_tokens = [tokenizer.decode([tid.item()]) for tid in topk_ids[0]]
            else:
                target_token = f"[{target_id}]"
                predicted_tokens = [f"[{tid.item()}]" for tid in topk_ids[0]]

            # Check if correct
            correct = topk_ids[0, 0].item() == target_id
            in_top5 = target_id in topk_ids[0].tolist()

            predictions.append({
                'layer': layer,
                'position': position,
                'target_id': target_id,
                'target_token': target_token,
                'predictions': [
                    {'token': tok, 'prob': prob.item(), 'id': tid.item()}
                    for tok, prob, tid in zip(predicted_tokens, topk_probs[0], topk_ids[0])
                ],
                'correct': correct,
                'in_top5': in_top5,
                'metadata': {
                    'cot_steps': test_dataset.metadata['cot_steps'][idx],
                    'cot_tokens': test_dataset.cot_target_token_ids[idx]
                }
            })

    return predictions


def print_results(
    layer_results: Dict[int, Dict[str, float]],
    position_results: Dict[int, Dict[str, float]],
    sample_preds: List[Dict],
    logger
):
    """Print evaluation results in a formatted way."""

    logger.info("\n" + "="*70)
    logger.info("PERFORMANCE BY LAYER")
    logger.info("="*70)
    logger.info(f"{'Layer':<8} {'Samples':<10} {'Loss':<10} {'Top-1':<10} {'Top-5':<10} {'Top-10':<10}")
    logger.info("-"*70)

    for layer_idx in sorted(layer_results.keys()):
        metrics = layer_results[layer_idx]
        logger.info(
            f"{layer_idx:<8} "
            f"{metrics['num_samples']:<10} "
            f"{metrics['loss']:<10.4f} "
            f"{metrics['top1_accuracy']:<10.2%} "
            f"{metrics['top5_accuracy']:<10.2%} "
            f"{metrics['top10_accuracy']:<10.2%}"
        )

    logger.info("\n" + "="*70)
    logger.info("PERFORMANCE BY POSITION")
    logger.info("="*70)
    logger.info(f"{'Position':<10} {'Samples':<10} {'Loss':<10} {'Top-1':<10} {'Top-5':<10} {'Top-10':<10}")
    logger.info("-"*70)

    for pos in sorted(position_results.keys()):
        metrics = position_results[pos]
        logger.info(
            f"{pos:<10} "
            f"{metrics['num_samples']:<10} "
            f"{metrics['loss']:<10.4f} "
            f"{metrics['top1_accuracy']:<10.2%} "
            f"{metrics['top5_accuracy']:<10.2%} "
            f"{metrics['top10_accuracy']:<10.2%}"
        )

    logger.info("\n" + "="*70)
    logger.info("SAMPLE PREDICTIONS (First 5)")
    logger.info("="*70)

    for i, pred in enumerate(sample_preds[:5]):
        logger.info(f"\nSample {i+1}:")
        logger.info(f"  Layer {pred['layer']}, Position {pred['position']}")
        logger.info(f"  CoT Steps: {pred['metadata']['cot_steps']}")
        logger.info(f"  Target: '{pred['target_token']}' (ID: {pred['target_id']})")
        logger.info(f"  Correct: {pred['correct']} | In Top-5: {pred['in_top5']}")
        logger.info(f"  Top-5 Predictions:")
        for j, p in enumerate(pred['predictions'], 1):
            logger.info(f"    {j}. '{p['token']}' ({p['prob']:.2%})")


def main():
    # Setup
    set_seed(42)
    device = get_device({'compute': {'device': 'cuda'}})

    # Create output directory
    output_dir = Path("results/cot_alignment")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging({'output': {'output_dir': str(output_dir), 'verbose': True}})
    logger.info("="*70)
    logger.info("COT TOKEN ALIGNMENT EVALUATION")
    logger.info("="*70)

    # Load test dataset
    logger.info("\nLoading test dataset...")
    test_dataset = CoTAlignmentDataset("data/cot_test_all_layers.pt")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Load trained model
    logger.info("\nLoading trained model...")
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
    model_wrapper.load_tuned_lens("models/cot_alignment/tuned_lens_all_layers_best.pt")

    # Evaluate by layer
    logger.info("\nEvaluating performance by layer...")
    layer_results = evaluate_by_layer(model_wrapper, test_loader, device)

    # Evaluate by position
    logger.info("\nEvaluating performance by position...")
    position_results = evaluate_by_position(model_wrapper, test_loader, device)

    # Generate sample predictions
    logger.info("\nGenerating sample predictions...")
    sample_preds = sample_predictions(model_wrapper, test_dataset, device, num_samples=20, top_k=5)

    # Print results
    print_results(layer_results, position_results, sample_preds, logger)

    # Save results to JSON
    results = {
        'layer_performance': {str(k): v for k, v in layer_results.items()},
        'position_performance': {str(k): v for k, v in position_results.items()},
        'sample_predictions': sample_preds
    }

    output_file = output_dir / "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")
    logger.info("\nEvaluation complete!")


if __name__ == '__main__':
    main()
