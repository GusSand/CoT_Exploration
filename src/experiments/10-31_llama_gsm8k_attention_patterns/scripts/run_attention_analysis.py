"""
Main script for CoT Attention Pattern Analysis (Experiment 3)

Analyzes whether CoT positions attend to each other sequentially or in parallel.
"""

import sys
import os
import json
import torch
import numpy as np
import wandb
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from core.model_loader import load_codi_model
from core.attention_extractor import AttentionExtractor
from core.attention_metrics import compute_all_metrics, aggregate_metrics_across_examples
from core.visualizations import (
    plot_layer_wise_heatmaps,
    plot_aggregated_attention,
    plot_metrics_comparison,
    plot_layer_evolution
)


def load_data(data_path, test_mode=False, test_size=10):
    """Load prepared question pairs"""
    with open(data_path, 'r') as f:
        data = json.load(f)

    if test_mode:
        data = data[:test_size]

    print(f"Loaded {len(data)} question pairs")
    return data


def prepare_codi_input(question, tokenizer, num_latent, device="cuda"):
    """
    Prepare input for CODI model with continuous thought placeholder tokens

    Args:
        question: Question string
        tokenizer: Tokenizer
        num_latent: Number of continuous thought tokens
        device: Device

    Returns:
        input_ids: Tensor of input token IDs including BoT/EoT markers
        attention_mask: Attention mask tensor
        cot_positions: List of positions for continuous thought tokens
    """
    # Format question with user header
    question_prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>"

    # Tokenize question
    question_tokens = tokenizer.encode(question_prompt, add_special_tokens=False)

    # Get special token IDs
    # BoT = Beginning of Thought, EoT = End of Thought
    think_header = "<|start_header_id|>think<|end_header_id|>\n\n"
    bot_tokens = tokenizer.encode(think_header, add_special_tokens=False)
    eot_token = tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0]

    # Create input with BoT, latent placeholders, and EoT
    # Format: [question tokens] [BoT tokens] [PAD] [PAD] ... [PAD] [EoT]
    input_ids = question_tokens + bot_tokens + [tokenizer.pad_token_id] * num_latent + [eot_token]

    # Positions of continuous thought tokens (the PAD tokens between BoT and EoT)
    bot_end_pos = len(question_tokens) + len(bot_tokens)
    cot_positions = list(range(bot_end_pos, bot_end_pos + num_latent))

    # Convert to tensor
    input_ids = torch.tensor([input_ids], device=device)
    attention_mask = torch.ones_like(input_ids)

    return input_ids, attention_mask, cot_positions


def main():
    print("="*80)
    print("CoT Attention Pattern Analysis - Experiment 3")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Test Mode: {config.TEST_MODE}")
    print(f"Device: {config.DEVICE}")
    print("="*80)

    # Create output directories
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.VIZ_DIR, exist_ok=True)

    # Initialize W&B (filter out non-serializable config values)
    config_dict = {k: v for k, v in vars(config).items()
                   if not k.startswith('_') and isinstance(v, (int, float, str, bool, list, dict, type(None)))}
    wandb.init(
        project=config.WANDB_PROJECT,
        entity=config.WANDB_ENTITY,
        name=f"{config.EXPERIMENT_NAME}_{'test' if config.TEST_MODE else 'full'}",
        config=config_dict
    )

    # Load data
    data = load_data(
        config.DATA_PATH,
        test_mode=config.TEST_MODE,
        test_size=config.TEST_SUBSET_SIZE
    )

    # Load model
    print("\nLoading CODI model...")
    model, tokenizer = load_codi_model(
        config.CHECKPOINT_PATH,
        config.MODEL_NAME,
        num_latent=config.NUM_LATENT,
        device=config.DEVICE
    )

    # Create attention extractor
    extractor = AttentionExtractor(
        model,
        num_layers=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS,
        num_latent=config.NUM_LATENT
    )

    # Register hooks to capture attention
    extractor.register_hooks()

    # Storage for results
    all_attention_patterns = {}  # example_idx -> layer_idx -> attention matrix
    all_metrics = []  # List of metrics dicts (one per example)

    print(f"\nAnalyzing attention patterns for {len(data)} examples...")

    for idx, pair in enumerate(tqdm(data, desc="Processing examples")):
        # Get question text
        question_text = pair['clean']['question']

        # Prepare CODI input with placeholder CoT tokens
        try:
            input_ids, attention_mask, cot_positions = prepare_codi_input(
                question_text,
                tokenizer,
                config.NUM_LATENT,
                config.DEVICE
            )
        except Exception as e:
            print(f"\nSkipping example {idx}: Failed to prepare input - {e}")
            continue

        # Extract attention patterns
        cot_attention = extractor.extract_cot_attention(
            input_ids,
            attention_mask,
            cot_positions
        )

        # Aggregate across heads if configured
        if config.AGGREGATE_HEADS:
            cot_attention = extractor.aggregate_across_heads(cot_attention)

        # Store patterns
        all_attention_patterns[idx] = cot_attention

        # Compute metrics for aggregated attention (if configured)
        if config.AGGREGATE_LAYERS:
            aggregated = extractor.aggregate_across_layers(cot_attention)
            metrics = compute_all_metrics(aggregated)
        else:
            # Use last layer for metrics
            last_layer_attn = cot_attention[config.NUM_LAYERS - 1]
            metrics = compute_all_metrics(last_layer_attn)

        all_metrics.append(metrics)

        # Log to W&B
        wandb.log({
            'example_idx': idx,
            'sequential_score': metrics['sequential_score'],
            'self_attention_score': metrics['self_attention_score'],
            'mean_entropy': metrics['mean_entropy'],
            'forward_attention': metrics['forward_attention'],
            'backward_attention': metrics['backward_attention']
        })

    # Remove hooks
    extractor.remove_hooks()

    print(f"\nProcessed {len(all_attention_patterns)} examples successfully")

    # Aggregate metrics across all examples
    print("\nComputing aggregate statistics...")
    aggregate_stats = aggregate_metrics_across_examples(all_metrics)

    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"Sequential Score (N→N-1): {aggregate_stats['sequential_score_mean']:.4f} ± {aggregate_stats['sequential_score_std']:.4f}")
    print(f"Self-Attention Score (N→N): {aggregate_stats['self_attention_mean']:.4f} ± {aggregate_stats['self_attention_std']:.4f}")
    print(f"Attention Entropy: {aggregate_stats['entropy_mean']:.3f} ± {aggregate_stats['entropy_std']:.3f} bits")
    print(f"  (Max possible: {np.log2(config.NUM_LATENT):.3f} bits)")
    print(f"Forward Attention: {aggregate_stats['forward_attention_mean']:.4f}")
    print(f"Backward Attention: {aggregate_stats['backward_attention_mean']:.4f}")
    print("="*80)

    # Log aggregate stats to W&B
    wandb.log(aggregate_stats)

    # Save results
    results_file = os.path.join(config.RESULTS_DIR, 'attention_analysis_results.json')
    # Filter config to only serializable values
    config_dict = {k: v for k, v in vars(config).items()
                   if not k.startswith('_') and isinstance(v, (int, float, str, bool, list, dict, type(None)))}
    with open(results_file, 'w') as f:
        json.dump({
            'aggregate_statistics': aggregate_stats,
            'per_example_metrics': all_metrics,
            'config': config_dict
        }, f, indent=2)
    print(f"\nSaved results to {results_file}")

    # Create visualizations
    print("\nGenerating visualizations...")

    # Average attention across all examples for each layer
    avg_attention_by_layer = {}
    for layer_idx in range(config.NUM_LAYERS):
        layer_patterns = []
        for example_attn in all_attention_patterns.values():
            if layer_idx in example_attn:
                layer_patterns.append(example_attn[layer_idx])
        if layer_patterns:
            avg_attention_by_layer[layer_idx] = np.mean(layer_patterns, axis=0)

    # 1. Layer-wise heatmaps
    layer_viz_dir = os.path.join(config.VIZ_DIR, 'layer_wise')
    plot_layer_wise_heatmaps(
        avg_attention_by_layer,
        layer_viz_dir,
        cmap=config.CMAP
    )

    # 2. Aggregated attention across all layers
    avg_across_all = np.mean([attn for attn in avg_attention_by_layer.values()], axis=0)
    aggregated_path = os.path.join(config.VIZ_DIR, 'aggregated_attention.png')
    plot_aggregated_attention(
        avg_across_all,
        'Average CoT Attention Pattern (All Layers & Examples)',
        aggregated_path,
        cmap=config.CMAP
    )

    # 3. Metrics comparison
    metrics_path = os.path.join(config.VIZ_DIR, 'metrics_comparison.png')
    plot_metrics_comparison(aggregate_stats, metrics_path)

    # 4. Layer evolution plots
    for metric_name in ['sequential', 'entropy', 'self_attention']:
        evolution_path = os.path.join(config.VIZ_DIR, f'{metric_name}_evolution.png')
        plot_layer_evolution(avg_attention_by_layer, metric_name, evolution_path)

    print("\nExperiment complete!")
    print(f"Results saved to: {config.RESULTS_DIR}")
    print(f"Visualizations saved to: {config.VIZ_DIR}")

    wandb.finish()


if __name__ == "__main__":
    main()
