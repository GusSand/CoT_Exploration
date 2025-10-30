"""
Story 2-4: Run layer-wise activation patching experiment

This script:
1. Loads CODI model
2. For each clean/corrupted pair:
   - Extracts clean activations at all layers
   - Runs baseline corrupted (no patching)
   - Patches corrupted with clean activations at each layer
   - Computes KL divergence between patched and baseline
3. Saves results with W&B logging
"""

import json
import sys
import os
import torch
import wandb
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add experiment directory to path
exp_dir = Path(__file__).parent.parent
sys.path.insert(0, str(exp_dir))

import config
from core.model_loader import (
    load_codi_model,
    prepare_codi_input,
    extract_activations_at_layer
)
from core.activation_patcher import (
    run_with_patching,
    extract_answer_logits
)
from core.metrics import (
    compute_kl_divergence,
    compute_logit_difference,
    compute_prediction_change
)


def setup_wandb():
    """Initialize Weights & Biases logging"""
    wandb.init(
        project=config.WANDB_PROJECT,
        entity=config.WANDB_ENTITY,
        name=f"{config.EXPERIMENT_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "model": config.MODEL_NAME,
            "checkpoint": config.CHECKPOINT_PATH,
            "num_layers": config.NUM_LAYERS,
            "num_latent": config.NUM_LATENT,
            "batch_size": config.BATCH_SIZE,
            "device": config.DEVICE,
            "seed": config.SEED,
        }
    )
    print("✓ W&B initialized")


def load_prepared_pairs():
    """Load prepared pairs from Story 1"""
    pairs_path = os.path.join(config.RESULTS_DIR, "prepared_pairs.json")
    with open(pairs_path, 'r') as f:
        pairs = json.load(f)
    print(f"✓ Loaded {len(pairs)} prepared pairs")
    return pairs


def extract_clean_activations(model, tokenizer, clean_question, num_layers):
    """
    Extract clean activations at all layers for CoT tokens

    Args:
        model: CODI model
        tokenizer: Tokenizer
        clean_question: Clean question string
        num_layers: Number of layers to extract from

    Returns:
        clean_acts: Dict mapping layer_idx -> activations tensor
        input_ids: Input IDs used
        cot_positions: Positions of CoT tokens
    """
    # Prepare input
    input_ids, cot_positions = prepare_codi_input(
        clean_question,
        tokenizer,
        bot_id=model.bot_id,
        eot_id=model.eot_id,
        num_latent=model.num_latent,
        device=config.DEVICE
    )

    # Create attention mask
    attention_mask = torch.ones_like(input_ids)

    # Extract activations at each layer
    clean_acts = {}
    for layer_idx in range(num_layers):
        acts = extract_activations_at_layer(
            model,
            input_ids,
            attention_mask,
            layer_idx,
            positions=cot_positions
        )
        clean_acts[layer_idx] = acts

    return clean_acts, input_ids, cot_positions


def run_baseline_corrupted(model, tokenizer, corrupted_question):
    """
    Run baseline corrupted model (no patching)

    Returns:
        output: Model output
        input_ids: Input IDs
        cot_positions: CoT token positions
    """
    # Prepare input
    input_ids, cot_positions = prepare_codi_input(
        corrupted_question,
        tokenizer,
        bot_id=model.bot_id,
        eot_id=model.eot_id,
        num_latent=model.num_latent,
        device=config.DEVICE
    )

    # Create attention mask
    attention_mask = torch.ones_like(input_ids)

    # Run forward pass
    with torch.no_grad():
        output = model.codi(input_ids=input_ids, attention_mask=attention_mask)

    return output, input_ids, cot_positions


def run_patching_for_pair(model, tokenizer, pair_data, num_layers):
    """
    Run patching experiment for a single pair

    Args:
        model: CODI model
        tokenizer: Tokenizer
        pair_data: Dict with 'clean' and 'corrupted' data
        num_layers: Number of layers

    Returns:
        results: Dict with KL divergence for each layer
    """
    pair_id = pair_data['pair_id']
    clean_q = pair_data['clean']['question']
    corrupt_q = pair_data['corrupted']['question']

    # Extract clean activations at all layers
    clean_acts, _, cot_positions = extract_clean_activations(
        model, tokenizer, clean_q, num_layers
    )

    # Run baseline corrupted (no patching)
    baseline_output, corrupt_input_ids, corrupt_cot_pos = run_baseline_corrupted(
        model, tokenizer, corrupt_q
    )

    # Determine answer token positions
    # Answer comes after EoT token
    eot_pos = corrupt_cot_pos[-1] + 1  # Position after last CoT token (which is EoT)
    answer_start_pos = eot_pos
    # We'll look at next 10 tokens as potential answer tokens
    answer_length = min(10, baseline_output.logits.shape[1] - answer_start_pos)

    baseline_answer_logits = baseline_output.logits[:, answer_start_pos:answer_start_pos+answer_length, :]

    # Create attention mask
    attention_mask = torch.ones_like(corrupt_input_ids)

    # Patch at each layer and compute metrics
    results = {
        'pair_id': pair_id,
        'clean_question': clean_q,
        'corrupted_question': corrupt_q,
        'clean_answer': pair_data['clean']['answer'],
        'corrupted_answer': pair_data['corrupted']['answer'],
        'layer_results': []
    }

    for layer_idx in range(num_layers):
        # Get clean activations for this layer
        clean_layer_acts = clean_acts[layer_idx].to(config.DEVICE)

        # Run with patching
        patched_output = run_with_patching(
            model,
            corrupt_input_ids,
            attention_mask,
            layer_idx,
            corrupt_cot_pos,
            clean_layer_acts
        )

        # Extract answer logits
        patched_answer_logits = patched_output.logits[:, answer_start_pos:answer_start_pos+answer_length, :]

        # Compute metrics
        kl_div, kl_per_pos = compute_kl_divergence(patched_answer_logits, baseline_answer_logits)
        l2_diff = compute_logit_difference(patched_answer_logits, baseline_answer_logits)
        changed, change_rate = compute_prediction_change(patched_answer_logits, baseline_answer_logits)

        results['layer_results'].append({
            'layer': layer_idx,
            'kl_divergence': kl_div,
            'l2_difference': l2_diff,
            'prediction_change_rate': change_rate
        })

    return results


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("STORY 2-4: Layer-wise Activation Patching Experiment")
    print("="*80 + "\n")

    # Set random seed
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # Setup W&B
    print("Initializing W&B...")
    setup_wandb()

    # Load model
    print("\nLoading CODI model...")
    model, tokenizer = load_codi_model(
        config.CHECKPOINT_PATH,
        config.MODEL_NAME,
        num_latent=config.NUM_LATENT,
        device=config.DEVICE
    )

    # Load prepared pairs
    print("\nLoading prepared pairs...")
    pairs = load_prepared_pairs()

    # Use subset if in test mode
    if config.TEST_MODE:
        pairs = pairs[:config.TEST_SUBSET_SIZE]
        print(f"⚠️  TEST MODE: Using subset of {len(pairs)} pairs")

    # Run patching for each pair
    print(f"\nRunning patching experiment for {len(pairs)} pairs...")
    print(f"  Testing {config.NUM_LAYERS} layers per pair")
    print(f"  Total forward passes: {len(pairs) * (config.NUM_LAYERS + 1)}\n")

    all_results = []

    for pair in tqdm(pairs, desc="Processing pairs"):
        try:
            result = run_patching_for_pair(model, tokenizer, pair, config.NUM_LAYERS)
            all_results.append(result)

            # Log to W&B
            pair_id = result['pair_id']
            for layer_result in result['layer_results']:
                wandb.log({
                    f"pair_{pair_id}/layer_{layer_result['layer']}/kl_div": layer_result['kl_divergence'],
                    f"pair_{pair_id}/layer_{layer_result['layer']}/l2_diff": layer_result['l2_difference'],
                    f"pair_{pair_id}/layer_{layer_result['layer']}/pred_change": layer_result['prediction_change_rate'],
                })

        except Exception as e:
            print(f"\n❌ Error processing pair {pair['pair_id']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    output_path = os.path.join(config.RESULTS_DIR, "patching_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Saved results to: {output_path}")

    # Compute aggregate statistics
    print("\n" + "="*80)
    print("Aggregate Statistics")
    print("="*80 + "\n")

    # Aggregate KL divergence by layer
    layer_kl_means = {}
    layer_kl_stds = {}

    for layer_idx in range(config.NUM_LAYERS):
        kl_values = [r['layer_results'][layer_idx]['kl_divergence'] for r in all_results]
        layer_kl_means[layer_idx] = np.mean(kl_values)
        layer_kl_stds[layer_idx] = np.std(kl_values)

        print(f"Layer {layer_idx:2d}: KL = {layer_kl_means[layer_idx]:.4f} ± {layer_kl_stds[layer_idx]:.4f}")

    # Identify critical layers (top 5 by mean KL)
    critical_layers = sorted(layer_kl_means.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\nTop 5 Critical Layers (by mean KL divergence):")
    for layer, kl in critical_layers:
        print(f"  Layer {layer}: KL = {kl:.4f}")

    # Log aggregate statistics to W&B
    for layer_idx in range(config.NUM_LAYERS):
        wandb.log({
            f"aggregate/layer_{layer_idx}/mean_kl": layer_kl_means[layer_idx],
            f"aggregate/layer_{layer_idx}/std_kl": layer_kl_stds[layer_idx],
        })

    # Save aggregate statistics
    aggregate_stats = {
        'layer_kl_means': layer_kl_means,
        'layer_kl_stds': layer_kl_stds,
        'critical_layers': [{'layer': l, 'kl': k} for l, k in critical_layers]
    }

    aggregate_path = os.path.join(config.RESULTS_DIR, "aggregate_statistics.json")
    with open(aggregate_path, 'w') as f:
        json.dump(aggregate_stats, f, indent=2)

    print(f"\n✓ Saved aggregate statistics to: {aggregate_path}")

    print("\n" + "="*80)
    print("✓ PATCHING EXPERIMENT COMPLETE")
    print("="*80 + "\n")

    wandb.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
