"""
Position-wise Activation Patching Experiment

This script tests all (layer, position) combinations individually to create
a fine-grained map of position-specific importance in continuous thought tokens.

For each pair:
1. Extract clean activations at all 16 layers × 5 positions
2. Run baseline corrupted (no patching)
3. For each (layer, position): patch ONLY that position and measure effects
4. Save results with all 4 metrics
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
from core.single_position_patcher import run_with_single_position_patching
from core.metrics import (
    compute_kl_divergence,
    compute_logit_difference,
    compute_prediction_change,
    compute_answer_logit_difference,
    tokenize_answer
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
            "dataset": "filtered_57_pairs",
        }
    )
    print("✓ W&B initialized")


def load_prepared_pairs():
    """Load filtered prepared pairs (57 pairs)"""
    with open(config.DATA_PATH, 'r') as f:
        pairs = json.load(f)
    print(f"✓ Loaded {len(pairs)} filtered pairs")
    return pairs


def extract_clean_activations_all_positions(model, tokenizer, clean_question, num_layers, num_positions):
    """
    Extract clean activations at all layers and positions

    Args:
        model: CODI model
        tokenizer: Tokenizer
        clean_question: Clean question string
        num_layers: Number of layers (16)
        num_positions: Number of CoT positions (5)

    Returns:
        clean_acts: Dict[layer_idx][position_idx] -> activation tensor [1, hidden_dim]
        input_ids: Input IDs used
        cot_positions: List of CoT token positions
    """
    # Prepare input
    input_ids, cot_positions = prepare_codi_input(
        clean_question,
        tokenizer,
        bot_id=model.bot_id,
        eot_id=model.eot_id,
        num_latent=num_positions,
        device=config.DEVICE
    )

    # Create attention mask
    attention_mask = torch.ones_like(input_ids)

    # Extract activations at each layer for all CoT positions
    clean_acts = {}
    for layer_idx in range(num_layers):
        # Extract all CoT positions for this layer
        acts = extract_activations_at_layer(
            model,
            input_ids,
            attention_mask,
            layer_idx,
            positions=cot_positions
        )
        # acts shape: [batch, num_positions, hidden_dim]

        # Store each position separately
        clean_acts[layer_idx] = {}
        for pos_idx in range(num_positions):
            # Extract single position: [batch, hidden_dim]
            clean_acts[layer_idx][pos_idx] = acts[:, pos_idx, :]

    return clean_acts, input_ids, cot_positions


def run_baseline_corrupted(model, tokenizer, corrupted_question, num_latent):
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
        num_latent=num_latent,
        device=config.DEVICE
    )

    # Create attention mask
    attention_mask = torch.ones_like(input_ids)

    # Run forward pass
    with torch.no_grad():
        output = model.codi(input_ids=input_ids, attention_mask=attention_mask)

    return output, input_ids, cot_positions


def run_position_patching_for_pair(model, tokenizer, pair_data, num_layers, num_positions):
    """
    Run position-wise patching for a single pair

    Tests all (layer, position) combinations: 16 × 5 = 80 patches

    Args:
        model: CODI model
        tokenizer: Tokenizer
        pair_data: Dict with 'clean' and 'corrupted' data
        num_layers: 16
        num_positions: 5

    Returns:
        results: Dict with results for all (layer, position) combinations
    """
    pair_id = pair_data['pair_id']
    clean_q = pair_data['clean']['question']
    corrupt_q = pair_data['corrupted']['question']
    clean_ans = pair_data['clean']['answer']
    corrupt_ans = pair_data['corrupted']['answer']

    # Extract clean activations at all layers and positions
    clean_acts, _, _ = extract_clean_activations_all_positions(
        model, tokenizer, clean_q, num_layers, num_positions
    )

    # Run baseline corrupted (no patching)
    baseline_output, corrupt_input_ids, corrupt_cot_pos = run_baseline_corrupted(
        model, tokenizer, corrupt_q, num_positions
    )

    # Determine answer token positions
    eot_pos = corrupt_cot_pos[-1] + 1
    answer_start_pos = eot_pos
    answer_length = min(10, baseline_output.logits.shape[1] - answer_start_pos)
    baseline_answer_logits = baseline_output.logits[:, answer_start_pos:answer_start_pos+answer_length, :]

    # Tokenize answers for answer_logit_difference metric
    clean_ans_tokens = tokenize_answer(clean_ans, tokenizer)
    corrupt_ans_tokens = tokenize_answer(corrupt_ans, tokenizer)

    # Create attention mask
    attention_mask = torch.ones_like(corrupt_input_ids)

    # Results structure
    results = {
        'pair_id': pair_id,
        'clean_question': clean_q,
        'corrupted_question': corrupt_q,
        'clean_answer': clean_ans,
        'corrupted_answer': corrupt_ans,
        'layer_position_results': {}
    }

    # Patch at each (layer, position) combination
    total_combinations = num_layers * num_positions
    pbar = tqdm(total=total_combinations, desc=f"Pair {pair_id}", leave=False)

    for layer_idx in range(num_layers):
        results['layer_position_results'][layer_idx] = {}

        for pos_idx in range(num_positions):
            # Get clean activation for this (layer, position)
            clean_act = clean_acts[layer_idx][pos_idx].to(config.DEVICE)

            # Run with single position patching
            patched_output = run_with_single_position_patching(
                model,
                corrupt_input_ids,
                attention_mask,
                layer_idx,
                corrupt_cot_pos[pos_idx],  # Actual position in sequence
                clean_act
            )

            # Extract answer logits
            patched_answer_logits = patched_output.logits[:, answer_start_pos:answer_start_pos+answer_length, :]

            # Compute all metrics
            kl_div, _ = compute_kl_divergence(patched_answer_logits, baseline_answer_logits)
            l2_diff = compute_logit_difference(patched_answer_logits, baseline_answer_logits)
            _, change_rate = compute_prediction_change(patched_answer_logits, baseline_answer_logits)
            ans_diff, clean_score, corrupt_score = compute_answer_logit_difference(
                patched_output.logits,
                clean_ans_tokens,
                corrupt_ans_tokens,
                answer_start_pos
            )

            # Store results
            results['layer_position_results'][layer_idx][pos_idx] = {
                'kl_divergence': kl_div,
                'l2_difference': l2_diff,
                'prediction_change_rate': change_rate,
                'answer_logit_diff': ans_diff,
                'clean_answer_score': clean_score,
                'corrupted_answer_score': corrupt_score
            }

            pbar.update(1)

            # Clear cache periodically
            if (layer_idx * num_positions + pos_idx) % 20 == 0:
                torch.cuda.empty_cache()

    pbar.close()

    return results


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("POSITION-WISE ACTIVATION PATCHING EXPERIMENT")
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
    print("\nLoading filtered pairs...")
    pairs = load_prepared_pairs()

    # Use subset if in test mode
    if config.TEST_MODE:
        pairs = pairs[:config.TEST_SUBSET_SIZE]
        print(f"⚠️  TEST MODE: Using subset of {len(pairs)} pairs")

    # Run position-wise patching for each pair
    print(f"\nRunning position-wise patching for {len(pairs)} pairs...")
    print(f"  Testing {config.NUM_LAYERS} layers × {config.NUM_LATENT} positions = {config.NUM_LAYERS * config.NUM_LATENT} combinations per pair")
    print(f"  Total forward passes: {len(pairs) * (config.NUM_LAYERS * config.NUM_LATENT + config.NUM_LAYERS + 1)}\n")

    all_results = []

    for pair in tqdm(pairs, desc="Processing pairs"):
        try:
            result = run_position_patching_for_pair(
                model, tokenizer, pair, config.NUM_LAYERS, config.NUM_LATENT
            )
            all_results.append(result)

            # Log to W&B
            pair_id = result['pair_id']
            for layer_idx in range(config.NUM_LAYERS):
                for pos_idx in range(config.NUM_LATENT):
                    metrics = result['layer_position_results'][layer_idx][pos_idx]
                    wandb.log({
                        f"pair_{pair_id}/layer_{layer_idx}/pos_{pos_idx}/kl_div": metrics['kl_divergence'],
                        f"pair_{pair_id}/layer_{layer_idx}/pos_{pos_idx}/answer_logit_diff": metrics['answer_logit_diff'],
                    })

            # Clear cache between pairs
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n❌ Error processing pair {pair['pair_id']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    output_path = os.path.join(config.RESULTS_DIR, "position_patching_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Saved results to: {output_path}")

    # Compute aggregate statistics
    print("\n" + "="*80)
    print("Aggregate Statistics")
    print("="*80 + "\n")

    # Aggregate metrics by (layer, position)
    layer_pos_kl_means = {}
    layer_pos_ans_diff_means = {}

    for layer_idx in range(config.NUM_LAYERS):
        layer_pos_kl_means[layer_idx] = {}
        layer_pos_ans_diff_means[layer_idx] = {}

        for pos_idx in range(config.NUM_LATENT):
            kl_values = [r['layer_position_results'][layer_idx][pos_idx]['kl_divergence']
                        for r in all_results]
            ans_diff_values = [r['layer_position_results'][layer_idx][pos_idx]['answer_logit_diff']
                              for r in all_results]

            layer_pos_kl_means[layer_idx][pos_idx] = np.mean(kl_values)
            layer_pos_ans_diff_means[layer_idx][pos_idx] = np.mean(ans_diff_values)

            print(f"Layer {layer_idx:2d}, Pos {pos_idx}: KL = {layer_pos_kl_means[layer_idx][pos_idx]:.4f}, "
                  f"Ans Diff = {layer_pos_ans_diff_means[layer_idx][pos_idx]:+.4f}")

    # Find top-5 critical (layer, position) combinations by KL divergence
    all_combos = []
    for layer_idx in range(config.NUM_LAYERS):
        for pos_idx in range(config.NUM_LATENT):
            all_combos.append((layer_idx, pos_idx, layer_pos_kl_means[layer_idx][pos_idx]))

    top_5_critical = sorted(all_combos, key=lambda x: x[2], reverse=True)[:5]

    print(f"\nTop 5 Critical (Layer, Position) Combinations by KL Divergence:")
    for layer, pos, kl in top_5_critical:
        ans_diff = layer_pos_ans_diff_means[layer][pos]
        print(f"  Layer {layer:2d}, Position {pos}: KL = {kl:.4f}, Ans Diff = {ans_diff:+.4f}")

    # Log aggregate statistics to W&B
    for layer_idx in range(config.NUM_LAYERS):
        for pos_idx in range(config.NUM_LATENT):
            wandb.log({
                f"aggregate/layer_{layer_idx}/pos_{pos_idx}/mean_kl": layer_pos_kl_means[layer_idx][pos_idx],
                f"aggregate/layer_{layer_idx}/pos_{pos_idx}/mean_ans_diff": layer_pos_ans_diff_means[layer_idx][pos_idx],
            })

    # Save aggregate statistics
    aggregate_stats = {
        'layer_position_kl_means': layer_pos_kl_means,
        'layer_position_ans_diff_means': layer_pos_ans_diff_means,
        'top_5_critical': [{'layer': l, 'position': p, 'kl_div': k, 'ans_diff': layer_pos_ans_diff_means[l][p]}
                          for l, p, k in top_5_critical]
    }

    aggregate_path = os.path.join(config.RESULTS_DIR, "aggregate_statistics.json")
    with open(aggregate_path, 'w') as f:
        json.dump(aggregate_stats, f, indent=2)

    print(f"\n✓ Saved aggregate statistics to: {aggregate_path}")

    print("\n" + "="*80)
    print("✓ POSITION-WISE PATCHING EXPERIMENT COMPLETE")
    print("="*80 + "\n")

    wandb.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
