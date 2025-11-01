"""
Main script for Experiment 2: Iterative CoT Activation Patching

This experiment tests whether CoT positions have sequential dependencies by:
1. Patching positions iteratively (0→1→2→3→4) with activation chaining
2. Comparing to parallel patching (all at once)
3. Measuring activation similarity between generated and clean activations
"""

import json
import sys
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import wandb

# Add experiment directory to path
exp_dir = Path(__file__).parent.parent
sys.path.insert(0, str(exp_dir))

import config
from core.model_loader import (
    load_codi_model,
    prepare_codi_input
)
from core.iterative_patcher import IterativePatcher, ParallelPatcher
from core.metrics import (
    compute_kl_divergence,
    compute_logit_difference,
    compute_prediction_change,
    compute_answer_logit_difference,
    compute_activation_similarity,
    tokenize_answer
)


def load_prepared_pairs():
    """Load filtered prepared pairs (57 pairs)"""
    with open(config.DATA_PATH, 'r') as f:
        pairs = json.load(f)
    print(f"✓ Loaded {len(pairs)} filtered pairs")
    return pairs


def load_activation_cache():
    """Load clean activations cache from Experiment 1"""
    cache_data = torch.load(config.ACTIVATION_CACHE_PATH, map_location='cpu')
    clean_activations = cache_data['clean_activations']
    metadata = cache_data['metadata']

    print(f"✓ Loaded activation cache")
    print(f"  {metadata['num_pairs']} pairs × {metadata['num_layers']} layers × {metadata['num_positions']} positions")
    print(f"  Model: {metadata['model']}")

    return clean_activations, metadata


def run_baseline_corrupted(model, tokenizer, corrupted_question, num_positions):
    """
    Run baseline with corrupted input (no patching)

    Returns:
        baseline_logits: Logits from corrupted run
        input_ids: Prepared input IDs
        attention_mask: Attention mask
        cot_positions: CoT token positions
    """
    # Prepare input
    input_ids, cot_positions = prepare_codi_input(
        corrupted_question,
        tokenizer,
        bot_id=model.bot_id,
        eot_id=model.eot_id,
        num_latent=num_positions,
        device=config.DEVICE
    )

    attention_mask = torch.ones_like(input_ids)

    # Forward pass
    with torch.no_grad():
        outputs = model.codi(input_ids=input_ids, attention_mask=attention_mask)
        baseline_logits = outputs.logits

    return baseline_logits, input_ids, attention_mask, cot_positions


def test_single_pair(
    model,
    tokenizer,
    pair,
    layer_idx,
    clean_activations_cache,
    num_positions
):
    """
    Test a single (clean, corrupted) pair with both iterative and parallel patching

    Args:
        model: CODI model
        tokenizer: Tokenizer
        pair: Pair dictionary
        layer_idx: Layer to patch
        clean_activations_cache: Cache of clean activations
        num_positions: Number of CoT positions

    Returns:
        result: Dictionary with all results for this pair at this layer
    """
    pair_id = pair['pair_id']
    clean_question = pair['clean']['question']
    corrupted_question = pair['corrupted']['question']
    clean_answer = pair['clean']['answer']
    corrupted_answer = pair['corrupted']['answer']

    # Tokenize answers
    clean_answer_tokens = tokenize_answer(clean_answer, tokenizer)
    corrupted_answer_tokens = tokenize_answer(corrupted_answer, tokenizer)

    # Get clean activations for this pair and layer
    pair_clean_acts = clean_activations_cache[pair_id][layer_idx]

    # Run baseline (corrupted, no patching)
    baseline_logits, input_ids, attention_mask, cot_positions = run_baseline_corrupted(
        model, tokenizer, corrupted_question, num_positions
    )

    # Answer position (after EoT token)
    answer_start_pos = cot_positions[-1] + 1

    # === 1. Iterative Patching ===
    iterative_patcher = IterativePatcher(
        model=model,
        layer_idx=layer_idx,
        clean_activations=pair_clean_acts,
        num_positions=num_positions
    )

    iter_results = iterative_patcher.run_iterative_patching(
        input_ids, attention_mask, cot_positions
    )
    iter_logits = iter_results['logits']
    iter_trajectory = iter_results['trajectory']
    iter_generated_acts = iter_results['generated_activations']

    # Compute metrics for iterative
    iter_kl, _ = compute_kl_divergence(iter_logits, baseline_logits)
    iter_l2 = compute_logit_difference(iter_logits, baseline_logits)
    _, iter_change_rate = compute_prediction_change(iter_logits, baseline_logits)
    iter_ans_diff, iter_clean_score, iter_corr_score = compute_answer_logit_difference(
        iter_logits, clean_answer_tokens, corrupted_answer_tokens, answer_start_pos
    )

    # Compute activation similarities
    # Compare generated activations to clean references
    activation_similarities = {}
    for pos_idx in range(1, num_positions):  # Positions 1-4 (we generate these)
        if pos_idx in iter_generated_acts:
            gen_act = iter_generated_acts[pos_idx]
            clean_act = pair_clean_acts[pos_idx]
            cos_sim, l2_dist = compute_activation_similarity(gen_act, clean_act)
            activation_similarities[pos_idx] = {
                'cosine_similarity': cos_sim,
                'l2_distance': l2_dist
            }

    # === 2. Parallel Patching ===
    parallel_patcher = ParallelPatcher(
        model=model,
        layer_idx=layer_idx,
        clean_activations=pair_clean_acts,
        num_positions=num_positions
    )

    parallel_logits = parallel_patcher.run_parallel_patching(
        input_ids, attention_mask, cot_positions
    )

    # Compute metrics for parallel
    parallel_kl, _ = compute_kl_divergence(parallel_logits, baseline_logits)
    parallel_l2 = compute_logit_difference(parallel_logits, baseline_logits)
    _, parallel_change_rate = compute_prediction_change(parallel_logits, baseline_logits)
    parallel_ans_diff, parallel_clean_score, parallel_corr_score = compute_answer_logit_difference(
        parallel_logits, clean_answer_tokens, corrupted_answer_tokens, answer_start_pos
    )

    # Package results
    result = {
        'pair_id': pair_id,
        'layer_idx': layer_idx,
        'iterative': {
            'kl_divergence': iter_kl,
            'l2_logit_diff': iter_l2,
            'prediction_change_rate': iter_change_rate,
            'answer_logit_diff': iter_ans_diff,
            'answer_clean_score': iter_clean_score,
            'answer_corrupted_score': iter_corr_score,
            'activation_similarities': activation_similarities
        },
        'parallel': {
            'kl_divergence': parallel_kl,
            'l2_logit_diff': parallel_l2,
            'prediction_change_rate': parallel_change_rate,
            'answer_logit_diff': parallel_ans_diff,
            'answer_clean_score': parallel_clean_score,
            'answer_corrupted_score': parallel_corr_score
        },
        'baseline': {
            'clean_answer': clean_answer,
            'corrupted_answer': corrupted_answer
        }
    }

    return result


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("EXPERIMENT 2: ITERATIVE COT ACTIVATION PATCHING")
    print("="*80 + "\n")

    # Initialize W&B
    wandb.init(
        project=config.WANDB_PROJECT,
        entity=config.WANDB_ENTITY,
        name=config.EXPERIMENT_NAME,
        config={
            'model': config.MODEL_NAME,
            'num_layers': config.NUM_LAYERS,
            'num_positions': config.NUM_LATENT,
            'test_mode': config.TEST_MODE,
            'strategies': config.PATCH_STRATEGIES
        }
    )

    # Set random seed
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # Load model
    print("Loading CODI model...")
    model, tokenizer = load_codi_model(
        config.CHECKPOINT_PATH,
        config.MODEL_NAME,
        num_latent=config.NUM_LATENT,
        device=config.DEVICE
    )

    # Load prepared pairs
    print("\nLoading filtered pairs...")
    pairs = load_prepared_pairs()

    # Load activation cache
    print("\nLoading activation cache...")
    clean_activations_cache, metadata = load_activation_cache()

    # Filter to test subset if TEST_MODE
    if config.TEST_MODE:
        pairs = pairs[:config.TEST_SUBSET_SIZE]
        print(f"\n⚠ TEST_MODE: Using {len(pairs)} pairs")

    print(f"\nRunning experiment on {len(pairs)} pairs × {config.NUM_LAYERS} layers")
    print(f"Total tests: {len(pairs) * config.NUM_LAYERS}")

    # Run experiment
    all_results = []

    with tqdm(total=len(pairs) * config.NUM_LAYERS, desc="Testing pairs") as pbar:
        for pair in pairs:
            for layer_idx in range(config.NUM_LAYERS):
                result = test_single_pair(
                    model, tokenizer, pair, layer_idx,
                    clean_activations_cache, config.NUM_LATENT
                )

                all_results.append(result)

                # Log to W&B
                wandb.log({
                    'pair_id': result['pair_id'],
                    'layer': layer_idx,
                    'iter_kl_div': result['iterative']['kl_divergence'],
                    'iter_ans_diff': result['iterative']['answer_logit_diff'],
                    'parallel_kl_div': result['parallel']['kl_divergence'],
                    'parallel_ans_diff': result['parallel']['answer_logit_diff']
                })

                pbar.update(1)

            # Clear cache periodically
            if (pair['pair_id'] + 1) % 10 == 0:
                torch.cuda.empty_cache()

    # Save results
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(config.RESULTS_DIR, "iterative_patching_results.json")

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Saved results to: {results_path}")

    # Compute aggregate statistics
    aggregate_stats = compute_aggregate_statistics(all_results)

    stats_path = os.path.join(config.RESULTS_DIR, "aggregate_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(aggregate_stats, f, indent=2)

    print(f"✓ Saved statistics to: {stats_path}")

    # Log final summary to W&B
    wandb.log({
        'final/iter_mean_kl': aggregate_stats['iterative']['mean_kl_divergence'],
        'final/iter_mean_ans_diff': aggregate_stats['iterative']['mean_answer_logit_diff'],
        'final/parallel_mean_kl': aggregate_stats['parallel']['mean_kl_divergence'],
        'final/parallel_mean_ans_diff': aggregate_stats['parallel']['mean_answer_logit_diff']
    })

    wandb.finish()

    print("\n" + "="*80)
    print("✓ EXPERIMENT 2 COMPLETE")
    print("="*80 + "\n")

    return 0


def compute_aggregate_statistics(results):
    """Compute aggregate statistics across all results"""
    iter_kls = []
    iter_ans_diffs = []
    parallel_kls = []
    parallel_ans_diffs = []

    # Activation similarity statistics (by position and layer)
    act_sim_by_pos_layer = {}  # {pos_idx: {layer: [cosine_sims]}}

    for result in results:
        iter_kls.append(result['iterative']['kl_divergence'])
        iter_ans_diffs.append(result['iterative']['answer_logit_diff'])
        parallel_kls.append(result['parallel']['kl_divergence'])
        parallel_ans_diffs.append(result['parallel']['answer_logit_diff'])

        # Activation similarities
        layer_idx = result['layer_idx']
        for pos_idx, sim_data in result['iterative']['activation_similarities'].items():
            if pos_idx not in act_sim_by_pos_layer:
                act_sim_by_pos_layer[pos_idx] = {}
            if layer_idx not in act_sim_by_pos_layer[pos_idx]:
                act_sim_by_pos_layer[pos_idx][layer_idx] = []

            act_sim_by_pos_layer[pos_idx][layer_idx].append(sim_data['cosine_similarity'])

    # Compute means
    stats = {
        'iterative': {
            'mean_kl_divergence': float(np.mean(iter_kls)),
            'std_kl_divergence': float(np.std(iter_kls)),
            'mean_answer_logit_diff': float(np.mean(iter_ans_diffs)),
            'std_answer_logit_diff': float(np.std(iter_ans_diffs))
        },
        'parallel': {
            'mean_kl_divergence': float(np.mean(parallel_kls)),
            'std_kl_divergence': float(np.std(parallel_kls)),
            'mean_answer_logit_diff': float(np.mean(parallel_ans_diffs)),
            'std_answer_logit_diff': float(np.std(parallel_ans_diffs))
        },
        'activation_similarity_by_position_layer': {}
    }

    # Average activation similarity per (position, layer)
    for pos_idx, layer_data in act_sim_by_pos_layer.items():
        stats['activation_similarity_by_position_layer'][str(pos_idx)] = {}
        for layer_idx, sims in layer_data.items():
            stats['activation_similarity_by_position_layer'][str(pos_idx)][str(layer_idx)] = {
                'mean': float(np.mean(sims)),
                'std': float(np.std(sims))
            }

    return stats


if __name__ == "__main__":
    sys.exit(main())
