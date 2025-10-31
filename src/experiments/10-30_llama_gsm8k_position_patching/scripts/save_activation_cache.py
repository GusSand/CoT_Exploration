"""
Save clean activations cache for reuse in Experiment 2

This script extracts and saves clean activations for all 57 pairs at all
layers and positions, creating a ~37 MB cache file that Experiment 2 can reuse.
"""

import json
import sys
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add experiment directory to path
exp_dir = Path(__file__).parent.parent
sys.path.insert(0, str(exp_dir))

import config
from core.model_loader import (
    load_codi_model,
    prepare_codi_input,
    extract_activations_at_layer
)


def load_prepared_pairs():
    """Load filtered prepared pairs (57 pairs)"""
    with open(config.DATA_PATH, 'r') as f:
        pairs = json.load(f)
    print(f"✓ Loaded {len(pairs)} filtered pairs")
    return pairs


def extract_and_save_clean_activations(model, tokenizer, pairs, num_layers, num_positions):
    """
    Extract clean activations for all pairs and save to cache

    Args:
        model: CODI model
        tokenizer: Tokenizer
        pairs: List of pair dictionaries
        num_layers: 16
        num_positions: 5

    Returns:
        cache_path: Path to saved cache file
    """
    clean_activations_cache = {}

    print(f"\nExtracting clean activations for {len(pairs)} pairs...")
    print(f"  {num_layers} layers × {num_positions} positions per pair")
    print(f"  Total extractions: {len(pairs) * num_layers}\n")

    for pair in tqdm(pairs, desc="Processing pairs"):
        pair_id = pair['pair_id']
        clean_question = pair['clean']['question']

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

        # Extract activations at each layer
        pair_acts = {}
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

            # Store each position separately as numpy arrays
            pair_acts[layer_idx] = {}
            for pos_idx in range(num_positions):
                # Convert to float32 first (BFloat16 not supported by numpy)
                pair_acts[layer_idx][pos_idx] = acts[:, pos_idx, :].float().cpu().numpy()

        clean_activations_cache[pair_id] = pair_acts

        # Clear cache periodically
        if (pair_id + 1) % 10 == 0:
            torch.cuda.empty_cache()

    # Save cache
    cache_data = {
        'clean_activations': clean_activations_cache,
        'metadata': {
            'num_pairs': len(pairs),
            'num_layers': num_layers,
            'num_positions': num_positions,
            'hidden_dim': config.HIDDEN_DIM,
            'model': config.MODEL_NAME,
            'checkpoint': config.CHECKPOINT_PATH
        }
    }

    cache_path = os.path.join(config.RESULTS_DIR, "clean_activations_cache.pt")
    torch.save(cache_data, cache_path)

    # Calculate file size
    file_size_mb = os.path.getsize(cache_path) / (1024 * 1024)

    print(f"\n✓ Saved activation cache to: {cache_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Structure: {len(pairs)} pairs × {num_layers} layers × {num_positions} positions")

    return cache_path


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("SAVE CLEAN ACTIVATIONS CACHE FOR EXPERIMENT 2")
    print("="*80 + "\n")

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

    # Extract and save activations
    cache_path = extract_and_save_clean_activations(
        model, tokenizer, pairs,
        config.NUM_LAYERS, config.NUM_LATENT
    )

    print("\n" + "="*80)
    print("✓ ACTIVATION CACHE SAVED SUCCESSFULLY")
    print("="*80 + "\n")

    print("This cache will be reused by Experiment 2 (iterative patching)")
    print("to avoid re-extracting clean activations.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
