"""
Create CoT Token Alignment Dataset for Tuned Lens Training.

This script transforms the enriched CoT data into a format suitable for training
Tuned Lens to decode continuous thought positions into their corresponding CoT tokens.

Approach (Uniform Split - Option A):
- Each problem has ~N CoT tokens and 6 continuous thought positions
- Uniformly distribute CoT tokens across the 6 positions
- Position i gets tokens from slice [i*N//6 : (i+1)*N//6]

Example:
  CoT tokens: ['322', '=', '322', '2', '*', '322', '=', '644', '644', '-', '19', '=', '625']
  Position 0: ['322', '=']
  Position 1: ['322', '2']
  Position 2: ['*', '322']
  Position 3: ['=', '644']
  Position 4: ['644', '-']
  Position 5: ['19', '=', '625']
"""

import torch
from pathlib import Path
from typing import Dict, List
import numpy as np
from transformers import AutoTokenizer

# Paths
ENRICHED_DATA_PATH = "/home/paperspace/dev/CoT_Exploration/src/experiments/sae_cot_decoder/results"
OUTPUT_PATH = Path("data")
OUTPUT_PATH.mkdir(exist_ok=True)

# Layer to focus on (layer 15 = final layer)
TARGET_LAYER = 15

# Model for tokenizer
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


def uniform_split_cot_tokens(cot_token_ids: List[int], num_positions: int = 6) -> List[List[int]]:
    """
    Uniformly split CoT tokens across continuous thought positions.

    Args:
        cot_token_ids: List of token IDs from the CoT sequence
        num_positions: Number of CT positions (default 6)

    Returns:
        List of token ID lists, one per CT position
    """
    n = len(cot_token_ids)
    assignments = []

    for pos in range(num_positions):
        start_idx = (pos * n) // num_positions
        end_idx = ((pos + 1) * n) // num_positions

        # Get assigned tokens for this position
        assigned_tokens = cot_token_ids[start_idx:end_idx]

        # Handle edge case: if no tokens assigned, use padding token
        if len(assigned_tokens) == 0:
            assigned_tokens = [0]  # Use 0 as padding

        assignments.append(assigned_tokens)

    return assignments


def create_alignment_dataset(
    data_path: str,
    output_name: str,
    all_layers: bool = True
):
    """
    Create CoT alignment dataset from enriched data.

    Args:
        data_path: Path to enriched_train/test_data_with_cot.pt
        output_name: Name for output file (e.g., 'cot_train')
        all_layers: If True, include all layers; if False, use TARGET_LAYER only
    """
    print(f"\n{'='*70}")
    print(f"Creating CoT Alignment Dataset: {output_name}")
    print(f"{'='*70}")

    # Load enriched data
    print(f"\nLoading data from: {data_path}")
    data = torch.load(data_path, weights_only=False)

    print(f"Total samples: {len(data['hidden_states'])}")

    if all_layers:
        print("Using ALL 16 layers (0-15)...")
        # Use all samples
        filtered_hidden_states = data['hidden_states']
        filtered_layers = data['metadata']['layers']
        filtered_positions = data['metadata']['positions']
        filtered_problem_ids = data['metadata']['problem_ids']
        filtered_difficulties = data['metadata']['difficulties']
        filtered_cot_token_ids = data['metadata']['cot_token_ids']
        filtered_cot_steps = data['metadata']['cot_steps']
    else:
        print(f"Filtering to layer {TARGET_LAYER}...")
        # Filter to target layer only
        layers = data['metadata']['layers']
        layer_indices = [i for i, l in enumerate(layers) if l == TARGET_LAYER]

        print(f"Samples in layer {TARGET_LAYER}: {len(layer_indices)}")

        # Extract layer-specific data
        filtered_hidden_states = data['hidden_states'][torch.tensor(layer_indices)]
        filtered_layers = [data['metadata']['layers'][i] for i in layer_indices]
        filtered_positions = [data['metadata']['positions'][i] for i in layer_indices]
        filtered_problem_ids = [data['metadata']['problem_ids'][i] for i in layer_indices]
        filtered_difficulties = [data['metadata']['difficulties'][i] for i in layer_indices]
        filtered_cot_token_ids = [data['metadata']['cot_token_ids'][i] for i in layer_indices]
        filtered_cot_steps = [data['metadata']['cot_steps'][i] for i in layer_indices]

    # Assign CoT tokens to each CT position
    print("\nAssigning CoT tokens to continuous thought positions...")

    cot_targets = []  # Will store assigned token IDs for each sample
    cot_target_counts = []  # Track how many tokens per sample

    for i, (pos, cot_tokens) in enumerate(zip(filtered_positions, filtered_cot_token_ids)):
        # Split CoT tokens across 6 positions
        position_assignments = uniform_split_cot_tokens(cot_tokens, num_positions=6)

        # Get tokens for this specific position
        assigned_tokens = position_assignments[pos]

        cot_targets.append(assigned_tokens)
        cot_target_counts.append(len(assigned_tokens))

        # Show first few examples
        if i < 3:
            print(f"\nSample {i}:")
            print(f"  Position: {pos}")
            print(f"  CoT steps: {filtered_cot_steps[i]}")
            print(f"  Total CoT tokens: {len(cot_tokens)}")
            print(f"  Assigned to pos {pos}: {assigned_tokens} ({len(assigned_tokens)} tokens)")

    # Statistics
    print(f"\n{'='*70}")
    print("Dataset Statistics:")
    print(f"{'='*70}")
    print(f"Total samples: {len(filtered_hidden_states)}")

    if all_layers:
        print(f"\nSamples per layer:")
        for layer in range(16):
            count = sum(1 for l in filtered_layers if l == layer)
            print(f"  Layer {layer:2d}: {count:5d} samples")

    print(f"\nSamples per position:")
    for pos in range(6):
        count = sum(1 for p in filtered_positions if p == pos)
        print(f"  Position {pos}: {count} samples")

    print(f"\nCoT tokens per sample:")
    print(f"  Mean: {np.mean(cot_target_counts):.2f}")
    print(f"  Min: {np.min(cot_target_counts)}")
    print(f"  Max: {np.max(cot_target_counts)}")
    print(f"  Median: {np.median(cot_target_counts):.2f}")

    # Save dataset
    if all_layers:
        output_file = OUTPUT_PATH / f"{output_name}_all_layers.pt"
    else:
        output_file = OUTPUT_PATH / f"{output_name}_layer{TARGET_LAYER}.pt"

    aligned_data = {
        'hidden_states': filtered_hidden_states,
        'cot_target_token_ids': cot_targets,  # List of lists
        'metadata': {
            'problem_ids': filtered_problem_ids,
            'layers': filtered_layers,
            'positions': filtered_positions,
            'difficulties': filtered_difficulties,
            'cot_token_ids': filtered_cot_token_ids,
            'cot_steps': filtered_cot_steps,
            'num_cot_tokens': cot_target_counts,
        },
        'config': {
            'all_layers': all_layers,
            'layer_range': [0, 15] if all_layers else [TARGET_LAYER, TARGET_LAYER],
            'num_positions': 6,
            'split_method': 'uniform',
            'source_file': str(data_path),
        }
    }

    torch.save(aligned_data, output_file)
    print(f"\nSaved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

    return aligned_data


def main():
    """Create both train and test CoT alignment datasets."""

    # Create train dataset (ALL LAYERS)
    train_data = create_alignment_dataset(
        data_path=f"{ENRICHED_DATA_PATH}/enriched_train_data_with_cot.pt",
        output_name="cot_train",
        all_layers=True
    )

    # Create test dataset (ALL LAYERS)
    test_data = create_alignment_dataset(
        data_path=f"{ENRICHED_DATA_PATH}/enriched_test_data_with_cot.pt",
        output_name="cot_test",
        all_layers=True
    )

    print(f"\n{'='*70}")
    print("CoT Alignment Datasets Created!")
    print(f"{'='*70}")
    print(f"Train samples: {len(train_data['hidden_states'])}")
    print(f"Test samples: {len(test_data['hidden_states'])}")
    print(f"Layers included: 0-15 (all 16 layers)")
    print(f"\nNext step: Train Tuned Lens with these CoT targets")
    print(f"  python train_cot_alignment_all_layers.py")


if __name__ == "__main__":
    main()
