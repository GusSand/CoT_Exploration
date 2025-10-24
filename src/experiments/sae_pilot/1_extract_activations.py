"""
Story 1.1: Extract Activations for SAE Training

This script converts pre-extracted continuous thought activations from the
operation circuits experiment into the format needed for SAE training.

Reuses existing activations from L4 (early), L8 (middle), L14 (late)
to save ~90 minutes of GPU extraction time.

Input: src/experiments/operation_circuits/results/continuous_thoughts_full_600.json
Output: src/experiments/sae_pilot/data/sae_training_activations.pt

Shape: [10,800, 2048] = 600 problems × 3 layers × 6 tokens × 2048 hidden_dim
"""

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime


def load_continuous_thoughts(input_path: str):
    """Load pre-extracted continuous thoughts from JSON."""
    print(f"Loading continuous thoughts from {input_path}...")

    with open(input_path, 'r') as f:
        data = json.load(f)

    print(f"  Loaded {len(data)} problems")

    return data


def convert_to_sae_format(data: list, output_path: str):
    """
    Convert continuous thoughts to SAE training format.

    Args:
        data: List of dicts with 'thoughts' key
        output_path: Path to save tensor

    Returns:
        Dict with activations tensor and metadata
    """
    print("\nConverting to SAE format...")

    # Collect all activations
    activations_list = []
    metadata = {
        'problems': [],
        'layers': [],
        'tokens': [],
        'operation_types': []
    }

    layer_names = ['early', 'middle', 'late']  # L4, L8, L14
    layer_mapping = {'early': 4, 'middle': 8, 'late': 14}

    for problem_idx, problem in enumerate(data):
        thoughts = problem['thoughts']

        for layer_name in layer_names:
            layer_thoughts = thoughts[layer_name]  # List of 6 tokens

            for token_idx, token_vector in enumerate(layer_thoughts):
                # Append activation vector
                activations_list.append(token_vector)

                # Track metadata
                metadata['problems'].append(problem_idx)
                metadata['layers'].append(layer_mapping[layer_name])
                metadata['tokens'].append(token_idx)
                metadata['operation_types'].append(problem['operation_type'])

    # Convert to tensor
    activations_tensor = torch.tensor(activations_list, dtype=torch.float32)

    print(f"  Shape: {activations_tensor.shape}")
    print(f"  Total vectors: {len(activations_list)}")
    print(f"  Memory: {activations_tensor.element_size() * activations_tensor.nelement() / 1024**2:.1f} MB")

    # Save
    print(f"\nSaving to {output_path}...")
    output_data = {
        'activations': activations_tensor,
        'metadata': metadata,
        'info': {
            'num_problems': len(data),
            'num_layers': len(layer_names),
            'layer_names': layer_names,
            'layer_indices': layer_mapping,
            'num_tokens': 6,
            'hidden_dim': 2048,
            'total_vectors': activations_tensor.shape[0],
            'extraction_date': datetime.now().isoformat()
        }
    }

    torch.save(output_data, output_path)

    # Print statistics
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"Problems: {output_data['info']['num_problems']}")
    print(f"Layers: {output_data['info']['num_layers']} (L4, L8, L14)")
    print(f"Tokens per layer: {output_data['info']['num_tokens']}")
    print(f"Hidden dimension: {output_data['info']['hidden_dim']}")
    print(f"Total activation vectors: {output_data['info']['total_vectors']:,}")
    print(f"Tensor shape: {activations_tensor.shape}")
    print(f"File size: {activations_tensor.element_size() * activations_tensor.nelement() / 1024**2:.1f} MB")

    # Operation type distribution
    from collections import Counter
    op_counts = Counter(metadata['operation_types'])
    print("\nOperation type distribution (per vector):")
    for op_type, count in sorted(op_counts.items()):
        print(f"  {op_type}: {count:,} ({count/len(metadata['operation_types'])*100:.1f}%)")

    print("="*60)

    return output_data


def main():
    # Paths
    project_root = Path(__file__).parent.parent.parent.parent
    input_path = project_root / "src/experiments/operation_circuits/results/continuous_thoughts_full_600.json"
    output_path = project_root / "src/experiments/sae_pilot/data/sae_training_activations.pt"

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_continuous_thoughts(input_path)

    # Convert and save
    output_data = convert_to_sae_format(data, output_path)

    print(f"\n✅ Success! Activations ready for SAE training at:")
    print(f"   {output_path}")

    return output_data


if __name__ == "__main__":
    main()
