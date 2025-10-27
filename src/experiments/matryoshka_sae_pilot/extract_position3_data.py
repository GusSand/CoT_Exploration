"""
Extract Position 3 activation data for Matryoshka SAE training.

Extracts Position 3 continuous thought activations across all 16 layers
from the full training dataset.
"""

import torch
import numpy as np
from pathlib import Path
import json


def extract_position3_data(
    input_path: str,
    output_path: str,
    position: int = 3
):
    """Extract single position data from full activation dataset.

    Args:
        input_path: Path to full_train_activations.pt
        output_path: Path to save position-specific data
        position: Position index to extract (0-5)
    """
    print(f"Loading full training data from {input_path}...")
    data = torch.load(input_path)

    activations = data['activations']  # (573888, 2048)
    metadata = data['metadata']
    config = data['config']

    print(f"\nFull dataset:")
    print(f"  Total activations: {activations.shape}")
    print(f"  Problems: {config['num_problems']}")
    print(f"  Layers: {config['num_layers']}")
    print(f"  Positions: {config['num_ct_tokens']}")

    # Extract position indices
    positions = np.array(metadata['positions'])
    layers = np.array(metadata['layers'])
    problem_ids = metadata['problem_ids']

    # Filter to position 3
    position_mask = (positions == position)

    # Extract position 3 activations
    position_activations = activations[position_mask]
    position_layers = layers[position_mask]
    position_problem_ids = [problem_ids[i] for i in range(len(problem_ids)) if position_mask[i]]

    print(f"\nPosition {position} dataset:")
    print(f"  Activations: {position_activations.shape}")
    print(f"  Expected: {config['num_problems'] * config['num_layers']} = {config['num_problems']} problems × {config['num_layers']} layers")

    # Verify we got all layers for all problems
    assert position_activations.shape[0] == config['num_problems'] * config['num_layers'], \
        f"Expected {config['num_problems'] * config['num_layers']} vectors, got {position_activations.shape[0]}"

    # Create output data structure
    output_data = {
        'activations': position_activations,
        'metadata': {
            'position': position,
            'layers': position_layers.tolist(),
            'problem_ids': position_problem_ids,
            'num_problems': config['num_problems'],
            'num_layers': config['num_layers']
        },
        'config': {
            'position': position,
            'num_problems': config['num_problems'],
            'num_layers': config['num_layers'],
            'hidden_size': config['hidden_size']
        },
        'source': input_path
    }

    # Save
    print(f"\nSaving to {output_path}...")
    torch.save(output_data, output_path)

    # Calculate file size
    file_size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"Saved {file_size_mb:.1f} MB")

    return output_data


if __name__ == "__main__":
    # Paths
    base_dir = Path(__file__).parent
    input_path = base_dir.parent / "sae_cot_decoder/data/full_train_activations.pt"
    output_path = base_dir / "data/position_3_activations.pt"

    print("=" * 60)
    print("Extracting Position 3 Activation Data")
    print("=" * 60)

    # Extract position 3
    output_data = extract_position3_data(
        input_path=str(input_path),
        output_path=str(output_path),
        position=3
    )

    print("\n" + "=" * 60)
    print("✓ Position 3 data extracted successfully!")
    print("=" * 60)
    print(f"\nOutput: {output_path}")
    print(f"Vectors: {output_data['activations'].shape[0]:,}")
    print(f"Problems: {output_data['config']['num_problems']:,}")
    print(f"Layers: {output_data['config']['num_layers']}")
    print(f"Position: {output_data['config']['position']}")
