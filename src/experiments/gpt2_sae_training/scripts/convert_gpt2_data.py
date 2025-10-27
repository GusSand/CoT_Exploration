"""
Convert GPT-2 shared data (JSON) to activation tensors (PT) for SAE training.

Reads: gpt2_predictions_1000.json (1.7GB JSON)
Writes:
  - gpt2_full_train_activations.pt (~800 samples)
  - gpt2_full_val_activations.pt (~200 samples)
"""

import json
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def convert_gpt2_data():
    """Convert GPT-2 JSON data to PT format."""

    print("="*80)
    print("CONVERTING GPT-2 DATA TO PT FORMAT")
    print("="*80)

    # Paths
    json_path = Path("src/experiments/gpt2_shared_data/gpt2_predictions_1000_checkpoint_1000.json")
    output_dir = Path("src/experiments/gpt2_sae_training/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load JSON data
    print(f"\n[1/3] Loading JSON data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    metadata = data['metadata']
    samples = data['samples']

    print(f"  Loaded {len(samples)} samples")
    print(f"  Layers: {metadata['n_layers']}")
    print(f"  Tokens: {metadata['n_tokens']}")
    print(f"  Hidden dim: {metadata['hidden_dim']}")

    # Extract all activations
    print(f"\n[2/3] Converting to tensors...")
    all_activations = []
    all_metadata = {
        'problem_ids': [],
        'layers': [],
        'positions': [],
    }

    for sample in samples:
        problem_id = sample['id']
        thoughts = sample['thoughts']

        # Extract activations for all layers and positions
        for layer_idx in range(12):
            layer_key = f'layer_{layer_idx}'
            layer_activations = thoughts[layer_key]  # List of 6 tokens

            for position_idx, activation in enumerate(layer_activations):
                # Convert to tensor
                act_tensor = torch.tensor(activation, dtype=torch.float32)
                all_activations.append(act_tensor)

                # Store metadata
                all_metadata['problem_ids'].append(problem_id)
                all_metadata['layers'].append(layer_idx)
                all_metadata['positions'].append(position_idx)

    # Stack into single tensor
    activations_tensor = torch.stack(all_activations)

    print(f"  Total samples: {len(activations_tensor):,}")
    print(f"  Shape: {activations_tensor.shape}")
    print(f"  Expected: {len(samples) * 12 * 6} = {len(samples)} problems × 12 layers × 6 tokens")

    # Split into train/test (80/20)
    print(f"\n[3/3] Splitting into train/test (80/20)...")

    # Get unique problem IDs
    unique_problem_ids = list(range(len(samples)))
    train_ids, test_ids = train_test_split(
        unique_problem_ids,
        train_size=0.8,
        random_state=42,
        shuffle=True
    )

    print(f"  Train problems: {len(train_ids)}")
    print(f"  Test problems: {len(test_ids)}")

    # Create masks for train/test
    problem_ids_array = np.array(all_metadata['problem_ids'])
    train_mask = np.isin(problem_ids_array, train_ids)
    test_mask = np.isin(problem_ids_array, test_ids)

    # Split data
    train_activations = activations_tensor[train_mask]
    test_activations = activations_tensor[test_mask]

    train_metadata = {
        'problem_ids': [pid for pid, m in zip(all_metadata['problem_ids'], train_mask) if m],
        'layers': [l for l, m in zip(all_metadata['layers'], train_mask) if m],
        'positions': [p for p, m in zip(all_metadata['positions'], train_mask) if m],
    }

    test_metadata = {
        'problem_ids': [pid for pid, m in zip(all_metadata['problem_ids'], test_mask) if m],
        'layers': [l for l, m in zip(all_metadata['layers'], test_mask) if m],
        'positions': [p for p, m in zip(all_metadata['positions'], test_mask) if m],
    }

    print(f"  Train samples: {len(train_activations):,}")
    print(f"  Test samples: {len(test_activations):,}")

    # Save train data
    train_data = {
        'activations': train_activations,
        'metadata': train_metadata,
        'config': {
            'model': 'gpt2',
            'num_problems': len(train_ids),
            'num_layers': 12,
            'num_ct_tokens': 6,
            'hidden_size': 768
        }
    }

    train_path = output_dir / "gpt2_full_train_activations.pt"
    print(f"\n  Saving train data to: {train_path}")
    torch.save(train_data, train_path)
    train_size_mb = train_path.stat().st_size / 1e6
    print(f"  ✓ Saved ({train_size_mb:.1f} MB)")

    # Save test data
    test_data = {
        'activations': test_activations,
        'metadata': test_metadata,
        'config': {
            'model': 'gpt2',
            'num_problems': len(test_ids),
            'num_layers': 12,
            'num_ct_tokens': 6,
            'hidden_size': 768
        }
    }

    test_path = output_dir / "gpt2_full_val_activations.pt"
    print(f"\n  Saving test data to: {test_path}")
    torch.save(test_data, test_path)
    test_size_mb = test_path.stat().st_size / 1e6
    print(f"  ✓ Saved ({test_size_mb:.1f} MB)")

    # Summary
    print("\n" + "="*80)
    print("CONVERSION COMPLETE!")
    print("="*80)
    print(f"  Train: {train_activations.shape}")
    print(f"  Test: {test_activations.shape}")
    print(f"  Output directory: {output_dir}")
    print("="*80)

    return train_data, test_data


if __name__ == '__main__':
    convert_gpt2_data()
