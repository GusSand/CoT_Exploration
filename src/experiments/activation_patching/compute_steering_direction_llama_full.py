#!/usr/bin/env python3
"""
Compute LLaMA Steering Direction

Computes steering direction as: direction = correct_mean - wrong_mean
For each of the 3 layers (early, middle, late).

Usage:
    python compute_steering_direction_llama.py
"""

import json
import torch
from pathlib import Path


def compute_direction_for_layer(layer_name: str, activations_dir: Path):
    """Compute steering direction for a specific layer.

    Args:
        layer_name: 'early', 'middle', or 'late'
        activations_dir: Path to steering_activations_llama_full directory

    Returns:
        torch.Tensor: [6, 2048] steering direction
    """
    layer_dir = activations_dir / layer_name

    # Load all correct activations
    correct_files = list(layer_dir.glob("*_train_correct.pt"))
    correct_activations = [torch.load(f) for f in correct_files]
    correct_stack = torch.stack(correct_activations)  # [N_correct, 6, 2048]

    # Load all wrong activations
    wrong_files = list(layer_dir.glob("*_train_wrong.pt"))
    wrong_activations = [torch.load(f) for f in wrong_files]
    wrong_stack = torch.stack(wrong_activations)  # [N_wrong, 6, 2048]

    # Compute means
    correct_mean = correct_stack.mean(dim=0)  # [6, 2048]
    wrong_mean = wrong_stack.mean(dim=0)      # [6, 2048]

    # Steering direction
    direction = correct_mean - wrong_mean  # [6, 2048]

    return direction, correct_mean, wrong_mean, len(correct_files), len(wrong_files)


def main():
    """Compute steering directions for all layers."""
    print("="*80)
    print("COMPUTE LLAMA STEERING DIRECTIONS")
    print("="*80)

    activations_dir = Path(__file__).parent / 'results' / 'steering_activations_llama_full'

    # Process all three layers
    for layer_name in ['early', 'middle', 'late']:
        print(f"\n{'='*80}")
        print(f"LAYER: {layer_name.upper()}")
        print("="*80)

        direction, correct_mean, wrong_mean, n_correct, n_wrong = compute_direction_for_layer(
            layer_name, activations_dir
        )

        # Compute statistics
        direction_norm = torch.norm(direction).item()
        correct_norm = torch.norm(correct_mean).item()
        wrong_norm = torch.norm(wrong_mean).item()

        print(f"\nSamples:")
        print(f"  Correct: {n_correct}")
        print(f"  Wrong:   {n_wrong}")

        print(f"\nActivation norms:")
        print(f"  Correct mean: {correct_norm:.4f}")
        print(f"  Wrong mean:   {wrong_norm:.4f}")

        print(f"\nSteering direction:")
        print(f"  Norm: {direction_norm:.4f}")
        print(f"  Shape: {direction.shape}")

        # Save
        output_dir = activations_dir / layer_name
        torch.save({
            'direction': direction,
            'correct_mean': correct_mean,
            'wrong_mean': wrong_mean,
            'n_correct': n_correct,
            'n_wrong': n_wrong,
            'direction_norm': direction_norm,
            'correct_norm': correct_norm,
            'wrong_norm': wrong_norm
        }, output_dir / 'steering_direction.pt')

        print(f"\n✓ Saved to {output_dir / 'steering_direction.pt'}")

    print("\n" + "="*80)
    print("✅ STEERING DIRECTIONS COMPUTED")
    print("="*80)
    print("\nDirection = correct_mean - wrong_mean")
    print("Shape: [6, 2048] for 6 latent tokens × 2048 hidden dim")


if __name__ == "__main__":
    main()
