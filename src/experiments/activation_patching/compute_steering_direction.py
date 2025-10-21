#!/usr/bin/env python3
"""
Compute Reasoning Direction for Steering

This script:
1. Loads CORRECT and WRONG activations
2. Computes mean activations for each group
3. Computes reasoning direction = correct_mean - wrong_mean
4. Saves direction for use in steering experiments
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
ACTIVATIONS_DIR = BASE_DIR / "results" / "steering_activations"
OUTPUT_DIR = ACTIVATIONS_DIR
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_activations():
    """Load CORRECT and WRONG activations."""
    print("="*80)
    print("LOADING ACTIVATIONS")
    print("="*80)

    correct_file = ACTIVATIONS_DIR / "correct_activations_middle.npz"
    wrong_file = ACTIVATIONS_DIR / "wrong_activations_middle.npz"

    print(f"\nLoading CORRECT from: {correct_file}")
    correct_data = np.load(correct_file)
    correct_activations = correct_data['activations']
    correct_ids = correct_data['pair_ids']

    print(f"Loading WRONG from: {wrong_file}")
    wrong_data = np.load(wrong_file)
    wrong_activations = wrong_data['activations']
    wrong_ids = wrong_data['pair_ids']

    print(f"\nCORRECT shape: {correct_activations.shape}")
    print(f"WRONG shape: {wrong_activations.shape}")
    print(f"CORRECT IDs: {len(correct_ids)}")
    print(f"WRONG IDs: {len(wrong_ids)}")

    return correct_activations, wrong_activations, correct_ids, wrong_ids


def compute_direction(correct_activations, wrong_activations):
    """Compute reasoning direction."""
    print("\n" + "="*80)
    print("COMPUTING REASONING DIRECTION")
    print("="*80)

    # Compute means
    correct_mean = np.mean(correct_activations, axis=0)  # Shape: [6, 768]
    wrong_mean = np.mean(wrong_activations, axis=0)      # Shape: [6, 768]

    print(f"\nCorrect mean shape: {correct_mean.shape}")
    print(f"Wrong mean shape: {wrong_mean.shape}")

    # Compute direction
    direction = correct_mean - wrong_mean  # Shape: [6, 768]

    print(f"Direction shape: {direction.shape}")
    print(f"Direction magnitude per token:")
    for i in range(6):
        mag = np.linalg.norm(direction[i])
        print(f"  Token {i}: {mag:.4f}")

    # Overall magnitude
    total_mag = np.linalg.norm(direction)
    print(f"\nTotal direction magnitude: {total_mag:.4f}")

    # Normalize to unit vector (optional, for visualization)
    direction_normalized = direction / (total_mag + 1e-8)

    return direction, direction_normalized, correct_mean, wrong_mean


def visualize_direction(direction, correct_mean, wrong_mean):
    """Visualize the reasoning direction."""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    # Figure 1: Per-token direction magnitude
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Token-wise magnitude
    ax = axes[0]
    token_mags = [np.linalg.norm(direction[i]) for i in range(6)]
    ax.bar(range(6), token_mags, color='#3498db', alpha=0.8)
    ax.set_xlabel('Latent Token Position', fontsize=12)
    ax.set_ylabel('Direction Magnitude', fontsize=12)
    ax.set_title('Reasoning Direction Strength by Token', fontsize=13, fontweight='bold')
    ax.set_xticks(range(6))
    ax.grid(alpha=0.3, axis='y')

    # Correct vs Wrong mean magnitudes
    ax = axes[1]
    correct_mags = [np.linalg.norm(correct_mean[i]) for i in range(6)]
    wrong_mags = [np.linalg.norm(wrong_mean[i]) for i in range(6)]

    x = np.arange(6)
    width = 0.35
    ax.bar(x - width/2, correct_mags, width, label='CORRECT', color='#2ecc71', alpha=0.8)
    ax.bar(x + width/2, wrong_mags, width, label='WRONG', color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Latent Token Position', fontsize=12)
    ax.set_ylabel('Mean Activation Magnitude', fontsize=12)
    ax.set_title('Activation Magnitude: CORRECT vs WRONG', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    fig_file = FIGURES_DIR / 'reasoning_direction_magnitude.png'
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {fig_file}")

    # Figure 2: Direction heatmap
    fig, ax = plt.subplots(figsize=(10, 4))

    # Show first 50 dimensions for visualization
    direction_sample = direction[:, :50]

    im = ax.imshow(direction_sample, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
    ax.set_xlabel('Hidden Dimension (first 50 of 768)', fontsize=12)
    ax.set_ylabel('Latent Token Position', fontsize=12)
    ax.set_title('Reasoning Direction Heatmap (sample)', fontsize=13, fontweight='bold')
    ax.set_yticks(range(6))
    plt.colorbar(im, ax=ax, label='Direction Value')

    plt.tight_layout()
    fig_file = FIGURES_DIR / 'reasoning_direction_heatmap.png'
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {fig_file}")

    print(f"\nAll visualizations saved to: {FIGURES_DIR}")


def save_direction(direction, direction_normalized, correct_mean, wrong_mean, correct_ids, wrong_ids):
    """Save reasoning direction and metadata."""
    print("\n" + "="*80)
    print("SAVING REASONING DIRECTION")
    print("="*80)

    # Save direction as .npz
    direction_file = OUTPUT_DIR / 'reasoning_direction.npz'
    np.savez_compressed(
        direction_file,
        direction=direction,
        direction_normalized=direction_normalized,
        correct_mean=correct_mean,
        wrong_mean=wrong_mean
    )
    print(f"✓ Saved direction: {direction_file}")

    # Save metadata
    metadata = {
        'num_correct_problems': len(correct_ids),
        'num_wrong_problems': len(wrong_ids),
        'direction_shape': list(direction.shape),
        'total_magnitude': float(np.linalg.norm(direction)),
        'token_magnitudes': {
            f'token_{i}': float(np.linalg.norm(direction[i]))
            for i in range(6)
        },
        'layer_name': 'middle',
        'hidden_dim': 768
    }

    metadata_file = OUTPUT_DIR / 'reasoning_direction_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_file}")

    return metadata


def main():
    """Main pipeline."""
    print("="*80)
    print("COMPUTING STEERING DIRECTION")
    print("="*80)

    # Load activations
    correct_activations, wrong_activations, correct_ids, wrong_ids = load_activations()

    # Compute direction
    direction, direction_normalized, correct_mean, wrong_mean = compute_direction(
        correct_activations, wrong_activations
    )

    # Visualize
    visualize_direction(direction, correct_mean, wrong_mean)

    # Save
    metadata = save_direction(
        direction, direction_normalized, correct_mean, wrong_mean,
        correct_ids, wrong_ids
    )

    # Print summary
    print("\n" + "="*80)
    print("✅ REASONING DIRECTION COMPUTED")
    print("="*80)
    print(f"✅ Based on {metadata['num_correct_problems']} CORRECT + {metadata['num_wrong_problems']} WRONG problems")
    print(f"✅ Direction shape: {metadata['direction_shape']}")
    print(f"✅ Total magnitude: {metadata['total_magnitude']:.4f}")
    print(f"✅ Saved to: {OUTPUT_DIR}")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Run baseline test set evaluation")
    print("2. Test steering with amplification (alpha > 0)")
    print("3. Test steering with suppression (alpha < 0)")
    print("="*80)


if __name__ == "__main__":
    main()
