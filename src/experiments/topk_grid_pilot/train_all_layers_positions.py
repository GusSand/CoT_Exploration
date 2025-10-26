"""
Train TopK SAE Grid across multiple layers and positions.

Strategy:
1. Layer 14, all positions (0-5)
2. Layers 10-15, all positions (0-5)

Total: 6 layers × 6 positions × 12 SAEs = 432 SAEs

Parallel execution:
- 3 processes per (layer, position): one per latent_dim
- Estimated time: ~2-3 hours for all 432 SAEs

Usage:
    # Train all
    python train_all_layers_positions.py

    # Train specific layer
    python train_all_layers_positions.py --layer 14

    # Train specific position
    python train_all_layers_positions.py --position 3
"""

import argparse
import subprocess
import time
from pathlib import Path


def train_layer_position(layer, position, dry_run=False):
    """
    Train 12 SAEs for a specific (layer, position) pair.

    Uses parallel training: 3 processes (one per latent_dim).
    """
    print(f"\n{'=' * 80}")
    print(f"Training Layer {layer}, Position {position} (12 SAEs)")
    print(f"{'=' * 80}\n")

    if dry_run:
        print("  [DRY RUN] Would train 12 SAEs in parallel...")
        return

    # Launch 3 parallel processes (one per latent_dim)
    processes = []
    for latent_dim in [512, 1024, 2048]:
        cmd = [
            'python',
            'src/experiments/topk_grid_pilot/train_grid.py',
            '--latent_dim', str(latent_dim),
            '--position', str(position),
            '--layer', str(layer),
        ]

        print(f"  Launching: latent_dim={latent_dim}")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append((latent_dim, proc))

    # Wait for all to complete
    print(f"\n  Waiting for 3 processes to complete...")
    start_time = time.time()

    for latent_dim, proc in processes:
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            print(f"\n  ERROR: latent_dim={latent_dim} failed!")
            print(f"  STDERR: {stderr}")
        else:
            print(f"  ✓ latent_dim={latent_dim} complete")

    elapsed = time.time() - start_time
    print(f"\n  Layer {layer}, Position {position} complete in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=None,
                        help='Train only this layer (default: all 10-15)')
    parser.add_argument('--position', type=int, default=None,
                        help='Train only this position (default: all 0-5)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print what would be trained without training')
    args = parser.parse_args()

    # Determine layers and positions to train
    if args.layer is not None:
        layers = [args.layer]
    else:
        layers = list(range(16))  # All layers 0-15

    if args.position is not None:
        positions = [args.position]
    else:
        positions = [0, 1, 2, 3, 4, 5]

    # Count total SAEs
    num_layer_position_pairs = len(layers) * len(positions)
    num_saes = num_layer_position_pairs * 12

    print("=" * 80)
    print("TopK SAE Multi-Layer Multi-Position Training")
    print("=" * 80)
    print(f"  Layers: {layers}")
    print(f"  Positions: {positions}")
    print(f"  Layer-Position pairs: {num_layer_position_pairs}")
    print(f"  Total SAEs: {num_saes}")
    print(f"  Estimated time: {num_layer_position_pairs * 0.5:.1f} - {num_layer_position_pairs * 1:.1f} minutes")
    print("=" * 80)

    if args.dry_run:
        print("\n[DRY RUN MODE] - No training will occur\n")

    # Train each (layer, position) pair
    overall_start = time.time()
    completed = 0

    for layer in layers:
        for position in positions:
            # Check if already trained
            results_dir = Path('src/experiments/topk_grid_pilot/results')
            existing_files = list(results_dir.glob(f'pos{position}_layer{layer}_d*_k*.pt'))

            if len(existing_files) == 12:
                print(f"\n✓ Layer {layer}, Position {position} already trained (12 SAEs found)")
                completed += 1
                continue

            train_layer_position(layer, position, dry_run=args.dry_run)
            completed += 1

            # Progress update
            elapsed = time.time() - overall_start
            avg_time = elapsed / completed
            remaining = (num_layer_position_pairs - completed) * avg_time

            print(f"\n  Progress: {completed}/{num_layer_position_pairs} pairs complete")
            print(f"  Elapsed: {elapsed/60:.1f} min, Remaining: {remaining/60:.1f} min")

    total_time = time.time() - overall_start
    print("\n" + "=" * 80)
    print(f"All training complete!")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  SAEs trained: {num_saes}")
    print("=" * 80)


if __name__ == '__main__':
    main()
