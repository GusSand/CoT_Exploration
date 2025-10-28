#!/usr/bin/env python3
"""
Attention Aggregator - Story 1.3

Aggregate attention patterns across problems and compute statistics.

Usage:
    python 3_aggregate_attention.py [--model MODEL]

Output:
    ../results/{model}/attention_patterns_avg.npy  # [L, H, 6, 6]
    ../results/{model}/attention_stats.json
"""
import json
import numpy as np
import argparse
from pathlib import Path


def aggregate_attention(model: str = 'llama') -> None:
    """
    Aggregate attention patterns across all problems.

    Args:
        model: Model name ('llama' or 'gpt2')
    """
    print("=" * 80)
    print("ATTENTION AGGREGATOR - Story 1.3")
    print("=" * 80)

    # Load raw attention patterns
    results_dir = Path(__file__).parent.parent / 'results' / model
    raw_path = results_dir / 'attention_patterns_raw.npy'
    metadata_path = results_dir / 'attention_metadata.json'

    print(f"\nLoading attention patterns from {raw_path}...")
    attention_raw = np.load(raw_path).astype(np.float32)  # Convert to float32 for computation
    print(f"✓ Loaded attention patterns")
    print(f"  Shape: {attention_raw.shape}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    n_problems, n_layers, n_heads, _, _ = attention_raw.shape
    print(f"  Problems: {n_problems}")
    print(f"  Layers: {n_layers}")
    print(f"  Heads: {n_heads}")

    # Compute mean across problems
    print("\nComputing mean attention across problems...")
    attention_avg = np.mean(attention_raw, axis=0)  # [L, H, 6, 6]
    print(f"✓ Mean attention computed: {attention_avg.shape}")

    # Compute standard deviation to check consistency
    print("\nComputing standard deviation (consistency check)...")
    attention_std = np.std(attention_raw, axis=0)  # [L, H, 6, 6]
    mean_std = attention_std.mean()
    print(f"✓ Standard deviation computed")
    print(f"  Mean std across all positions: {mean_std:.4f}")

    if mean_std < 0.2:
        print(f"  ✓ Patterns are consistent (std < 0.2)")
    else:
        print(f"  ⚠️  High variance (std >= 0.2) - patterns may be unstable")

    # Identify top 20 heads by maximum attention weight
    print("\nIdentifying top 20 heads by attention strength...")
    max_attention_per_head = attention_avg.max(axis=(2, 3))  # [L, H]

    # Flatten to get all heads
    head_scores = []
    for layer in range(n_layers):
        for head in range(n_heads):
            head_scores.append({
                'layer': layer,
                'head': head,
                'max_attention': float(max_attention_per_head[layer, head]),
                'mean_attention': float(attention_avg[layer, head].mean()),
                'std_attention': float(attention_std[layer, head].mean())
            })

    # Sort by max attention
    head_scores.sort(key=lambda x: x['max_attention'], reverse=True)
    top_20_heads = head_scores[:20]

    print(f"✓ Top 20 heads identified")
    print(f"\n  Top 5 heads:")
    for i, head_info in enumerate(top_20_heads[:5]):
        print(f"    {i+1}. L{head_info['layer']}H{head_info['head']}: "
              f"max={head_info['max_attention']:.3f}, "
              f"mean={head_info['mean_attention']:.3f}")

    # Compute per-position statistics
    print("\nComputing per-position statistics...")
    position_stats = []
    for pos in range(6):
        # Incoming attention to this position (across all layers/heads)
        # Sum across source positions (axis=3) to get total attention TO each destination
        incoming = attention_avg[:, :, :, pos].sum(axis=2).mean()  # Average across layers/heads

        # Outgoing attention from this position
        outgoing = attention_avg[:, :, pos, :].sum(axis=2).mean()

        position_stats.append({
            'position': pos,
            'avg_incoming_attention': float(incoming),
            'avg_outgoing_attention': float(outgoing)
        })

    print(f"✓ Per-position statistics computed")

    # Save results
    print("\nSaving aggregated attention...")
    avg_path = results_dir / 'attention_patterns_avg.npy'
    np.save(avg_path, attention_avg.astype(np.float16))  # Save as float16 to reduce size
    print(f"✓ Saved: {avg_path}")
    print(f"  Shape: {attention_avg.shape}")
    print(f"  Size: {avg_path.stat().st_size / 1024:.1f} KB")

    # Save statistics
    stats = {
        'model': model,
        'n_problems': n_problems,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'consistency': {
            'mean_std': float(mean_std),
            'max_std': float(attention_std.max()),
            'min_std': float(attention_std.min()),
            'consistent': bool(mean_std < 0.2)
        },
        'top_20_heads': top_20_heads,
        'position_stats': position_stats,
        'overall_stats': {
            'mean_attention': float(attention_avg.mean()),
            'max_attention': float(attention_avg.max()),
            'min_attention': float(attention_avg.min())
        }
    }

    stats_path = results_dir / 'attention_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved: {stats_path}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nConsistency across problems:")
    print(f"  Mean std: {mean_std:.4f} {'✓' if mean_std < 0.2 else '⚠️'}")
    print(f"  Max std: {attention_std.max():.4f}")

    print(f"\nAttention range:")
    print(f"  Min: {attention_avg.min():.4f}")
    print(f"  Mean: {attention_avg.mean():.4f}")
    print(f"  Max: {attention_avg.max():.4f}")

    print(f"\nTop head: L{top_20_heads[0]['layer']}H{top_20_heads[0]['head']}")
    print(f"  Max attention: {top_20_heads[0]['max_attention']:.4f}")

    print(f"\nPer-position incoming attention:")
    for ps in position_stats:
        print(f"  Position {ps['position']}: {ps['avg_incoming_attention']:.4f}")

    print("\n" + "=" * 80)
    print("STORY 1.3 COMPLETE ✓")
    print("=" * 80)
    print(f"\nAggregated attention from {n_problems} problems")
    print(f"Output: {avg_path}")
    print(f"Statistics: {stats_path}")
    print("\nNext step: Run Story 1.4 to create visualizations")
    print("  python 4_visualize_heatmaps.py")


def main():
    parser = argparse.ArgumentParser(description='Aggregate attention patterns')
    parser.add_argument('--model', type=str, default='llama',
                        choices=['llama', 'gpt2'],
                        help='Model to aggregate (llama or gpt2)')
    args = parser.parse_args()

    aggregate_attention(model=args.model)


if __name__ == '__main__':
    main()
