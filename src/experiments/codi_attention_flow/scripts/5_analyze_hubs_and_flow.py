#!/usr/bin/env python3
"""
Hub and Flow Analyzer - Story 1.5

Analyze hub positions, sequential flow, and skip connections.

Usage:
    python 5_analyze_hubs_and_flow.py [--model MODEL]

Output:
    ../results/{model}/attention_summary.json
    ../figures/{model}/3_hub_analysis.png
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path


def analyze_hubs_and_flow(model: str = 'llama') -> None:
    """
    Analyze hub positions, sequential flow, and skip connections.

    Args:
        model: Model name ('llama' or 'gpt2')
    """
    print("=" * 80)
    print("HUB AND FLOW ANALYZER - Story 1.5")
    print("=" * 80)

    # Load aggregated attention
    results_dir = Path(__file__).parent.parent / 'results' / model
    figures_dir = Path(__file__).parent.parent / 'figures' / model

    avg_path = results_dir / 'attention_patterns_avg.npy'
    print(f"\nLoading attention patterns from {avg_path}...")
    attention_avg = np.load(avg_path).astype(np.float32)
    print(f"✓ Loaded: {attention_avg.shape}")

    n_layers, n_heads, _, _ = attention_avg.shape

    # Compute hub scores (average incoming attention per position)
    print("\n" + "=" * 80)
    print("HUB ANALYSIS")
    print("=" * 80)

    # For each position, sum incoming attention across all source positions
    # Then average across all layers and heads
    hub_scores = []
    for pos in range(6):
        # Incoming attention to position 'pos' from all other positions
        # Sum across source positions (axis=3), then average across layers/heads
        incoming = attention_avg[:, :, :, pos].sum(axis=2).mean()
        hub_scores.append(float(incoming))

    # Identify hub position
    hub_position = int(np.argmax(hub_scores))
    hub_score = hub_scores[hub_position]

    print(f"\nHub scores (avg incoming attention per position):")
    for pos in range(6):
        marker = " ← HUB" if pos == hub_position else ""
        print(f"  Position {pos}: {hub_scores[pos]:.4f}{marker}")

    # Compute ratio vs uniform baseline
    uniform_baseline = 1.0 / 6  # 0.167
    hub_ratio = hub_score / uniform_baseline

    print(f"\nHub position: {hub_position}")
    print(f"  Hub score: {hub_score:.4f}")
    print(f"  Uniform baseline: {uniform_baseline:.4f}")
    print(f"  Ratio: {hub_ratio:.2f}×")

    if hub_ratio > 2.0:
        print(f"  ✓ Strong hub (>{2.0}× uniform)")
    else:
        print(f"  ⚠️  Weak hub (<{2.0}× uniform)")

    # Sequential flow analysis
    print("\n" + "=" * 80)
    print("SEQUENTIAL FLOW ANALYSIS")
    print("=" * 80)

    # Measure attention from position i to position i-1
    sequential_scores = []
    for pos in range(1, 6):  # Skip position 0 (no previous position)
        # Attention from position 'pos' to position 'pos-1'
        # Average across all layers and heads
        attn_to_prev = attention_avg[:, :, pos, pos-1].mean()
        sequential_scores.append(float(attn_to_prev))

    avg_sequential = np.mean(sequential_scores)

    print(f"\nSequential attention (i → i-1):")
    for i, score in enumerate(sequential_scores, start=1):
        print(f"  Position {i} → {i-1}: {score:.4f}")

    print(f"\nAverage sequential flow: {avg_sequential:.4f}")

    # Determine if there's sequential flow
    # High sequential flow if avg > 0.3 (arbitrary threshold, 2× uniform baseline)
    has_sequential_flow = avg_sequential > 0.3

    print(f"\nSequential flow present? {has_sequential_flow}")
    if has_sequential_flow:
        print(f"  ✓ Yes (avg {avg_sequential:.3f} > 0.3)")
    else:
        print(f"  ✗ No (avg {avg_sequential:.3f} <= 0.3)")

    # Does position 5 attend to position 4?
    attn_5_to_4 = attention_avg[:, :, 5, 4].mean()
    print(f"\nDoes position 5 attend to position 4? {attn_5_to_4:.4f}")
    if attn_5_to_4 > 0.3:
        print(f"  ✓ Yes (strongly, {attn_5_to_4:.3f} > 0.3)")
    elif attn_5_to_4 > 0.15:
        print(f"  ~ Yes (moderately, {attn_5_to_4:.3f} > 0.15)")
    else:
        print(f"  ✗ No (weakly, {attn_5_to_4:.3f} <= 0.15)")

    # Skip connection analysis
    print("\n" + "=" * 80)
    print("SKIP CONNECTION ANALYSIS")
    print("=" * 80)

    # Measure attention from position 5 to positions 0-2 (skipping 3-4)
    skip_scores = []
    for target_pos in range(3):  # Positions 0, 1, 2
        attn = attention_avg[:, :, 5, target_pos].mean()
        skip_scores.append(float(attn))

    avg_skip = np.mean(skip_scores)

    print(f"\nSkip connections (position 5 → early positions):")
    for i, score in enumerate(skip_scores):
        print(f"  Position 5 → {i}: {score:.4f}")

    print(f"\nAverage skip connection: {avg_skip:.4f}")

    # Determine if there are skip connections
    has_skip_connections = avg_skip > 0.1

    print(f"\nSkip connections present? {has_skip_connections}")
    if has_skip_connections:
        print(f"  ✓ Yes (avg {avg_skip:.3f} > 0.1)")
    else:
        print(f"  ✗ No (avg {avg_skip:.3f} <= 0.1)")

    # Create hub visualization
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Hub scores
    ax1 = axes[0]
    positions = range(6)
    bars = ax1.bar(positions, hub_scores, color=['red' if i == hub_position else 'steelblue' for i in positions])
    ax1.axhline(y=uniform_baseline, color='gray', linestyle='--', label='Uniform baseline')
    ax1.set_xlabel('Position', fontsize=12)
    ax1.set_ylabel('Avg Incoming Attention', fontsize=12)
    ax1.set_title('Hub Analysis - Incoming Attention per Position', fontsize=13, fontweight='bold')
    ax1.set_xticks(positions)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Sequential flow
    ax2 = axes[1]
    seq_positions = list(range(1, 6))
    ax2.plot(seq_positions, sequential_scores, marker='o', linewidth=2, markersize=8, color='green')
    ax2.axhline(y=0.3, color='gray', linestyle='--', label='Threshold (0.3)')
    ax2.set_xlabel('Position i', fontsize=12)
    ax2.set_ylabel('Attention i → i-1', fontsize=12)
    ax2.set_title('Sequential Flow - Attention to Previous Position', fontsize=13, fontweight='bold')
    ax2.set_xticks(seq_positions)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Skip connections
    ax3 = axes[2]
    skip_positions = [0, 1, 2]
    ax3.bar(skip_positions, skip_scores, color='orange')
    ax3.axhline(y=0.1, color='gray', linestyle='--', label='Threshold (0.1)')
    ax3.set_xlabel('Target Position', fontsize=12)
    ax3.set_ylabel('Attention from Position 5', fontsize=12)
    ax3.set_title('Skip Connections - Position 5 → Early Positions', fontsize=13, fontweight='bold')
    ax3.set_xticks(skip_positions)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    plt.suptitle(f'{model.upper()} - Hub, Flow, and Skip Connection Analysis',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig3_path = figures_dir / '3_hub_analysis.png'
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Saved visualization: {fig3_path}")
    print(f"  Size: {fig3_path.stat().st_size / 1024:.1f} KB")

    # Save summary
    summary = {
        'model': model,
        'hub_analysis': {
            'hub_position': hub_position,
            'hub_score': hub_score,
            'uniform_baseline': uniform_baseline,
            'hub_ratio': float(hub_ratio),
            'is_strong_hub': hub_ratio > 2.0,
            'all_hub_scores': hub_scores
        },
        'sequential_flow': {
            'has_sequential_flow': bool(has_sequential_flow),
            'avg_sequential_attention': float(avg_sequential),
            'sequential_scores': sequential_scores,
            'position_5_to_4': float(attn_5_to_4)
        },
        'skip_connections': {
            'has_skip_connections': bool(has_skip_connections),
            'avg_skip_attention': float(avg_skip),
            'skip_scores': skip_scores
        },
        'answers': {
            'which_position_is_hub': int(hub_position),
            'is_there_sequential_flow': bool(has_sequential_flow),
            'does_pos5_attend_to_pos4': bool(attn_5_to_4 > 0.15),
            'are_there_skip_connections': bool(has_skip_connections)
        }
    }

    summary_path = results_dir / 'attention_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary: {summary_path}")

    # Final summary
    print("\n" + "=" * 80)
    print("STORY 1.5 COMPLETE ✓")
    print("=" * 80)

    print(f"\n📊 KEY FINDINGS:")
    print(f"\n1. Hub Position: Position {hub_position}")
    print(f"   - Hub score: {hub_score:.3f} ({hub_ratio:.1f}× uniform baseline)")
    print(f"   - {['Weak', 'STRONG'][hub_ratio > 2.0]} hub detected")

    print(f"\n2. Sequential Flow: {'YES ✓' if has_sequential_flow else 'NO ✗'}")
    print(f"   - Avg attention to previous position: {avg_sequential:.3f}")
    print(f"   - Position 5 → Position 4: {attn_5_to_4:.3f}")

    print(f"\n3. Skip Connections: {'YES ✓' if has_skip_connections else 'NO ✗'}")
    print(f"   - Avg attention from pos 5 to pos 0-2: {avg_skip:.3f}")

    print(f"\nOutput files:")
    print(f"  - Summary: {summary_path}")
    print(f"  - Visualization: {fig3_path}")

    print(f"\n{'='*80}")
    print("PHASE 1 COMPLETE - All 5 stories done!")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Analyze hubs and flow patterns')
    parser.add_argument('--model', type=str, default='llama',
                        choices=['llama', 'gpt2'],
                        help='Model to analyze (llama or gpt2)')
    args = parser.parse_args()

    analyze_hubs_and_flow(model=args.model)


if __name__ == '__main__':
    main()
