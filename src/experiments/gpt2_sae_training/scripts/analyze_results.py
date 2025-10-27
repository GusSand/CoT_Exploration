"""
Analyze GPT-2 TopK SAE parameter sweep results.

Loads metrics from all 8 configs and identifies the "sweet spot" based on:
1. Explained Variance ≥ 70% (reconstruction quality)
2. Lowest Feature Death Rate (feature utilization)
3. Balanced sparsity (not too dense)
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


def load_all_metrics() -> List[Dict]:
    """Load metrics from all 8 checkpoints."""
    results_dir = Path("src/experiments/gpt2_sae_training/results")

    configs = [
        (192, 20),
        (192, 40),
        (256, 30),
        (256, 50),
        (256, 75),
        (384, 75),
        (512, 100),
        (512, 150),
    ]

    all_metrics = []

    for latent_dim, k in configs:
        checkpoint_path = results_dir / f"gpt2_pos3_layer8_d{latent_dim}_k{k}.pt"

        if not checkpoint_path.exists():
            print(f"⚠️  Missing checkpoint: {checkpoint_path}")
            continue

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        metrics = checkpoint['metrics']
        config = checkpoint['config']

        # Compute derived metrics
        sparsity_pct = (k / latent_dim) * 100
        expansion_ratio = latent_dim / 768  # GPT-2 has 768 hidden dims

        all_metrics.append({
            'latent_dim': latent_dim,
            'k': k,
            'sparsity_pct': sparsity_pct,
            'expansion_ratio': expansion_ratio,
            'explained_variance': metrics['explained_variance'],
            'feature_death_rate': metrics['feature_death_rate'],
            'reconstruction_loss': metrics['reconstruction_loss'],
            'l0_mean': metrics['l0_mean'],
            'l0_std': metrics['l0_std'],
            'mean_activation': metrics['mean_activation'],
            'max_activation': metrics['max_activation'],
            'train_time_sec': metrics['train_time_sec'],
        })

    return all_metrics


def identify_sweet_spot(metrics: List[Dict]) -> Tuple[Dict, str]:
    """
    Identify the sweet spot configuration.

    Criteria:
    1. Explained Variance ≥ 70%
    2. Among those, lowest Feature Death Rate
    3. If tie, prefer medium sparsity (15-25%)

    Returns:
        (sweet_spot_config, rationale)
    """

    # Filter by explained variance threshold
    ev_threshold = 0.70
    candidates = [m for m in metrics if m['explained_variance'] >= ev_threshold]

    if not candidates:
        # If none meet threshold, take best EV
        candidates = [max(metrics, key=lambda x: x['explained_variance'])]
        rationale_prefix = f"⚠️  No configs met EV≥{ev_threshold}, selected best EV instead.\n"
    else:
        rationale_prefix = f"✓ {len(candidates)}/8 configs meet EV≥{ev_threshold}\n"

    # Among candidates, find lowest feature death rate
    sweet_spot = min(candidates, key=lambda x: x['feature_death_rate'])

    # Build rationale
    rationale = rationale_prefix
    rationale += f"✓ Selected d={sweet_spot['latent_dim']}, K={sweet_spot['k']} based on:\n"
    rationale += f"  1. Explained Variance: {sweet_spot['explained_variance']:.3f} ({'✓' if sweet_spot['explained_variance'] >= 0.70 else '✗'} ≥70%)\n"
    rationale += f"  2. Feature Death Rate: {sweet_spot['feature_death_rate']:.3f} (lowest among candidates)\n"
    rationale += f"  3. Sparsity: {sweet_spot['sparsity_pct']:.1f}% (K={sweet_spot['k']}/{sweet_spot['latent_dim']})\n"
    rationale += f"  4. Expansion: {sweet_spot['expansion_ratio']:.2f}x (vs 768 input dims)\n"

    return sweet_spot, rationale


def create_comparison_table(metrics: List[Dict]) -> pd.DataFrame:
    """Create a comparison table of all configs."""

    df = pd.DataFrame(metrics)

    # Reorder columns
    df = df[[
        'latent_dim', 'k', 'sparsity_pct', 'expansion_ratio',
        'explained_variance', 'feature_death_rate',
        'reconstruction_loss', 'l0_mean', 'train_time_sec'
    ]]

    # Sort by explained variance descending
    df = df.sort_values('explained_variance', ascending=False)

    return df


def main():
    print("="*80)
    print("GPT-2 TOPK SAE PARAMETER SWEEP - RESULTS ANALYSIS")
    print("="*80)

    # Load all metrics
    print("\n[1/3] Loading metrics from 8 checkpoints...")
    all_metrics = load_all_metrics()
    print(f"  Loaded {len(all_metrics)} configs")

    # Create comparison table
    print("\n[2/3] Creating comparison table...")
    df = create_comparison_table(all_metrics)

    print("\n" + "="*80)
    print("RESULTS TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print()

    # Identify sweet spot
    print("\n[3/3] Identifying sweet spot...")
    sweet_spot, rationale = identify_sweet_spot(all_metrics)

    print("\n" + "="*80)
    print("SWEET SPOT IDENTIFICATION")
    print("="*80)
    print(rationale)
    print()

    # Save results
    output_dir = Path("src/experiments/gpt2_sae_training/results")
    analysis_path = output_dir / "analysis_summary.json"

    with open(analysis_path, 'w') as f:
        json.dump({
            'all_configs': all_metrics,
            'sweet_spot': sweet_spot,
            'rationale': rationale,
            'comparison_table': df.to_dict(orient='records')
        }, f, indent=2)

    print(f"✓ Analysis saved to: {analysis_path}")
    print()

    # Print sweet spot again for clarity
    print("="*80)
    print(f"SWEET SPOT: d={sweet_spot['latent_dim']}, K={sweet_spot['k']}")
    print("="*80)
    print(f"  Explained Variance: {sweet_spot['explained_variance']:.1%}")
    print(f"  Feature Death Rate: {sweet_spot['feature_death_rate']:.1%}")
    print(f"  Sparsity: {sweet_spot['sparsity_pct']:.1f}%")
    print(f"  L0 Norm: {sweet_spot['l0_mean']:.1f}")
    print("="*80)

    return sweet_spot, df


if __name__ == '__main__':
    main()
