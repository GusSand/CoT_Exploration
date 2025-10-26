"""
Analyze quality patterns across all layers and positions.

Loads all trained SAE metrics and identifies patterns:
1. Which layers have best reconstruction quality?
2. Which positions have best reconstruction quality?
3. Layer × position interaction effects
4. Feature death patterns
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd


def load_all_metrics():
    """Load metrics for all (layer, position) pairs."""
    results_dir = Path('src/experiments/topk_grid_pilot/results')

    all_data = []

    for json_file in sorted(results_dir.glob('grid_metrics_pos*_layer*_latent*.json')):
        with open(json_file, 'r') as f:
            data = json.load(f)

        metadata = data['metadata']
        position = metadata['position']
        layer = metadata['layer']

        for latent_dim_str, k_results in data['results'].items():
            latent_dim = int(latent_dim_str)

            for k_str, metrics in k_results.items():
                k = int(k_str)

                all_data.append({
                    'layer': layer,
                    'position': position,
                    'latent_dim': latent_dim,
                    'k': k,
                    **metrics
                })

    return pd.DataFrame(all_data)


def analyze_layer_effects(df):
    """Analyze which layers have best quality."""
    print("=" * 80)
    print("LAYER EFFECTS (averaged across positions, K=20, d=1024)")
    print("=" * 80)

    subset = df[(df['k'] == 20) & (df['latent_dim'] == 1024)]

    layer_stats = subset.groupby('layer').agg({
        'explained_variance': ['mean', 'std'],
        'feature_death_rate': ['mean', 'std'],
        'reconstruction_loss': ['mean', 'std']
    }).round(4)

    print(layer_stats)
    print()

    # Find best layers
    best_ev = subset.groupby('layer')['explained_variance'].mean().idxmax()
    worst_ev = subset.groupby('layer')['explained_variance'].mean().idxmin()

    print(f"Best layer (EV): {best_ev}")
    print(f"Worst layer (EV): {worst_ev}")
    print()


def analyze_position_effects(df):
    """Analyze which positions have best quality."""
    print("=" * 80)
    print("POSITION EFFECTS (averaged across layers, K=20, d=1024)")
    print("=" * 80)

    subset = df[(df['k'] == 20) & (df['latent_dim'] == 1024)]

    pos_stats = subset.groupby('position').agg({
        'explained_variance': ['mean', 'std'],
        'feature_death_rate': ['mean', 'std'],
        'reconstruction_loss': ['mean', 'std']
    }).round(4)

    print(pos_stats)
    print()

    # Find best positions
    best_ev = subset.groupby('position')['explained_variance'].mean().idxmax()
    worst_ev = subset.groupby('position')['explained_variance'].mean().idxmin()

    print(f"Best position (EV): {best_ev}")
    print(f"Worst position (EV): {worst_ev}")
    print()


def analyze_layer_position_matrix(df):
    """Create layer × position quality matrix."""
    print("=" * 80)
    print("LAYER × POSITION MATRIX (EV for K=20, d=1024)")
    print("=" * 80)

    subset = df[(df['k'] == 20) & (df['latent_dim'] == 1024)]

    pivot = subset.pivot_table(
        values='explained_variance',
        index='layer',
        columns='position',
        aggfunc='mean'
    ).round(3)

    print(pivot)
    print()

    # Find best combination
    best_idx = subset['explained_variance'].idxmax()
    best_row = subset.loc[best_idx]
    print(f"Best (layer, position): ({int(best_row['layer'])}, {int(best_row['position'])}) - EV={best_row['explained_variance']:.4f}")
    print()


def analyze_k_patterns(df):
    """Analyze K vs quality tradeoff across all layers/positions."""
    print("=" * 80)
    print("K VALUE EFFECTS (averaged across all layers/positions, d=1024)")
    print("=" * 80)

    subset = df[df['latent_dim'] == 1024]

    k_stats = subset.groupby('k').agg({
        'explained_variance': ['mean', 'std', 'min', 'max'],
        'feature_death_rate': ['mean', 'std', 'min', 'max']
    }).round(4)

    print(k_stats)
    print()


def save_summary(df):
    """Save summary statistics to JSON."""
    output_file = Path('src/experiments/topk_grid_pilot/results/analysis_summary.json')

    # Compute aggregates
    summary = {
        'total_saes': len(df),
        'layers': sorted(df['layer'].unique().tolist()),
        'positions': sorted(df['position'].unique().tolist()),
        'k_values': sorted(df['k'].unique().tolist()),
        'latent_dims': sorted(df['latent_dim'].unique().tolist()),

        'overall_stats': {
            'explained_variance': {
                'mean': float(df['explained_variance'].mean()),
                'std': float(df['explained_variance'].std()),
                'min': float(df['explained_variance'].min()),
                'max': float(df['explained_variance'].max())
            },
            'feature_death_rate': {
                'mean': float(df['feature_death_rate'].mean()),
                'std': float(df['feature_death_rate'].std()),
                'min': float(df['feature_death_rate'].min()),
                'max': float(df['feature_death_rate'].max())
            }
        },

        'best_configs': {
            'highest_ev': df.loc[df['explained_variance'].idxmax()].to_dict(),
            'lowest_death': df.loc[df['feature_death_rate'].idxmin()].to_dict()
        }
    }

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {output_file}")


def main():
    print("Loading all SAE metrics...\n")

    df = load_all_metrics()

    print(f"Loaded {len(df)} SAE configurations")
    print(f"  Layers: {sorted(df['layer'].unique())}")
    print(f"  Positions: {sorted(df['position'].unique())}")
    print(f"  K values: {sorted(df['k'].unique())}")
    print(f"  Latent dims: {sorted(df['latent_dim'].unique())}")
    print()

    analyze_layer_effects(df)
    analyze_position_effects(df)
    analyze_layer_position_matrix(df)
    analyze_k_patterns(df)
    save_summary(df)

    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
