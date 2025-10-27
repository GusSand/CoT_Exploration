"""
Validate Matryoshka SAE quality metrics and compare to ReLU baseline.

Computes comprehensive metrics for each hierarchy level and compares
to the ReLU SAE pilot results.
"""

import torch
import json
from pathlib import Path
from matryoshka_sae import MatryoshkaSAE


def load_model(model_path: str, device: str = 'cuda'):
    """Load trained Matryoshka SAE.

    Args:
        model_path: Path to saved model checkpoint
        device: Device to load on

    Returns:
        Loaded model
    """
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path)

    config = checkpoint['config']
    model = MatryoshkaSAE(
        input_dim=config['input_dim'],
        levels=config['levels'],
        l1_coefficient=config['l1_coefficient'],
        level_weights=config['level_weights']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  Loaded from epoch {checkpoint['epoch'] + 1}")
    print(f"  Val loss: {checkpoint['val_loss']:.6f}")

    return model, checkpoint


def validate_on_data(model, data_path: str, device: str = 'cuda'):
    """Run validation on test data.

    Args:
        model: Trained Matryoshka SAE
        data_path: Path to position_3_activations.pt
        device: Device to run on

    Returns:
        Dictionary of validation metrics
    """
    print(f"\nLoading validation data from {data_path}...")
    data = torch.load(data_path)
    activations = data['activations']

    # Use last 20% as validation
    n_samples = activations.shape[0]
    n_val = int(n_samples * 0.2)
    val_data = activations[-n_val:].to(device)

    print(f"  Validation samples: {val_data.shape[0]:,}")

    # Run forward pass
    print("\nComputing validation metrics...")
    with torch.no_grad():
        reconstructions, features = model(val_data)
        metrics = model.compute_metrics(val_data, reconstructions, features)

    return metrics


def compare_to_relu_baseline(matryoshka_metrics: dict):
    """Compare Matryoshka results to ReLU SAE baseline.

    Args:
        matryoshka_metrics: Validation metrics from Matryoshka SAE

    Returns:
        Comparison dictionary
    """
    # ReLU SAE baseline from sae_pilot (from validation_report.json)
    relu_baseline = {
        'explained_variance': 0.7862,  # 78.6%
        'feature_death_rate': 0.9697,  # 97.0%
        'l0_norm': 23.34,
        'n_features': 8192
    }

    print("\n" + "=" * 60)
    print("Matryoshka SAE vs ReLU SAE Baseline")
    print("=" * 60)

    comparison = {
        'relu_baseline': relu_baseline,
        'matryoshka': {},
        'improvements': {}
    }

    # Print comparison table
    print(f"\n{'Metric':<25} {'ReLU Baseline':<15} {'Level 1':<15} {'Level 2':<15} {'Level 3':<15}")
    print("-" * 85)

    # Explained Variance
    relu_ev = relu_baseline['explained_variance']
    m1_ev = matryoshka_metrics['level_1']['explained_variance']
    m2_ev = matryoshka_metrics['level_2']['explained_variance']
    m3_ev = matryoshka_metrics['level_3']['explained_variance']

    print(f"{'Explained Variance':<25} {relu_ev:>14.1%} {m1_ev:>14.1%} {m2_ev:>14.1%} {m3_ev:>14.1%}")

    # Feature Death Rate
    relu_death = relu_baseline['feature_death_rate']
    m1_death = matryoshka_metrics['level_1']['feature_death_rate']
    m2_death = matryoshka_metrics['level_2']['feature_death_rate']
    m3_death = matryoshka_metrics['level_3']['feature_death_rate']

    print(f"{'Feature Death Rate':<25} {relu_death:>14.1%} {m1_death:>14.1%} {m2_death:>14.1%} {m3_death:>14.1%}")

    # L0 Norm
    relu_l0 = relu_baseline['l0_norm']
    m1_l0 = matryoshka_metrics['level_1']['l0_norm']
    m2_l0 = matryoshka_metrics['level_2']['l0_norm']
    m3_l0 = matryoshka_metrics['level_3']['l0_norm']

    print(f"{'L0 Norm (Active Features)':<25} {relu_l0:>14.1f} {m1_l0:>14.1f} {m2_l0:>14.1f} {m3_l0:>14.1f}")

    # Number of Features
    relu_n_feat = relu_baseline['n_features']
    m1_n_feat = matryoshka_metrics['level_1']['features']
    m2_n_feat = matryoshka_metrics['level_2']['features']
    m3_n_feat = matryoshka_metrics['level_3']['features']

    print(f"{'Total Features':<25} {relu_n_feat:>14,} {m1_n_feat:>14,} {m2_n_feat:>14,} {m3_n_feat:>14,}")

    # Active Features Count
    relu_active = int(relu_n_feat * (1 - relu_death))
    m1_active = matryoshka_metrics['level_1']['active_features']
    m2_active = matryoshka_metrics['level_2']['active_features']
    m3_active = matryoshka_metrics['level_3']['active_features']

    print(f"{'Active Features':<25} {relu_active:>14,} {m1_active:>14,} {m2_active:>14,} {m3_active:>14,}")

    print("\n" + "=" * 60)
    print("Key Improvements")
    print("=" * 60)

    # Feature death improvement (lower is better)
    death_improvement = (relu_death - m3_death) * 100
    print(f"Feature Death Reduction: {death_improvement:+.1f} percentage points")
    print(f"  ReLU: {relu_death:.1%} → Matryoshka L3: {m3_death:.1%}")

    # EV comparison (higher is better)
    ev_change = (m3_ev - relu_ev) * 100
    print(f"\nExplained Variance Change: {ev_change:+.1f} percentage points")
    print(f"  ReLU: {relu_ev:.1%} → Matryoshka L3: {m3_ev:.1%}")

    # Active features comparison
    active_improvement = m3_active - relu_active
    print(f"\nActive Features Change: {active_improvement:+,}")
    print(f"  ReLU: {relu_active:,} → Matryoshka L3: {m3_active:,}")

    # Save comparison
    comparison['matryoshka'] = matryoshka_metrics
    comparison['improvements'] = {
        'feature_death_reduction_pct': death_improvement,
        'ev_change_pct': ev_change,
        'active_features_increase': active_improvement
    }

    return comparison


if __name__ == "__main__":
    # Paths
    base_dir = Path(__file__).parent
    model_path = base_dir / "models/pos3_hierarchical.pt"
    data_path = base_dir / "data/position_3_activations.pt"
    output_path = base_dir / "results/validation_metrics.json"

    print("=" * 60)
    print("Matryoshka SAE Validation")
    print("=" * 60)

    # Load model
    model, checkpoint = load_model(str(model_path))

    # Validate
    metrics = validate_on_data(model, str(data_path))

    # Compare to baseline
    comparison = compare_to_relu_baseline(metrics)

    # Save results
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\nValidation results saved to: {output_path}")
    print("\n✓ Validation complete!")
