"""
Feature Validation via Ablation Experiments.

Validates SAE feature interpretations by ablating features and measuring impact:
1. General feature importance: Top features should have large impact
2. Operation-level features: Specialized features should affect specific operations

Usage:
    python validate_features.py --layer 14 --position 3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, 'src/experiments/topk_grid_pilot')
from topk_sae import TopKAutoencoder
from causal_interventions import FeatureInterventionEngine


def load_models_and_data(layer, position, k=100, latent_dim=512):
    """Load SAE, activations, and feature labels."""
    print(f"\n{'='*80}")
    print(f"Loading Models and Data")
    print(f"{'='*80}\n")

    # Load SAE
    ckpt_path = f'src/experiments/topk_grid_pilot/results/checkpoints/pos{position}_layer{layer}_d{latent_dim}_k{k}.pt'
    ckpt = torch.load(ckpt_path, weights_only=False)
    sae = TopKAutoencoder(
        input_dim=ckpt['config']['input_dim'],
        latent_dim=ckpt['config']['latent_dim'],
        k=ckpt['config']['k']
    )
    sae.load_state_dict(ckpt['model_state_dict'])
    sae.eval()
    print(f"✓ SAE loaded: Layer {layer}, Position {position}")

    # Load validation data
    val_data = torch.load('src/experiments/sae_cot_decoder/data/full_val_activations.pt', weights_only=False)
    positions = np.array(val_data['metadata']['positions'])
    layers = np.array(val_data['metadata']['layers'])
    mask = (positions == position) & (layers == layer)
    activations = val_data['activations'][mask]
    print(f"✓ Validation data loaded: {len(activations)} samples")

    # Load feature labels from Story 1
    labels_path = f'src/experiments/llama_sae_hierarchy/feature_labels_layer{layer}_pos{position}.json'
    with open(labels_path, 'r') as f:
        labels_data = json.load(f)
    print(f"✓ Feature labels loaded: {len(labels_data['features'])} features\n")

    return sae, activations, labels_data


def validate_general_features(engine, activations, feature_labels, top_n=10):
    """
    Validate that top general features have significant impact when ablated.

    Args:
        engine: FeatureInterventionEngine
        activations: Validation activations
        feature_labels: Feature label data from Story 1
        top_n: Number of top features to test

    Returns:
        results: List of validation results
    """
    print(f"{'='*80}")
    print(f"Validation 1: General Feature Importance (Top {top_n} Features)")
    print(f"{'='*80}\n")

    print(f"Testing hypothesis: Top features should have large impact when ablated\n")
    print(f"{'Rank':<6} {'Feature':<10} {'Act %':<10} {'Mean Impact':<15} {'Max Impact':<15} {'Status'}")
    print(f"{'-'*80}")

    results = []

    for feat_data in feature_labels['features'][:top_n]:
        feat_idx = feat_data['feature_id']
        rank = feat_data['rank']

        # Measure impact
        impact = engine.measure_feature_impact(activations, feat_idx, metric='all')

        # Classify impact magnitude
        mean_impact = impact['mean_abs_diff']
        if mean_impact > 0.1:
            status = "✓ HIGH"
        elif mean_impact > 0.01:
            status = "○ MEDIUM"
        else:
            status = "✗ LOW"

        # Store result
        result = {
            'rank': rank,
            'feature_id': feat_idx,
            'activation_freq': impact['feature_activation_freq'],
            'mean_impact': mean_impact,
            'max_impact': impact['max_abs_diff'],
            'classification': status
        }
        results.append(result)

        # Print
        act_pct = f"{impact['feature_activation_freq']*100:.1f}%"
        print(f"{rank:<6} {feat_idx:<10} {act_pct:<10} {mean_impact:<15.6f} {impact['max_abs_diff']:<15.6f} {status}")

    print(f"{'-'*80}\n")

    # Summary
    high_impact = sum(1 for r in results if "HIGH" in r['classification'])
    medium_impact = sum(1 for r in results if "MEDIUM" in r['classification'])
    low_impact = sum(1 for r in results if "LOW" in r['classification'])

    print(f"Summary:")
    print(f"  High impact: {high_impact}/{top_n} ({100*high_impact/top_n:.0f}%)")
    print(f"  Medium impact: {medium_impact}/{top_n} ({100*medium_impact/top_n:.0f}%)")
    print(f"  Low impact: {low_impact}/{top_n} ({100*low_impact/top_n:.0f}%)")
    print()

    # Validation
    if high_impact >= top_n * 0.7:  # 70% should be high impact
        print(f"✓ VALIDATION PASSED: {high_impact}/{top_n} features have high impact (≥70%)")
    else:
        print(f"✗ VALIDATION FAILED: Only {high_impact}/{top_n} features have high impact (<70%)")
    print()

    return results


def validate_specialized_features(engine, activations, layer, position):
    """
    Validate operation-specialized features found in Story 2.

    Args:
        engine: FeatureInterventionEngine
        activations: Validation activations
        layer: Layer index
        position: Position index

    Returns:
        results: List of validation results
    """
    print(f"{'='*80}")
    print(f"Validation 2: Operation-Specialized Features")
    print(f"{'='*80}\n")

    # Load specialized features from Story 2
    analysis_path = f'src/experiments/llama_sae_hierarchy/activation_analysis_layer{layer}_pos{position}_rank400-512.json'

    if not Path(analysis_path).exists():
        print(f"⊘ SKIPPED: No specialized feature analysis found")
        print(f"  Expected: {analysis_path}\n")
        return []

    with open(analysis_path, 'r') as f:
        analysis_data = json.load(f)

    specialized = analysis_data['specialized_features']

    if len(specialized) == 0:
        print(f"⊘ SKIPPED: No specialized features found in analysis\n")
        return []

    print(f"Testing {len(specialized)} specialized features:\n")
    print(f"{'Feature':<10} {'Type':<25} {'Act %':<10} {'Impact':<15} {'Status'}")
    print(f"{'-'*80}")

    results = []

    for feat_data in specialized:
        feat_idx = feat_data['feature_id']
        spec_type = feat_data['specialization']['type']
        spec_desc = feat_data['specialization']['description']

        # Measure impact
        impact = engine.measure_feature_impact(activations, feat_idx, metric='all')

        # Classify (specialized features may have lower impact due to rarity)
        mean_impact = impact['mean_abs_diff']
        if mean_impact > 0.01:
            status = "✓ MEASURABLE"
        else:
            status = "○ MINIMAL"

        result = {
            'feature_id': feat_idx,
            'specialization_type': spec_type,
            'description': spec_desc,
            'activation_freq': impact['feature_activation_freq'],
            'mean_impact': mean_impact,
            'classification': status
        }
        results.append(result)

        # Print
        act_pct = f"{impact['feature_activation_freq']*100:.1f}%"
        print(f"{feat_idx:<10} {spec_desc:<25} {act_pct:<10} {mean_impact:<15.6f} {status}")

    print(f"{'-'*80}\n")

    # Summary
    measurable = sum(1 for r in results if "MEASURABLE" in r['classification'])

    print(f"Summary:")
    print(f"  Measurable impact: {measurable}/{len(specialized)}")
    print(f"  Note: Specialized features are rare (0.1-0.3% activation)")
    print(f"        Low impact expected due to infrequent activation\n")

    return results


def main():
    parser = argparse.ArgumentParser(description='Feature Validation via Ablation')
    parser.add_argument('--layer', type=int, default=14, help='Layer index')
    parser.add_argument('--position', type=int, default=3, help='Position index')
    parser.add_argument('--top_n', type=int, default=10, help='Number of top features to validate')

    args = parser.parse_args()

    # Load models and data
    sae, activations, feature_labels = load_models_and_data(args.layer, args.position)

    # Create intervention engine
    engine = FeatureInterventionEngine(sae)

    # Run sanity checks
    print(f"Running sanity checks...")
    sanity_results = engine.run_sanity_checks(activations[:100], verbose=False)
    passed = sum(1 for v in sanity_results.values() if v is True)
    total = sum(1 for v in sanity_results.values() if v is not None)
    print(f"✓ Sanity checks: {passed}/{total} passed\n")

    # Validation 1: General features
    general_results = validate_general_features(
        engine, activations, feature_labels, args.top_n
    )

    # Validation 2: Specialized features
    specialized_results = validate_specialized_features(
        engine, activations, args.layer, args.position
    )

    # Save results (convert any tensors to floats)
    def convert_tensors(obj):
        """Recursively convert tensors to floats."""
        if isinstance(obj, torch.Tensor):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_tensors(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tensors(v) for v in obj]
        else:
            return obj

    output = {
        'metadata': {
            'layer': args.layer,
            'position': args.position,
            'num_samples': len(activations),
            'sanity_checks': convert_tensors(sanity_results),
            'validation_date': '2025-10-27'
        },
        'general_features': convert_tensors(general_results),
        'specialized_features': convert_tensors(specialized_results)
    }

    output_path = f'src/experiments/llama_sae_hierarchy/validation_results_layer{args.layer}_pos{args.position}.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
