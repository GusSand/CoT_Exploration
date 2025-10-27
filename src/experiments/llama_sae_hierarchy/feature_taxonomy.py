"""
Feature Taxonomy & Labeling for LLaMA TopK SAE.

Identifies and categorizes SAE features by abstraction level:
- Operation-level: Addition, multiplication, division, etc.
- Value-level: Specific numbers like 12, 50, 100
- Mixed: Features showing both operation and value patterns

Usage:
    python feature_taxonomy.py --layer 14 --position 3 --top_n 20
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add topk_grid_pilot to path for TopK SAE
sys.path.insert(0, 'src/experiments/topk_grid_pilot')
from topk_sae import TopKAutoencoder


def load_sae_and_data(layer, position, k=100, latent_dim=512):
    """
    Load SAE model and validation data.

    Args:
        layer: Layer index (0-15)
        position: Position index (0-5)
        k: TopK sparsity level (default: 100)
        latent_dim: Dictionary size (default: 512)

    Returns:
        model: Loaded TopK SAE model
        activations: Validation activations for this layer/position
        problems: List of problem metadata with CoT sequences
    """
    print(f"\n{'='*80}")
    print(f"Loading SAE: Layer {layer}, Position {position}, K={k}, d={latent_dim}")
    print(f"{'='*80}\n")

    # Load SAE checkpoint
    ckpt_path = f'src/experiments/topk_grid_pilot/results/checkpoints/pos{position}_layer{layer}_d{latent_dim}_k{k}.pt'
    print(f"Loading checkpoint: {ckpt_path}")

    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, weights_only=False)

    # Initialize model
    model = TopKAutoencoder(
        input_dim=ckpt['config']['input_dim'],
        latent_dim=ckpt['config']['latent_dim'],
        k=ckpt['config']['k']
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"  Model loaded: {ckpt['config']['input_dim']} → {ckpt['config']['latent_dim']} (K={ckpt['config']['k']})")

    # Load validation data
    val_data_path = 'src/experiments/sae_cot_decoder/data/full_val_activations.pt'
    print(f"\nLoading validation data: {val_data_path}")

    if not Path(val_data_path).exists():
        raise FileNotFoundError(f"Validation data not found: {val_data_path}")

    val_data = torch.load(val_data_path, weights_only=False)

    # Filter to target layer and position
    positions = np.array(val_data['metadata']['positions'])
    layers = np.array(val_data['metadata']['layers'])

    mask = (positions == position) & (layers == layer)
    activations = val_data['activations'][mask]

    print(f"  Total samples in validation set: {len(val_data['activations'])}")
    print(f"  Filtered samples (layer={layer}, pos={position}): {len(activations)}")

    # Extract problem metadata
    problem_ids = [val_data['metadata']['problem_ids'][i] for i, m in enumerate(mask) if m]
    cot_sequences = [val_data['metadata']['cot_sequences'][i] for i, m in enumerate(mask) if m]

    # Create problem list
    problems = []
    for pid, cot in zip(problem_ids, cot_sequences):
        problems.append({
            'problem_id': pid,
            'cot_sequence': cot,
            'cot_text': ' | '.join(cot)  # Concatenate for pattern matching
        })

    print(f"  Problem metadata extracted: {len(problems)} problems\n")

    return model, activations, problems


def detect_patterns(cot_samples):
    """
    Detect patterns in CoT sequences using heuristics.

    Args:
        cot_samples: List of CoT text samples

    Returns:
        patterns: Set of detected patterns
    """
    patterns = set()

    # Count pattern occurrences
    addition_count = sum(1 for cot in cot_samples if '+' in cot or 'sum' in cot.lower())
    subtraction_count = sum(1 for cot in cot_samples if '-' in cot or 'difference' in cot.lower())
    multiplication_count = sum(1 for cot in cot_samples if '*' in cot or 'multiply' in cot.lower() or '×' in cot)
    division_count = sum(1 for cot in cot_samples if '/' in cot or 'divide' in cot.lower() or '÷' in cot)

    # Round numbers (100, 200, 500, 1000)
    round_numbers_count = sum(1 for cot in cot_samples
                             if any(str(n) in cot for n in [100, 200, 500, 1000]))

    # Specific numbers (check common values)
    specific_numbers = {}
    for num in [12, 20, 30, 50, 100]:
        count = sum(1 for cot in cot_samples if str(num) in cot)
        if count >= 5:  # Threshold for specific number
            specific_numbers[num] = count

    # Add patterns if threshold met (≥3 occurrences)
    if addition_count >= 3:
        patterns.add('addition')
    if subtraction_count >= 3:
        patterns.add('subtraction')
    if multiplication_count >= 3:
        patterns.add('multiplication')
    if division_count >= 3:
        patterns.add('division')
    if round_numbers_count >= 3:
        patterns.add('round_numbers')

    for num, count in specific_numbers.items():
        patterns.add(f'number_{num}')

    return patterns


def classify_feature_type(patterns):
    """
    Classify feature as operation-level, value-level, or mixed.

    Args:
        patterns: Set of detected patterns

    Returns:
        feature_type: 'operation-level', 'value-level', 'mixed', or 'unknown'
        confidence: 'high', 'medium', 'low', or 'unknown'
    """
    operation_patterns = {'addition', 'subtraction', 'multiplication', 'division'}
    value_patterns = {p for p in patterns if p.startswith('number_') or p == 'round_numbers'}

    has_operation = bool(operation_patterns & patterns)
    has_value = bool(value_patterns)

    if has_operation and has_value:
        return 'mixed', 'medium'
    elif has_operation:
        return 'operation-level', 'high' if len(operation_patterns & patterns) >= 2 else 'medium'
    elif has_value:
        return 'value-level', 'high' if len(value_patterns) >= 2 else 'medium'
    else:
        return 'unknown', 'unknown'


def analyze_features(model, activations, problems, top_n=20):
    """
    Analyze top N features and create taxonomy.

    Args:
        model: TopK SAE model
        activations: Validation activations
        problems: Problem metadata
        top_n: Number of top features to analyze

    Returns:
        feature_labels: List of feature analysis dicts
    """
    print(f"\n{'='*80}")
    print(f"Analyzing Top {top_n} Features")
    print(f"{'='*80}\n")

    print(f"Running SAE on {len(activations)} samples...")

    # Run SAE on all activations
    with torch.no_grad():
        _, sparse, _ = model(activations)

    # Compute feature statistics
    feature_activation_freq = (sparse != 0).float().mean(dim=0)  # How often each feature fires
    feature_mean_magnitude = sparse.abs().mean(dim=0)  # Average magnitude when active

    # Get top features by activation frequency
    top_features_idx = torch.argsort(feature_activation_freq, descending=True)[:top_n]

    print(f"Top {top_n} features identified by activation frequency\n")
    print(f"{'Rank':<6} {'Feature':<10} {'Act Freq':<12} {'Mean Mag':<12} {'Type':<18} {'Confidence':<12} {'Patterns'}")
    print(f"{'-'*100}")

    feature_labels = []

    for rank, feat_idx in enumerate(top_features_idx, 1):
        feat_idx = feat_idx.item()

        # Get samples where this feature is active
        active_mask = sparse[:, feat_idx] != 0
        active_indices = torch.where(active_mask)[0]

        # Get top samples by activation magnitude
        if len(active_indices) > 0:
            active_magnitudes = sparse[active_indices, feat_idx].abs()
            top_k = min(100, len(active_indices))
            top_magnitudes, top_local_indices = torch.topk(active_magnitudes, k=top_k)
            top_global_indices = active_indices[top_local_indices]

            # Extract CoT samples
            top_samples = []
            cot_texts = []
            for idx in top_global_indices:
                idx = idx.item()
                top_samples.append({
                    'problem_id': problems[idx]['problem_id'],
                    'cot': problems[idx]['cot_text'],
                    'activation': float(sparse[idx, feat_idx])
                })
                cot_texts.append(problems[idx]['cot_text'])

            # Detect patterns
            patterns = detect_patterns(cot_texts)

            # Classify feature type
            feature_type, confidence = classify_feature_type(patterns)

        else:
            top_samples = []
            patterns = set()
            feature_type = 'unknown'
            confidence = 'unknown'

        # Store feature analysis
        feature_analysis = {
            'rank': rank,
            'feature_id': feat_idx,
            'activation_freq': float(feature_activation_freq[feat_idx]),
            'mean_magnitude': float(feature_mean_magnitude[feat_idx]),
            'interpretation': {
                'type': feature_type,
                'confidence': confidence,
                'detected_patterns': sorted(list(patterns)),
                'description': generate_description(patterns, feature_type)
            },
            'top_samples': top_samples[:10]  # Store only top 10 for brevity
        }

        feature_labels.append(feature_analysis)

        # Print summary
        patterns_str = ', '.join(sorted(patterns)) if patterns else 'None'
        if len(patterns_str) > 40:
            patterns_str = patterns_str[:37] + '...'

        print(f"{rank:<6} {feat_idx:<10} {feature_analysis['activation_freq']:<12.4f} "
              f"{feature_analysis['mean_magnitude']:<12.3f} {feature_type:<18} "
              f"{confidence:<12} {patterns_str}")

    print(f"{'-'*100}\n")

    # Summary statistics
    type_counts = {}
    confidence_counts = {}
    for f in feature_labels:
        ftype = f['interpretation']['type']
        conf = f['interpretation']['confidence']
        type_counts[ftype] = type_counts.get(ftype, 0) + 1
        confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

    print(f"Summary Statistics:")
    print(f"  Feature Types:")
    for ftype, count in sorted(type_counts.items()):
        print(f"    {ftype}: {count}/{top_n} ({100*count/top_n:.1f}%)")
    print(f"  Confidence Levels:")
    for conf, count in sorted(confidence_counts.items()):
        print(f"    {conf}: {count}/{top_n} ({100*count/top_n:.1f}%)")
    print()

    return feature_labels


def generate_description(patterns, feature_type):
    """Generate human-readable description from patterns."""
    if not patterns:
        return "No clear pattern detected"

    descriptions = []

    # Operation patterns
    ops = []
    if 'addition' in patterns:
        ops.append('addition')
    if 'subtraction' in patterns:
        ops.append('subtraction')
    if 'multiplication' in patterns:
        ops.append('multiplication')
    if 'division' in patterns:
        ops.append('division')

    if ops:
        descriptions.append(f"Operations: {', '.join(ops)}")

    # Number patterns
    nums = [p.replace('number_', '') for p in patterns if p.startswith('number_')]
    if nums:
        descriptions.append(f"Numbers: {', '.join(nums)}")

    if 'round_numbers' in patterns:
        descriptions.append("Round numbers (100, 200, 500, 1000)")

    return '; '.join(descriptions) if descriptions else "Mixed patterns"


def save_results(feature_labels, layer, position, output_dir='src/experiments/llama_sae_hierarchy'):
    """Save feature taxonomy to JSON."""
    output_path = Path(output_dir) / f'feature_labels_layer{layer}_pos{position}.json'

    # Create metadata
    output = {
        'metadata': {
            'layer': layer,
            'position': position,
            'sae_config': 'K=100, d=512',
            'num_features': len(feature_labels),
            'analysis_date': '2025-10-27'
        },
        'features': feature_labels
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {output_path}\n")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Feature Taxonomy & Labeling')
    parser.add_argument('--layer', type=int, default=14, help='Layer index (0-15)')
    parser.add_argument('--position', type=int, default=3, help='Position index (0-5)')
    parser.add_argument('--top_n', type=int, default=20, help='Number of top features to analyze')
    parser.add_argument('--k', type=int, default=100, help='TopK sparsity level')
    parser.add_argument('--latent_dim', type=int, default=512, help='Dictionary size')

    args = parser.parse_args()

    # Load model and data
    model, activations, problems = load_sae_and_data(
        args.layer, args.position, args.k, args.latent_dim
    )

    # Analyze features
    feature_labels = analyze_features(model, activations, problems, args.top_n)

    # Save results
    output_path = save_results(feature_labels, args.layer, args.position)

    print(f"{'='*80}")
    print(f"Feature Taxonomy Complete!")
    print(f"{'='*80}")
    print(f"Analyzed {args.top_n} features from Layer {args.layer}, Position {args.position}")
    print(f"Results: {output_path}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
