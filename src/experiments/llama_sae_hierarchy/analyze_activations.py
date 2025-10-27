"""
Feature Activation Analysis - Find Specialized Features.

Analyzes mid-frequency features (rank 50-200) to find specialized features:
- Operation-specific: Activate on multiplication but not addition
- Value-specific: Activate on problems with "12" but not "50"

Usage:
    python analyze_activations.py --layer 14 --position 3 --start_rank 50 --end_rank 200
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add topk_grid_pilot to path
sys.path.insert(0, 'src/experiments/topk_grid_pilot')
from topk_sae import TopKAutoencoder


def load_sae_and_data(layer, position, k=100, latent_dim=512):
    """Load SAE model and validation data."""
    print(f"\n{'='*80}")
    print(f"Loading SAE: Layer {layer}, Position {position}, K={k}, d={latent_dim}")
    print(f"{'='*80}\n")

    # Load SAE checkpoint - check both locations
    ckpt_paths = [
        f'src/experiments/llama_sae_hierarchy/checkpoints/pos{position}_layer{layer}_d{latent_dim}_k{k}.pt',
        f'src/experiments/topk_grid_pilot/results/checkpoints/pos{position}_layer{layer}_d{latent_dim}_k{k}.pt'
    ]

    ckpt_path = None
    for path in ckpt_paths:
        if Path(path).exists():
            ckpt_path = path
            break

    if ckpt_path is None:
        raise FileNotFoundError(f"Checkpoint not found in any location: {ckpt_paths}")

    ckpt = torch.load(ckpt_path, weights_only=False)
    model = TopKAutoencoder(
        input_dim=ckpt['config']['input_dim'],
        latent_dim=ckpt['config']['latent_dim'],
        k=ckpt['config']['k']
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Load validation data
    val_data_path = 'src/experiments/sae_cot_decoder/data/full_val_activations.pt'

    if not Path(val_data_path).exists():
        raise FileNotFoundError(f"Validation data not found: {val_data_path}")

    val_data = torch.load(val_data_path, weights_only=False)
    positions = np.array(val_data['metadata']['positions'])
    layers = np.array(val_data['metadata']['layers'])

    mask = (positions == position) & (layers == layer)
    activations = val_data['activations'][mask]

    problem_ids = [val_data['metadata']['problem_ids'][i] for i, m in enumerate(mask) if m]
    cot_sequences = [val_data['metadata']['cot_sequences'][i] for i, m in enumerate(mask) if m]

    problems = []
    for pid, cot in zip(problem_ids, cot_sequences):
        problems.append({
            'problem_id': pid,
            'cot_sequence': cot,
            'cot_text': ' | '.join(cot)
        })

    print(f"Loaded {len(activations)} samples\n")

    return model, activations, problems


def compute_specialization_scores(cot_texts):
    """
    Compute specialization scores for operations and values.

    Returns:
        dict with operation and value specialization scores
    """
    # Count occurrences
    addition = sum(1 for cot in cot_texts if '+' in cot or 'sum' in cot.lower())
    subtraction = sum(1 for cot in cot_texts if '-' in cot or 'difference' in cot.lower())
    multiplication = sum(1 for cot in cot_texts if '*' in cot or 'multiply' in cot.lower() or '×' in cot)
    division = sum(1 for cot in cot_texts if '/' in cot or 'divide' in cot.lower() or '÷' in cot)

    total = len(cot_texts)

    # Operation percentages
    op_scores = {
        'addition': 100.0 * addition / total if total > 0 else 0,
        'subtraction': 100.0 * subtraction / total if total > 0 else 0,
        'multiplication': 100.0 * multiplication / total if total > 0 else 0,
        'division': 100.0 * division / total if total > 0 else 0,
    }

    # Value counts
    value_counts = {}
    for num in [12, 20, 30, 50, 100]:
        count = sum(1 for cot in cot_texts if str(num) in cot)
        value_counts[str(num)] = count

    return {
        'operations': op_scores,
        'values': value_counts,
        'total_samples': total
    }


def classify_specialization(scores):
    """
    Determine if feature is specialized.

    A feature is specialized if:
    - Operation-specialized: One operation >70%, others <30%
    - Value-specialized: One value appears in >50% samples, others <20%
    - General: Multiple operations/values above thresholds
    """
    ops = scores['operations']
    vals = scores['values']
    total = scores['total_samples']

    # Check operation specialization
    op_values = list(ops.values())
    max_op = max(op_values)
    max_op_name = [k for k, v in ops.items() if v == max_op][0]
    other_ops = [v for v in op_values if v != max_op]

    is_op_specialized = max_op > 70 and all(v < 30 for v in other_ops)

    # Check value specialization
    val_percentages = {k: 100.0 * v / total for k, v in vals.items() if total > 0}
    max_val_pct = max(val_percentages.values()) if val_percentages else 0
    max_val_name = [k for k, v in val_percentages.items() if v == max_val_pct][0] if val_percentages else None
    other_val_pcts = [v for v in val_percentages.values() if v != max_val_pct]

    is_val_specialized = max_val_pct > 50 and all(v < 20 for v in other_val_pcts)

    # Classification
    if is_op_specialized and is_val_specialized:
        return 'highly-specialized', f"{max_op_name} + number {max_val_name}"
    elif is_op_specialized:
        return 'operation-specialized', max_op_name
    elif is_val_specialized:
        return 'value-specialized', f"number {max_val_name}"
    else:
        return 'general', 'mixed patterns'


def analyze_feature_range(model, activations, problems, start_rank, end_rank):
    """
    Analyze features in specified rank range.

    Args:
        model: TopK SAE model
        activations: Validation activations
        problems: Problem metadata
        start_rank: Starting rank (e.g., 50)
        end_rank: Ending rank (e.g., 200)

    Returns:
        feature_analyses: List of analyzed features with specialization info
    """
    print(f"\n{'='*80}")
    print(f"Analyzing Features: Rank {start_rank} to {end_rank}")
    print(f"{'='*80}\n")

    # Run SAE
    with torch.no_grad():
        _, sparse, _ = model(activations)

    # Compute feature statistics
    feature_activation_freq = (sparse != 0).float().mean(dim=0)
    feature_mean_magnitude = sparse.abs().mean(dim=0)

    # Get features in rank range
    all_features_idx = torch.argsort(feature_activation_freq, descending=True)
    features_in_range = all_features_idx[start_rank-1:end_rank]  # -1 for 0-indexing

    print(f"Analyzing {len(features_in_range)} features...\n")
    print(f"{'Rank':<6} {'Feature':<10} {'Act %':<10} {'Type':<25} {'Description'}")
    print(f"{'-'*95}")

    feature_analyses = []
    specialized_features = []

    for local_idx, feat_idx in enumerate(features_in_range):
        feat_idx = feat_idx.item()
        rank = start_rank + local_idx

        # Get active samples
        active_mask = sparse[:, feat_idx] != 0
        active_indices = torch.where(active_mask)[0]

        if len(active_indices) == 0:
            continue

        # Get top 100 samples by magnitude
        active_magnitudes = sparse[active_indices, feat_idx].abs()
        top_k = min(100, len(active_indices))
        top_magnitudes, top_local_indices = torch.topk(active_magnitudes, k=top_k)
        top_global_indices = active_indices[top_local_indices]

        # Extract CoT texts
        cot_texts = [problems[idx.item()]['cot_text'] for idx in top_global_indices]

        # Compute specialization scores
        scores = compute_specialization_scores(cot_texts)

        # Classify specialization
        spec_type, spec_desc = classify_specialization(scores)

        # Store analysis
        analysis = {
            'rank': rank,
            'feature_id': feat_idx,
            'activation_freq': float(feature_activation_freq[feat_idx]),
            'mean_magnitude': float(feature_mean_magnitude[feat_idx]),
            'specialization': {
                'type': spec_type,
                'description': spec_desc,
                'scores': scores
            }
        }

        feature_analyses.append(analysis)

        # Track specialized features
        if spec_type != 'general':
            specialized_features.append(analysis)

            # Store top samples for specialized features
            top_samples = []
            for idx in top_global_indices[:10]:
                idx = idx.item()
                top_samples.append({
                    'problem_id': problems[idx]['problem_id'],
                    'cot': problems[idx]['cot_text'],
                    'activation': float(sparse[idx, feat_idx])
                })
            analysis['top_samples'] = top_samples

        # Print summary
        act_pct = f"{100*analysis['activation_freq']:.1f}%"
        print(f"{rank:<6} {feat_idx:<10} {act_pct:<10} {spec_type:<25} {spec_desc}")

    print(f"{'-'*95}\n")

    # Summary statistics
    type_counts = {}
    for f in feature_analyses:
        ftype = f['specialization']['type']
        type_counts[ftype] = type_counts.get(ftype, 0) + 1

    print(f"Summary:")
    print(f"  Total features analyzed: {len(feature_analyses)}")
    print(f"  Specialized features found: {len(specialized_features)}")
    print(f"  Feature types:")
    for ftype, count in sorted(type_counts.items()):
        print(f"    {ftype}: {count} ({100*count/len(feature_analyses):.1f}%)")
    print()

    return feature_analyses, specialized_features


def select_swap_pairs(specialized_features):
    """
    Select candidate feature pairs for swap experiments.

    Criteria:
    - Value-specialized features with different target numbers
    - Operation-specialized features with different operations
    """
    print(f"\n{'='*80}")
    print(f"Selecting Swap Pairs")
    print(f"{'='*80}\n")

    # Group by specialization type
    value_features = {}
    operation_features = {}

    for f in specialized_features:
        spec_type = f['specialization']['type']
        spec_desc = f['specialization']['description']

        if spec_type == 'value-specialized':
            # Extract number from description (e.g., "number 12" -> "12")
            num = spec_desc.replace('number ', '')
            if num not in value_features:
                value_features[num] = []
            value_features[num].append(f)

        elif spec_type == 'operation-specialized':
            # Extract operation (e.g., "multiplication")
            op = spec_desc
            if op not in operation_features:
                operation_features[op] = []
            operation_features[op].append(f)

    swap_pairs = []

    # Create value-based pairs
    value_numbers = list(value_features.keys())
    if len(value_numbers) >= 2:
        for i in range(len(value_numbers)):
            for j in range(i+1, len(value_numbers)):
                num_a = value_numbers[i]
                num_b = value_numbers[j]

                # Pick highest activation frequency from each group
                feat_a = sorted(value_features[num_a], key=lambda x: x['activation_freq'], reverse=True)[0]
                feat_b = sorted(value_features[num_b], key=lambda x: x['activation_freq'], reverse=True)[0]

                swap_pairs.append({
                    'pair_type': 'value',
                    'feature_a': feat_a['feature_id'],
                    'feature_b': feat_b['feature_id'],
                    'description_a': f"number {num_a}",
                    'description_b': f"number {num_b}",
                    'prediction': f"Swapping {num_a}↔{num_b} features should change answers"
                })

    # Create operation-based pairs
    operations = list(operation_features.keys())
    if len(operations) >= 2:
        for i in range(len(operations)):
            for j in range(i+1, len(operations)):
                op_a = operations[i]
                op_b = operations[j]

                feat_a = sorted(operation_features[op_a], key=lambda x: x['activation_freq'], reverse=True)[0]
                feat_b = sorted(operation_features[op_b], key=lambda x: x['activation_freq'], reverse=True)[0]

                swap_pairs.append({
                    'pair_type': 'operation',
                    'feature_a': feat_a['feature_id'],
                    'feature_b': feat_b['feature_id'],
                    'description_a': op_a,
                    'description_b': op_b,
                    'prediction': f"Swapping {op_a}↔{op_b} features should affect operation type"
                })

    if swap_pairs:
        print(f"Found {len(swap_pairs)} candidate swap pairs:\n")
        for i, pair in enumerate(swap_pairs, 1):
            print(f"{i}. {pair['pair_type'].upper()} pair:")
            print(f"   Feature {pair['feature_a']} ({pair['description_a']}) ↔ "
                  f"Feature {pair['feature_b']} ({pair['description_b']})")
            print(f"   Prediction: {pair['prediction']}\n")
    else:
        print("No swap pairs found. Need more specialized features.\n")

    return swap_pairs


def save_results(feature_analyses, specialized_features, swap_pairs, layer, position, start_rank, end_rank, k=None, output_dir='src/experiments/llama_sae_hierarchy'):
    """Save activation analysis results."""
    if k and k != 100:
        output_path = Path(output_dir) / f'activation_analysis_layer{layer}_pos{position}_rank{start_rank}-{end_rank}_k{k}.json'
    else:
        output_path = Path(output_dir) / f'activation_analysis_layer{layer}_pos{position}_rank{start_rank}-{end_rank}.json'

    output = {
        'metadata': {
            'layer': layer,
            'position': position,
            'sae_config': 'K=100, d=512',
            'rank_range': [start_rank, end_rank],
            'total_features_analyzed': len(feature_analyses),
            'specialized_features_found': len(specialized_features),
            'swap_pairs_generated': len(swap_pairs),
            'analysis_date': '2025-10-27'
        },
        'all_features': feature_analyses,
        'specialized_features': specialized_features,
        'swap_pairs': swap_pairs
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}\n")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Feature Activation Analysis')
    parser.add_argument('--layer', type=int, default=14, help='Layer index (0-15)')
    parser.add_argument('--position', type=int, default=3, help='Position index (0-5)')
    parser.add_argument('--start_rank', type=int, default=50, help='Starting rank')
    parser.add_argument('--end_rank', type=int, default=200, help='Ending rank')
    parser.add_argument('--k', type=int, default=100, help='TopK sparsity level')
    parser.add_argument('--latent_dim', type=int, default=512, help='Dictionary size')

    args = parser.parse_args()

    # Load model and data
    model, activations, problems = load_sae_and_data(
        args.layer, args.position, args.k, args.latent_dim
    )

    # Analyze feature range
    feature_analyses, specialized_features = analyze_feature_range(
        model, activations, problems, args.start_rank, args.end_rank
    )

    # Select swap pairs
    swap_pairs = select_swap_pairs(specialized_features)

    # Save results
    output_path = save_results(
        feature_analyses, specialized_features, swap_pairs,
        args.layer, args.position, args.start_rank, args.end_rank, k=args.k
    )

    print(f"Analysis complete!")
    print(f"  Features analyzed: {len(feature_analyses)}")
    print(f"  Specialized features: {len(specialized_features)}")
    print(f"  Swap pairs: {len(swap_pairs)}")
    print(f"  Output: {output_path}\n")


if __name__ == '__main__':
    main()
