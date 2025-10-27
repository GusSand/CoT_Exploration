"""
Analyze and compare semantic interpretability of features from Layer 3 vs Layer 14.

Goal: Determine if clean early layers or messy late layers produce more
semantically meaningful features for mechanistic interpretability.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, 'src/experiments/topk_grid_pilot')
from topk_sae import TopKAutoencoder


def load_sae_and_data(layer, position, k, latent_dim):
    """Load SAE model and corresponding validation data."""
    # Load model
    ckpt_path = f'src/experiments/topk_grid_pilot/results/pos{position}_layer{layer}_d{latent_dim}_k{k}.pt'
    ckpt = torch.load(ckpt_path, weights_only=False)

    model = TopKAutoencoder(
        input_dim=ckpt['config']['input_dim'],
        latent_dim=ckpt['config']['latent_dim'],
        k=ckpt['config']['k']
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Load validation data
    val_data = torch.load('src/experiments/sae_cot_decoder/data/full_val_activations.pt', weights_only=False)
    positions = np.array(val_data['metadata']['positions'])
    layers = np.array(val_data['metadata']['layers'])

    mask = (positions == position) & (layers == layer)
    activations = val_data['activations'][mask]

    # Extract problem metadata for the filtered samples
    problem_ids = [val_data['metadata']['problem_ids'][i] for i, m in enumerate(mask) if m]
    cot_sequences = [val_data['metadata']['cot_sequences'][i] for i, m in enumerate(mask) if m]

    # Create problem dict with CoT sequences as content
    problems = []
    for pid, cot in zip(problem_ids, cot_sequences):
        problems.append({
            'problem_id': pid,
            'question': ' | '.join(cot),  # Use CoT as question proxy
            'cot_sequence': cot
        })

    return model, activations, problems


def analyze_features(model, activations, problems, top_n=20):
    """
    Analyze top N features by finding what activates them.

    Returns: List of feature analysis dicts with semantic interpretations.
    """
    print(f"  Analyzing {activations.shape[0]} samples...")

    # Run SAE on all activations
    with torch.no_grad():
        _, sparse, _ = model(activations)

    # Compute feature statistics
    feature_activation_freq = (sparse != 0).float().mean(dim=0)  # How often each feature fires
    feature_mean_magnitude = sparse.abs().mean(dim=0)  # Average magnitude when active

    # Get top features by activation frequency
    top_features_idx = torch.argsort(feature_activation_freq, descending=True)[:top_n]

    feature_analyses = []

    for rank, feat_idx in enumerate(top_features_idx, 1):
        feat_idx = feat_idx.item()

        # Find samples where this feature is active
        active_mask = sparse[:, feat_idx] != 0
        active_samples = torch.where(active_mask)[0]

        if len(active_samples) == 0:
            continue

        # Get top 5 samples with highest activation
        activations_for_feat = sparse[active_mask, feat_idx]
        top_5_idx = torch.argsort(activations_for_feat, descending=True)[:5]
        top_5_samples = active_samples[top_5_idx]

        # Get problem data for these samples
        sample_problems = []
        for sample_idx in top_5_samples[:5]:
            prob = problems[sample_idx.item()]
            question_str = prob['question']
            sample_problems.append({
                'question': question_str[:100] + '...' if len(question_str) > 100 else question_str,
                'problem_id': prob['problem_id'],
                'activation': sparse[sample_idx, feat_idx].item()
            })

        feature_analyses.append({
            'feature_idx': feat_idx,
            'rank': rank,
            'activation_freq': feature_activation_freq[feat_idx].item(),
            'mean_magnitude': feature_mean_magnitude[feat_idx].item(),
            'num_activations': len(active_samples),
            'top_samples': sample_problems
        })

    return feature_analyses


def interpret_feature(feature_analysis):
    """
    Try to interpret what a feature represents based on its top activations.

    This is a heuristic - looks for patterns in questions.
    """
    # Look at all questions
    questions = [s['question'] for s in feature_analysis['top_samples']]

    # Simple heuristics for interpretation
    interpretations = []

    # Check for mathematical operations
    if sum('+' in q for q in questions) >= 3:
        interpretations.append('Addition operations')
    if sum('-' in q for q in questions) >= 3:
        interpretations.append('Subtraction operations')
    if sum('*' in q or 'multiply' in q.lower() for q in questions) >= 3:
        interpretations.append('Multiplication operations')
    if sum('/' in q or 'divide' in q.lower() for q in questions) >= 3:
        interpretations.append('Division operations')

    # Check for numbers
    if sum(any(str(n) in q for n in [100, 200, 500, 1000]) for q in questions) >= 3:
        interpretations.append('Round/large numbers')

    # Check for keywords
    keywords = ['how many', 'total', 'sum', 'difference', 'product', 'each', 'per']
    for kw in keywords:
        if sum(kw in q.lower() for q in questions) >= 3:
            interpretations.append(f'"{kw}" questions')

    if not interpretations:
        interpretations.append('Unknown/Mixed pattern')

    return interpretations


def main():
    print("=" * 80)
    print("Feature Semantic Analysis: Layer 3 vs Layer 14")
    print("=" * 80)
    print()

    # Configuration (using sweet spot: K=100, d=512)
    position = 3
    k = 100
    latent_dim = 512

    results = {}

    for layer in [3, 14]:
        print(f"\n{'=' * 80}")
        print(f"ANALYZING LAYER {layer} (K={k}, d={latent_dim}, Position {position})")
        print(f"{'=' * 80}\n")

        # Load model and data
        print("Loading model and data...")
        model, activations, problems = load_sae_and_data(layer, position, k, latent_dim)

        # Analyze features
        print("Analyzing top 20 features...")
        feature_analyses = analyze_features(model, activations, problems, top_n=20)

        # Interpret features
        print(f"\nTop 10 Features for Layer {layer}:")
        print("-" * 80)

        for i, feat in enumerate(feature_analyses[:10], 1):
            interpretations = interpret_feature(feat)

            print(f"\n{i}. Feature {feat['feature_idx']}:")
            print(f"   Activation Frequency: {feat['activation_freq']:.1%}")
            print(f"   Mean Magnitude: {feat['mean_magnitude']:.3f}")
            print(f"   Interpretations: {', '.join(interpretations)}")
            print(f"   Sample questions:")
            for j, sample in enumerate(feat['top_samples'][:3], 1):
                print(f"     {j}. {sample['question'][:80]}...")

        results[f'layer_{layer}'] = feature_analyses

    # Compare interpretability
    print("\n" + "=" * 80)
    print("INTERPRETABILITY COMPARISON")
    print("=" * 80)
    print()

    for layer in [3, 14]:
        features = results[f'layer_{layer}']

        # Count how many features have clear interpretations
        clear_interp = 0
        for feat in features[:10]:
            interps = interpret_feature(feat)
            if 'Unknown/Mixed pattern' not in interps:
                clear_interp += 1

        print(f"Layer {layer}:")
        print(f"  Features with clear patterns: {clear_interp}/10")
        print(f"  Average activation frequency: {np.mean([f['activation_freq'] for f in features[:10]]):.1%}")
        print()

    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
