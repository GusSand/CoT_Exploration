"""
Neuronpedia-style feature visualization for TopK SAE.

Creates interactive HTML dashboard showing:
1. Max activating samples per feature
2. Token correlation heatmap
3. Layer selectivity analysis

Usage:
    python visualize_features_neuronpedia.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import chi2_contingency

sys.path.insert(0, 'src/experiments/topk_grid_pilot')
from topk_sae import TopKAutoencoder


def load_sae_and_data(layer=14, position=3, k=100, latent_dim=512):
    """Load SAE model and validation data."""
    print(f"Loading SAE model: Layer {layer}, Position {position}, K={k}, d={latent_dim}")

    # Load model
    ckpt_path = f'src/experiments/topk_grid_pilot/results/checkpoints/pos{position}_layer{layer}_d{latent_dim}_k{k}.pt'
    ckpt = torch.load(ckpt_path, weights_only=False)

    model = TopKAutoencoder(
        input_dim=ckpt['config']['input_dim'],
        latent_dim=ckpt['config']['latent_dim'],
        k=ckpt['config']['k']
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Load validation data
    print("Loading validation data...")
    val_data = torch.load('src/experiments/sae_cot_decoder/data/full_val_activations.pt', weights_only=False)
    positions = np.array(val_data['metadata']['positions'])
    layers = np.array(val_data['metadata']['layers'])

    mask = (positions == position) & (layers == layer)
    activations = val_data['activations'][mask]

    # Extract metadata
    problem_ids = [val_data['metadata']['problem_ids'][i] for i, m in enumerate(mask) if m]
    cot_sequences = [val_data['metadata']['cot_sequences'][i] for i, m in enumerate(mask) if m]

    print(f"Loaded {len(activations)} samples for Layer {layer}, Position {position}")

    return model, activations, problem_ids, cot_sequences


def extract_features(model, activations):
    """Extract sparse feature activations."""
    print("Extracting sparse features...")

    with torch.no_grad():
        _, sparse_features, _ = model(activations)

    return sparse_features


def analyze_token_correlations(sparse_features, cot_sequences, top_n_features=10, activation_threshold_percentile=75):
    """
    Analyze which CoT tokens correlate with each feature.

    Returns dict of feature_id -> list of (token, enrichment, p_value)
    """
    print(f"\nAnalyzing token correlations for top {top_n_features} features...")

    # Identify top features by activation frequency
    activation_frequency = (sparse_features != 0).float().mean(dim=0)
    top_features = activation_frequency.argsort(descending=True)[:top_n_features]

    # Tokenize CoT sequences
    all_tokens = []
    for seq in cot_sequences:
        tokens = []
        for step in seq:
            # Split calculation steps into tokens
            # E.g., "16-3-4=9" -> ["16", "-", "3", "-", "4", "=", "9"]
            import re
            step_tokens = re.findall(r'\d+|[+\-*/=]', step)
            tokens.extend(step_tokens)
        all_tokens.append(tokens)

    # Build vocabulary
    vocab = set()
    for tokens in all_tokens:
        vocab.update(tokens)
    vocab = sorted(list(vocab))

    print(f"Vocabulary size: {len(vocab)} unique tokens")

    # For each feature, find correlated tokens
    feature_correlations = {}

    for feat_id in top_features:
        feat_id = feat_id.item()

        # Activation threshold
        activations = sparse_features[:, feat_id]
        threshold = torch.quantile(activations, activation_threshold_percentile / 100.0)
        active_mask = activations > threshold

        # Token enrichment analysis
        token_enrichments = []

        for token in vocab:
            # Count samples with/without token
            has_token = np.array([token in tokens for tokens in all_tokens])

            # 2x2 contingency table
            active_with_token = (active_mask.numpy() & has_token).sum()
            active_without_token = (active_mask.numpy() & ~has_token).sum()
            inactive_with_token = (~active_mask.numpy() & has_token).sum()
            inactive_without_token = (~active_mask.numpy() & ~has_token).sum()

            contingency = [
                [active_with_token, active_without_token],
                [inactive_with_token, inactive_without_token]
            ]

            # Chi-squared test
            if active_with_token > 0:
                chi2, p_value, _, _ = chi2_contingency(contingency)

                # Enrichment: fraction of active samples with token
                enrichment = active_with_token / max(1, active_mask.sum().item())

                if p_value < 0.01:  # Significant
                    token_enrichments.append({
                        'token': token,
                        'enrichment': enrichment,
                        'p_value': p_value,
                        'active_count': int(active_with_token),
                        'inactive_count': int(inactive_with_token)
                    })

        # Sort by enrichment
        token_enrichments.sort(key=lambda x: x['enrichment'], reverse=True)

        feature_correlations[feat_id] = {
            'activation_frequency': activation_frequency[feat_id].item(),
            'top_tokens': token_enrichments[:20],  # Top 20
            'num_active_samples': active_mask.sum().item()
        }

        print(f"  Feature {feat_id}: {len(token_enrichments)} significant tokens, "
              f"{active_mask.sum().item()} active samples")

    return feature_correlations, top_features.tolist()


def find_max_activating_samples(sparse_features, problem_ids, cot_sequences, feature_ids, top_k=10):
    """Find top K samples that activate each feature most strongly."""
    print(f"\nFinding top {top_k} activating samples per feature...")

    max_activations = {}

    for feat_id in feature_ids:
        activations = sparse_features[:, feat_id]

        # Top K samples
        topk_values, topk_indices = torch.topk(activations, k=min(top_k, len(activations)))

        samples = []
        for val, idx in zip(topk_values, topk_indices):
            if val > 0:  # Only include if actually activated
                samples.append({
                    'problem_id': problem_ids[idx],
                    'cot_sequence': cot_sequences[idx],
                    'activation': val.item(),
                    'sample_idx': idx.item()
                })

        max_activations[feat_id] = samples

    return max_activations


def analyze_layer_selectivity(model_base, feature_ids, position=3, layers=list(range(16))):
    """
    Analyze which layers each feature is most selective to.

    Loads SAE for same feature across all layers.
    """
    print(f"\nAnalyzing layer selectivity for {len(feature_ids)} features across {len(layers)} layers...")

    # Load validation data once
    val_data = torch.load('src/experiments/sae_cot_decoder/data/full_val_activations.pt', weights_only=False)
    positions_meta = np.array(val_data['metadata']['positions'])
    layers_meta = np.array(val_data['metadata']['layers'])

    layer_selectivity = {}

    for feat_id in feature_ids:
        layer_means = {}

        for layer in layers:
            try:
                # Load SAE for this layer
                ckpt_path = f'src/experiments/topk_grid_pilot/results/checkpoints/pos{position}_layer{layer}_d512_k100.pt'
                ckpt = torch.load(ckpt_path, weights_only=False)

                sae = TopKAutoencoder(
                    input_dim=ckpt['config']['input_dim'],
                    latent_dim=ckpt['config']['latent_dim'],
                    k=ckpt['config']['k']
                )
                sae.load_state_dict(ckpt['model_state_dict'])
                sae.eval()

                # Extract activations for this layer
                mask = (positions_meta == position) & (layers_meta == layer)
                activations = val_data['activations'][mask]

                # Get feature activations
                with torch.no_grad():
                    _, sparse_features, _ = sae(activations)

                # Mean activation for this feature
                layer_means[layer] = sparse_features[:, feat_id].mean().item()

            except FileNotFoundError:
                layer_means[layer] = 0.0

        # Compute selectivity index (normalized entropy)
        values = np.array(list(layer_means.values()))
        if values.sum() > 0:
            probs = values / values.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(len(layers))
            selectivity_index = 1.0 - (entropy / max_entropy)
        else:
            selectivity_index = 0.0

        most_selective_layer = max(layer_means, key=layer_means.get)

        layer_selectivity[feat_id] = {
            'layer_means': layer_means,
            'selectivity_index': selectivity_index,
            'most_selective_layer': most_selective_layer
        }

        print(f"  Feature {feat_id}: Most selective to Layer {most_selective_layer} "
              f"(index={selectivity_index:.3f})")

    return layer_selectivity


def save_feature_data(feature_correlations, max_activations, layer_selectivity, feature_ids, output_path):
    """Save extracted feature data for visualization."""
    print(f"\nSaving feature data to {output_path}...")

    data = {
        'feature_ids': [int(f) for f in feature_ids],
        'features': {}
    }

    for feat_id in feature_ids:
        data['features'][int(feat_id)] = {
            'correlations': feature_correlations.get(feat_id, {}),
            'max_activations': max_activations.get(feat_id, []),
            'layer_selectivity': layer_selectivity.get(feat_id, {})
        }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✓ Saved feature data for {len(feature_ids)} features")


def main():
    """Extract feature data for visualization."""
    # Configuration
    layer = 14
    position = 3
    k = 100
    latent_dim = 512

    # Create output directory
    output_dir = Path('src/experiments/topk_grid_pilot/visualizations')
    output_dir.mkdir(exist_ok=True)

    # Step 1: Load model and data
    model, activations, problem_ids, cot_sequences = load_sae_and_data(
        layer=layer, position=position, k=k, latent_dim=latent_dim
    )

    # Step 2: Extract features
    sparse_features = extract_features(model, activations)

    # Step 3: Load specialized feature IDs (from find_interesting_features.py)
    specialized_file = output_dir / 'specialized_features.txt'
    if specialized_file.exists():
        print(f"\nLoading specialized feature IDs from: {specialized_file}")
        with open(specialized_file, 'r') as f:
            top_feature_ids = [int(line.strip()) for line in f]
        print(f"Loaded {len(top_feature_ids)} specialized features")
    else:
        print(f"\nWARNING: {specialized_file} not found. Using top 10 most frequent features.")
        activation_frequency = (sparse_features != 0).float().mean(dim=0)
        top_feature_ids = activation_frequency.argsort(descending=True)[:10].tolist()

    # Step 4: Analyze token correlations for these specific features
    feature_correlations = {}
    for feat_id in top_feature_ids:
        activations_feat = sparse_features[:, feat_id]
        threshold = torch.quantile(activations_feat[activations_feat > 0], 0.75) if (activations_feat > 0).any() else 0
        active_mask = activations_feat > threshold

        # Token enrichment analysis (simplified from analyze_token_correlations)
        all_tokens = []
        for seq in cot_sequences:
            tokens = []
            for step in seq:
                import re
                step_tokens = re.findall(r'\d+|[+\-*/=]', step)
                tokens.extend(step_tokens)
            all_tokens.append(tokens)

        vocab = set()
        for tokens in all_tokens:
            vocab.update(tokens)
        vocab = sorted(list(vocab))

        token_enrichments = []
        for token in vocab:
            has_token = np.array([token in tokens for tokens in all_tokens])

            active_with_token = (active_mask.numpy() & has_token).sum()
            active_without_token = (active_mask.numpy() & ~has_token).sum()
            inactive_with_token = (~active_mask.numpy() & has_token).sum()
            inactive_without_token = (~active_mask.numpy() & ~has_token).sum()

            contingency = [
                [active_with_token, active_without_token],
                [inactive_with_token, inactive_without_token]
            ]

            if active_with_token > 0:
                from scipy.stats import chi2_contingency
                chi2, p_value, _, _ = chi2_contingency(contingency)
                enrichment = active_with_token / max(1, active_mask.sum().item())

                if p_value < 0.01:
                    token_enrichments.append({
                        'token': token,
                        'enrichment': enrichment,
                        'p_value': p_value,
                        'active_count': int(active_with_token),
                        'inactive_count': int(inactive_with_token)
                    })

        token_enrichments.sort(key=lambda x: x['enrichment'], reverse=True)

        activation_frequency = (sparse_features[:, feat_id] != 0).float().mean().item()
        feature_correlations[feat_id] = {
            'activation_frequency': activation_frequency,
            'top_tokens': token_enrichments[:20],
            'num_active_samples': active_mask.sum().item()
        }

        print(f"  Feature {feat_id}: {len(token_enrichments)} significant tokens, "
              f"{active_mask.sum().item()} active samples, "
              f"{activation_frequency:.1%} activation freq")

    # Step 4: Find max activating samples
    max_activations = find_max_activating_samples(
        sparse_features, problem_ids, cot_sequences, top_feature_ids, top_k=10
    )

    # Step 5: Analyze layer selectivity
    layer_selectivity = analyze_layer_selectivity(
        model, top_feature_ids, position=position
    )

    # Step 6: Save feature data
    output_path = output_dir / 'feature_data.json'
    save_feature_data(
        feature_correlations, max_activations, layer_selectivity,
        top_feature_ids, output_path
    )

    print("\n" + "="*80)
    print("✓ Feature data extraction complete!")
    print(f"✓ Output: {output_path}")
    print("="*80)

    return output_path


if __name__ == '__main__':
    main()
