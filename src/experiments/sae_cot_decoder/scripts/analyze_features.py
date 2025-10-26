"""
Feature Interpretability Analysis for SAE-based Continuous Thought Decoder.

Analyzes:
1. Feature-CoT token correlations
2. Layer selectivity
3. Position-specific patterns
4. Feature catalog generation

Usage:
    python analyze_features.py
"""

import torch
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
from scipy.stats import chi2_contingency
from transformers import AutoTokenizer

from sae_model import SparseAutoencoder


def load_trained_saes(model_dir: Path, device: str = 'cuda') -> Dict[int, SparseAutoencoder]:
    """Load all trained SAE models."""
    saes = {}
    for position in range(6):
        model_path = model_dir / f'pos_{position}_best.pt'
        if not model_path.exists():
            model_path = model_dir / f'pos_{position}_final.pt'

        sae = SparseAutoencoder(input_dim=2048, n_features=2048, l1_coefficient=0.0005)
        sae.load_state_dict(torch.load(model_path, map_location=device))
        sae = sae.to(device)
        sae.eval()

        saes[position] = sae

    return saes


def extract_all_features(
    saes: Dict[int, SparseAutoencoder],
    data: Dict,
    device: str = 'cuda'
) -> Dict:
    """Extract SAE features for all samples.

    Returns:
        Dictionary with features and metadata for each position
    """
    position_features = {}

    for position in range(6):
        print(f"\nExtracting features for position {position}...")

        # Get samples for this position
        position_indices = [
            i for i, pos in enumerate(data['metadata']['positions'])
            if pos == position
        ]

        hidden_states = data['hidden_states'][position_indices]
        sae = saes[position]

        # Extract features in batches
        batch_size = 4096
        all_features = []

        with torch.no_grad():
            for i in tqdm(range(0, len(hidden_states), batch_size), desc=f"Position {position}"):
                batch = hidden_states[i:i+batch_size].to(device)
                _, features = sae(batch)
                all_features.append(features.cpu())

        all_features = torch.cat(all_features, dim=0)

        # Store features and metadata
        position_features[position] = {
            'features': all_features,  # (N, 2048)
            'problem_ids': [data['metadata']['problem_ids'][i] for i in position_indices],
            'layers': [data['metadata']['layers'][i] for i in position_indices],
            'cot_steps': [data['metadata']['cot_steps'][i] for i in position_indices],
            'cot_token_ids': [data['metadata']['cot_token_ids'][i] for i in position_indices],
            'num_cot_steps': [data['metadata']['num_cot_steps'][i] for i in position_indices]
        }

        print(f"  Extracted {len(all_features)} feature vectors")

    return position_features


def analyze_feature_cot_correlations(
    position_features: Dict,
    tokenizer,
    top_k: int = 100
) -> Dict:
    """Analyze correlation between features and CoT tokens.

    For each feature:
    - Which CoT tokens appear when it's active?
    - Statistical significance (chi-squared test)
    """
    print(f"\n{'='*70}")
    print("FEATURE-COT TOKEN CORRELATION ANALYSIS")
    print(f"{'='*70}")

    feature_correlations = {}

    for position in range(6):
        print(f"\nAnalyzing position {position}...")

        features = position_features[position]['features']  # (N, 2048)
        cot_token_ids = position_features[position]['cot_token_ids']

        position_correlations = []

        # Analyze each feature
        for feature_id in tqdm(range(2048), desc=f"Position {position}"):
            feature_activations = features[:, feature_id].numpy()

            # Threshold for "active"
            activation_threshold = np.percentile(feature_activations[feature_activations > 0], 75) if (feature_activations > 0).any() else 0

            if activation_threshold == 0:
                # Dead feature
                continue

            # Samples where this feature is active
            active_samples = feature_activations > activation_threshold

            # Collect CoT tokens for active vs inactive samples
            active_tokens = []
            inactive_tokens = []

            for i, is_active in enumerate(active_samples):
                tokens = cot_token_ids[i]
                if is_active:
                    active_tokens.extend(tokens)
                else:
                    inactive_tokens.extend(tokens)

            if len(active_tokens) == 0:
                continue

            # Find tokens enriched in active samples
            active_token_counts = Counter(active_tokens)
            inactive_token_counts = Counter(inactive_tokens)

            # Chi-squared test for top tokens
            enriched_tokens = []
            for token_id, active_count in active_token_counts.most_common(20):
                inactive_count = inactive_token_counts.get(token_id, 0)

                # Only test if token appears enough times
                if active_count < 5:
                    continue

                # Chi-squared test
                contingency_table = [
                    [active_count, len(active_tokens) - active_count],
                    [inactive_count, len(inactive_tokens) - inactive_count]
                ]

                try:
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

                    if p_value < 0.01:  # Significant correlation
                        enriched_tokens.append({
                            'token_id': int(token_id),
                            'token_str': tokenizer.decode([token_id]),
                            'active_count': int(active_count),
                            'inactive_count': int(inactive_count),
                            'enrichment': float(active_count / (inactive_count + 1)),
                            'p_value': float(p_value)
                        })
                except:
                    continue

            if len(enriched_tokens) > 0:
                # Sort by enrichment
                enriched_tokens.sort(key=lambda x: x['enrichment'], reverse=True)

                position_correlations.append({
                    'feature_id': feature_id,
                    'position': position,
                    'activation_threshold': float(activation_threshold),
                    'num_active_samples': int(active_samples.sum()),
                    'enriched_tokens': enriched_tokens[:10],  # Top 10
                    'interpretability_score': len(enriched_tokens)
                })

        # Sort by interpretability score
        position_correlations.sort(key=lambda x: x['interpretability_score'], reverse=True)
        feature_correlations[position] = position_correlations[:top_k]

        print(f"  Found {len(position_correlations)} interpretable features")

    return feature_correlations


def analyze_layer_selectivity(position_features: Dict) -> Dict:
    """Analyze which layers each feature is most selective for."""
    print(f"\n{'='*70}")
    print("LAYER SELECTIVITY ANALYSIS")
    print(f"{'='*70}")

    layer_selectivity = {}

    for position in range(6):
        print(f"\nAnalyzing position {position}...")

        features = position_features[position]['features']  # (N, 2048)
        layers = np.array(position_features[position]['layers'])

        position_selectivity = []

        for feature_id in tqdm(range(2048), desc=f"Position {position}"):
            feature_activations = features[:, feature_id].numpy()

            if (feature_activations > 0).sum() < 10:
                # Too few activations
                continue

            # Calculate mean activation per layer
            layer_means = {}
            for layer in range(16):
                layer_mask = layers == layer
                if layer_mask.sum() > 0:
                    layer_means[layer] = feature_activations[layer_mask].mean()
                else:
                    layer_means[layer] = 0

            # Selectivity index (entropy-based)
            total_activation = sum(layer_means.values())
            if total_activation > 0:
                probs = np.array([v / total_activation for v in layer_means.values()])
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                max_entropy = np.log(16)  # 16 layers
                selectivity = 1 - (entropy / max_entropy)

                # Most selective layer
                most_selective_layer = max(layer_means, key=layer_means.get)

                position_selectivity.append({
                    'feature_id': feature_id,
                    'selectivity_index': float(selectivity),
                    'most_selective_layer': int(most_selective_layer),
                    'layer_means': {int(k): float(v) for k, v in layer_means.items()}
                })

        # Sort by selectivity
        position_selectivity.sort(key=lambda x: x['selectivity_index'], reverse=True)
        layer_selectivity[position] = position_selectivity

        print(f"  Analyzed {len(position_selectivity)} features")

    return layer_selectivity


def generate_feature_catalog(
    feature_correlations: Dict,
    layer_selectivity: Dict,
    output_path: Path
):
    """Generate comprehensive feature catalog."""
    print(f"\n{'='*70}")
    print("GENERATING FEATURE CATALOG")
    print(f"{'='*70}")

    catalog = {
        'summary': {
            'total_features': 2048 * 6,
            'interpretable_features_per_position': {}
        },
        'positions': {}
    }

    for position in range(6):
        # Combine correlation and selectivity data
        correlation_dict = {f['feature_id']: f for f in feature_correlations.get(position, [])}
        selectivity_dict = {f['feature_id']: f for f in layer_selectivity.get(position, [])}

        combined_features = []
        for feature_id in set(list(correlation_dict.keys()) + list(selectivity_dict.keys())):
            feature_info = {
                'feature_id': feature_id,
                'position': position
            }

            if feature_id in correlation_dict:
                feature_info.update(correlation_dict[feature_id])

            if feature_id in selectivity_dict:
                feature_info['selectivity'] = selectivity_dict[feature_id]

            combined_features.append(feature_info)

        # Sort by interpretability
        combined_features.sort(
            key=lambda x: x.get('interpretability_score', 0),
            reverse=True
        )

        catalog['positions'][position] = {
            'total_features': 2048,
            'interpretable_features': len(combined_features),
            'top_100_features': combined_features[:100]
        }

        catalog['summary']['interpretable_features_per_position'][position] = len(combined_features)

        print(f"Position {position}: {len(combined_features)} interpretable features")

    # Save catalog
    with open(output_path, 'w') as f:
        json.dump(catalog, f, indent=2)

    print(f"\n✓ Feature catalog saved: {output_path}")

    return catalog


def main():
    print("="*70)
    print("SAE FEATURE INTERPRETABILITY ANALYSIS")
    print("="*70)

    # Paths
    base_dir = Path("/home/paperspace/dev/CoT_Exploration")
    model_dir = base_dir / "src/experiments/sae_cot_decoder/models"
    data_dir = base_dir / "src/experiments/sae_cot_decoder/results"
    analysis_dir = base_dir / "src/experiments/sae_cot_decoder/analysis"
    analysis_dir.mkdir(exist_ok=True)

    test_data_path = data_dir / "enriched_test_data_with_cot.pt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    # Load SAEs
    print("\nLoading trained SAEs...")
    saes = load_trained_saes(model_dir, device)
    print(f"✓ Loaded {len(saes)} SAE models")

    # Load test data
    print("\nLoading test data...")
    test_data = torch.load(test_data_path, weights_only=False)
    print(f"✓ Loaded {len(test_data['hidden_states'])} samples")

    # Extract features
    position_features = extract_all_features(saes, test_data, device)

    # Save extracted features
    features_path = analysis_dir / "extracted_features.pt"
    torch.save(position_features, features_path)
    print(f"\n✓ Saved extracted features: {features_path}")

    # Analyze feature-CoT correlations
    feature_correlations = analyze_feature_cot_correlations(
        position_features,
        tokenizer,
        top_k=200
    )

    # Save correlations
    correlations_path = analysis_dir / "feature_cot_correlations.json"
    with open(correlations_path, 'w') as f:
        json.dump(feature_correlations, f, indent=2)
    print(f"\n✓ Saved correlations: {correlations_path}")

    # Analyze layer selectivity
    layer_selectivity = analyze_layer_selectivity(position_features)

    # Save selectivity
    selectivity_path = analysis_dir / "layer_selectivity.json"
    with open(selectivity_path, 'w') as f:
        json.dump(layer_selectivity, f, indent=2)
    print(f"✓ Saved selectivity: {selectivity_path}")

    # Generate feature catalog
    catalog_path = analysis_dir / "feature_catalog.json"
    catalog = generate_feature_catalog(
        feature_correlations,
        layer_selectivity,
        catalog_path
    )

    # Print summary
    print(f"\n{'='*70}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print(f"Total features analyzed: {2048 * 6:,}")
    for pos, count in catalog['summary']['interpretable_features_per_position'].items():
        print(f"  Position {pos}: {count} interpretable features")

    print(f"\n{'='*70}")
    print("FEATURE ANALYSIS COMPLETE!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
