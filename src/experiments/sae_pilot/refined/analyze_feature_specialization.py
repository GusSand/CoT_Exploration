"""
Deep feature analysis: operation/layer/token specialization patterns.

Validates whether SAE features discover operation-specific patterns
consistent with operation circuits findings.

Key questions:
1. Do specific features specialize in multiplication vs addition vs mixed?
2. Do features show layer preferences (early/middle/late)?
3. Do features show token position preferences (0-5)?
4. Are operation-specific features concentrated in Token 1 L8?
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
import json
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    """Simple sparse autoencoder."""

    def __init__(self, input_dim: int = 2048, n_features: int = 2048):
        super().__init__()
        self.input_dim = input_dim
        self.n_features = n_features
        self.encoder = nn.Linear(input_dim, n_features, bias=True)
        self.decoder = nn.Linear(n_features, input_dim, bias=True)

    def encode(self, x):
        return torch.relu(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


def load_sae(weights_path: str, device='cuda'):
    """Load trained SAE from checkpoint."""
    checkpoint = torch.load(weights_path, map_location=device)
    sae = SparseAutoencoder(
        input_dim=checkpoint['config']['input_dim'],
        n_features=checkpoint['config']['n_features']
    )
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.to(device)
    sae.eval()
    return sae, checkpoint


def compute_feature_statistics(features_np, metadata):
    """
    Compute comprehensive statistics for each feature.

    Returns dict with per-feature:
    - operation_specialization: {op_type: mean_activation}
    - layer_specialization: {layer: mean_activation}
    - token_specialization: {token: mean_activation}
    - selectivity_scores: how operation-specific the feature is
    """
    n_features = features_np.shape[1]
    n_vectors = features_np.shape[0]

    print(f"\nComputing feature statistics for {n_features:,} features...")

    feature_stats = {
        'operation_acts': defaultdict(lambda: {'mixed': [], 'pure_addition': [], 'pure_multiplication': []}),
        'layer_acts': defaultdict(lambda: {4: [], 8: [], 14: []}),
        'token_acts': defaultdict(lambda: {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}),
        'usage_count': np.zeros(n_features),
        'mean_activation': np.zeros(n_features),
        'max_activation': np.zeros(n_features)
    }

    # Collect activations by context
    for vec_idx in range(n_vectors):
        op_type = metadata['operation_types'][vec_idx]
        layer = metadata['layers'][vec_idx]
        token = metadata['tokens'][vec_idx]

        activations = features_np[vec_idx]

        for feat_idx in range(n_features):
            act = activations[feat_idx]

            if act > 0:  # Feature is active
                feature_stats['usage_count'][feat_idx] += 1
                feature_stats['operation_acts'][feat_idx][op_type].append(act)
                feature_stats['layer_acts'][feat_idx][layer].append(act)
                feature_stats['token_acts'][feat_idx][token].append(act)

                feature_stats['mean_activation'][feat_idx] += act
                feature_stats['max_activation'][feat_idx] = max(
                    feature_stats['max_activation'][feat_idx], act
                )

    # Compute means
    feature_stats['mean_activation'] /= n_vectors

    print(f"  Active features: {(feature_stats['usage_count'] > 0).sum()} / {n_features}")
    print(f"  Mean usage: {feature_stats['usage_count'].mean():.1f} vectors")

    return feature_stats


def compute_selectivity_scores(feature_stats, n_features):
    """
    Compute how selective each feature is for operations/layers/tokens.

    Selectivity = max_mean / mean_of_means
    High selectivity (>2) = feature strongly prefers specific context
    """

    selectivity = {
        'operation': np.zeros(n_features),
        'layer': np.zeros(n_features),
        'token': np.zeros(n_features),
        'operation_preference': {},
        'layer_preference': {},
        'token_preference': {}
    }

    for feat_idx in range(n_features):
        # Operation selectivity
        op_means = {}
        for op_type in ['mixed', 'pure_addition', 'pure_multiplication']:
            acts = feature_stats['operation_acts'][feat_idx][op_type]
            op_means[op_type] = np.mean(acts) if len(acts) > 0 else 0

        if sum(op_means.values()) > 0:
            max_mean = max(op_means.values())
            avg_mean = np.mean(list(op_means.values()))
            selectivity['operation'][feat_idx] = max_mean / (avg_mean + 1e-10)
            selectivity['operation_preference'][feat_idx] = max(op_means, key=op_means.get)

        # Layer selectivity
        layer_means = {}
        for layer in [4, 8, 14]:
            acts = feature_stats['layer_acts'][feat_idx][layer]
            layer_means[layer] = np.mean(acts) if len(acts) > 0 else 0

        if sum(layer_means.values()) > 0:
            max_mean = max(layer_means.values())
            avg_mean = np.mean(list(layer_means.values()))
            selectivity['layer'][feat_idx] = max_mean / (avg_mean + 1e-10)
            selectivity['layer_preference'][feat_idx] = max(layer_means, key=layer_means.get)

        # Token selectivity
        token_means = {}
        for token in range(6):
            acts = feature_stats['token_acts'][feat_idx][token]
            token_means[token] = np.mean(acts) if len(acts) > 0 else 0

        if sum(token_means.values()) > 0:
            max_mean = max(token_means.values())
            avg_mean = np.mean(list(token_means.values()))
            selectivity['token'][feat_idx] = max_mean / (avg_mean + 1e-10)
            selectivity['token_preference'][feat_idx] = max(token_means, key=token_means.get)

    return selectivity


def find_operation_specific_features(feature_stats, selectivity, n_features, threshold=2.0, min_usage=10):
    """
    Identify features highly selective for specific operations.

    Returns dict: {operation: [feature_indices]}
    """
    operation_features = {
        'mixed': [],
        'pure_addition': [],
        'pure_multiplication': []
    }

    for feat_idx in range(n_features):
        # Must be sufficiently used
        if feature_stats['usage_count'][feat_idx] < min_usage:
            continue

        # Must be selective
        if selectivity['operation'][feat_idx] < threshold:
            continue

        preferred_op = selectivity['operation_preference'].get(feat_idx)
        if preferred_op:
            operation_features[preferred_op].append(feat_idx)

    return operation_features


def analyze_token1_l8_hypothesis(feature_stats, selectivity, n_features, min_usage=10):
    """
    Test if operation-specific features concentrate in Token 1 L8.

    Hypothesis from operation circuits: Token 1 L8 is most discriminative.
    Question: Do SAE features show same pattern?
    """
    print("\n" + "="*80)
    print("TESTING OPERATION CIRCUITS HYPOTHESIS: Token 1 L8 Concentration")
    print("="*80)

    # Find highly operation-selective features
    op_selective_features = []
    for feat_idx in range(n_features):
        if feature_stats['usage_count'][feat_idx] >= min_usage:
            if selectivity['operation'][feat_idx] >= 2.0:
                op_selective_features.append(feat_idx)

    print(f"\nFound {len(op_selective_features)} operation-selective features (selectivity ≥ 2.0)")

    # Count preferences for these selective features
    token_prefs = Counter([selectivity['token_preference'].get(f) for f in op_selective_features])
    layer_prefs = Counter([selectivity['layer_preference'].get(f) for f in op_selective_features])

    print(f"\nToken preferences:")
    for token in range(6):
        count = token_prefs.get(token, 0)
        pct = 100 * count / len(op_selective_features) if op_selective_features else 0
        marker = " ⭐" if token == 1 else ""
        print(f"  Token {token}: {count:3d} features ({pct:5.1f}%){marker}")

    print(f"\nLayer preferences:")
    for layer in [4, 8, 14]:
        count = layer_prefs.get(layer, 0)
        pct = 100 * count / len(op_selective_features) if op_selective_features else 0
        marker = " ⭐" if layer == 8 else ""
        print(f"  Layer {layer}: {count:3d} features ({pct:5.1f}%){marker}")

    # Compute Token 1 × L8 concentration
    token1_l8_count = sum(
        1 for f in op_selective_features
        if selectivity['token_preference'].get(f) == 1
        and selectivity['layer_preference'].get(f) == 8
    )

    print(f"\nToken 1 × Layer 8 features: {token1_l8_count} ({100*token1_l8_count/len(op_selective_features):.1f}%)")
    print(f"Expected if random: {100/18:.1f}% (1/18 positions)")

    # Verdict
    enrichment = (token1_l8_count / len(op_selective_features)) / (1/18) if op_selective_features else 0
    print(f"Enrichment factor: {enrichment:.2f}x")

    if enrichment > 2.0:
        print("✅ HYPOTHESIS SUPPORTED: Token 1 L8 is enriched for operation-specific features")
    elif enrichment > 1.5:
        print("⚠️ WEAK SUPPORT: Token 1 L8 shows some enrichment")
    else:
        print("❌ HYPOTHESIS NOT SUPPORTED: No special enrichment in Token 1 L8")

    return {
        'token_preferences': dict(token_prefs),
        'layer_preferences': dict(layer_prefs),
        'token1_l8_count': token1_l8_count,
        'total_selective': len(op_selective_features),
        'enrichment': enrichment
    }


def visualize_feature_specialization(feature_stats, selectivity, operation_features, n_features, output_dir: Path):
    """Generate comprehensive visualizations of feature specialization."""
    print("\nGenerating visualizations...")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Selectivity distributions
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(selectivity['operation'][selectivity['operation'] > 0], bins=50, alpha=0.7, color='blue')
    ax1.axvline(x=2.0, color='red', linestyle='--', label='Threshold (2.0)')
    ax1.set_xlabel('Operation Selectivity')
    ax1.set_ylabel('Number of Features')
    ax1.set_title('Operation Selectivity Distribution')
    ax1.legend()
    ax1.set_xlim([0, 10])

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(selectivity['layer'][selectivity['layer'] > 0], bins=50, alpha=0.7, color='green')
    ax2.axvline(x=2.0, color='red', linestyle='--', label='Threshold (2.0)')
    ax2.set_xlabel('Layer Selectivity')
    ax2.set_ylabel('Number of Features')
    ax2.set_title('Layer Selectivity Distribution')
    ax2.legend()
    ax2.set_xlim([0, 10])

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(selectivity['token'][selectivity['token'] > 0], bins=50, alpha=0.7, color='purple')
    ax3.axvline(x=2.0, color='red', linestyle='--', label='Threshold (2.0)')
    ax3.set_xlabel('Token Selectivity')
    ax3.set_ylabel('Number of Features')
    ax3.set_title('Token Selectivity Distribution')
    ax3.legend()
    ax3.set_xlim([0, 10])

    # 2. Operation-specific feature counts
    ax4 = fig.add_subplot(gs[1, 0])
    ops = ['Mixed', 'Addition', 'Multiplication']
    counts = [
        len(operation_features['mixed']),
        len(operation_features['pure_addition']),
        len(operation_features['pure_multiplication'])
    ]
    colors_ops = ['gray', 'blue', 'orange']
    bars = ax4.bar(ops, counts, color=colors_ops, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Number of Selective Features')
    ax4.set_title('Operation-Specific Features (Selectivity ≥ 2.0)')
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    # 3. Layer preferences
    ax5 = fig.add_subplot(gs[1, 1])
    layer_prefs = Counter([selectivity['layer_preference'].get(f) for f in range(n_features)
                           if feature_stats['usage_count'][f] >= 10])
    layers = [4, 8, 14]
    layer_counts = [layer_prefs.get(l, 0) for l in layers]
    colors_layers = ['lightblue', 'blue', 'darkblue']
    bars = ax5.bar(['Early (L4)', 'Middle (L8)', 'Late (L14)'], layer_counts,
                   color=colors_layers, alpha=0.7, edgecolor='black')
    ax5.set_ylabel('Number of Features')
    ax5.set_title('Layer Preferences (Active Features)')
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    # 4. Token preferences
    ax6 = fig.add_subplot(gs[1, 2])
    token_prefs = Counter([selectivity['token_preference'].get(f) for f in range(n_features)
                          if feature_stats['usage_count'][f] >= 10])
    tokens = list(range(6))
    token_counts = [token_prefs.get(t, 0) for t in tokens]
    colors_tokens = ['lightgreen' if t == 1 else 'green' for t in tokens]
    bars = ax6.bar([f'T{t}' for t in tokens], token_counts,
                   color=colors_tokens, alpha=0.7, edgecolor='black')
    ax6.set_ylabel('Number of Features')
    ax6.set_title('Token Position Preferences (Active Features)')
    ax6.set_xlabel('Token Position')
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=8)

    # 5. Joint operation × layer heatmap
    ax7 = fig.add_subplot(gs[2, :2])

    # Build heatmap data
    ops_list = ['mixed', 'pure_addition', 'pure_multiplication']
    layers_list = [4, 8, 14]
    heatmap_data = np.zeros((len(ops_list), len(layers_list)))

    for op_idx, op in enumerate(ops_list):
        for feat in operation_features[op]:
            layer_pref = selectivity['layer_preference'].get(feat)
            if layer_pref in layers_list:
                layer_idx = layers_list.index(layer_pref)
                heatmap_data[op_idx, layer_idx] += 1

    sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax7,
                xticklabels=['Early (L4)', 'Middle (L8)', 'Late (L14)'],
                yticklabels=['Mixed', 'Addition', 'Multiplication'],
                cbar_kws={'label': 'Feature Count'})
    ax7.set_title('Operation-Specific Features by Layer Preference')
    ax7.set_ylabel('Operation Type')
    ax7.set_xlabel('Layer Preference')

    # 6. Summary statistics
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')

    total_features = n_features
    active_features = (feature_stats['usage_count'] > 0).sum()
    selective_features = (selectivity['operation'] >= 2.0).sum()

    summary_text = f"""
FEATURE SUMMARY

Total Features: {total_features:,}
Active Features: {active_features:,} ({100*active_features/total_features:.1f}%)
Operation-Selective: {selective_features:,} ({100*selective_features/total_features:.1f}%)

Operation-Specific:
  Mixed: {len(operation_features['mixed'])}
  Addition: {len(operation_features['pure_addition'])}
  Multiplication: {len(operation_features['pure_multiplication'])}

Most Common Preferences:
  Layer: L{max(layer_prefs, key=layer_prefs.get) if layer_prefs else '?'} ({max(layer_prefs.values()) if layer_prefs else 0} features)
  Token: T{max(token_prefs, key=token_prefs.get) if token_prefs else '?'} ({max(token_prefs.values()) if token_prefs else 0} features)
    """

    ax8.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center')

    plt.suptitle('SAE Feature Specialization Analysis (Refined 2048)', fontsize=14, fontweight='bold')

    # Save
    for ext in ['png', 'pdf']:
        save_path = output_dir / f'feature_specialization.{ext}'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sae_weights', type=str,
                        default='src/experiments/sae_pilot/refined/sae_weights.pt')
    parser.add_argument('--activations', type=str,
                        default='src/experiments/sae_pilot/data/sae_training_activations.pt')
    parser.add_argument('--output_dir', type=str,
                        default='src/experiments/sae_pilot/refined')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)

    # Load SAE
    print("Loading refined SAE...")
    sae, checkpoint = load_sae(args.sae_weights, device)
    print(f"  Config: {checkpoint['config']['n_features']} features, "
          f"L1={checkpoint['config']['l1_coefficient']}")

    # Load activations
    print("\nLoading activations...")
    activation_data = torch.load(args.activations)
    activations = activation_data['activations']
    metadata = activation_data['metadata']
    print(f"  Activations shape: {activations.shape}")
    print(f"  Problems: {len(set(metadata['problems']))}")
    print(f"  Layers: {sorted(set(metadata['layers']))}")
    print(f"  Tokens: {sorted(set(metadata['tokens']))}")

    # Extract SAE features
    print("\nExtracting SAE features...")
    with torch.no_grad():
        x = activations.to(device)
        _, features = sae(x)
        features_np = features.cpu().numpy()

    print(f"  Features shape: {features_np.shape}")
    print(f"  Sparsity: {(features_np > 0).mean()*100:.2f}% nonzero")

    # Compute statistics
    n_features = checkpoint['config']['n_features']
    feature_stats = compute_feature_statistics(features_np, metadata)
    selectivity = compute_selectivity_scores(feature_stats, n_features)

    # Find operation-specific features
    print("\n" + "="*80)
    print("OPERATION-SPECIFIC FEATURES")
    print("="*80)

    operation_features = find_operation_specific_features(feature_stats, selectivity, n_features)

    for op_type, features in operation_features.items():
        print(f"\n{op_type}: {len(features)} features")
        if len(features) > 0:
            # Show top 5 by max activation
            top_5 = sorted(features, key=lambda f: feature_stats['max_activation'][f], reverse=True)[:5]
            print(f"  Top 5 features: {top_5}")
            for f in top_5:
                layer_pref = selectivity['layer_preference'].get(f, '?')
                token_pref = selectivity['token_preference'].get(f, '?')
                max_act = feature_stats['max_activation'][f]
                print(f"    Feature {f}: L{layer_pref} T{token_pref}, max_act={max_act:.3f}")

    # Test Token 1 L8 hypothesis
    token1_l8_analysis = analyze_token1_l8_hypothesis(feature_stats, selectivity, n_features)

    # Visualize
    visualize_feature_specialization(feature_stats, selectivity, operation_features, n_features, output_dir)

    # Save results
    results = {
        'n_features': int(n_features),
        'active_features': int((feature_stats['usage_count'] > 0).sum()),
        'operation_selective_features': {
            'mixed': len(operation_features['mixed']),
            'pure_addition': len(operation_features['pure_addition']),
            'pure_multiplication': len(operation_features['pure_multiplication'])
        },
        'selectivity_stats': {
            'operation_mean': float(selectivity['operation'][selectivity['operation'] > 0].mean()),
            'layer_mean': float(selectivity['layer'][selectivity['layer'] > 0].mean()),
            'token_mean': float(selectivity['token'][selectivity['token'] > 0].mean())
        },
        'token1_l8_hypothesis': token1_l8_analysis
    }

    results_path = output_dir / 'feature_specialization_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Analysis complete! Results saved to {results_path}")

    # Final summary
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print(f"Active Features: {results['active_features']:,} / {results['n_features']:,} "
          f"({100*results['active_features']/results['n_features']:.1f}%)")
    print(f"\nOperation-Specific Features (selectivity ≥ 2.0):")
    print(f"  Multiplication: {results['operation_selective_features']['pure_multiplication']}")
    print(f"  Addition: {results['operation_selective_features']['pure_addition']}")
    print(f"  Mixed: {results['operation_selective_features']['mixed']}")
    print(f"\nToken 1 L8 Enrichment: {token1_l8_analysis['enrichment']:.2f}x")
    print("="*80)


if __name__ == "__main__":
    main()
