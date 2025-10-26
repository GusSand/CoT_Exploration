"""
Visualize specific features from full dataset SAE models.
Creates detailed activation and token correlation visualizations.
"""
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer
import sys

sys.path.append(str(Path(__file__).parent.parent.parent / "operation_circuits"))
from sae_model import SparseAutoencoder

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models_full_dataset"
ANALYSIS_DIR = BASE_DIR / "analysis"
VIZ_DIR = ANALYSIS_DIR / "visualizations" / "full_dataset_7473_samples"
VIZ_DIR.mkdir(exist_ok=True, parents=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Load test data
print("Loading test data...")
test_data_path = BASE_DIR / "results" / "enriched_test_data_with_cot.pt"
test_data = torch.load(test_data_path, weights_only=False)
activations = test_data['hidden_states'].to(device)
metadata = test_data['metadata']

# Load feature catalog
catalog_path = ANALYSIS_DIR / "feature_catalog_full_dataset.json"
with open(catalog_path) as f:
    catalog = json.load(f)

print(f"✓ Loaded {len(activations)} test samples")


def visualize_feature(position, feature_id):
    """Create detailed visualization for a specific feature."""
    print(f"\n{'='*80}")
    print(f"VISUALIZING POSITION {position}, FEATURE {feature_id}")
    print(f"{'='*80}")

    # Load SAE model
    model_path = MODELS_DIR / f"pos_{position}_final.pt"
    sae = SparseAutoencoder(input_dim=2048, n_features=2048, l1_coefficient=0.0005).to(device)
    sae.load_state_dict(torch.load(model_path, map_location=device))
    sae.eval()

    # Filter samples for this position
    pos_indices = [i for i, p in enumerate(metadata['positions']) if p == position]
    pos_activations = activations[pos_indices]
    pos_cot_token_ids = [metadata['cot_token_ids'][i] for i in pos_indices]

    print(f"  Analyzing {len(pos_activations)} samples from position {position}")

    # Extract features
    with torch.no_grad():
        _, features = sae(pos_activations)

    # Get activations for this specific feature
    feature_acts = features[:, feature_id].cpu().numpy()

    # Find top activating samples
    threshold = np.percentile(feature_acts, 90)
    top_samples = np.where(feature_acts > threshold)[0]

    print(f"  Top activating samples (90th percentile): {len(top_samples)}")
    print(f"  Threshold: {threshold:.4f}")

    # Count tokens in top activating samples
    token_counts_top = Counter()
    for sample_idx in top_samples:
        tokens = pos_cot_token_ids[sample_idx]
        if isinstance(tokens, list):
            for token in tokens:
                token_counts_top[token] += 1

    # Count tokens in all samples
    token_counts_all = Counter()
    for tokens in pos_cot_token_ids:
        if isinstance(tokens, list):
            for token in tokens:
                token_counts_all[token] += 1

    # Calculate enrichment
    enriched_tokens = []
    for token, count_top in token_counts_top.most_common(15):
        count_all = token_counts_all[token]
        freq_top = count_top / len(top_samples)
        freq_all = count_all / len(pos_activations)
        enrichment = freq_top / freq_all if freq_all > 0 else 0

        token_str = tokenizer.decode([token]) if isinstance(token, int) else str(token)
        enriched_tokens.append({
            "token": token_str,
            "enrichment": enrichment,
            "count_top": count_top,
            "freq_top": freq_top
        })

    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Activation distribution
    ax1 = fig.add_subplot(gs[0, :])
    ax1.hist(feature_acts, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(threshold, color='red', linestyle='--', linewidth=2,
               label=f'90th percentile: {threshold:.3f}')
    ax1.set_xlabel('Feature Activation', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title(f'Position {position}, Feature {feature_id}: Activation Distribution',
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Token enrichment
    ax2 = fig.add_subplot(gs[1, 0])
    tokens = [t['token'] for t in enriched_tokens[:10]]
    enrichments = [t['enrichment'] * 100 for t in enriched_tokens[:10]]

    bars = ax2.barh(range(len(tokens)), enrichments, color='coral', alpha=0.8, edgecolor='black')
    ax2.set_yticks(range(len(tokens)))
    ax2.set_yticklabels([f"'{t}'" for t in tokens], fontsize=10)
    ax2.set_xlabel('Enrichment %', fontsize=11, fontweight='bold')
    ax2.set_title('Top 10 Enriched Tokens', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, enrichments)):
        ax2.text(val, i, f' {val:.0f}%', va='center', fontsize=9, fontweight='bold')

    # 3. Token frequency in top samples
    ax3 = fig.add_subplot(gs[1, 1])
    freqs = [t['freq_top'] * 100 for t in enriched_tokens[:10]]

    bars = ax3.barh(range(len(tokens)), freqs, color='lightgreen', alpha=0.8, edgecolor='black')
    ax3.set_yticks(range(len(tokens)))
    ax3.set_yticklabels([f"'{t}'" for t in tokens], fontsize=10)
    ax3.set_xlabel('Frequency in Top Samples (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Token Frequency (Top Activations)', fontsize=12, fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, freqs)):
        ax3.text(val, i, f' {val:.1f}%', va='center', fontsize=9, fontweight='bold')

    # 4. Example activations (top 15 samples)
    ax4 = fig.add_subplot(gs[2, :])
    top_15_indices = np.argsort(feature_acts)[-15:][::-1]
    top_15_acts = feature_acts[top_15_indices]

    bars = ax4.bar(range(15), top_15_acts, color='mediumpurple', alpha=0.8, edgecolor='black')
    ax4.set_xlabel('Sample Rank', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Activation Strength', fontsize=11, fontweight='bold')
    ax4.set_title('Top 15 Activations', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(15))
    ax4.set_xticklabels([f'#{i+1}' for i in range(15)], fontsize=9)
    ax4.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_15_acts)):
        ax4.text(i, val, f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle(f'Feature {feature_id} Analysis (Position {position}, Full Dataset Model)',
                fontsize=16, fontweight='bold', y=0.995)

    output_path = VIZ_DIR / f"feature_detail_pos{position}_f{feature_id}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()

    # Print summary
    print(f"\n📊 Feature {feature_id} Summary:")
    print(f"  Mean activation: {feature_acts.mean():.4f}")
    print(f"  Max activation: {feature_acts.max():.4f}")
    print(f"  90th percentile: {threshold:.4f}")
    print(f"\n🔝 Top 5 Enriched Tokens:")
    for i, t in enumerate(enriched_tokens[:5], 1):
        print(f"  {i}. '{t['token']}': {t['enrichment']*100:.1f}% enrichment "
              f"({t['count_top']} occurrences in top samples)")


if __name__ == "__main__":
    features_to_visualize = [
        (1, 1000),  # Position 1, Feature 1000
        (1, 1377),  # Position 1, Feature 1377
        (1, 1412),  # Position 1, Feature 1412
    ]

    print("="*80)
    print("FEATURE VISUALIZATION - FULL DATASET MODELS")
    print("="*80)

    for position, feature_id in features_to_visualize:
        visualize_feature(position, feature_id)

    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE!")
    print(f"Saved to: {VIZ_DIR}")
    print("="*80)
