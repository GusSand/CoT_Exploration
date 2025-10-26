"""
Create CODI Figure 6-style visualizations showing feature-CoT token correlations.

This script creates:
1. Feature-token correlation heatmap
2. Example-specific feature activation visualization
3. Top features per important token (0-9, *, =)
"""

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer

# Paths
BASE_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = BASE_DIR / "analysis"
DATA_DIR = BASE_DIR / "data"
VIZ_DIR = ANALYSIS_DIR / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)

# Load data
print("Loading data...")
with open(ANALYSIS_DIR / "feature_catalog.json", "r") as f:
    catalog = json.load(f)

with open(ANALYSIS_DIR / "feature_cot_correlations.json", "r") as f:
    correlations = json.load(f)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Load test data for examples
test_data = torch.load(DATA_DIR / "enriched_test_data_with_cot.pt")

# Load extracted features
features_data = torch.load(ANALYSIS_DIR / "extracted_features.pt")


def create_token_feature_heatmap(position=0, top_k_tokens=20, top_k_features=30):
    """
    Create heatmap showing which features correlate with which tokens.
    Similar to CODI Figure 6 panel showing feature-token associations.
    """
    print(f"\nCreating feature-token heatmap for position {position}...")

    # Get top features for this position
    pos_features = catalog["positions"][str(position)]["top_100_features"]

    # Collect all tokens and their feature correlations
    token_feature_map = defaultdict(lambda: defaultdict(float))

    for feature in pos_features[:top_k_features]:
        feature_id = feature["feature_id"]
        for token_info in feature["enriched_tokens"]:
            token = token_info["token_str"]
            enrichment = token_info["enrichment"]
            p_value = token_info["p_value"]

            # Use -log10(p_value) as correlation strength
            # Cap at 100 to handle extremely small p-values
            strength = min(-np.log10(p_value), 100)
            token_feature_map[token][feature_id] = strength

    # Get top tokens by total correlation strength
    token_scores = {token: sum(features.values())
                   for token, features in token_feature_map.items()}
    top_tokens = sorted(token_scores.keys(), key=lambda t: token_scores[t], reverse=True)[:top_k_tokens]

    # Get feature IDs
    feature_ids = [f["feature_id"] for f in pos_features[:top_k_features]]

    # Build matrix
    matrix = np.zeros((len(top_tokens), len(feature_ids)))
    for i, token in enumerate(top_tokens):
        for j, feature_id in enumerate(feature_ids):
            matrix[i, j] = token_feature_map[token].get(feature_id, 0)

    # Create heatmap
    plt.figure(figsize=(16, 10))
    sns.heatmap(
        matrix,
        xticklabels=[f"F{fid}" for fid in feature_ids],
        yticklabels=top_tokens,
        cmap="YlOrRd",
        cbar_kws={"label": "-log10(p-value)"},
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title(f"Position {position}: Feature-Token Correlation Heatmap\n"
              f"(Brighter = Stronger Statistical Association)", fontsize=14, fontweight='bold')
    plt.xlabel("Feature ID", fontsize=12)
    plt.ylabel("CoT Token", fontsize=12)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"position_{position}_feature_token_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {VIZ_DIR / f'position_{position}_feature_token_heatmap.png'}")
    plt.close()


def visualize_example_features(example_idx=0, position=0):
    """
    Visualize which features activate for a specific example.
    Shows: problem → CoT → which features fire → what they detect
    """
    print(f"\nVisualizing features for example {example_idx}, position {position}...")

    # Get example data
    # Find samples for this position
    position_mask = test_data['positions'] == position
    position_indices = torch.where(position_mask)[0]

    if example_idx >= len(position_indices):
        print(f"  Example {example_idx} not found for position {position}")
        return

    sample_idx = position_indices[example_idx].item()
    problem_id = test_data['problem_ids'][sample_idx].item()
    cot_sequence = test_data['cot_sequences'][sample_idx]

    # Get feature activations
    feature_key = f"position_{position}"
    if feature_key not in features_data:
        print(f"  Features not found for position {position}")
        return

    # Get features for this sample
    sample_features = features_data[feature_key][example_idx % features_data[feature_key].shape[0]]

    # Get top activated features
    top_k = 15
    top_feature_indices = torch.argsort(sample_features, descending=True)[:top_k]
    top_activations = sample_features[top_feature_indices]

    # Get interpretable info for top features
    pos_catalog = catalog["positions"][str(position)]["top_100_features"]
    feature_lookup = {f["feature_id"]: f for f in pos_catalog}

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Left panel: Feature activations
    feature_labels = []
    feature_colors = []
    for idx, activation in zip(top_feature_indices.numpy(), top_activations.numpy()):
        if idx in feature_lookup:
            # Get top token for this feature
            top_token = feature_lookup[idx]["enriched_tokens"][0]["token_str"]
            enrichment = feature_lookup[idx]["enriched_tokens"][0]["enrichment"]
            feature_labels.append(f"F{idx}: '{top_token}' ({enrichment:.1%})")
            feature_colors.append('green')
        else:
            feature_labels.append(f"F{idx}: Unknown")
            feature_colors.append('gray')

    ax1.barh(range(top_k), top_activations.numpy(), color=feature_colors, alpha=0.7)
    ax1.set_yticks(range(top_k))
    ax1.set_yticklabels(feature_labels, fontsize=9)
    ax1.set_xlabel("Feature Activation Strength", fontsize=11)
    ax1.set_title(f"Top {top_k} Activated Features\n(Position {position}, Example {example_idx})",
                  fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()

    # Right panel: CoT sequence and feature matches
    ax2.axis('off')

    # Display CoT
    cot_text = f"Problem ID: {problem_id}\n\n"
    cot_text += "Chain of Thought:\n"
    for i, step in enumerate(cot_sequence):
        cot_text += f"  {i+1}. {step}\n"

    cot_text += f"\n\nTop Detected Tokens (from features):\n"

    # Collect all detected tokens
    detected_tokens = defaultdict(list)
    for idx, activation in zip(top_feature_indices.numpy(), top_activations.numpy()):
        if idx in feature_lookup and activation > 0.1:  # threshold
            for token_info in feature_lookup[idx]["enriched_tokens"][:3]:
                token = token_info["token_str"]
                enrichment = token_info["enrichment"]
                detected_tokens[token].append((idx, activation, enrichment))

    # Show top detected tokens
    for i, (token, detections) in enumerate(list(detected_tokens.items())[:10]):
        avg_activation = np.mean([d[1] for d in detections])
        max_enrichment = max([d[2] for d in detections])
        feature_ids = [d[0] for d in detections]
        cot_text += f"  '{token}': {len(detections)} features, "
        cot_text += f"avg_act={avg_activation:.2f}, enrich={max_enrichment:.1%}\n"

    ax2.text(0.05, 0.95, cot_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"example_{example_idx}_position_{position}_features.png",
                dpi=300, bbox_inches='tight')
    print(f"  Saved: {VIZ_DIR / f'example_{example_idx}_position_{position}_features.png'}")
    plt.close()


def create_token_specific_analysis(tokens=["0", "1", "2", "*", "="], max_features=10):
    """
    For each important token, show top features that detect it.
    This creates individual panels similar to CODI Figure 6.
    """
    print(f"\nCreating token-specific feature analysis...")

    n_tokens = len(tokens)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, token in enumerate(tokens):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Collect features that detect this token across all positions
        token_features = []

        for position in range(6):
            pos_features = catalog["positions"][str(position)]["top_100_features"]

            for feature in pos_features:
                for token_info in feature["enriched_tokens"]:
                    if token_info["token_str"] == token:
                        score = -np.log10(token_info["p_value"])
                        token_features.append({
                            "position": position,
                            "feature_id": feature["feature_id"],
                            "enrichment": token_info["enrichment"],
                            "p_value": token_info["p_value"],
                            "score": min(score, 100)
                        })
                        break  # Only count feature once per token

        # Sort by score
        token_features.sort(key=lambda x: x["score"], reverse=True)
        top_features = token_features[:max_features]

        if not top_features:
            ax.text(0.5, 0.5, f"No features found\nfor token '{token}'",
                   ha='center', va='center', fontsize=12)
            ax.set_title(f"Token: '{token}'", fontweight='bold')
            ax.axis('off')
            continue

        # Create bar chart
        labels = [f"Pos{f['position']}-F{f['feature_id']}" for f in top_features]
        scores = [f['score'] for f in top_features]
        enrichments = [f['enrichment'] for f in top_features]

        colors = plt.cm.viridis([e for e in enrichments])

        ax.barh(range(len(labels)), scores, color=colors, alpha=0.8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("-log10(p-value)", fontsize=9)
        ax.set_title(f"Token: '{token}'\nTop {len(top_features)} Features",
                    fontweight='bold', fontsize=11)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

        # Add enrichment percentages
        for i, (score, enrich) in enumerate(zip(scores, enrichments)):
            ax.text(score, i, f' {enrich:.0%}',
                   va='center', fontsize=7, color='black')

    # Remove extra subplots
    for idx in range(len(tokens), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle("Top Features Detecting Each Important Token\n(Color intensity = enrichment strength)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "token_specific_features.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {VIZ_DIR / 'token_specific_features.png'}")
    plt.close()


def create_cross_position_comparison(token="0", max_features=10):
    """
    Show how a specific token is detected across all 6 positions.
    This reveals position-specific specialization.
    """
    print(f"\nCreating cross-position comparison for token '{token}'...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for position in range(6):
        ax = axes[position]

        # Get features for this position that detect the token
        pos_features = catalog["positions"][str(position)]["top_100_features"]

        token_detectors = []
        for feature in pos_features:
            for token_info in feature["enriched_tokens"]:
                if token_info["token_str"] == token:
                    token_detectors.append({
                        "feature_id": feature["feature_id"],
                        "enrichment": token_info["enrichment"],
                        "p_value": token_info["p_value"],
                        "score": min(-np.log10(token_info["p_value"]), 100)
                    })
                    break

        token_detectors.sort(key=lambda x: x["score"], reverse=True)
        top_detectors = token_detectors[:max_features]

        if not top_detectors:
            ax.text(0.5, 0.5, f"No '{token}' detectors",
                   ha='center', va='center', fontsize=10)
            ax.set_title(f"Position {position}", fontweight='bold')
            ax.axis('off')
            continue

        # Create visualization
        labels = [f"F{d['feature_id']}" for d in top_detectors]
        enrichments = [d['enrichment'] * 100 for d in top_detectors]
        scores = [d['score'] for d in top_detectors]

        x = np.arange(len(labels))
        width = 0.35

        ax2 = ax.twinx()

        bars1 = ax.bar(x - width/2, enrichments, width, label='Enrichment %',
                      color='steelblue', alpha=0.7)
        bars2 = ax2.bar(x + width/2, scores, width, label='-log10(p)',
                       color='coral', alpha=0.7)

        ax.set_ylabel('Enrichment %', fontsize=9, color='steelblue')
        ax2.set_ylabel('-log10(p-value)', fontsize=9, color='coral')
        ax.set_xlabel('Feature ID', fontsize=9)
        ax.set_title(f"Position {position}\n{len(top_detectors)} '{token}' detectors",
                    fontweight='bold', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.tick_params(axis='y', labelcolor='steelblue')
        ax2.tick_params(axis='y', labelcolor='coral')
        ax.grid(axis='y', alpha=0.3)

        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=7)

    plt.suptitle(f"How Token '{token}' is Detected Across All 6 Positions\n"
                 f"(Blue = Enrichment, Orange = Statistical Significance)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"cross_position_token_{token}_comparison.png",
                dpi=300, bbox_inches='tight')
    print(f"  Saved: {VIZ_DIR / f'cross_position_token_{token}_comparison.png'}")
    plt.close()


if __name__ == "__main__":
    print("="*70)
    print("CREATING CODI FIGURE 6-STYLE VISUALIZATIONS")
    print("="*70)

    # 1. Feature-token heatmap (most similar to Figure 6)
    for position in [0, 1, 3, 5]:  # Sample positions
        create_token_feature_heatmap(position=position, top_k_tokens=20, top_k_features=30)

    # 2. Example-specific feature activations
    for example_idx in [0, 5, 10]:
        for position in [0, 3]:
            visualize_example_features(example_idx=example_idx, position=position)

    # 3. Token-specific analysis (like Figure 6 panels)
    create_token_specific_analysis(tokens=["0", "1", "2", "5", "*", "="])

    # 4. Cross-position comparison
    for token in ["0", "1", "*"]:
        create_cross_position_comparison(token=token, max_features=8)

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print(f"All figures saved to: {VIZ_DIR}")
    print("="*70)
