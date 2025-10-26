"""
Create CODI Figure 6-style visualizations using only JSON data.
Works without the large .pt files.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Paths
BASE_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = BASE_DIR / "analysis"
VIZ_DIR = ANALYSIS_DIR / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)

# Load data
print("Loading feature catalog...")
with open(ANALYSIS_DIR / "feature_catalog.json", "r") as f:
    catalog = json.load(f)


def create_token_feature_heatmap(position=0, top_k_tokens=20, top_k_features=30):
    """
    Create heatmap showing which features correlate with which tokens.
    This is the main Figure 6-style visualization.
    """
    print(f"\nCreating feature-token heatmap for position {position}...")

    # Get top features for this position
    pos_features = catalog["positions"][str(position)]["top_100_features"][:top_k_features]

    # Collect all tokens and their feature correlations
    token_feature_map = defaultdict(lambda: defaultdict(float))

    for feature in pos_features:
        feature_id = feature["feature_id"]
        for token_info in feature["enriched_tokens"]:
            token = token_info["token_str"]
            p_value = token_info["p_value"]

            # Use -log10(p_value) as correlation strength (cap at 100)
            strength = min(-np.log10(max(p_value, 1e-300)), 100)
            token_feature_map[token][feature_id] = strength

    # Get top tokens by total correlation strength
    token_scores = {token: sum(features.values())
                   for token, features in token_feature_map.items()}
    top_tokens = sorted(token_scores.keys(), key=lambda t: token_scores[t], reverse=True)[:top_k_tokens]

    # Get feature IDs
    feature_ids = [f["feature_id"] for f in pos_features]

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
        yticklabels=[f"'{t}'" for t in top_tokens],
        cmap="YlOrRd",
        cbar_kws={"label": "-log10(p-value)"},
        linewidths=0.5,
        linecolor='lightgray',
        vmin=0,
        vmax=50
    )
    plt.title(f"Position {position}: Feature-CoT Token Correlation Matrix\n"
              f"(Brighter = Stronger Statistical Association)",
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("Feature ID", fontsize=12, labelpad=10)
    plt.ylabel("CoT Token", fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"position_{position}_feature_token_heatmap.png",
                dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: position_{position}_feature_token_heatmap.png")
    plt.close()


def create_token_specific_analysis(tokens=["0", "1", "2", "*", "=", "5"], max_features=10):
    """
    For each important token, show top features that detect it across all positions.
    Similar to individual panels in CODI Figure 6.
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
                        score = min(-np.log10(max(token_info["p_value"], 1e-300)), 100)
                        token_features.append({
                            "position": position,
                            "feature_id": feature["feature_id"],
                            "enrichment": token_info["enrichment"],
                            "p_value": token_info["p_value"],
                            "score": score
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

        # Color by enrichment
        colors = plt.cm.viridis(np.array(enrichments))

        bars = ax.barh(range(len(labels)), scores, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("-log10(p-value)", fontsize=10)
        ax.set_title(f"Token: '{token}'\nTop {len(top_features)} Features",
                    fontweight='bold', fontsize=11)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

        # Add enrichment percentages as text
        for i, (score, enrich) in enumerate(zip(scores, enrichments)):
            ax.text(score + 1, i, f'{enrich:.0%}',
                   va='center', fontsize=8, fontweight='bold')

    # Remove extra subplots
    for idx in range(len(tokens), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle("Top Features Detecting Each Important CoT Token\n"
                 "(Bar color intensity = enrichment strength)",
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "token_specific_features.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: token_specific_features.png")
    plt.close()


def create_cross_position_comparison(token="0", max_features=8):
    """
    Show how a specific token is detected across all 6 positions.
    Reveals position-specific specialization.
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
                        "score": min(-np.log10(max(token_info["p_value"], 1e-300)), 100)
                    })
                    break

        token_detectors.sort(key=lambda x: x["score"], reverse=True)
        top_detectors = token_detectors[:max_features]

        if not top_detectors:
            ax.text(0.5, 0.5, f"No '{token}'\ndetectors",
                   ha='center', va='center', fontsize=11)
            ax.set_title(f"Position {position}", fontweight='bold', fontsize=12)
            ax.axis('off')
            continue

        # Create dual-axis visualization
        labels = [f"F{d['feature_id']}" for d in top_detectors]
        enrichments = [d['enrichment'] * 100 for d in top_detectors]
        scores = [d['score'] for d in top_detectors]

        x = np.arange(len(labels))
        width = 0.35

        ax2 = ax.twinx()

        bars1 = ax.bar(x - width/2, enrichments, width, label='Enrichment %',
                      color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        bars2 = ax2.bar(x + width/2, scores, width, label='-log10(p)',
                       color='coral', alpha=0.7, edgecolor='black', linewidth=0.5)

        ax.set_ylabel('Enrichment %', fontsize=10, color='steelblue', fontweight='bold')
        ax2.set_ylabel('-log10(p-value)', fontsize=10, color='coral', fontweight='bold')
        ax.set_xlabel('Feature ID', fontsize=10)
        ax.set_title(f"Position {position}\n{len(top_detectors)} '{token}' detectors",
                    fontweight='bold', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.tick_params(axis='y', labelcolor='steelblue')
        ax2.tick_params(axis='y', labelcolor='coral')
        ax.grid(axis='y', alpha=0.3)

        # Add values on bars
        for i, (bar, val) in enumerate(zip(bars1, enrichments)):
            ax.text(bar.get_x() + bar.get_width()/2, val, f'{val:.0f}',
                   ha='center', va='bottom', fontsize=7, fontweight='bold')

    plt.suptitle(f"How Token '{token}' is Detected Across All 6 Continuous Thought Positions\n"
                 f"(Blue bars = Enrichment %, Orange bars = Statistical Significance)",
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"cross_position_token_{token}_comparison.png",
                dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: cross_position_token_{token}_comparison.png")
    plt.close()


def create_feature_detail_view(position=0, feature_id=1155):
    """
    Deep dive into a single feature showing all its token correlations.
    """
    print(f"\nCreating detailed view for Feature {feature_id}, Position {position}...")

    # Find the feature
    pos_features = catalog["positions"][str(position)]["top_100_features"]
    feature = None
    for f in pos_features:
        if f["feature_id"] == feature_id:
            feature = f
            break

    if not feature:
        print(f"  Feature {feature_id} not found in position {position}")
        return

    # Get all enriched tokens
    tokens_data = feature["enriched_tokens"][:15]  # Top 15 tokens

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Enrichment bars
    tokens = [t["token_str"] for t in tokens_data]
    enrichments = [t["enrichment"] * 100 for t in tokens_data]
    colors = plt.cm.RdYlGn(np.array(enrichments) / max(enrichments))

    bars = ax1.barh(range(len(tokens)), enrichments, color=colors,
                   alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(range(len(tokens)))
    ax1.set_yticklabels([f"'{t}'" for t in tokens], fontsize=10)
    ax1.set_xlabel("Enrichment (%)", fontsize=11, fontweight='bold')
    ax1.set_title(f"Feature {feature_id} (Position {position})\nToken Enrichment",
                 fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()

    # Add values
    for i, (bar, val) in enumerate(zip(bars, enrichments)):
        ax1.text(val + 1, i, f'{val:.1f}%',
                va='center', fontsize=9, fontweight='bold')

    # Right: Statistical significance
    p_values = [t["p_value"] for t in tokens_data]
    log_p = [-np.log10(max(p, 1e-300)) for p in p_values]

    # Cap at 100 for visualization
    log_p_capped = [min(lp, 100) for lp in log_p]

    colors2 = plt.cm.YlOrRd(np.array(log_p_capped) / max(log_p_capped))

    bars2 = ax2.barh(range(len(tokens)), log_p_capped, color=colors2,
                    alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(range(len(tokens)))
    ax2.set_yticklabels([f"'{t}'" for t in tokens], fontsize=10)
    ax2.set_xlabel("-log10(p-value)", fontsize=11, fontweight='bold')
    ax2.set_title(f"Statistical Significance\n(Higher = Stronger Association)",
                 fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()

    # Add values (show actual p-values)
    for i, (bar, lp, p) in enumerate(zip(bars2, log_p_capped, p_values)):
        if lp >= 100:
            label = "p<10⁻¹⁰⁰"
        elif lp >= 10:
            label = f"p<10⁻{int(lp)}"
        else:
            label = f"p={p:.2e}"
        ax2.text(lp + 2, i, label, va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"feature_detail_pos{position}_f{feature_id}.png",
                dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: feature_detail_pos{position}_f{feature_id}.png")
    plt.close()


if __name__ == "__main__":
    print("="*70)
    print("CREATING CODI FIGURE 6-STYLE VISUALIZATIONS")
    print("(Using feature catalog JSON only)")
    print("="*70)

    # 1. Feature-token heatmaps (most similar to Figure 6)
    print("\n[1/4] Creating feature-token heatmaps...")
    for position in [0, 1, 3, 5]:
        create_token_feature_heatmap(position=position, top_k_tokens=20, top_k_features=30)

    # 2. Token-specific analysis
    print("\n[2/4] Creating token-specific analysis...")
    create_token_specific_analysis(tokens=["0", "1", "2", "5", "*", "="])

    # 3. Cross-position comparison
    print("\n[3/4] Creating cross-position comparisons...")
    for token in ["0", "1", "*", "="]:
        create_cross_position_comparison(token=token, max_features=8)

    # 4. Feature detail views
    print("\n[4/4] Creating feature detail views...")
    create_feature_detail_view(position=0, feature_id=1155)  # The "000" detector
    create_feature_detail_view(position=0, feature_id=745)   # The "810/900" detector

    print("\n" + "="*70)
    print("✓ VISUALIZATION COMPLETE!")
    print(f"All figures saved to: {VIZ_DIR}")
    print("="*70)
    print("\nGenerated visualizations:")
    print("  - Feature-token heatmaps (4 positions)")
    print("  - Token-specific feature analysis (6 tokens)")
    print("  - Cross-position comparisons (4 tokens)")
    print("  - Feature detail views (2 features)")
    print("="*70)
