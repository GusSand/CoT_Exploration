"""
Create detailed visualizations for specific high-value features.

This script generates feature detail plots showing token enrichment and
statistical significance for user-specified features.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = BASE_DIR / "analysis"
VIZ_DIR = ANALYSIS_DIR / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)

# Load catalog
print("Loading feature catalog...")
with open(ANALYSIS_DIR / "feature_catalog.json", "r") as f:
    catalog = json.load(f)

def create_feature_detail_plot(position, feature_id, max_tokens=10):
    """Create detailed visualization for a specific feature.

    Args:
        position: Position index (0-5)
        feature_id: Feature ID to visualize
        max_tokens: Number of top tokens to show
    """
    # Find the feature in catalog
    pos_features = catalog["positions"][str(position)]["top_100_features"]

    feature = None
    for feat in pos_features:
        if feat["feature_id"] == feature_id:
            feature = feat
            break

    if not feature:
        print(f"⚠️  Feature {feature_id} not found in Position {position}")
        return

    # Extract token data
    tokens = []
    enrichments = []
    p_values = []

    for token_info in feature["enriched_tokens"][:max_tokens]:
        tokens.append(token_info["token_str"])
        enrichments.append(token_info["enrichment"] * 100)  # Convert to percentage
        p_values.append(token_info["p_value"])

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Enrichment percentages
    y_pos = np.arange(len(tokens))

    # Color bars by enrichment level
    colors1 = plt.cm.YlOrRd(np.array(enrichments) / max(enrichments))

    bars1 = ax1.barh(y_pos, enrichments, color=colors1)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"'{t}'" for t in tokens])
    ax1.set_xlabel('Enrichment (%)', fontsize=12)
    ax1.set_title(f'Feature {feature_id} (Position {position})\nToken Enrichment',
                  fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, enrichments)):
        ax1.text(val + max(enrichments)*0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=10)

    # Subplot 2: Statistical significance
    # Convert p-values to -log10(p) for visualization
    neg_log_p = [-np.log10(max(p, 1e-300)) for p in p_values]

    # Color bars by significance
    colors2 = plt.cm.RdPu(np.array(neg_log_p) / max(neg_log_p))

    bars2 = ax2.barh(y_pos, neg_log_p, color=colors2)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"'{t}'" for t in tokens])
    ax2.set_xlabel('-log10(p-value)', fontsize=12)
    ax2.set_title('Statistical Significance\n(Higher = Stronger Association)',
                  fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    # Add p-value labels
    for i, (bar, pval, neg_log) in enumerate(zip(bars2, p_values, neg_log_p)):
        if pval < 0.01:
            label = f"p<10^{int(-np.log10(max(pval, 1e-300)))}"
        else:
            label = f"p={pval:.2e}"
        ax2.text(neg_log + max(neg_log_p)*0.02, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=9)

    plt.tight_layout()

    # Save
    output_path = VIZ_DIR / f"feature_detail_pos{position}_f{feature_id}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()

    # Print summary
    print(f"\n  Feature {feature_id} (Position {position}) Summary:")
    print(f"    Top token: '{tokens[0]}' ({enrichments[0]:.1f}% enrichment, p<10^{int(-np.log10(max(p_values[0], 1e-300)))})")
    print(f"    Selectivity: {feature.get('layer_selectivity', 'N/A')}")
    top_3_str = ', '.join([f"'{t}'" for t in tokens[:3]])
    print(f"    Top 3 tokens: {top_3_str}")


# Create visualizations for requested features
print("\n" + "="*80)
print("CREATING FEATURE DETAIL VISUALIZATIONS")
print("="*80)

# Position 1: F148 (270% enrichment for "0")
print("\n[1/2] Position 1, Feature 148 (Top '0' detector - 270% enrichment)")
create_feature_detail_plot(position=1, feature_id=148, max_tokens=10)

# Position 3: F1893 (292% enrichment for "0")
print("\n[2/2] Position 3, Feature 1893 (Top '0' detector - 292% enrichment)")
create_feature_detail_plot(position=3, feature_id=1893, max_tokens=10)

print("\n" + "="*80)
print("VISUALIZATIONS COMPLETE!")
print("="*80)
print(f"\nSaved to: {VIZ_DIR}")
print("\nThese features are the most '0-obsessed' at their respective positions!")
print("  F148 (Pos 1): 270% enrichment → Early operation encoding")
print("  F1893 (Pos 3): 292% enrichment → Mid-calculation encoding")
