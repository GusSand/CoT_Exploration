"""
Create updated cross_position_token_0_comparison visualization using FULL DATASET models.
Same style as original but with full dataset feature analysis.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = BASE_DIR / "analysis"
VIZ_DIR = ANALYSIS_DIR / "visualizations"

# Load full dataset feature catalog
catalog_path = ANALYSIS_DIR / "feature_catalog_full_dataset.json"
with open(catalog_path) as f:
    catalog = json.load(f)

def create_cross_position_comparison(token="0", max_features=8):
    """
    Show how token '0' is detected across all 6 positions using FULL DATASET models.
    Matches the original visualization style.
    """
    print(f"\nCreating cross-position comparison for token '{token}' (Full Dataset Models)...")

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
            ax.set_title(f"Position {position}\n{len(token_detectors)} '{token}' detectors",
                        fontweight='bold')
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

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars1, enrichments)):
            if val > 10:  # Only show labels for significant values
                ax.text(bar.get_x() + bar.get_width()/2, val,
                       f'{int(val)}',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')

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

    plt.suptitle(f"How Token '{token}' is Detected Across All 6 Continuous Thought Positions\n"
                 f"(Blue bars = Enrichment %, Orange bars = Statistical Significance)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = VIZ_DIR / f"cross_position_token_{token}_comparison_updated.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    print("="*80)
    print("CREATING UPDATED CROSS-POSITION TOKEN '0' COMPARISON")
    print("Using FULL DATASET models (7,473 problems)")
    print("="*80)

    create_cross_position_comparison(token="0", max_features=8)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
