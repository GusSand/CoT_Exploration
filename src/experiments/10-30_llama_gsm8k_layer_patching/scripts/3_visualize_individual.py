"""
Story 5: Generate individual example heatmaps

This script creates a heatmap for each example showing the effect of
patching at each layer on KL divergence.
"""

import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add experiment directory to path
exp_dir = Path(__file__).parent.parent
sys.path.insert(0, str(exp_dir))

import config


def load_results():
    """Load patching results"""
    results_path = os.path.join(config.RESULTS_DIR, "patching_results.json")
    with open(results_path, 'r') as f:
        results = json.load(f)
    print(f"✓ Loaded results for {len(results)} pairs")
    return results


def create_individual_heatmap(result, output_dir):
    """
    Create heatmap for a single example

    Args:
        result: Result dict for one pair
        output_dir: Directory to save heatmap
    """
    pair_id = result['pair_id']
    clean_q = result['clean_question']
    corrupt_q = result['corrupted_question']
    clean_ans = result['clean_answer']
    corrupt_ans = result['corrupted_answer']

    # Extract KL divergences
    layers = [lr['layer'] for lr in result['layer_results']]
    kl_divs = [lr['kl_divergence'] for lr in result['layer_results']]

    # Create figure
    fig, ax = plt.subplots(figsize=(config.FIG_WIDTH, config.FIG_HEIGHT))

    # Plot KL divergence by layer
    ax.plot(layers, kl_divs, marker='o', linewidth=2, markersize=6)
    ax.fill_between(layers, 0, kl_divs, alpha=0.3)

    # Styling
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('KL Divergence', fontsize=12, fontweight='bold')
    ax.set_title(f'Pair {pair_id}: Layer-wise Patching Effect\n' +
                 f'Clean Answer: {clean_ans} | Corrupted Answer: {corrupt_ans}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, len(layers) - 0.5)
    ax.set_ylim(bottom=0)

    # Add question snippets as text
    clean_snippet = clean_q[:60] + "..." if len(clean_q) > 60 else clean_q
    corrupt_snippet = corrupt_q[:60] + "..." if len(corrupt_q) > 60 else corrupt_q

    fig.text(0.1, 0.02, f'Clean: {clean_snippet}', fontsize=8, style='italic')
    fig.text(0.1, 0.005, f'Corrupt: {corrupt_snippet}', fontsize=8, style='italic')

    # Highlight critical layer (max KL)
    max_layer = layers[np.argmax(kl_divs)]
    max_kl = max(kl_divs)
    ax.axvline(max_layer, color='red', linestyle='--', alpha=0.5, label=f'Critical Layer: {max_layer}')
    ax.legend()

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Save
    output_path = os.path.join(output_dir, f"pair_{pair_id:03d}_patching_effect.png")
    plt.savefig(output_path, dpi=config.DPI, bbox_inches='tight')
    plt.close()

    return output_path


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("STORY 5: Generate Individual Example Heatmaps")
    print("="*80 + "\n")

    # Load results
    results = load_results()

    # Create output directory
    output_dir = os.path.join(config.RESULTS_DIR, "individual_heatmaps")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating heatmaps for {len(results)} pairs...")

    # Generate heatmap for each pair
    for i, result in enumerate(results):
        output_path = create_individual_heatmap(result, output_dir)
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{len(results)} heatmaps")

    print(f"\n✓ Generated {len(results)} heatmaps")
    print(f"✓ Saved to: {output_dir}")

    print("\n" + "="*80)
    print("✓ INDIVIDUAL HEATMAPS COMPLETE")
    print("="*80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
