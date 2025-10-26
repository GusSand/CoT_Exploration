"""
Create an updated cross-position comparison for Position 0 (Token 0)
comparing baseline (800 problems) vs full dataset (7,473 problems).
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = BASE_DIR / "analysis"
VIZ_DIR = ANALYSIS_DIR / "visualizations"

# Load baseline metrics
baseline_path = ANALYSIS_DIR / "sae_training_summary.json"
with open(baseline_path) as f:
    baseline = json.load(f)

# Load full dataset metrics
full_data_path = ANALYSIS_DIR / "sae_training_results_full_data.json"
with open(full_data_path) as f:
    full_data = json.load(f)

# Extract Position 0 metrics
pos0_baseline = baseline["0"]["final_metrics"]
pos0_full = full_data["0"]

# Metrics to compare
metrics = {
    "Explained Variance": {
        "baseline": pos0_baseline["explained_variance"],
        "full": pos0_full["explained_variance"][0] if isinstance(pos0_full["explained_variance"], list) else pos0_full["explained_variance"]
    },
    "Validation Loss": {
        "baseline": pos0_baseline["test_loss"],
        "full": pos0_full["val_loss"][0] if isinstance(pos0_full["val_loss"], list) else pos0_full["val_loss"]
    },
    "Feature Death Rate": {
        "baseline": pos0_baseline["feature_death_rate"],
        "full": pos0_full["feature_death_rate"][0] if isinstance(pos0_full["feature_death_rate"], list) else pos0_full["feature_death_rate"]
    },
    "L0 Sparsity": {
        "baseline": pos0_baseline["l0_norm"],
        "full": pos0_full["l0_norm"][0] if isinstance(pos0_full["l0_norm"], list) else pos0_full["l0_norm"]
    }
}

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

metric_names = list(metrics.keys())
better_higher = [True, False, False, True]  # EV and L0 higher is better, Loss and Death Rate lower is better

for idx, (metric_name, better) in enumerate(zip(metric_names, better_higher)):
    ax = axes[idx]

    baseline_val = metrics[metric_name]["baseline"]
    full_val = metrics[metric_name]["full"]

    # Bar chart
    bars = ax.bar([0, 1], [baseline_val, full_val],
                   color=['#FF6B6B', '#4ECDC4'],
                   edgecolor='black', linewidth=2, width=0.5, alpha=0.8)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, [baseline_val, full_val])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Calculate improvement
    if better:
        improvement = ((full_val - baseline_val) / baseline_val) * 100
    else:
        improvement = -((full_val - baseline_val) / baseline_val) * 100

    # Add improvement annotation
    mid_x = 0.5
    y_max = max(baseline_val, full_val)
    y_min = min(baseline_val, full_val)
    mid_y = (y_max + y_min) / 2

    arrow_props = dict(arrowstyle='->', lw=2,
                      color='green' if improvement > 0 else 'red')
    ax.annotate(f'{improvement:+.1f}%',
                xy=(1, full_val), xytext=(mid_x, mid_y),
                fontsize=14, fontweight='bold',
                color='green' if improvement > 0 else 'red',
                ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor='green' if improvement > 0 else 'red', linewidth=2))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Baseline\n(800 problems)', 'Full Dataset\n(7,473 problems)'],
                       fontsize=11, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f'Position 0: {metric_name}', fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Position 0 (Token 0): Baseline vs Full Dataset Comparison',
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save
output_path = VIZ_DIR / "cross_position_token_0_comparison_updated.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

# Print summary
print("\n" + "="*80)
print("POSITION 0 (TOKEN 0) COMPARISON SUMMARY")
print("="*80)
for metric_name, better in zip(metric_names, better_higher):
    baseline_val = metrics[metric_name]["baseline"]
    full_val = metrics[metric_name]["full"]

    if better:
        improvement = ((full_val - baseline_val) / baseline_val) * 100
    else:
        improvement = -((full_val - baseline_val) / baseline_val) * 100

    print(f"\n{metric_name}:")
    print(f"  Baseline:    {baseline_val:.4f}")
    print(f"  Full Dataset: {full_val:.4f}")
    print(f"  Improvement: {improvement:+.1f}%")

print("\n" + "="*80)
