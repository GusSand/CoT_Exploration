"""
Create comprehensive visualizations comparing baseline vs full dataset SAE training.
Also re-analyze key features to see if they hold with the new models.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from collections import Counter

# Paths
BASE_DIR = Path(__file__).parent.parent
VIZ_DIR = BASE_DIR / "analysis" / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)

# Load baseline and full dataset training results
baseline_path = BASE_DIR / "results" / "sae_training_results.json"
full_data_path = BASE_DIR / "analysis" / "sae_training_results_full_data.json"

with open(baseline_path, 'r') as f:
    baseline = json.load(f)
with open(full_data_path, 'r') as f:
    full_data = json.load(f)

print("="*80)
print("CREATING COMPREHENSIVE COMPARISON VISUALIZATIONS")
print("="*80)

# ============================================================================
# 1. Training Curves Comparison (Baseline vs Full Dataset)
# ============================================================================
print("\n[1/4] Creating training curves comparison...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
positions = list(range(6))

for idx, position in enumerate(positions):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]

    pos_key = str(position)

    # Plot baseline
    if pos_key in baseline:
        epochs_baseline = list(range(1, len(baseline[pos_key]['explained_variance']) + 1))
        ax.plot(epochs_baseline, baseline[pos_key]['explained_variance'],
                'r-', alpha=0.7, linewidth=2, label='800 Problems')

    # Plot full dataset
    if pos_key in full_data:
        epochs_full = list(range(1, len(full_data[pos_key]['explained_variance']) + 1))
        ax.plot(epochs_full, full_data[pos_key]['explained_variance'],
                'g-', alpha=0.7, linewidth=2, label='7,473 Problems')

    # Add 70% threshold line
    ax.axhline(y=0.70, color='blue', linestyle='--', alpha=0.5, linewidth=1, label='70% Target')

    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Explained Variance', fontsize=10)
    ax.set_title(f'Position {position}', fontweight='bold', fontsize=12)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])

plt.suptitle('Explained Variance: Baseline vs Full Dataset Training',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
output_path = VIZ_DIR / "training_curves_comparison.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {output_path}")
plt.close()

# ============================================================================
# 2. Feature Death Rate Comparison
# ============================================================================
print("\n[2/4] Creating feature death rate comparison...")

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(positions))
width = 0.35

baseline_death = [baseline[str(p)]['feature_death_rate'][-1] for p in positions]
full_death = [full_data[str(p)]['feature_death_rate'][-1] for p in positions]

bars1 = ax.bar(x - width/2, baseline_death, width, label='800 Problems',
               color='salmon', alpha=0.8)
bars2 = ax.bar(x + width/2, full_death, width, label='7,473 Problems',
               color='lightgreen', alpha=0.8)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Position', fontsize=12)
ax.set_ylabel('Feature Death Rate', fontsize=12)
ax.set_title('Feature Death Rate Comparison\n(Lower is Better)',
             fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(positions)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
output_path = VIZ_DIR / "death_rate_comparison.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {output_path}")
plt.close()

# ============================================================================
# 3. Cross-Position Comparison (Updated)
# ============================================================================
print("\n[3/4] Creating cross-position comparison...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Metrics to compare
metrics = {
    'Explained Variance': ('explained_variance', 'higher'),
    'Validation Loss': ('val_loss', 'lower'),
    'Feature Death Rate': ('feature_death_rate', 'lower'),
    'L0 Sparsity': ('l0_norm', 'neutral')
}

for idx, (metric_name, (metric_key, direction)) in enumerate(metrics.items()):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]

    baseline_vals = []
    full_vals = []

    for p in positions:
        pos_key = str(p)
        if pos_key in baseline:
            if metric_key == 'val_loss':
                # Use test_loss for baseline
                baseline_vals.append(baseline[pos_key]['test_loss'][-1])
            else:
                baseline_vals.append(baseline[pos_key][metric_key][-1])
        else:
            baseline_vals.append(0)

        if pos_key in full_data:
            full_vals.append(full_data[pos_key][metric_key][-1])
        else:
            full_vals.append(0)

    x = np.arange(len(positions))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_vals, width, label='800 Problems',
                   color='lightcoral', alpha=0.8)
    bars2 = ax.bar(x + width/2, full_vals, width, label='7,473 Problems',
                   color='lightgreen', alpha=0.8)

    ax.set_xlabel('Position', fontsize=11)
    ax.set_ylabel(metric_name, fontsize=11)

    title = f'{metric_name}'
    if direction == 'higher':
        title += '\n(Higher is Better)'
    elif direction == 'lower':
        title += '\n(Lower is Better)'
    ax.set_title(title, fontweight='bold', fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Cross-Position Metrics: Baseline vs Full Dataset',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
output_path = VIZ_DIR / "cross_position_comparison_updated.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {output_path}")
plt.close()

# ============================================================================
# 4. Improvement Summary
# ============================================================================
print("\n[4/4] Creating improvement summary...")

fig, ax = plt.subplots(figsize=(12, 8))

improvements = []
for p in positions:
    pos_key = str(p)
    baseline_ev = baseline[pos_key]['explained_variance'][-1]
    full_ev = full_data[pos_key]['explained_variance'][-1]
    improvement = ((full_ev - baseline_ev) / baseline_ev) * 100
    improvements.append(improvement)

colors = ['red' if x < 0 else 'green' for x in improvements]
bars = ax.barh(positions, improvements, color=colors, alpha=0.7)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, improvements)):
    width = bar.get_width()
    label_x = width + (3 if width > 0 else -3)
    ax.text(label_x, bar.get_y() + bar.get_height()/2,
            f'{val:+.1f}%', ha='left' if width > 0 else 'right',
            va='center', fontweight='bold', fontsize=11)

ax.set_xlabel('Explained Variance Improvement (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Position', fontsize=12, fontweight='bold')
ax.set_title('Explained Variance Improvement: Full Dataset vs Baseline',
             fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.grid(axis='x', alpha=0.3)
ax.set_yticks(positions)

# Add average line
avg_improvement = np.mean(improvements)
ax.axvline(x=avg_improvement, color='blue', linestyle='--', linewidth=2,
           label=f'Average: {avg_improvement:+.1f}%')
ax.legend(fontsize=11)

plt.tight_layout()
output_path = VIZ_DIR / "improvement_summary.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {output_path}")
plt.close()

print("\n" + "="*80)
print("VISUALIZATION GENERATION COMPLETE!")
print("="*80)
print(f"\nGenerated files:")
print(f"  1. training_curves_comparison.png - Training curves across all positions")
print(f"  2. death_rate_comparison.png - Feature death rate improvements")
print(f"  3. cross_position_comparison_updated.png - All metrics comparison")
print(f"  4. improvement_summary.png - EV improvement summary")
print(f"\nAll visualizations saved to: {VIZ_DIR}")
