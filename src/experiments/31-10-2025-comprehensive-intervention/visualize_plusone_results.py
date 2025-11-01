#!/usr/bin/env python3
"""
Visualize plus-one discretization intervention results.
Compares: Baseline vs Discretize vs Discretize+1
"""
import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('plusone_comparison_results.json', 'r') as f:
    results = json.load(f)

# Extract data
datasets = ['clean', 'gsm8k_train', 'gsm8k_test']
dataset_labels = ['Clean\n(90.2% baseline)', 'GSM8K Train\n(86.4% baseline)', 'GSM8K Test\n(54.5% baseline)']
interventions = ['baseline', 'discretize', 'discretize_plusone']
intervention_labels = ['Baseline', 'Discretize', 'Discretize+1']

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ===== LEFT PLOT: Bar chart comparison =====
x = np.arange(len(datasets))
width = 0.25

baseline_accs = [results[ds]['baseline']['accuracy'] for ds in datasets]
discretize_accs = [results[ds]['discretize']['accuracy'] for ds in datasets]
plusone_accs = [results[ds]['discretize_plusone']['accuracy'] for ds in datasets]

bars1 = ax1.bar(x - width, baseline_accs, width, label='Baseline', color='#2ecc71', alpha=0.8)
bars2 = ax1.bar(x, discretize_accs, width, label='Discretize', color='#3498db', alpha=0.8)
bars3 = ax1.bar(x + width, plusone_accs, width, label='Discretize+1', color='#e74c3c', alpha=0.8)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Plus-One Discretization Intervention Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(dataset_labels, fontsize=10)
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, 100)

# ===== RIGHT PLOT: Performance degradation =====
baseline_ref = [100, 100, 100]  # Baseline as 100%
discretize_rel = [(results[ds]['discretize']['accuracy'] / results[ds]['baseline']['accuracy'] * 100)
                  for ds in datasets]
plusone_rel = [(results[ds]['discretize_plusone']['accuracy'] / results[ds]['baseline']['accuracy'] * 100)
               for ds in datasets]

bars_base = ax2.bar(x - width/2, baseline_ref, width, label='Baseline (100%)', color='#2ecc71', alpha=0.8)
bars_disc = ax2.bar(x + width/2, discretize_rel, width, label='Discretize', color='#3498db', alpha=0.8)
bars_plus = ax2.bar(x + width*1.5, plusone_rel, width, label='Discretize+1', color='#e74c3c', alpha=0.8)

# Add value labels
for bars, values in [(bars_disc, discretize_rel), (bars_plus, plusone_rel)]:
    for bar, val in zip(bars, values):
        height = bar.get_height()
        drop = 100 - val
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%\n({drop:.1f}% drop)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax2.set_ylabel('Relative Performance (% of Baseline)', fontsize=12, fontweight='bold')
ax2.set_title('Performance Degradation vs Baseline', fontsize=14, fontweight='bold')
ax2.set_xticks(x + width/2)
ax2.set_xticklabels(dataset_labels, fontsize=10)
ax2.legend(fontsize=11, loc='lower right')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim(0, 110)
ax2.axhline(y=100, color='black', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig('plusone_intervention_comparison.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: plusone_intervention_comparison.png")

# ===== Create detailed comparison table =====
fig2, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')

# Prepare table data
table_data = []
table_data.append(['Dataset', 'Baseline', 'Discretize', 'Δ from Baseline', 'Discretize+1', 'Δ from Baseline'])

for ds, ds_label in zip(datasets, ['Clean', 'GSM8K Train', 'GSM8K Test']):
    baseline = results[ds]['baseline']['accuracy']
    disc = results[ds]['discretize']['accuracy']
    plus = results[ds]['discretize_plusone']['accuracy']

    disc_delta = disc - baseline
    plus_delta = plus - baseline

    table_data.append([
        ds_label,
        f"{baseline:.1f}%",
        f"{disc:.1f}%",
        f"{disc_delta:+.1f}%",
        f"{plus:.1f}%",
        f"{plus_delta:+.1f}%"
    ])

# Add summary statistics
table_data.append(['', '', '', '', '', ''])
table_data.append(['Average Drop', '',
                  f"{np.mean([results[ds]['discretize']['accuracy'] - results[ds]['baseline']['accuracy'] for ds in datasets]):.1f}%", '',
                  f"{np.mean([results[ds]['discretize_plusone']['accuracy'] - results[ds]['baseline']['accuracy'] for ds in datasets]):.1f}%", ''])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header row
for i in range(6):
    cell = table[(0, i)]
    cell.set_facecolor('#34495e')
    cell.set_text_props(weight='bold', color='white')

# Style data rows
for i in range(1, 4):
    for j in range(6):
        cell = table[(i, j)]
        if j == 1:  # Baseline column
            cell.set_facecolor('#d5f4e6')
        elif j == 2 or j == 4:  # Intervention columns
            cell.set_facecolor('#ebf5fb')
        elif j == 3 or j == 5:  # Delta columns
            cell.set_facecolor('#fadbd8')

# Style summary row
for j in range(6):
    cell = table[(5, j)]
    cell.set_facecolor('#f8f9f9')
    cell.set_text_props(weight='bold')

plt.title('Plus-One Discretization: Detailed Comparison',
         fontsize=16, fontweight='bold', pad=20)
plt.savefig('plusone_intervention_table.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: plusone_intervention_table.png")

# Print summary statistics
print("\n" + "="*80)
print("PLUS-ONE DISCRETIZATION ANALYSIS")
print("="*80)

print("\n1. ABSOLUTE ACCURACIES:")
print("-" * 80)
for ds, ds_label in zip(datasets, ['Clean', 'GSM8K Train', 'GSM8K Test']):
    baseline = results[ds]['baseline']['accuracy']
    disc = results[ds]['discretize']['accuracy']
    plus = results[ds]['discretize_plusone']['accuracy']
    print(f"{ds_label:15s} | Baseline: {baseline:5.1f}% | Discretize: {disc:5.1f}% | Discretize+1: {plus:5.1f}%")

print("\n2. PERFORMANCE DEGRADATION (from baseline):")
print("-" * 80)
for ds, ds_label in zip(datasets, ['Clean', 'GSM8K Train', 'GSM8K Test']):
    baseline = results[ds]['baseline']['accuracy']
    disc = results[ds]['discretize']['accuracy']
    plus = results[ds]['discretize_plusone']['accuracy']

    disc_drop = baseline - disc
    plus_drop = baseline - plus
    disc_rel = disc / baseline * 100
    plus_rel = plus / baseline * 100

    print(f"{ds_label:15s} | Discretize: -{disc_drop:5.1f}% ({disc_rel:5.1f}% of baseline) | "
          f"Discretize+1: -{plus_drop:5.1f}% ({plus_rel:5.1f}% of baseline)")

print("\n3. DISCRETIZE+1 vs DISCRETIZE:")
print("-" * 80)
for ds, ds_label in zip(datasets, ['Clean', 'GSM8K Train', 'GSM8K Test']):
    disc = results[ds]['discretize']['accuracy']
    plus = results[ds]['discretize_plusone']['accuracy']

    diff = plus - disc
    rel = plus / disc * 100

    print(f"{ds_label:15s} | Discretize+1 is {diff:+.1f}% ({rel:.1f}% of Discretize performance)")

print("\n4. KEY FINDINGS:")
print("-" * 80)
avg_disc_drop = np.mean([results[ds]['baseline']['accuracy'] - results[ds]['discretize']['accuracy'] for ds in datasets])
avg_plus_drop = np.mean([results[ds]['baseline']['accuracy'] - results[ds]['discretize_plusone']['accuracy'] for ds in datasets])
additional_drop = avg_plus_drop - avg_disc_drop

print(f"• Average discretize drop from baseline: -{avg_disc_drop:.1f}%")
print(f"• Average discretize+1 drop from baseline: -{avg_plus_drop:.1f}%")
print(f"• Additional degradation from +1 shift: -{additional_drop:.1f}%")
print(f"\nDiscretize+1 is ~{(avg_plus_drop / avg_disc_drop):.2f}x worse than regular discretize")

print("\n" + "="*80)
