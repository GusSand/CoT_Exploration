#!/usr/bin/env python3
"""
Simple visualization of plus-one discretization results with average ablation baseline.
"""
import json
import matplotlib.pyplot as plt
import numpy as np

# Load plus-one results
with open('plusone_comparison_results.json', 'r') as f:
    plusone_results = json.load(f)

# Load full results for average ablation baseline
with open('intervention_comparison_results/full_results_clean_132_examples.json', 'r') as f:
    clean_full = json.load(f)
with open('intervention_comparison_results/full_results_gsm8k_train_132_examples.json', 'r') as f:
    train_full = json.load(f)
with open('intervention_comparison_results/full_results_gsm8k_test_132_examples.json', 'r') as f:
    test_full = json.load(f)

# Extract average ablation (numbers only) from full results
def get_average_accuracy(full_results):
    for condition in full_results['conditions']:
        if condition['intervention_type'] == 'average' and condition['intervention_scope'] == 'numbers':
            return condition['accuracy']
    return None

avg_clean = get_average_accuracy(clean_full)
avg_train = get_average_accuracy(train_full)
avg_test = get_average_accuracy(test_full)

# Prepare data
datasets = ['Clean', 'GSM8K Train', 'GSM8K Test']
x = np.arange(len(datasets))
width = 0.18

# Extract accuracies
baseline = [plusone_results['clean']['baseline']['accuracy'],
           plusone_results['gsm8k_train']['baseline']['accuracy'],
           plusone_results['gsm8k_test']['baseline']['accuracy']]

average_abl = [avg_clean, avg_train, avg_test]

discretize = [plusone_results['clean']['discretize']['accuracy'],
             plusone_results['gsm8k_train']['discretize']['accuracy'],
             plusone_results['gsm8k_test']['discretize']['accuracy']]

plusone = [plusone_results['clean']['discretize_plusone']['accuracy'],
          plusone_results['gsm8k_train']['discretize_plusone']['accuracy'],
          plusone_results['gsm8k_test']['discretize_plusone']['accuracy']]

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

# Create bars
bars1 = ax.bar(x - 1.5*width, baseline, width, label='Baseline',
              color='#27ae60', alpha=0.85, edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x - 0.5*width, average_abl, width, label='Average Ablation',
              color='#f39c12', alpha=0.85, edgecolor='black', linewidth=1.2)
bars3 = ax.bar(x + 0.5*width, discretize, width, label='Discretize',
              color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.2)
bars4 = ax.bar(x + 1.5*width, plusone, width, label='Discretize+1',
              color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.2)

# Add value labels on bars
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# Styling
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
ax.set_title('Intervention Comparison: Baseline vs Average vs Discretize vs Discretize+1',
            fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=13)
ax.legend(fontsize=12, loc='upper right', framealpha=0.95)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_ylim(0, 105)
ax.set_axisbelow(True)

# Add baseline reference lines
for i, (baseline_val, dataset_name) in enumerate(zip(baseline, datasets)):
    ax.axhline(y=baseline_val, xmin=(i-0.4)/len(datasets), xmax=(i+0.4)/len(datasets),
              color='gray', linestyle=':', linewidth=1.5, alpha=0.5)

plt.tight_layout()
plt.savefig('avg_intervention_simple.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: avg_intervention_simple.png")

# Print summary
print("\n" + "="*80)
print("INTERVENTION COMPARISON SUMMARY")
print("="*80)
print(f"{'Dataset':15s} {'Baseline':>10s} {'Avg Abl':>10s} {'Discretize':>10s} {'Disc+1':>10s}")
print("-"*80)
for i, ds in enumerate(datasets):
    print(f"{ds:15s} {baseline[i]:9.1f}% {average_abl[i]:9.1f}% {discretize[i]:9.1f}% {plusone[i]:9.1f}%")

print("\n" + "="*80)
print("DEGRADATION FROM BASELINE")
print("="*80)
print(f"{'Dataset':15s} {'Avg Abl':>10s} {'Discretize':>10s} {'Disc+1':>10s}")
print("-"*80)
for i, ds in enumerate(datasets):
    avg_drop = baseline[i] - average_abl[i]
    disc_drop = baseline[i] - discretize[i]
    plus_drop = baseline[i] - plusone[i]
    print(f"{ds:15s} {avg_drop:9.1f}% {disc_drop:9.1f}% {plus_drop:9.1f}%")

print("\nAverage degradation from baseline:")
print(f"  Average Ablation: -{np.mean([baseline[i] - average_abl[i] for i in range(3)]):.1f}%")
print(f"  Discretize:       -{np.mean([baseline[i] - discretize[i] for i in range(3)]):.1f}%")
print(f"  Discretize+1:     -{np.mean([baseline[i] - plusone[i] for i in range(3)]):.1f}%")
print("="*80)
