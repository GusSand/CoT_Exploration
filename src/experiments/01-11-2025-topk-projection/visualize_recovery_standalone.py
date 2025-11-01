"""
Standalone visualization: Performance Recovery from Average Ablation
"""

import json
import numpy as np
import matplotlib.pyplot as plt

# Load results from all three datasets
datasets = {
    'clean': 'full_results_clean_132_examples.json',
    'gsm8k_train': 'full_results_gsm8k_train_132_examples.json',
    'gsm8k_test': 'full_results_gsm8k_test_132_examples.json'
}

results = {}
for dataset_name, filename in datasets.items():
    with open(filename, 'r') as f:
        results[dataset_name] = json.load(f)

# Extract k values and accuracies
k_values = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]

data = {}
for dataset_name, dataset_results in results.items():
    data[dataset_name] = {
        'baseline': None,
        'average': None,
        'proj': []
    }

    for condition in dataset_results['conditions']:
        interv_type = condition['intervention_type']
        accuracy = condition['accuracy']

        if interv_type == 'baseline':
            data[dataset_name]['baseline'] = accuracy
        elif interv_type == 'average':
            data[dataset_name]['average'] = accuracy
        elif interv_type.startswith('proj'):
            k = int(interv_type[4:])
            data[dataset_name]['proj'].append((k, accuracy))

    # Sort by k
    data[dataset_name]['proj'].sort(key=lambda x: x[0])

# Create standalone recovery plot
fig, ax = plt.subplots(figsize=(14, 9))

colors = {'clean': '#2ecc71', 'gsm8k_train': '#3498db', 'gsm8k_test': '#e74c3c'}
labels = {'clean': 'Clean Dataset (Easy)', 'gsm8k_train': 'GSM8K Train', 'gsm8k_test': 'GSM8K Test'}

for dataset_name in ['clean', 'gsm8k_train', 'gsm8k_test']:
    baseline = data[dataset_name]['baseline']
    average = data[dataset_name]['average']
    k_vals = [x[0] for x in data[dataset_name]['proj']]

    # Recovery = (proj_acc - avg_acc) / (baseline - avg_acc) * 100
    recovery = []
    for k, acc in data[dataset_name]['proj']:
        rec = (acc - average) / (baseline - average) * 100
        recovery.append(rec)

    ax.plot(k_vals, recovery, 'o-', linewidth=3.5, markersize=10,
             color=colors[dataset_name], label=labels[dataset_name], alpha=0.85)

# Add 100% recovery reference line
ax.axhline(y=100, color='black', linestyle='--', linewidth=2.5, alpha=0.6,
            label='Full Recovery (= Baseline)')

# Add shaded region showing "good recovery" zone (>80%)
ax.axhspan(80, 120, alpha=0.1, color='green', label='Strong Recovery Zone (>80%)')

ax.set_xlabel('K (Number of Top Vocabulary Embeddings)', fontsize=16, fontweight='bold')
ax.set_ylabel('Performance Recovery (%)', fontsize=16, fontweight='bold')
ax.set_title('Recovery from Average Ablation via Top-K Projection\n[(Proj@K - Avg) / (Baseline - Avg)] × 100%',
              fontsize=18, fontweight='bold', pad=20)

ax.legend(fontsize=14, loc='lower right', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
ax.set_xscale('log')
ax.set_xticks(k_values)
ax.set_xticklabels([str(k) for k in k_values])
ax.tick_params(labelsize=13)
ax.set_ylim([0, 110])

# Add annotations for final k=50 values
for dataset_name in ['clean', 'gsm8k_train', 'gsm8k_test']:
    baseline = data[dataset_name]['baseline']
    average = data[dataset_name]['average']
    proj50_acc = data[dataset_name]['proj'][-1][1]
    recovery_50 = (proj50_acc - average) / (baseline - average) * 100

    # Annotate the final point
    ax.annotate(f'{recovery_50:.1f}%',
                xy=(50, recovery_50),
                xytext=(10, -5 if dataset_name == 'clean' else 5),
                textcoords='offset points',
                fontsize=12,
                fontweight='bold',
                color=colors[dataset_name],
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=colors[dataset_name], alpha=0.8))

plt.tight_layout()
plt.savefig('recovery_from_ablation_standalone.png', dpi=300, bbox_inches='tight')
print("Saved: recovery_from_ablation_standalone.png")

# Print recovery statistics
print("\n" + "="*80)
print("PERFORMANCE RECOVERY FROM AVERAGE ABLATION")
print("="*80)
print("\nRecovery Formula: (Proj@K - Average) / (Baseline - Average) × 100%")
print("\nThis measures how much of the performance lost to average ablation")
print("is recovered by projecting onto top-K vocabulary embeddings.")
print("\n" + "="*80)

for dataset_name, label in [('clean', 'Clean Dataset'),
                             ('gsm8k_train', 'GSM8K Train'),
                             ('gsm8k_test', 'GSM8K Test')]:
    baseline = data[dataset_name]['baseline']
    average = data[dataset_name]['average']
    loss = baseline - average

    print(f"\n{label}:")
    print(f"  Baseline:        {baseline:.1f}%")
    print(f"  Average Ablation: {average:.1f}%")
    print(f"  Loss:            {loss:.1f} points")
    print(f"\n  Recovery by K:")

    for k, acc in data[dataset_name]['proj']:
        rec = (acc - average) / (baseline - average) * 100
        recovered_points = acc - average
        print(f"    K={k:2d}:  {rec:5.1f}% ({recovered_points:+5.1f} of {loss:.1f} points)")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

print("\n1. Recovery Efficiency at K=50:")
for dataset_name, label in [('clean', 'Clean'), ('gsm8k_train', 'Train'), ('gsm8k_test', 'Test')]:
    baseline = data[dataset_name]['baseline']
    average = data[dataset_name]['average']
    proj50_acc = data[dataset_name]['proj'][-1][1]
    recovery = (proj50_acc - average) / (baseline - average) * 100
    print(f"   {label:12s}: {recovery:.1f}% recovery")

print("\n2. Plateau Analysis (K where recovery > 80%):")
for dataset_name, label in [('clean', 'Clean'), ('gsm8k_train', 'Train'), ('gsm8k_test', 'Test')]:
    baseline = data[dataset_name]['baseline']
    average = data[dataset_name]['average']

    for k, acc in data[dataset_name]['proj']:
        rec = (acc - average) / (baseline - average) * 100
        if rec >= 80:
            print(f"   {label:12s}: K={k} ({rec:.1f}% recovery)")
            break
    else:
        print(f"   {label:12s}: Never reaches 80%")

print("\n3. Diminishing Returns (improvement K=1 to K=50):")
for dataset_name, label in [('clean', 'Clean'), ('gsm8k_train', 'Train'), ('gsm8k_test', 'Test')]:
    baseline = data[dataset_name]['baseline']
    average = data[dataset_name]['average']

    proj1_acc = data[dataset_name]['proj'][0][1]
    proj50_acc = data[dataset_name]['proj'][-1][1]

    recovery_1 = (proj1_acc - average) / (baseline - average) * 100
    recovery_50 = (proj50_acc - average) / (baseline - average) * 100
    improvement = recovery_50 - recovery_1

    print(f"   {label:12s}: {recovery_1:.1f}% -> {recovery_50:.1f}% (+{improvement:.1f} points)")

print("\n" + "="*80)
