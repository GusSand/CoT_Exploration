"""
Visualize Top-K Projection Intervention Results
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

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))

# 1. Line plot: Accuracy vs K for all datasets
ax1 = plt.subplot(2, 3, (1, 2))
colors = {'clean': '#2ecc71', 'gsm8k_train': '#3498db', 'gsm8k_test': '#e74c3c'}
labels = {'clean': 'Clean Dataset', 'gsm8k_train': 'GSM8K Train', 'gsm8k_test': 'GSM8K Test'}

for dataset_name in ['clean', 'gsm8k_train', 'gsm8k_test']:
    k_vals = [x[0] for x in data[dataset_name]['proj']]
    acc_vals = [x[1] for x in data[dataset_name]['proj']]

    ax1.plot(k_vals, acc_vals, 'o-', linewidth=2.5, markersize=8,
             color=colors[dataset_name], label=labels[dataset_name], alpha=0.8)

    # Add baseline line
    baseline = data[dataset_name]['baseline']
    ax1.axhline(y=baseline, color=colors[dataset_name], linestyle='--',
                linewidth=1.5, alpha=0.5)

    # Add average ablation line
    average = data[dataset_name]['average']
    ax1.axhline(y=average, color=colors[dataset_name], linestyle=':',
                linewidth=1.5, alpha=0.3)

ax1.set_xlabel('K (Top-K Vocabulary Embeddings)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax1.set_title('Top-K Projection: Accuracy vs K\n(Dashed = Baseline, Dotted = Average Ablation)',
              fontsize=15, fontweight='bold')
ax1.legend(fontsize=11, loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')
ax1.set_xticks(k_values)
ax1.set_xticklabels([str(k) for k in k_values])
ax1.set_ylim([0, 100])

# 2. Bar chart: All conditions for each dataset
ax2 = plt.subplot(2, 3, 3)
x_pos = np.arange(len(k_values))
width = 0.25

clean_proj = [x[1] for x in data['clean']['proj']]
train_proj = [x[1] for x in data['gsm8k_train']['proj']]
test_proj = [x[1] for x in data['gsm8k_test']['proj']]

bars1 = ax2.bar(x_pos - width, clean_proj, width, label='Clean',
                color=colors['clean'], alpha=0.8)
bars2 = ax2.bar(x_pos, train_proj, width, label='GSM8K Train',
                color=colors['gsm8k_train'], alpha=0.8)
bars3 = ax2.bar(x_pos + width, test_proj, width, label='GSM8K Test',
                color=colors['gsm8k_test'], alpha=0.8)

ax2.set_xlabel('K Value', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Projection@K Accuracy by Dataset', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([str(k) for k in k_values], rotation=45)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim([0, 100])

# 3. Recovery percentage from baseline
ax3 = plt.subplot(2, 3, 4)
for dataset_name in ['clean', 'gsm8k_train', 'gsm8k_test']:
    baseline = data[dataset_name]['baseline']
    average = data[dataset_name]['average']
    k_vals = [x[0] for x in data[dataset_name]['proj']]

    # Recovery = (proj_acc - avg_acc) / (baseline - avg_acc) * 100
    recovery = []
    for k, acc in data[dataset_name]['proj']:
        rec = (acc - average) / (baseline - average) * 100
        recovery.append(rec)

    ax3.plot(k_vals, recovery, 'o-', linewidth=2.5, markersize=8,
             color=colors[dataset_name], label=labels[dataset_name], alpha=0.8)

ax3.axhline(y=100, color='black', linestyle='--', linewidth=1.5, alpha=0.5,
            label='Full Recovery')
ax3.set_xlabel('K (Top-K Vocabulary Embeddings)', fontsize=13, fontweight='bold')
ax3.set_ylabel('Recovery (%)', fontsize=13, fontweight='bold')
ax3.set_title('Performance Recovery from Average Ablation\n[(Proj - Avg) / (Baseline - Avg)] x 100%',
              fontsize=14, fontweight='bold')
ax3.legend(fontsize=11, loc='lower right')
ax3.grid(True, alpha=0.3)
ax3.set_xscale('log')
ax3.set_xticks(k_values)
ax3.set_xticklabels([str(k) for k in k_values])
ax3.set_ylim([0, 120])

# 4. Comparison: Baseline vs Average vs Proj@50
ax4 = plt.subplot(2, 3, 5)
datasets_list = ['Clean', 'GSM8K Train', 'GSM8K Test']
baseline_vals = [data['clean']['baseline'], data['gsm8k_train']['baseline'],
                 data['gsm8k_test']['baseline']]
average_vals = [data['clean']['average'], data['gsm8k_train']['average'],
                data['gsm8k_test']['average']]
proj50_vals = [data['clean']['proj'][-1][1], data['gsm8k_train']['proj'][-1][1],
               data['gsm8k_test']['proj'][-1][1]]

x_pos = np.arange(len(datasets_list))
width = 0.25

bars1 = ax4.bar(x_pos - width, baseline_vals, width, label='Baseline',
                color='#95a5a6', alpha=0.8)
bars2 = ax4.bar(x_pos, average_vals, width, label='Average Ablation',
                color='#e67e22', alpha=0.8)
bars3 = ax4.bar(x_pos + width, proj50_vals, width, label='Proj@50',
                color='#9b59b6', alpha=0.8)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax4.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax4.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax4.set_title('Intervention Comparison: Three Key Conditions', fontsize=14, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(datasets_list)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim([0, 100])

# 5. Gap from baseline
ax5 = plt.subplot(2, 3, 6)
for dataset_name in ['clean', 'gsm8k_train', 'gsm8k_test']:
    baseline = data[dataset_name]['baseline']
    k_vals = [x[0] for x in data[dataset_name]['proj']]
    gaps = [baseline - x[1] for x in data[dataset_name]['proj']]

    ax5.plot(k_vals, gaps, 'o-', linewidth=2.5, markersize=8,
             color=colors[dataset_name], label=labels[dataset_name], alpha=0.8)

ax5.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
ax5.set_xlabel('K (Top-K Vocabulary Embeddings)', fontsize=13, fontweight='bold')
ax5.set_ylabel('Gap from Baseline (%)', fontsize=13, fontweight='bold')
ax5.set_title('Remaining Performance Gap\n(Baseline - Proj@K)',
              fontsize=14, fontweight='bold')
ax5.legend(fontsize=11, loc='upper right')
ax5.grid(True, alpha=0.3)
ax5.set_xscale('log')
ax5.set_xticks(k_values)
ax5.set_xticklabels([str(k) for k in k_values])
ax5.set_ylim([0, 50])

plt.tight_layout()
plt.savefig('topk_projection_visualization.png', dpi=300, bbox_inches='tight')
print("Saved: topk_projection_visualization.png")

# Print detailed statistics
print("\n" + "="*80)
print("TOP-K PROJECTION INTERVENTION RESULTS")
print("="*80)

for dataset_name, label in [('clean', 'Clean Dataset'),
                             ('gsm8k_train', 'GSM8K Train'),
                             ('gsm8k_test', 'GSM8K Test')]:
    print(f"\n{label}:")
    print(f"  Baseline:         {data[dataset_name]['baseline']:.1f}%")
    print(f"  Average ablation: {data[dataset_name]['average']:.1f}% (delta {data[dataset_name]['average'] - data[dataset_name]['baseline']:+.1f}%)")
    print(f"  Proj@1:           {data[dataset_name]['proj'][0][1]:.1f}% (delta {data[dataset_name]['proj'][0][1] - data[dataset_name]['baseline']:+.1f}%)")
    print(f"  Proj@50:          {data[dataset_name]['proj'][-1][1]:.1f}% (delta {data[dataset_name]['proj'][-1][1] - data[dataset_name]['baseline']:+.1f}%)")

    baseline = data[dataset_name]['baseline']
    average = data[dataset_name]['average']
    proj50 = data[dataset_name]['proj'][-1][1]
    recovery = (proj50 - average) / (baseline - average) * 100
    print(f"  Recovery@50:      {recovery:.1f}% of baseline performance")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)
print("\n1. Average Ablation Impact:")
for dataset_name, label in [('clean', 'Clean'), ('gsm8k_train', 'Train'), ('gsm8k_test', 'Test')]:
    drop = data[dataset_name]['baseline'] - data[dataset_name]['average']
    print(f"   {label:12s}: -{drop:.1f} points ({drop/data[dataset_name]['baseline']*100:.0f}% loss)")

print("\n2. Projection@50 Recovery:")
for dataset_name, label in [('clean', 'Clean'), ('gsm8k_train', 'Train'), ('gsm8k_test', 'Test')]:
    baseline = data[dataset_name]['baseline']
    proj50 = data[dataset_name]['proj'][-1][1]
    recovery_pct = proj50 / baseline * 100
    print(f"   {label:12s}: {recovery_pct:.1f}% of baseline performance")

print("\n3. Remaining Gap at K=50:")
for dataset_name, label in [('clean', 'Clean'), ('gsm8k_train', 'Train'), ('gsm8k_test', 'Test')]:
    gap = data[dataset_name]['baseline'] - data[dataset_name]['proj'][-1][1]
    print(f"   {label:12s}: {gap:.1f} points still missing")

print("\n4. Improvement Trend:")
for dataset_name, label in [('clean', 'Clean'), ('gsm8k_train', 'Train'), ('gsm8k_test', 'Test')]:
    proj1 = data[dataset_name]['proj'][0][1]
    proj50 = data[dataset_name]['proj'][-1][1]
    improvement = proj50 - proj1
    print(f"   {label:12s}: +{improvement:.1f} points from K=1 to K=50")

print("\n" + "="*80)
