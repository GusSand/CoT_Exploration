"""
Flattened visualization of Average Activation Intervention
Per-position metrics: next-position impact, downstream average impact, number decoding rate
"""

import json
import numpy as np
import matplotlib.pyplot as plt

# Load average activation results
with open("avg_intervention_cascade_statistics.json", 'r') as f:
    statistics = json.load(f)

# Load raw data to compute number decoding rate
with open("avg_intervention_cascade_raw.json", 'r') as f:
    raw_data = json.load(f)

num_positions = 7

# Extract token change matrix
token_change = np.zeros((num_positions, num_positions))

for int_stat in statistics['position_statistics']:
    int_pos = int_stat['intervention_position']
    for effect in int_stat['downstream_effects']:
        target_pos = effect['target_position']
        token_change[int_pos, target_pos] = effect['token_change_rate'] * 100

# Compute metrics for each position
next_position_impact = []
downstream_avg_impact = []
number_decode_rate = []

import re
number_regex = r'^\s?\d+'

for pos in range(num_positions):
    # 1. Impact on next position
    if pos < num_positions - 1:
        next_impact = token_change[pos, pos + 1]
    else:
        next_impact = 0  # No next position for position 6
    next_position_impact.append(next_impact)

    # 2. Average impact on all following positions
    if pos < num_positions - 1:
        following_impacts = token_change[pos, pos+1:]
        avg_downstream = np.mean(following_impacts)
    else:
        avg_downstream = 0  # No following positions for position 6
    downstream_avg_impact.append(avg_downstream)

    # 3. Number decoding rate at this position (baseline - no intervention)
    number_count = 0
    total_count = 0
    for example in raw_data:
        # Find baseline run (intervention_position = -1 or no intervention)
        baseline_intervention = example['interventions'][0]  # First is baseline with int_pos=-1
        pos_metric = baseline_intervention['position_metrics'][pos]
        token = pos_metric['baseline_token']
        is_number = bool(re.match(number_regex, token))
        if is_number:
            number_count += 1
        total_count += 1

    number_decode_rate.append((number_count / total_count) * 100)

# Create bar chart
fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(num_positions)
width = 0.25

bars1 = ax.bar(x - width, next_position_impact, width, label='Next Position Impact (%)',
               color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x, downstream_avg_impact, width, label='Avg Downstream Impact (%)',
               color='#3498db', alpha=0.8)
bars3 = ax.bar(x + width, number_decode_rate, width, label='Number Decoding Rate (%)',
               color='#2ecc71', alpha=0.8)

ax.set_xlabel('CoT Position', fontsize=14, fontweight='bold')
ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
ax.set_title('Per-Position Intervention Metrics\nAverage Activation Intervention (20 Examples)',
             fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'P{i}' for i in range(num_positions)])
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 100])

# Add value labels on bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{height:.0f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

plt.tight_layout()
plt.savefig('avg_intervention_flat.png', dpi=300, bbox_inches='tight')
print("Saved: avg_intervention_flat.png")

# Print detailed statistics
print("\n" + "="*80)
print("PER-POSITION INTERVENTION METRICS")
print("="*80)
print(f"\n{'Pos':<5} {'Next Impact':<15} {'Avg Downstream':<18} {'Number Decode':<15}")
print("-" * 80)

for i in range(num_positions):
    print(f"{i:<5} {next_position_impact[i]:>12.1f}%  {downstream_avg_impact[i]:>15.1f}%  {number_decode_rate[i]:>13.1f}%")

print("\n" + "="*80)
print("KEY OBSERVATIONS")
print("="*80)

# Find positions with high next-position impact
print("\nStrongest Next-Position Cascades (>40%):")
for i in range(num_positions):
    if next_position_impact[i] > 40:
        print(f"  Position {i} -> {i+1}: {next_position_impact[i]:.0f}% token change")

# Find positions with high downstream impact
print("\nStrongest Average Downstream Impact (>20%):")
for i in range(num_positions):
    if downstream_avg_impact[i] > 20:
        print(f"  Position {i}: {downstream_avg_impact[i]:.0f}% average impact on following positions")

# Identify position types by number decoding rate
print("\nPosition Types (by number decoding):")
print("  Operator positions (<20% numbers):", end=" ")
operator_positions = [i for i in range(num_positions) if number_decode_rate[i] < 20]
print(operator_positions if operator_positions else "None")

print("  Arithmetic positions (>80% numbers):", end=" ")
arithmetic_positions = [i for i in range(num_positions) if number_decode_rate[i] > 80]
print(arithmetic_positions if arithmetic_positions else "None")

print("  Mixed positions (20-80% numbers):", end=" ")
mixed_positions = [i for i in range(num_positions) if 20 <= number_decode_rate[i] <= 80]
print(mixed_positions if mixed_positions else "None")

print("\n" + "="*80)
